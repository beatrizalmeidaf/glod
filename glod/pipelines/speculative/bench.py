#!/usr/bin/env python3
"""Experimento D: escolher o draft da decodificacao especulativa SO pelo KL, e medir o relogio.

Etapa 1 (previsao, sem instanciar o sistema): para cada draft candidato
(quantizacoes/podas do alvo e modelos menores da familia), le do estudo de
fidelidade o KL e o TV por token contra o alvo e preve:
  aceitacao por amostragem     a_s = 1 - TV                  (identidade exata)
  aceitacao greedy por posicao a_g = 1 - kappa*sqrt(KL)      (lei; kappa do alvo)
  tokens por rodada (k)        E = (1 - a^(k+1)) / (1 - a)
  speedup previsto             E / (k*c + 1),  c = custo relativo do draft (bytes de peso)

Etapa 2 (medicao): decodificacao especulativa real, lote 1, com
torch.cuda.synchronize nos tempos; mede aceitacao, tokens/rodada e tokens/s
contra o alvo sozinho no mesmo hardware e nos mesmos prompts.

Etapa 3: ranking previsto x medido (Spearman) e R2 aceitacao prevista x medida.

    python -m glod spec-bench --target Qwen/Qwen3-14B --device cuda:0 \\
        --drafts Qwen/Qwen3-14B:u4 Qwen/Qwen3-14B:u3 Qwen/Qwen3-4B:raw Qwen/Qwen3-8B:raw
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

import numpy as np
import torch

from glod.core import compressors as C
from glod.core.elastic_depth import make_elastic_depth
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import _eos_ids
from glod.pipelines.tasks.closedloop import gsm8k_disjoint, speculative
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, slug
from glod.pipelines.fidelity.build_grid import apply_config

LOGGER = logging.getLogger("glod.spec")


def predict(target: str, draft_model: str, cfg: str, k: int) -> dict:
    ts, ds = slug(target), slug(draft_model)
    f = OUT / ts / ds / f"{cfg}.pt"
    ref = torch.load(OUT / ts / ts / "bf16.pt", map_location="cpu", mmap=True, weights_only=False)
    c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
    kl = c["kl"].double().mean().item()
    tv = c["tv"].double().mean().item()
    law = json.loads((OUT / "analysis" / "law.json").read_text())
    pts = [r for r in law["rows"] if r["ref"] == ts and r["model"] == ts and 1e-3 < r["kl"] < 0.2
           and r["family"] != "outro modelo"]
    kappa = float(np.median([r["flip"] / math.sqrt(r["kl"]) for r in pts]))
    a_g = max(0.0, 1 - kappa * math.sqrt(kl))
    a_s = 1 - tv
    tpr = lambda a: (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1
    # previsao 2 (ainda sem instanciar o sistema): simula rodadas sobre a sequencia REAL de
    # concordancia do teacher forcing no corpus do alvo - captura as rajadas que o modelo
    # geometrico i.i.d. ignora (a aceitacao e autocorrelacionada)
    corpus = json.loads((OUT / "corpora" / f"{ts}.json").read_text())
    agree = (c["top1"] == ref["top1"]).tolist()
    emitted = rounds = 0
    off = 0
    for g in corpus["gen_ids"]:
        n = len(g)
        seqf = agree[off:off + n]
        off += n
        t = 0
        while t < n:
            acc = 0
            while acc < min(k, n - t) and seqf[t + acc]:
                acc += 1
            step = min(acc + 1, n - t)
            emitted += step
            rounds += 1
            t += step
    tpr_sim = emitted / max(rounds, 1)
    return {"kl": kl, "tv": tv, "kappa_target": kappa, "pred_accept_greedy": a_g, "pred_accept_sample": a_s,
            "pred_tokens_per_round_greedy_tfsim": tpr_sim,
            "pred_tokens_per_round_greedy": tpr(a_g), "pred_tokens_per_round_sample": tpr(a_s),
            "flip_measured_tf": (c["top1"] != ref["top1"]).double().mean().item()}


@torch.no_grad()
def time_target_only(target, prompts, max_new: int, eos) -> tuple[float, int]:
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for ids in prompts:
        out = target.model.generate(torch.tensor([ids], device=target.device), max_new_tokens=max_new,
                                    do_sample=False, top_p=None, top_k=None,
                                    pad_token_id=target.tokenizer.pad_token_id)
        gen = out[0, len(ids):].tolist()
        cut = next((i + 1 for i, t in enumerate(gen) if t in eos), len(gen))
        n += cut
    torch.cuda.synchronize()
    return time.time() - t0, n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="Qwen/Qwen3-14B")
    p.add_argument("--drafts", nargs="+", required=True, help="modelo:config (config do estudo de fidelidade)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--calib-batch", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    # synchronize() sem device cria contexto na GPU 0, que pode estar cheia (GPUs compartilhadas)
    torch.cuda.set_device(torch.device(args.device))
    out_dir = OUT / "spec_bench" / slug(args.target)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = load_model(args.target, role="ref", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    eos = _eos_ids(target)
    corpus = json.loads((OUT / "corpora" / f"{slug(args.target)}.json").read_text())
    used = {r["prompt"] for r in corpus["records"]}
    recs = gsm8k_disjoint(target.tokenizer, args.n, used, args.cache_dir, seed=9001)
    prompts = [target.tokenizer(r["prompt"], add_special_tokens=False).input_ids for r in recs]
    # aquecimento
    time_target_only(target, prompts[:2], 16, eos)
    t_base, n_base = time_target_only(target, prompts, args.max_new_tokens, eos)
    base_tps = n_base / t_base
    LOGGER.info("alvo sozinho: %.2f tokens/s (%d tokens)", base_tps, n_base)
    target_bytes = sum(p_.numel() * p_.element_size() for p_ in target.model.parameters())

    prev = out_dir / "results.json"
    results = json.loads(prev.read_text()) if prev.exists() else {"drafts": {}}
    results.update({"target": args.target, "target_tokens_per_s": base_tps})
    for spec in args.drafts:
        if spec in results["drafts"] and "skipped" not in results["drafts"][spec]:
            LOGGER.info("[%s] ja medido, pulando", spec)
            continue
        try:
            run_draft(spec, args, target, eos, prompts, base_tps, target_bytes, results, out_dir)
        except torch.OutOfMemoryError as exc:  # GPUs compartilhadas: registra e segue
            LOGGER.warning("[%s] pulado por falta de memoria: %s", spec, str(exc)[:120])
            results["drafts"][spec] = {"skipped": "oom"}
            torch.cuda.empty_cache()
    _summaries(results)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return 0


def run_draft(spec, args, target, eos, prompts, base_tps, target_bytes, results, out_dir) -> None:
    if True:
        dmodel, cfg = spec.split(":")
        pred = predict(args.target, dmodel, cfg, args.k)
        draft = load_model(dmodel, role="draft", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
        bank = C.WeightBank(draft.decoder)
        ctrl = make_elastic_depth(draft.model)
        calib: dict = {}

        def calib_fn():
            if "x" not in calib:
                calib["x"] = C.wikitext_calibration(draft.tokenizer, 128, 512, cache_dir=f"{args.cache_dir}/datasets")
            return calib["x"]

        with apply_config("bf16" if cfg == "raw" else cfg, draft, bank, ctrl, calib_fn, args):
            row = {"prediction": pred}
            for mode in ("greedy", "sample"):
                gen = torch.Generator(device=args.device).manual_seed(0)
                speculative(target, draft, prompts[0], k=args.k, max_new=16, eos=eos, sampling=mode == "sample",
                            gen=gen)  # aquecimento
                torch.cuda.synchronize()
                t0 = time.time()
                tot = {"proposed": 0, "accepted": 0, "rounds": 0, "new_tokens": 0, "probs": []}
                for ids in prompts:
                    o = speculative(target, draft, ids, k=args.k, max_new=args.max_new_tokens, eos=eos,
                                    sampling=mode == "sample", gen=gen)
                    for key in ("proposed", "accepted", "rounds", "new_tokens"):
                        tot[key] += o[key]
                    tot["probs"] += o["accept_probs"]
                torch.cuda.synchronize()
                dt = time.time() - t0
                row[mode] = {"accept_rate": tot["accepted"] / tot["proposed"],
                             "tokens_per_round": tot["new_tokens"] / tot["rounds"],
                             "mean_accept_prob": float(np.mean(tot["probs"])) if tot["probs"] else None,
                             "tokens_per_s": tot["new_tokens"] / dt,
                             "speedup_wallclock": (tot["new_tokens"] / dt) / base_tps}
                LOGGER.info("[%s | %s] tokens/rodada %.3f (prev %.3f) | speedup real %.2fx", spec, mode,
                            row[mode]["tokens_per_round"], pred[f"pred_tokens_per_round_{mode}"],
                            row[mode]["speedup_wallclock"])
        row["draft_bytes_ratio"] = sum(p_.numel() * p_.element_size() for p_ in draft.model.parameters()) / target_bytes
        results["drafts"][spec] = row
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))
        del draft, bank
        torch.cuda.empty_cache()


def _summaries(results) -> None:
    names = [n for n, r in results["drafts"].items() if "skipped" not in r]
    if len(names) >= 3:
        from scipy.stats import spearmanr
        for mode in ("greedy", "sample"):
            pr = [results["drafts"][n]["prediction"][f"pred_tokens_per_round_{mode}"] for n in names]
            me = [results["drafts"][n][mode]["tokens_per_round"] for n in names]
            sp = [results["drafts"][n][mode]["speedup_wallclock"] for n in names]
            ss = 1 - np.sum((np.array(me) - np.array(pr)) ** 2) / np.sum((np.array(me) - np.mean(me)) ** 2)
            LOGGER.info("%s: R2 tokens/rodada previsto (so KL, iid) x medido %.3f | Spearman(previsto, speedup real) %.2f",
                        mode, ss, spearmanr(pr, sp)[0])
        pr2 = [results["drafts"][n]["prediction"]["pred_tokens_per_round_greedy_tfsim"] for n in names]
        me2 = [results["drafts"][n]["greedy"]["tokens_per_round"] for n in names]
        ss2 = 1 - np.sum((np.array(me2) - np.array(pr2)) ** 2) / np.sum((np.array(me2) - np.mean(me2)) ** 2)
        LOGGER.info("greedy: R2 tokens/rodada previsto (simulacao TF, com rajadas) x medido %.3f | Spearman %.2f",
                    ss2, spearmanr(pr2, me2)[0])


if __name__ == "__main__":
    raise SystemExit(main())
