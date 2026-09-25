#!/usr/bin/env python3
"""Experimento principal: metodos de compressao comparados com KL CASADO.

Desenho. Cada metodo M produz uma perturbacao dW_M = W_M - W0 (rodado uma vez
por modelo). A intensidade e variada por um fator continuo alpha:

    W(alpha) = W0 + alpha * dW_M

alpha e ajustado por bisseccao para que o KL por token atinja um alvo comum
(0.02 / 0.05 / 0.10 / 0.20). Isso separa a DIRECAO do erro (identidade do
metodo) da sua MAGNITUDE (o KL), que e a pergunta da tese. Os pontos nativos
(alpha = 1) tambem sao avaliados como ancoras deployaveis.

Depois, com o KL casado, cada configuracao e avaliada em tarefa (GSM8K e
MMLU-PT em geracao, prompts disjuntos do corpus de calibracao). A analise
(`--stage tost`) faz teste de EQUIVALENCIA (TOST) nas diferencas pareadas por
modelo entre familias, com margem declarada.

    python -m glod matched-kl run  --model Qwen/Qwen3-4B --device cuda:0
    python -m glod matched-kl tost --margin 0.03
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch

from glod.core import compressors as C
from glod.core.elastic_depth import make_elastic_depth
from glod.core.fidelity import score_fidelity, subset_index
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import GeneratedCorpus, generate_greedy
from glod.pipelines.tasks.closedloop import gsm8k_disjoint, parse_answer
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, slug
from glod.pipelines.fidelity.build_grid import apply_config
from glod.corpora.token_oracle import build_prompts_mmlu_pt
from glod.paths import MMLU_PT_CSV

LOGGER = logging.getLogger("glod.matched")
# KV cache fica de fora: e um hook em tempo de execucao, nao uma perturbacao de pesos,
# entao W0 + alpha*dW com dW = 0 nao a escala (bug real observado: alpha=8 -> KL 0.0004)
METHODS = ("u4", "gptq4", "awq4", "sgpt50", "wanda50", "mag40", "g4")


def small_corpus(corpus: GeneratedCorpus, n_seq: int) -> tuple[GeneratedCorpus, torch.Tensor]:
    """Sub-corpus com passo fixo: o corpus vem ordenado (GSM8K e depois MMLU-PT),
    entao pegar as primeiras sequencias casaria o alpha em um dominio so."""
    total = len(corpus.gen_ids)
    idx = list(range(0, total, max(total // n_seq, 1)))[:n_seq]
    offs = [0]
    for g in corpus.gen_ids:
        offs.append(offs[-1] + len(g))
    keep = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in idx])
    return (GeneratedCorpus(prompt_ids=[corpus.prompt_ids[i] for i in idx],
                            gen_ids=[corpus.gen_ids[i] for i in idx],
                            records=[corpus.records[i] for i in idx]), keep)


@torch.no_grad()
def apply_scaled(bank: C.WeightBank, deltas: dict[str, torch.Tensor], alpha: float) -> None:
    for t in bank.targets:
        w = t.original.to(t.module.weight.device).float()
        w += alpha * deltas[t.name].to(w.device).float()
        t.module.weight.copy_(w.to(t.module.weight.dtype))


def kl_of(loaded, corpus, ref, batch_size) -> float:
    res = score_fidelity(loaded, corpus, ref=ref, batch_size=batch_size, subset=None)
    return res["kl"].double().mean().item()


def stage_run(args) -> None:
    out_dir = OUT / "matched_kl" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text()))
    ref_full = torch.load(OUT / slug(args.model) / slug(args.model) / "bf16.pt")
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank = C.WeightBank(loaded.decoder)
    ctrl = make_elastic_depth(loaded.model)
    ctrl.allow_kv_cache = True
    probe, keep = small_corpus(corpus, args.probe_seqs)
    # a ordem do score_fidelity segue a ordem das sequencias do corpus dado, logo
    # basta recortar as linhas correspondentes da referencia
    ref_probe = {k: ref_full[k][keep] for k in ("top1", "topv", "topi")}
    calib_cache: dict = {}

    def calib_fn():
        if "x" not in calib_cache:
            calib_cache["x"] = C.wikitext_calibration(loaded.tokenizer, 128, 512,
                                                      cache_dir=f"{args.cache_dir}/datasets")
        return calib_cache["x"]

    used = {r["prompt"] for r in corpus.records}
    gsm = gsm8k_disjoint(loaded.tokenizer, args.n_task, used, args.cache_dir)
    mmlu = [r for r in build_prompts_mmlu_pt(loaded.tokenizer, MMLU_PT_CSV, 4 * args.n_task, seed=777)
            if r["prompt"] not in used][:args.n_task]

    def task_eval(tag: str) -> dict:
        recs = [dict(r) for r in gsm + mmlu]
        gen = generate_greedy(loaded, recs, max_new_tokens=args.max_new_tokens, batch_size=args.batch)
        out = {}
        for src in ("gsm8k", "mmlu_pt"):
            rows = [r for r in gen.records if r["source"] == src]
            if src == "gsm8k":
                ok = [parse_answer(r["completion"]) is not None
                      and abs(parse_answer(r["completion"]) - float(r["gold"])) < 1e-6 for r in rows]
            else:
                import re
                ok = []
                for r in rows:
                    m = re.findall(r"Resposta:\s*\**\(?([ABCD])", r["completion"])
                    ok.append(bool(m) and m[-1] == r["gold"])
            out[src] = {"acc": sum(ok) / len(ok), "n": len(ok),
                        "correct": [bool(x) for x in ok]}
        LOGGER.info("   [%s] gsm8k %.4f | mmlu_pt %.4f", tag, out["gsm8k"]["acc"], out["mmlu_pt"]["acc"])
        return out

    # baseline denso
    base_file = out_dir / "base.json"
    if not base_file.exists():
        bank.restore()
        base_file.write_text(json.dumps({"config": "bf16", "alpha": 0.0, "kl": 0.0, **task_eval("bf16")}))

    for method in args.methods:
        dfile = out_dir / f"{method}.json"
        if dfile.exists() and not args.overwrite:
            LOGGER.info("%s existe", method)
            continue
        t0 = time.time()
        with apply_config(method, loaded, bank, ctrl, calib_fn, args):
            deltas = {t.name: (t.module.weight.detach().float() - t.original.to(t.module.weight.device).float()
                               ).to("cpu", torch.float16) for t in bank.targets}
            native_kl = kl_of(loaded, probe, ref_probe, args.score_batch)
            native = {"alpha": 1.0, "kl_probe": native_kl}
            if not args.skip_native_task:
                native.update(task_eval(f"{method} nativo"))
        rows = {"method": method, "native": native, "seconds_setup": time.time() - t0, "targets": {}}
        for target in args.targets:
            # bracket com kl(lo) < alvo <= kl(hi); alpha pode passar de 1 se o
            # metodo nativo perturba menos que o alvo
            lo, hi = 0.0, 1.0
            apply_scaled(bank, deltas, hi)
            while kl_of(loaded, probe, ref_probe, args.score_batch) < target and hi < 8:
                lo, hi = hi, hi * 2
                apply_scaled(bank, deltas, hi)
            for _ in range(args.bisect):
                mid = 0.5 * (lo + hi)
                apply_scaled(bank, deltas, mid)
                if kl_of(loaded, probe, ref_probe, args.score_batch) < target:
                    lo = mid
                else:
                    hi = mid
            alpha = 0.5 * (lo + hi)
            apply_scaled(bank, deltas, alpha)
            kl_probe = kl_of(loaded, probe, ref_probe, args.score_batch)
            full = score_fidelity(loaded, corpus, ref=ref_full, batch_size=args.score_batch,
                                  subset=subset_index(corpus.n_tokens, 2048))
            rows["targets"][str(target)] = {
                "alpha": alpha, "kl_probe": kl_probe,
                "kl_full_corpus": full["kl"].double().mean().item(),
                "flip": (full["top1"] != ref_full["top1"]).double().mean().item(),
                **task_eval(f"{method} @KL={target}")}
            LOGGER.info("[%s] alvo %.3f -> alpha %.3f | KL %.4f | flip %.4f", method, target, alpha,
                        rows["targets"][str(target)]["kl_full_corpus"], rows["targets"][str(target)]["flip"])
            dfile.write_text(json.dumps(rows))
        bank.restore()
        del deltas
        torch.cuda.empty_cache()


def tost(diffs, margin: float) -> dict:
    """Dois testes t unilaterais: equivalencia se ambos p < 0.05 (|media| dentro de +-margin)."""
    import numpy as np
    from scipy import stats
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t_lo = (d.mean() + margin) / se
    t_hi = (d.mean() - margin) / se
    p_lo = stats.t.sf(t_lo, n - 1)          # H0: media <= -margin
    p_hi = stats.t.cdf(t_hi, n - 1)         # H0: media >= +margin
    return {"n": n, "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if n > 1 else float("nan"),
            "p_tost": float(max(p_lo, p_hi)), "equivalente": bool(max(p_lo, p_hi) < 0.05),
            "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}


def stage_tost(args) -> None:
    import numpy as np
    base_dir = OUT / "matched_kl"
    data = {}
    for mdir in sorted(base_dir.iterdir()):
        base = json.loads((mdir / "base.json").read_text()) if (mdir / "base.json").exists() else None
        if base is None:
            continue
        for f in sorted(mdir.glob("*.json")):
            if f.stem == "base" or f.stem.startswith("kv"):
                continue
            r = json.loads(f.read_text())
            for tgt, row in r["targets"].items():
                data.setdefault((mdir.name, tgt), {})[r["method"]] = {
                    "gsm8k": row["gsm8k"]["acc"] - base["gsm8k"]["acc"],
                    "mmlu_pt": row["mmlu_pt"]["acc"] - base["mmlu_pt"]["acc"],
                    "flip": row["flip"], "kl": row["kl_full_corpus"]}
    fams = {"quant": ("u4", "gptq4", "awq4", "g4"), "poda": ("sgpt50", "wanda50", "mag40")}
    print(f"{'modelo':26s} {'alvo':>5s} {'metodo':8s} {'KL':>7s} {'flip':>6s} {'dGSM8K':>8s} {'dMMLU':>7s}")
    per_pair = {"gsm8k": [], "mmlu_pt": [], "flip": []}
    for (model, tgt), methods in sorted(data.items()):
        for m, v in sorted(methods.items()):
            print(f"{model[:26]:26s} {tgt:>5s} {m:8s} {v['kl']:7.4f} {v['flip']:6.3f} "
                  f"{100*v['gsm8k']:+8.2f} {100*v['mmlu_pt']:+7.2f}")
        q = [methods[m] for m in fams["quant"] if m in methods]
        p = [methods[m] for m in fams["poda"] if m in methods]
        if q and p:
            for key in per_pair:
                per_pair[key].append(float(np.mean([x[key] for x in p]) - np.mean([x[key] for x in q])))
    print("\nTOST poda - quantizacao (pareado por modelo x alvo de KL), margem "
          f"{100*args.margin:.1f}pp / {args.margin_flip:.3f} em flips:")
    for key, mg in (("gsm8k", args.margin), ("mmlu_pt", args.margin), ("flip", args.margin_flip)):
        if len(per_pair[key]) >= 2:
            r = tost(per_pair[key], mg)
            print(f"   {key:8s} n={r['n']:2d} media {100*r['mean']:+6.2f} (IC95 [{100*r['ci95'][0]:+.2f},"
                  f"{100*r['ci95'][1]:+.2f}]) p_TOST={r['p_tost']:.4f} -> "
                  f"{'EQUIVALENTE' if r['equivalente'] else 'inconclusivo'}")
    (OUT / "analysis" / "matched_kl.json").write_text(json.dumps(
        {"per_pair": per_pair, "rows": {f"{k[0]}|{k[1]}": v for k, v in data.items()}}, indent=1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["run", "tost"])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--targets", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    p.add_argument("--probe-seqs", type=int, default=96)
    p.add_argument("--bisect", type=int, default=6)
    p.add_argument("--n-task", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--score-batch", type=int, default=8)
    p.add_argument("--calib-batch", type=int, default=8)
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--margin-flip", type=float, default=0.02)
    p.add_argument("--skip-native-task", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    {"run": stage_run, "tost": stage_tost}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
