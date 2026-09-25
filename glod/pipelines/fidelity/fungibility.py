#!/usr/bin/env python3
"""Experimento de cerco 1: o KL e fungivel entre partes da arquitetura?

Restringe uma perturbacao (RTN 3 bits ou ruido gaussiano de mesma variancia) a um
subconjunto de projecoes - atencao x MLP, em cada terco de profundidade - e ajusta
sua intensidade (alpha) para o MESMO KL alvo. Se o KL for fungivel, os flips (e kappa)
sao iguais para todos os subconjuntos; se a atencao ou as camadas iniciais forem
"mais nocivas por nat", aparecem kappas diferentes.

    python -m glod fungibility --model Qwen/Qwen3-4B --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys

import torch

from glod.core import compressors as C
from glod.core.elastic_depth import make_elastic_depth
from glod.core.fidelity import score_fidelity, subset_index
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import GeneratedCorpus
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, add_corpus_arg, ref_slug
from glod.pipelines.tasks.matched_kl import apply_scaled, kl_of, small_corpus

LOGGER = logging.getLogger("glod.fung")
ATTN = re.compile(r"(q_proj|k_proj|v_proj|o_proj|qkv_proj)$")
MLP = re.compile(r"(gate_proj|up_proj|down_proj|gate_up_proj)$")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    add_corpus_arg(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--perturb", nargs="+", default=["u3", "g3"])
    p.add_argument("--targets", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--probe-seqs", type=int, default=96)
    p.add_argument("--bisect", type=int, default=7)
    p.add_argument("--score-batch", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)

    ms = ref_slug(args.model, corpus=args.corpus)
    out_dir = OUT / "fungibility" / ms
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{ms}.json").read_text()))
    ref_full = torch.load(OUT / ms / ms / "bf16.pt")
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank = C.WeightBank(loaded.decoder)
    make_elastic_depth(loaded.model)
    probe, keep = small_corpus(corpus, args.probe_seqs)
    ref_probe = {k: ref_full[k][keep] for k in ("top1", "topv", "topi")}
    L = bank.num_layers
    thirds = {"inicio": range(0, L // 3), "meio": range(L // 3, 2 * L // 3), "fim": range(2 * L // 3, L)}
    subsets = {"tudo": lambda t: True}
    for part, rx in (("atencao", ATTN), ("mlp", MLP)):
        subsets[part] = (lambda t, rx=rx: bool(rx.search(t.name)))
        for tname, rng in thirds.items():
            subsets[f"{part}_{tname}"] = (lambda t, rx=rx, rng=rng: bool(rx.search(t.name)) and t.layer_idx in rng)

    results = json.loads((out_dir / "results.json").read_text()) if (out_dir / "results.json").exists() else {}
    for pert in args.perturb:
        bits = int(pert[1:])
        gen = torch.Generator(device=args.device).manual_seed(0)
        full_delta = {}
        for t in bank.targets:
            w = t.original.to(loaded.device).float()
            new = C.rtn(w, bits).float() if pert.startswith("u") else C.gaussian_like_rtn(w, bits, gen).float()
            full_delta[t.name] = (new - w).to("cpu", torch.float16)
        for sname, pred in subsets.items():
            key = f"{pert}|{sname}"
            if key in results and len(results[key]) == len(args.targets):
                continue
            deltas = {t.name: (full_delta[t.name] if pred(t) else torch.zeros(1, dtype=torch.float16))
                      for t in bank.targets}
            deltas = {n: (d if d.numel() > 1 else torch.zeros_like(full_delta[n])) for n, d in deltas.items()}
            n_proj = sum(pred(t) for t in bank.targets)
            rows = {}
            for target in args.targets:
                lo, hi = 0.0, 1.0
                apply_scaled(bank, deltas, hi)
                while kl_of(loaded, probe, ref_probe, args.score_batch) < target and hi < 64:
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
                res = score_fidelity(loaded, corpus, ref=ref_full, batch_size=args.score_batch,
                                     subset=subset_index(corpus.n_tokens, 1024))
                flip = (res["top1"] != ref_full["top1"]).double().mean().item()
                kl = res["kl"].double().mean().item()
                rows[str(target)] = {"alpha": alpha, "kl": kl, "flip": flip, "kappa": flip / max(kl, 1e-12) ** 0.5,
                                     "n_proj": n_proj}
                LOGGER.info("[%s | %-15s] alvo %.2f alpha %.3f -> KL %.4f flip %.4f kappa %.3f",
                            pert, sname, target, alpha, kl, flip, rows[str(target)]["kappa"])
            results[key] = rows
            bank.restore()
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))
        del full_delta
    # resumo: dispersao de kappa entre subconjuntos no mesmo alvo
    for pert in args.perturb:
        for target in args.targets:
            ks = {s: results[f"{pert}|{s}"][str(target)]["kappa"] for s in subsets if f"{pert}|{s}" in results}
            vals = list(ks.values())
            LOGGER.info("RESUMO %s alvo %.2f: kappa min %.3f max %.3f razao max/min %.2f | %s", pert, target,
                        min(vals), max(vals), max(vals) / min(vals),
                        " ".join(f"{s}={v:.3f}" for s, v in ks.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
