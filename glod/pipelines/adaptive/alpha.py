#!/usr/bin/env python3
"""Experimento de cerco 4: adaptatividade rompe a taxa fixa KL -> flips?

A tese diz que perturbacoes FIXAS de peso convertem KL em flips a uma taxa unica,
~6x abaixo do teto de um atacante por token. Aqui a MESMA direcao de perturbacao
(RTN/GPTQ de um modelo) e aplicada em K intensidades alpha, pontuadas por teacher
forcing, e cada token (ou sequencia) escolhe sua intensidade:

  oraculo    escolha por token que conhece flips e KL (Lagrangiano) - limite superior
  margem     escolha realizavel usando SO o gap top1-top2 do modelo denso (gate GLOD)
  sequencia  uma intensidade por sequencia (granularidade grossa)

em dois sentidos, no mesmo KL medio:
  maligno    maximiza flips   (rompe a taxa por cima? quanto do teto analitico alcanca?)
  benigno    minimiza flips   (o sentido util para compressao adaptativa)

Aproximacao declarada (a mesma da Fase 1c): cada intensidade usa o proprio prefixo
no teacher forcing; num sistema real o KV do prefixo misturaria intensidades.

    python -m glod adaptive-alpha --model Qwen/Qwen3-4B --device cuda:0 --method u3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch

from glod.core import compressors as C
from glod.core.elastic_depth import make_elastic_depth
from glod.core.fidelity import score_fidelity
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import GeneratedCorpus
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, slug
from glod.pipelines.fidelity.build_grid import apply_config
from glod.pipelines.tasks.matched_kl import apply_scaled

LOGGER = logging.getLogger("glod.adaptive")


def frontier_lagrange(F: np.ndarray, K: np.ndarray, budgets, sign: int) -> dict:
    """Escolha por token argmax_l sign*F - lam*K, varrendo lam; interpola nos orcamentos de KL."""
    pts = []
    for lam in np.concatenate([np.geomspace(1e-3, 1e3, 120), [0.0]]):
        choice = np.argmax(sign * F - lam * K, axis=0)
        idx = np.arange(F.shape[1])
        pts.append((K[choice, idx].mean(), F[choice, idx].mean()))
    return _pick(pts, budgets, sign)


def _pick(pts, budgets, sign: int) -> dict:
    """maligno: max flips com KL <= b. benigno: min flips com KL >= b (gastar o orcamento);
    'KL <= b' no benigno seria trivial (alpha = 0 da zero flips)."""
    pts = sorted(pts)
    ks, fs = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    out = {}
    for b in budgets:
        ok = ks <= b + 1e-12 if sign > 0 else ks >= b - 1e-12
        out[str(b)] = float((fs[ok].max() if sign > 0 else fs[ok].min()) if ok.any() else np.nan)
    return out


def frontier_signal(F, K, score, budgets, sign: int) -> dict:
    """Gate por um sinal escalar: tokens ordenados por `score` recebem a intensidade l,
    os demais alpha = 0; varre l e a fracao de tokens. maligno: ordem crescente do score
    (tokens mais frageis primeiro); benigno: decrescente (mais robustos primeiro)."""
    order = np.argsort(score) if sign > 0 else np.argsort(-score)
    T = F.shape[1]
    pts = []
    for l in range(1, F.shape[0]):
        cumK = np.cumsum(K[l, order]) / T
        cumF = np.cumsum(F[l, order]) / T
        step = max(T // 400, 1)
        pts += list(zip(cumK[::step], cumF[::step]))
    return _pick(pts, budgets, sign)


def frontier_sequence(F, K, seq, budgets, sign: int) -> dict:
    n_seq = int(seq.max()) + 1
    Fs = np.stack([np.bincount(seq, weights=F[l], minlength=n_seq) for l in range(F.shape[0])])
    Ks = np.stack([np.bincount(seq, weights=K[l], minlength=n_seq) for l in range(F.shape[0])])
    T = F.shape[1]
    pts = []
    for lam in np.concatenate([np.geomspace(1e-3, 1e3, 120), [0.0]]):
        choice = np.argmax(sign * Fs - lam * Ks, axis=0)
        idx = np.arange(n_seq)
        pts.append((Ks[choice, idx].sum() / T, Fs[choice, idx].sum() / T))
    return _pick(pts, budgets, sign)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--method", default="u3")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20])
    p.add_argument("--score-batch", type=int, default=8)
    p.add_argument("--calib-batch", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)

    out_dir = OUT / "adaptive" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"levels_{args.method}.pt"
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text()))
    ref = torch.load(OUT / slug(args.model) / slug(args.model) / "bf16.pt")
    seq = np.concatenate([np.full(len(g), i) for i, g in enumerate(corpus.gen_ids)])
    v = ref["topv"].double()
    gap = (v[:, 0] - v[:, 1]).numpy()

    if cache.exists():
        lv = torch.load(cache)
    else:
        loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
        bank = C.WeightBank(loaded.decoder)
        ctrl = make_elastic_depth(loaded.model)
        calib: dict = {}

        def calib_fn():
            if "x" not in calib:
                calib["x"] = C.wikitext_calibration(loaded.tokenizer, 128, 512, cache_dir=f"{args.cache_dir}/datasets")
            return calib["x"]

        with apply_config(args.method, loaded, bank, ctrl, calib_fn, args):
            deltas = {t.name: (t.module.weight.detach().float() - t.original.to(loaded.device).float()).to("cpu", torch.float16)
                      for t in bank.targets}
        lv = {"alphas": args.alphas, "flip": [], "kl": [], "entropy": []}
        for a in args.alphas:
            apply_scaled(bank, deltas, a)
            r = score_fidelity(loaded, corpus, ref=ref, batch_size=args.score_batch)
            lv["flip"].append((r["top1"] != ref["top1"]).to(torch.uint8))
            lv["kl"].append(r["kl"].float())
            lv["entropy"].append(r["entropy"].float())
            LOGGER.info("alpha %.2f: flip %.4f KL %.4f", a, lv["flip"][-1].double().mean(), lv["kl"][-1].double().mean())
        bank.restore()
        torch.save(lv, cache)

    if "entropy" not in lv:
        raise SystemExit(f"cache antigo sem entropia: apague {cache} e rode de novo")
    F = torch.stack(lv["flip"]).double().numpy()
    # sinal REALIZAVEL: entropia do modelo na intensidade maxima (a "base" barata que roda sempre)
    H_base = torch.stack(lv["entropy"])[-1].double().numpy()
    K = torch.stack(lv["kl"]).double().clamp(min=0).numpy()
    static = [(float(K[l].mean()), float(F[l].mean())) for l in range(len(lv["alphas"]))]
    sk, sf = np.array([s[0] for s in static]), np.array([s[1] for s in static])
    static_at = {str(b): float(np.interp(b, sk, sf)) for b in args.budgets}
    # teto analitico (atacante oraculo com liberdade total por token)
    p1, p2 = v[:, 0].exp(), v[:, 1].exp()
    m = (p1 + p2) / 2
    cost = (p1 * (p1 / m).log() + p2 * (p2 / m).log()).numpy()
    cum = np.cumsum(np.sort(cost)) / len(cost)
    ceiling = {str(b): float((cum <= b).mean()) for b in args.budgets}
    res = {
        "static": static_at, "ceiling": ceiling,
        "malign_oracle": frontier_lagrange(F, K, args.budgets, +1),
        "malign_margin_dense": frontier_signal(F, K, gap, args.budgets, +1),
        "malign_entropy_base": frontier_signal(F, K, -H_base, args.budgets, +1),
        "malign_sequence": frontier_sequence(F, K, seq, args.budgets, +1),
        "benign_oracle": frontier_lagrange(F, K, args.budgets, -1),
        "benign_margin_dense": frontier_signal(F, K, gap, args.budgets, -1),
        "benign_entropy_base": frontier_signal(F, K, -H_base, args.budgets, -1),
        "benign_sequence": frontier_sequence(F, K, seq, args.budgets, -1),
    }
    cols = ["static", "malign_sequence", "malign_entropy_base", "malign_margin_dense", "malign_oracle", "ceiling",
            "benign_sequence", "benign_entropy_base", "benign_margin_dense", "benign_oracle"]
    print(f"\n{args.model} | {args.method} | flips (%) com KL medio no orcamento")
    print("   KL " + " ".join(f"{c.replace('malign_', 'M:').replace('benign_', 'B:')[:12]:>12s}" for c in cols))
    for b in args.budgets:
        print(f"{b:5.2f} " + " ".join(f"{100 * res[c][str(b)]:12.2f}" for c in cols))
    (out_dir / f"results_{args.method}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
