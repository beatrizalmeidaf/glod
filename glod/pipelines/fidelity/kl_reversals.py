#!/usr/bin/env python3
"""Quantas comparacoes entre compressores o KL erra de ordem, e quantas a TV erra.

Para todo par de configuracoes (i, j) cujas taxas de flip diferem em pelo menos
`--min-gap` (10% por padrao: uma diferenca que importa para quem escolhe), o KL
"inverte" o par quando diz que o compressor com MAIS flips mudou MENOS a distribuicao:
sign(log KL_i - log KL_j) != sign(log flip_i - log flip_j). Idem para a TV.

Os pares sao separados pelo que um leitor de relatorios compara:
  same_ref            mesmo modelo, mesmo corpus (controle: o KL ordena bem aqui)
  same_model_x_corpus mesmo modelo, corpora diferentes
  x_model_same_corpus modelos diferentes, mesmo corpus
  x_model_x_corpus    modelos e corpora diferentes (dois relatorios quaisquer)
Intervalos: bootstrap sobre modelos (a unidade que se repete entre pares).

    python -m glod kl-reversals --csv data/measurements.csv   # grava OUT/analysis/kl_reversals.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from glod.paths import OUT

CATS = ("same_ref", "same_model_x_corpus", "x_model_same_corpus", "x_model_x_corpus")


def rates(model, corpus, lf, lk, lt, keep_models, min_gap):
    """Taxas de inversao por categoria, restritas aos modelos em keep_models (com repeticao)."""
    idx = np.concatenate([np.flatnonzero(model == m) for m in keep_models])
    m, c, f, k, t = model[idx], corpus[idx], lf[idx], lk[idx], lt[idx]
    df = f[:, None] - f[None, :]
    sel = np.triu(np.abs(df) > np.log1p(min_gap), 1)
    same_m = m[:, None] == m[None, :]
    same_c = c[:, None] == c[None, :]
    rev_k = np.sign(k[:, None] - k[None, :]) != np.sign(df)
    rev_t = np.sign(t[:, None] - t[None, :]) != np.sign(df)
    masks = {"same_ref": same_m & same_c, "same_model_x_corpus": same_m & ~same_c,
             "x_model_same_corpus": ~same_m & same_c, "x_model_x_corpus": ~same_m & ~same_c}
    out = {}
    for name, mk in masks.items():
        w = sel & mk
        n = int(w.sum())
        out[name] = {"n_pairs": n, "kl": float(rev_k[w].mean()) if n else None,
                     "tv": float(rev_t[w].mean()) if n else None}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/measurements.csv")
    p.add_argument("--min-gap", type=float, default=0.10)
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--out", default=str(OUT / "analysis" / "kl_reversals.json"))
    args = p.parse_args(argv)
    rows = [r for r in csv.DictReader(open(args.csv))
            if float(r["kl"]) > 0 and float(r["flip"]) > 0 and float(r["tv"]) > 0]
    model = np.array([r["model"] for r in rows]); corpus = np.array([r["corpus"] for r in rows])
    lf = np.log([float(r["flip"]) for r in rows]); lk = np.log([float(r["kl"]) for r in rows])
    lt = np.log([float(r["tv"]) for r in rows])
    models = sorted(set(model))
    point = rates(model, corpus, lf, lk, lt, models, args.min_gap)
    rng = np.random.default_rng(0)
    boots = {c: {"kl": [], "tv": []} for c in CATS}
    for _ in range(args.boot):
        pick = list(rng.choice(models, len(models)))
        b = rates(model, corpus, lf, lk, lt, pick, args.min_gap)
        for c in CATS:
            if b[c]["n_pairs"]:
                boots[c]["kl"].append(b[c]["kl"]); boots[c]["tv"].append(b[c]["tv"])
    for c in CATS:
        for s in ("kl", "tv"):
            v = boots[c][s]
            point[c][f"{s}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
        if point[c]["kl"]:
            point[c]["tv_over_kl"] = point[c]["tv"] / point[c]["kl"]
    res = {"n_configs": len(rows), "n_models": len(models), "min_gap": args.min_gap, "by_category": point}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    for c in CATS:
        v = point[c]
        print(f"{c:22s} pares {v['n_pairs']:7d}  KL inverte {100 * v['kl']:5.1f}% "
              f"[{100 * v['kl_ci'][0]:.1f}, {100 * v['kl_ci'][1]:.1f}]  TV inverte {100 * v['tv']:5.1f}% "
              f"[{100 * v['tv_ci'][0]:.1f}, {100 * v['tv_ci'][1]:.1f}]")
    print(f"gravado: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
