#!/usr/bin/env python3
"""Previsao de primeira ordem, so com o modelo denso, de flips / TV.

Modelo: o deslocamento dos logits dz_i (i no vocabulario) e isotropico entre tokens,
dz_i ~ N(0, s^2) i.i.d. (o paper mede que os compressores se alinham com as direcoes de
flip como ruido isotropico, App. A.6). Entao, em primeira ordem:

  flips = rho(0) E|dg| / 2 = rho(0) s / sqrt(pi),            dg = dz_1 - dz_2 ~ N(0, 2 s^2)
  TV_t  = 1/2 sum_i p_i |dz_i - sum_j p_j dz_j|,              Var = s^2 (1 - 2 p_i + S), S = sum p_j^2
        = s / sqrt(2 pi) * h(p),   h(p) = sum_i p_i sqrt(1 - 2 p_i + S)

e s cancela:

  flips / TV = sqrt(2) rho(0) / E_t[h(p_t)].

Duas consequencias exatas: (i) numa disputa so entre dois tokens, h = 2 sqrt(2) s(g) e a
razao vira rho(0) / (2 E[s(g)]) = R_flat; o termo h absorve os outros tokens, que o paper
separa em P_pair; (ii) com os logits divididos por T -> 0, rho_T(0) e E[h_T] escalam os
dois com T e a razao vai a 1, a ancora do paper.

Compara a previsao com flips/TV medido por referencia (tv_mechanism.json), em T = 1 e na
varredura de temperatura (top-64 renormalizado, como a medida).

    python -m glod tv-theory
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
from scipy import stats

from glod.paths import OUT
from glod.pipelines.fidelity.tv_mechanism import TEMPS, rho_c


def h_of(p: torch.Tensor, tail: torch.Tensor | None = None) -> torch.Tensor:
    S = (p * p).sum(-1)
    h = (p * (1 - 2 * p + S[:, None]).clamp_min(0).sqrt()).sum(-1)
    if tail is not None:                          # cauda: p_i ~ 0, cada termo ~ p_i sqrt(1 + S)
        h = h + tail * (1 + S).sqrt()
    return h


def predict(topv: torch.Tensor, T: float | None = None) -> dict:
    lp = topv.double()
    if T is None:                                 # T = 1 com a cauda fora do top-64
        p = lp.exp()
        tail = (1 - p.sum(-1)).clamp_min(0)
        g = lp[:, 0] - lp[:, 1]
        h = h_of(p, tail)
    else:                                         # top-64 renormalizado, como a medida em T
        lq = torch.log_softmax(lp / T, -1)
        g = lq[:, 0] - lq[:, 1]
        h = h_of(lq.exp())
    rho = rho_c(g)
    return {"rho0": rho, "E_h": h.mean().item(), "pred": math.sqrt(2) * rho / h.mean().item()}


def fit(pred, meas) -> dict:
    p, m = np.array(pred), np.array(meas)
    e = p / m - 1
    return {"n": len(p), "pred_median": float(np.median(p)), "pred_range": [float(p.min()), float(p.max())],
            "meas_median": float(np.median(m)), "meas_range": [float(m.min()), float(m.max())],
            "err_median": float(np.median(e)), "mape": float(np.mean(np.abs(e))), "max_abs_err": float(np.abs(e).max()),
            "pearson_log": float(stats.pearsonr(np.log(p), np.log(m)).statistic),
            "spearman": float(stats.spearmanr(p, m).statistic)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "analysis" / "tv_theory.json"))
    args = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    tvm = json.loads((OUT / "analysis" / "tv_mechanism.json").read_text())["refs"]
    tvm = tvm if isinstance(tvm, list) else list(tvm.values())
    rows, byT = [], {str(T): ([], []) for T in TEMPS}
    for r in tvm:
        d = r["ref"]
        topv = torch.load(OUT / d / d / "bf16.pt", weights_only=False)["topv"]
        x = predict(topv)
        rows.append({"ref": d, "corpus": r["corpus"], **x, "meas": r["med_flip_over_tv"], "r_flat": r["r_flat"]})
        for T in TEMPS:
            byT[str(T)][0].append(predict(topv, T)["pred"]); byT[str(T)][1].append(r["temperature"][str(T)]["ratio"])
    out = {"t1": fit([x["pred"] for x in rows], [x["meas"] for x in rows]),
           "r_flat_range": [min(x["r_flat"] for x in rows), max(x["r_flat"] for x in rows)],
           "by_corpus": {c: {"pred": float(np.median([x["pred"] for x in rows if x["corpus"] == c])),
                             "meas": float(np.median([x["meas"] for x in rows if x["corpus"] == c])),
                             "n": sum(x["corpus"] == c for x in rows)} for c in sorted({x["corpus"] for x in rows})},
           "temperature": {T: fit(*v) for T, v in byT.items()},
           "pooled_temperature": fit(sum((v[0] for v in byT.values()), []), sum((v[1] for v in byT.values()), [])),
           "rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("T=1", out["t1"])
    for c, v in out["by_corpus"].items():
        print(f"  {c:14s} pred {v['pred']:.3f} medido {v['meas']:.3f} (n={v['n']})")
    for T, v in out["temperature"].items():
        print(f"  T={T:4s} pred {v['pred_median']:.3f} medido {v['meas_median']:.3f} erro med {v['err_median']:+.3f} r(log) {v['pearson_log']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
