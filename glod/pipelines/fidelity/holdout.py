#!/usr/bin/env python3
"""Validacao fora da amostra: confere as previsoes registradas contra um corpus novo.

As previsoes (var/logs/r7/holdout_predictions.json) foram fixadas ANTES de gerar o
corpus, so com numeros das 42/52 referencias publicadas. Aqui, para cada referencia
com cabeca fp32 sob GLOD_RESULTS (a raiz isolada do corpus novo):

  P1  expoente do ajuste log flips x log KL por referencia dentro da faixa publicada;
  P2  desvio de cada familia em relacao a curva ajustada SEM ela dentro de +-5.5%;
  P3  mediana de flips/TV dentro da faixa publicada por referencia;
  P4  conversao por token c = E[TV_t]/E[sqrt(KL_t)] dentro da faixa publicada;
  P5  kappa previsto por J, kappa_hat = mediana(flips/TV) * mediana(c) * J, com as
      medianas publicadas, a menos de 15% do kappa medido.

    GLOD_RESULTS=<raiz do holdout> python -m glod holdout --pred <json> --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import numpy as np
import torch

from glod.paths import OUT

KL_WINDOW = (1e-3, 0.25)


def family(cfg: str) -> str:
    for pre, fam in (("gptq", "gptq"), ("awq", "awq"), ("sgpt", "sparsegpt"), ("wanda", "wanda"),
                     ("mag", "magnitude"), ("kv", "kv")):
        if cfg.startswith(pre):
            return fam
    if re.fullmatch(r"u\d+", cfg):
        return "rtn"
    if re.fullmatch(r"g\d+", cfg):
        return "gauss"
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    P = json.load(open(args.pred))
    torch.set_grad_enabled(False)
    out = {"predictions": P, "refs": {}}
    for d in sorted(os.listdir(OUT)):
        if "__fp32" not in d or not (OUT / d / d / "bf16.pt").exists():
            continue
        ref = torch.load(OUT / d / d / "bf16.pt", weights_only=False)
        rows = []
        for f in sorted((OUT / d / d).glob("*.pt")):
            if f.stem == "bf16":
                continue
            c = torch.load(f, weights_only=False)
            kl = c["kl"].double().clamp_min(0)
            tv = c["tv"].double()
            K = kl.mean().item()
            flip = (c["top1"] != ref["top1"]).double().mean().item()
            rows.append({"config": f.stem, "family": family(f.stem), "kl": K, "flip": flip, "tv": tv.mean().item(),
                         "c": tv.mean().item() / kl.sqrt().mean().item(),
                         "J": kl.sqrt().mean().item() / math.sqrt(K)})
        # P1: expoente em todas as configuracoes
        x = np.log([r["kl"] for r in rows]); y = np.log([r["flip"] for r in rows])
        slope = float(np.polyfit(x, y, 1)[0])
        # P2: desvio de familia com a curva ajustada sem a familia
        dev = {}
        for fam in sorted({r["family"] for r in rows}):
            tr = [r for r in rows if r["family"] != fam]
            te = [r for r in rows if r["family"] == fam]
            a, b = np.polyfit(np.log([r["kl"] for r in tr]), np.log([r["flip"] for r in tr]), 1)
            dev[fam] = float(np.exp(np.mean([math.log(r["flip"]) - (b + a * math.log(r["kl"])) for r in te])))
        w = [r for r in rows if KL_WINDOW[0] < r["kl"] < KL_WINDOW[1]]
        fot = float(np.median([r["flip"] / r["tv"] for r in w]))
        cc = float(np.median([r["c"] for r in w]))
        J = float(np.median([r["J"] for r in w]))
        kap = float(np.median([r["flip"] / math.sqrt(r["kl"]) for r in w]))
        k5 = P["P5_kappa_from_J"]
        khat = k5["median_flip_over_tv"] * k5["median_c"] * J
        out["refs"][d] = {
            "n_configs": len(rows), "n_window": len(w), "exponent": slope, "family_dev": dev,
            "flip_over_tv": fot, "c": cc, "J": J, "kappa": kap, "kappa_hat": khat,
            "P1": P["P1_exponent_range"][0] <= slope <= P["P1_exponent_range"][1],
            "P2": all(abs(v - 1) * 100 <= P["P2_family_dev_max_pct"] for v in dev.values()),
            "P2_max_dev_pct": float(100 * max(abs(v - 1) for v in dev.values())),
            "P3": P["P3_flip_over_tv_ref_range"][0] <= fot <= P["P3_flip_over_tv_ref_range"][1],
            "P4": P["P4_c_range"][0] <= cc <= P["P4_c_range"][1],
            "P5": abs(khat / kap - 1) <= k5["tolerance_rel"], "P5_rel_err": float(khat / kap - 1)}
        print(d, json.dumps(out["refs"][d]))
    json.dump(out, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
