#!/usr/bin/env python3
"""TV e especial, ou qualquer divergencia de 1a ordem serve? (controle de revisor)

Para cada configuracao das referencias com cabeca fp32 calcula, por token, sobre o
top-64 da referencia mais a cauda agregada: TV, KL, a distancia de Hellinger
H_t = sqrt(1 - sum sqrt(p q)) e a divergencia de Jensen-Shannon JS_t. Hellinger e TV
sao de 1a ordem no deslocamento dos logits; KL e JS sao de 2a. A pergunta e qual
estatistica converte em flips com menos dependencia da referencia:

  - razao flips/X (X de 1a ordem) ou flips/sqrt(X) (X de 2a ordem), por configuracao;
  - dispersao dessa razao entre referencias e erro de prever uma configuracao a partir
    das outras da mesma referencia (constante por referencia).

    python -m glod divergences                  # referencias do paper (fp32)
    GLOD_RESULTS=<holdout> python -m glod divergences --out <arquivo>
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from glod.paths import OUT

KL_WINDOW = (1e-3, 0.25)
STATS = ("tv", "hellinger", "sqrt_kl", "sqrt_js")


def dists(ref: dict, c: dict) -> dict:
    lp = ref["topv"].double()
    lq = c["at_ref"].double()
    p = torch.cat([lp.exp(), (1 - lp.exp().sum(-1)).clamp_min(1e-12)[:, None]], 1)
    q = torch.cat([lq.exp(), (1 - lq.exp().sum(-1)).clamp_min(1e-12)[:, None]], 1)
    q = q / q.sum(-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(-1)
    kl_qm = (q * (q / m).log()).sum(-1)
    js = (0.5 * kl_pm + 0.5 * kl_qm).clamp_min(0)
    bc = (p * q).sqrt().sum(-1).clamp(max=1.0)
    hel = (1 - bc).clamp_min(0).sqrt()
    return {"tv": c["tv"].double().mean().item(), "kl": c["kl"].double().clamp_min(0).mean().item(),
            "hellinger": hel.mean().item(), "js": js.mean().item()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "analysis" / "divergences.json"))
    args = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    rows = []
    for d in sorted(os.listdir(OUT)):
        if "__fp32" not in d or not (OUT / d / d / "bf16.pt").exists():
            continue
        ref = torch.load(OUT / d / d / "bf16.pt", weights_only=False)
        for f in sorted((OUT / d / d).glob("*.pt")):
            if f.stem == "bf16":
                continue
            c = torch.load(f, weights_only=False)
            if "at_ref" not in c:
                continue
            x = dists(ref, c)
            if not (KL_WINDOW[0] < x["kl"] < KL_WINDOW[1]):
                continue
            flip = (c["top1"] != ref["top1"]).double().mean().item()
            rows.append({"ref": d, "config": f.stem, "flip": flip, **x,
                         "r_tv": flip / x["tv"], "r_hellinger": flip / x["hellinger"],
                         "r_sqrt_kl": flip / math.sqrt(x["kl"]), "r_sqrt_js": flip / math.sqrt(x["js"])})
        print(d, sum(r["ref"] == d for r in rows), flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["ref"], []).append(r)
    summ = {"n_configs": len(rows), "n_refs": len(by)}
    for s in STATS:
        k = "r_" + s
        allv = np.array([r[k] for r in rows])
        med = [float(np.median([r[k] for r in v])) for v in by.values()]
        loo = []
        for v in by.values():
            if len(v) < 4:
                continue
            for i, r in enumerate(v):
                oth = np.median([x[k] for j, x in enumerate(v) if j != i])
                loo.append(abs(oth / r[k] - 1))
        summ[s] = {"median": float(np.median(allv)), "cv_all": float(allv.std(ddof=1) / allv.mean()),
                   "ref_min": min(med), "ref_max": max(med), "ref_span": max(med) / min(med),
                   "loo_median_err": float(np.median(loo)), "loo_p90_err": float(np.percentile(loo, 90))}
    print(json.dumps(summ, indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
