#!/usr/bin/env python3
"""A calibracao no mesmo split dos corpora wikitext muda a relacao flips x KL?

GPTQ/AWQ/SparseGPT/Wanda calibram com o treino do WikiText-2, de onde tambem saem os
corpora `wikitext` e `wikitext_nat`. Aqui comparamos, nas mesmas referencias, cada
configuracao calibrada no WikiText-2 com a mesma configuracao calibrada no C4
(`glod grid score --calib-source c4`, gravada em OUT/calib_c4/<ref>/): KL, flips e o
desvio de cada uma em relacao a curva log-log da referencia, ajustada nas familias sem
calibracao (RTN, ruido, KV, magnitude, remocao de camadas), que nao dependem do texto
de calibracao.

    python -m glod calib-control
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from glod.paths import OUT

AN = OUT / "analysis"
CALIBRATED = ("gptq", "awq", "sgpt", "wanda")


def point(ref: dict, c: dict) -> dict:
    return {"kl": c["kl"].double().mean().item(), "tv": c["tv"].double().mean().item(),
            "flip": (c["top1"] != ref["top1"]).double().mean().item()}


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    torch.set_grad_enabled(False)
    rows = []
    root = OUT / "calib_c4"
    for d in sorted(p.name for p in root.iterdir()) if root.exists() else []:
        ref = torch.load(OUT / d / d / "bf16.pt", weights_only=False)
        own = {}
        for f in sorted((OUT / d / d).glob("*.pt")):
            if f.stem != "bf16":
                own[f.stem] = point(ref, torch.load(f, weights_only=False))
        base = [v for k, v in own.items() if not k.startswith(CALIBRATED)]
        x = np.log([v["kl"] for v in base]); y = np.log([v["flip"] for v in base])
        a, b = np.polyfit(x, y, 1)

        def dev(v):
            return math.exp(math.log(v["flip"]) - (b + a * math.log(v["kl"])))
        for f in sorted((root / d).glob("*.pt")):
            if f.stem not in own:
                continue
            c4 = point(ref, torch.load(f, weights_only=False))
            wt = own[f.stem]
            rows.append({"ref": d, "config": f.stem, "kl_wikitext": wt["kl"], "kl_c4": c4["kl"],
                         "flip_wikitext": wt["flip"], "flip_c4": c4["flip"],
                         "dev_wikitext": dev(wt), "dev_c4": dev(c4),
                         "kappa_wikitext": wt["flip"] / math.sqrt(wt["kl"]),
                         "kappa_c4": c4["flip"] / math.sqrt(c4["kl"]),
                         "fot_wikitext": wt["flip"] / wt["tv"], "fot_c4": c4["flip"] / c4["tv"]})
            r = rows[-1]
            print(f"{d:36s} {f.stem:8s} KL wt {r['kl_wikitext']:.4f} c4 {r['kl_c4']:.4f} | "
                  f"desvio da curva wt {r['dev_wikitext']:.3f} c4 {r['dev_c4']:.3f}")
    summ = {}
    if rows:
        kr = np.array([r["kl_c4"] / r["kl_wikitext"] for r in rows])
        dd = np.array([r["dev_c4"] / r["dev_wikitext"] for r in rows])
        summ = {"n": len(rows), "n_refs": len({r["ref"] for r in rows}),
                "kl_ratio_median": float(np.median(kr)), "kl_ratio_range": [float(kr.min()), float(kr.max())],
                "dev_ratio_median": float(np.median(dd)), "dev_ratio_range": [float(dd.min()), float(dd.max())],
                "dev_wikitext_abs_median": float(np.median([abs(r["dev_wikitext"] - 1) for r in rows])),
                "dev_c4_abs_median": float(np.median([abs(r["dev_c4"] - 1) for r in rows])),
                "dev_c4_abs_max": float(max(abs(r["dev_c4"] - 1) for r in rows))}
        print(json.dumps(summ, indent=1))
    AN.mkdir(parents=True, exist_ok=True)
    (AN / "calib_control.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
