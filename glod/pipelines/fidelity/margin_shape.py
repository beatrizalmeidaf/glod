#!/usr/bin/env python3
"""O desvio do expoente em relacao a 1/2 acompanha a forma da densidade de margens?

O artigo atribui o desvio do expoente (0.46 a 0.58) ao termo de terceira ordem
desprezado. Esse termo tem uma assinatura: com densidade de margens
rho(g) ~ rho0 + rho1 g perto de zero, a taxa de flips e
    flips ~ 1/2 [rho0 E|dg| + 1/2 rho1 E dg^2],
e como E|dg| ~ sqrt(KL) e E dg^2 ~ KL, o expoente local sobe com rho1/rho0.
Medimos rho1/rho0 por ajuste linear do histograma das margens em (0, h) e
correlacionamos com o expoente ajustado de cada referencia.

h = 1 nat e o mesmo limiar do token fragil (phi) usado no resto do artigo;
0.5 nat entra como sensibilidade. As margens vem das referencias com cabeca fp32,
sem o atomo de empates do bf16.

    python -m glod margin-shape
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import torch
from scipy import stats

from glod.paths import OUT

AN = OUT / "analysis"
WINDOWS = (1.0, 0.5)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    fits = json.loads((AN / "law.json").read_text())["fits"]
    rows = []
    for rp in sorted(glob.glob(str(OUT / "*__fp32" / "*__fp32" / "bf16.pt"))):
        ref = os.path.basename(os.path.dirname(rp))[:-len("__fp32")]
        if ref not in fits:
            continue
        topv = torch.load(rp, map_location="cpu", mmap=True, weights_only=False)["topv"]
        g = (topv[:, 0] - topv[:, 1]).double().numpy()
        row = {"ref": ref, "corpus": ref.split("__")[1] if "__" in ref else "mix",
               "exponent": fits[ref]["top"]["slope"]}
        for h in WINDOWS:
            edges = np.linspace(0, h, 11)
            c, _ = np.histogram(g, edges)
            dens = c / len(g) / np.diff(edges)
            b1, b0 = np.polyfit((edges[:-1] + edges[1:]) / 2, dens, 1)
            row[f"slope_rel_{h}"] = float(b1 / b0)
        rows.append(row)

    out = {"n_refs": len(rows), "rows": rows}
    for h in WINDOWS:
        x = np.array([r[f"slope_rel_{h}"] for r in rows]); y = np.array([r["exponent"] for r in rows])
        rho, pv = stats.spearmanr(x, y)
        # dentro do corpus: o corpus move as duas coisas, entao tiramos a media de cada corpus
        byc = collections.defaultdict(list)
        for i, r in enumerate(rows):
            byc[r["corpus"]].append(i)
        xd, yd = x.copy(), y.copy()
        for idx in byc.values():
            xd[idx] -= x[idx].mean(); yd[idx] -= y[idx].mean()
        rw, pw = stats.spearmanr(xd, yd)
        out[f"h_{h}"] = {"spearman": float(rho), "p": float(pv),
                         "spearman_within_corpus": float(rw), "p_within_corpus": float(pw),
                         "slope_rel_min": float(x.min()), "slope_rel_max": float(x.max()),
                         "n_positive_slope": int((x > 0).sum())}
    out["exponent_min"] = float(min(r["exponent"] for r in rows))
    out["exponent_max"] = float(max(r["exponent"] for r in rows))
    out["n_exponent_above_half"] = int(sum(r["exponent"] > 0.5 for r in rows))
    (AN / "margin_shape.json").write_text(json.dumps(out, indent=1))

    for h in WINDOWS:
        v = out[f"h_{h}"]
        print(f"h = {h} nat: Spearman {v['spearman']:.2f} (p={v['p']:.2g}), dentro do corpus "
              f"{v['spearman_within_corpus']:.2f} (p={v['p_within_corpus']:.2g}); rho1/rho0 em "
              f"[{v['slope_rel_min']:.2f}, {v['slope_rel_max']:.2f}], {v['n_positive_slope']} positivos")
    print(f"{out['n_exponent_above_half']}/{out['n_refs']} expoentes acima de 1/2")
    print(f"-> {AN / 'margin_shape.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
