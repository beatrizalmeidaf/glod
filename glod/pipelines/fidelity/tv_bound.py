#!/usr/bin/env python3
"""Um limite formal: quantos flips um dado TV permite, pela densidade de margens do denso.

Proposicao. Se o arg-max muda na posicao t (de i1 para j), entao
    delta_t = p_t(i1) - p_t(i2) <= 2 TV_t.
Prova: q(j) >= q(i1) da  delta_t <= p(i1) - p(j) <= [p(i1) - q(i1)] + [q(j) - p(j)],
e cada colchete e no maximo TV_t (massa perdida e massa ganha somam TV cada uma).

Corolario (Markov). Para todo h > 0, com F(h) a fracao de posicoes com delta_t < h,
    flips <= F(h) + 2 TV / h,
e o limite B = min_h [F(h) + 2 TV/h] depende so do modelo denso e do TV medio. Se
F(h) <= rho h perto de zero, B <= 2 sqrt(2 rho TV): no pior caso os flips crescem como
a RAIZ do TV, nao linearmente - a razao ~1 medida e propriedade dos compressores, nao
uma garantia.

Este estagio verifica a proposicao posicao a posicao (com TV no top-64 + cauda, que
agrega a cauda e pode subestimar o TV completo) e mede quao apertado e o limite.

    python -m glod tv-bound            # grava OUT/analysis/tv_bound.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from glod.paths import OUT, corpus_of

H_GRID = np.logspace(-5, 0, 400)


def ref_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and d.name.endswith("__fp32") and (d / d.name / "bf16.pt").exists())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    root = Path(args.out)
    rows, viol_pos, flip_pos = [], 0, 0
    for d in ref_dirs(root):
        ref = torch.load(d / d.name / "bf16.pt", weights_only=False)
        lp = ref["topv"].double()
        delta = (lp[:, 0].exp() - lp[:, 1].exp()).numpy()
        ds = np.sort(delta)
        F = np.searchsorted(ds, H_GRID, side="left") / len(ds)
        for f in sorted((d / d.name).glob("*.pt")):
            if f.stem in ("bf16", "raw") or f.stem.startswith("ofc@"):
                continue
            r = torch.load(f, weights_only=False)
            flip = (r["top1"] != ref["top1"]).numpy()
            tv_t = r["tv"].double().numpy()
            fl, tv = float(flip.mean()), float(tv_t.mean())
            if fl == 0 or tv == 0:
                continue
            bad = flip & (tv_t < delta / 2 - 1e-6)          # proposicao violada nesta posicao
            viol_pos += int(bad.sum()); flip_pos += int(flip.sum())
            vals = F + 2 * tv / H_GRID
            k = int(vals.argmin())
            rows.append({"ref": d.name.replace("__fp32", ""), "corpus": corpus_of(d.name), "config": f.stem,
                         "flip": fl, "tv": tv, "kl": float(r["kl"].double().mean()),
                         "bound": float(min(1.0, vals[k])), "h_star": float(H_GRID[k]),
                         "pos_violations": int(bad.sum()), "n_flips": int(flip.sum())})
    if not rows:
        raise SystemExit(f"nenhuma referencia fp32 em {root}")
    tight = np.array([x["flip"] / x["bound"] for x in rows])
    over = [x for x in rows if x["flip"] > x["bound"] + 1e-9]
    # expoente do limite em TV por referencia (a raiz do pior caso aparece como ~0.5)
    slopes = []
    for ref in sorted({x["ref"] for x in rows}):
        g = [x for x in rows if x["ref"] == ref and x["bound"] < 1 and 1e-3 < x["kl"] < 2]
        if len(g) >= 4:
            slopes.append(float(np.polyfit(np.log([x["tv"] for x in g]), np.log([x["bound"] for x in g]), 1)[0]))
    small = [x for x in rows if 1e-3 < x["kl"] < 0.05]
    summ = {
        "n_refs": len({x["ref"] for x in rows}), "n_configs": len(rows),
        "pos_violation_share": viol_pos / max(flip_pos, 1), "n_flipped_positions": flip_pos,
        "n_configs_above_bound": len(over),
        "flip_over_bound": {"median": float(np.median(tight)), "p10": float(np.percentile(tight, 10)),
                            "p90": float(np.percentile(tight, 90)), "max": float(tight.max())},
        "flip_over_bound_kappa_window_median": float(np.median([x["flip"] / x["bound"] for x in small]))
        if small else None,
        "bound_tv_slope_median": float(np.median(slopes)) if slopes else None,
        "bound_tv_slope_range": [float(min(slopes)), float(max(slopes))] if slopes else None,
    }
    out = root / "analysis" / "tv_bound.json"
    out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))
    print(f"gravado: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
