#!/usr/bin/env python3
"""Por que as familias deixam de ser intercambiaveis no corpus de codigo? (post hoc)

O holdout pre-registrado mostrou desvios de familia 3x maiores no codigo quando a curva
e flips x KL. Tres perguntas, todas sobre os mesmos arquivos:

  1. A falha e do KL ou das familias? Refaz o desvio de cada familia (curva ajustada
     sem ela, por referencia) com o TV no lugar do sqrt(KL), no codigo e nas 42
     referencias fp32 publicadas.
  2. Mecanismo: pela identidade kappa = (flips/TV) * c * J, se flips/TV e c nao mudam
     entre familias, o desvio em sqrt(KL) e a diferenca de J (quao espalhada a
     divergencia esta entre tokens). Regressao do desvio (log) em log J, log(flips/TV)
     e log c de cada familia relativos as demais.
  3. Calibracao: GPTQ/AWQ/SparseGPT/Wanda recalibrados em codigo Python (codeparrot;
     `grid score --calib-source code`, raiz calib_code/) no lugar da calibracao
     WikiText. Se a calibracao fosse a causa, o desvio cairia.

    python -m glod holdout-mech
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from glod.paths import OUT
from glod.pipelines.fidelity.holdout import family

HOLDOUT = Path(os.environ.get("GLOD_HOLDOUT", "/local/user_beatrizalmeida/ews_results/holdout"))
CALIBRATED = ("gptq4", "awq4", "sgpt50", "wanda50")


def load_rows(root: Path, d: str, override: Path | None = None) -> list[dict]:
    base = root / d / d
    ref = torch.load(base / "bf16.pt", weights_only=False)
    rows = []
    for f in sorted(base.glob("*.pt")):
        if f.stem == "bf16":
            continue
        src = f
        if override is not None and f.stem in CALIBRATED and (override / f.name).exists():
            src = override / f.name
        c = torch.load(src, weights_only=False)
        kl = c["kl"].double().clamp_min(0)
        tv = c["tv"].double()
        flip = (c["top1"] != ref["top1"]).double().mean().item()
        rows.append({"cfg": f.stem, "fam": family(f.stem), "kl": kl.mean().item(), "tv": tv.mean().item(),
                     "flip": flip, "fot": flip / tv.mean().item(), "c": tv.mean().item() / kl.sqrt().mean().item(),
                     "J": kl.sqrt().mean().item() / math.sqrt(kl.mean().item())})
    return rows


def deviations(rows: list[dict], x: str) -> dict[str, float]:
    """Desvio (log) de cada familia em relacao a curva flips ~ x ajustada sem ela."""
    out = {}
    for fam in {r["fam"] for r in rows}:
        tr = [r for r in rows if r["fam"] != fam]; te = [r for r in rows if r["fam"] == fam]
        if len(tr) < 4:
            continue
        a, b = np.polyfit(np.log([r[x] for r in tr]), np.log([r["flip"] for r in tr]), 1)
        out[fam] = float(np.mean([math.log(r["flip"]) - (b + a * math.log(r[x])) for r in te]))
    return out


def summary(devs: list[float]) -> dict:
    a = np.abs(np.expm1(np.array(devs))) * 100
    return {"n": len(a), "median_pct": float(np.median(a)), "p90_pct": float(np.percentile(a, 90)),
            "max_pct": float(a.max())}


def refs_in(root: Path) -> list[str]:
    return [d for d in sorted(os.listdir(root)) if d.endswith("__fp32") and (root / d / d / "bf16.pt").exists()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "analysis" / "holdout_mech.json"))
    args = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    res: dict = {}
    per_ref: dict = {}
    for label, root in (("code", HOLDOUT), ("published", OUT)):
        dk, dt, maxk, maxt = [], [], [], []
        for d in refs_in(root):
            rows = load_rows(root, d)
            if len({r["fam"] for r in rows}) < 3:
                continue
            k, t = deviations(rows, "kl"), deviations(rows, "tv")
            dk += list(k.values()); dt += list(t.values())
            maxk.append(max(abs(math.expm1(v)) for v in k.values()) * 100)
            maxt.append(max(abs(math.expm1(v)) for v in t.values()) * 100)
            if label == "code":
                per_ref[d] = {"rows": rows, "dev_kl": k, "dev_tv": t}
        res[label] = {"sqrt_kl": {**summary(dk), "refs_max_le_5_5": float(np.mean(np.array(maxk) <= 5.5))},
                      "tv": {**summary(dt), "refs_max_le_5_5": float(np.mean(np.array(maxt) <= 5.5))},
                      "n_refs": len(maxk)}

    # 2. mecanismo no codigo
    pts = []
    for d, v in per_ref.items():
        rows = v["rows"]
        for fam, dev in v["dev_kl"].items():
            te = [r for r in rows if r["fam"] == fam]; tr = [r for r in rows if r["fam"] != fam]
            g = lambda k, S: float(np.mean([math.log(r[k]) for r in S]))  # noqa: E731
            pts.append({"ref": d, "fam": fam, "dev": dev, "dJ": g("J", te) - g("J", tr),
                        "dfot": g("fot", te) - g("fot", tr), "dc": g("c", te) - g("c", tr)})
    y = np.array([p["dev"] for p in pts])
    X = np.array([[p["dJ"], p["dfot"], p["dc"]] for p in pts])
    Xc = np.c_[np.ones(len(y)), X]
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    r2 = 1 - ((y - Xc @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fam_tab = {}
    for fam in sorted({p["fam"] for p in pts}):
        sel = [p for p in pts if p["fam"] == fam]
        fam_tab[fam] = {"dev_pct": float(100 * math.expm1(np.median([p["dev"] for p in sel]))),
                        "J_ratio": float(math.exp(np.median([p["dJ"] for p in sel]))),
                        "fot_ratio": float(math.exp(np.median([p["dfot"] for p in sel]))),
                        "c_ratio": float(math.exp(np.median([p["dc"] for p in sel])))}
    res["mechanism"] = {"n_pairs": len(pts), "r2": float(r2), "beta": beta.tolist(),
                        "pearson_dev_vs_sum": float(stats.pearsonr(X.sum(1), y).statistic),
                        "pearson_dev_vs_J": float(stats.pearsonr(X[:, 0], y).statistic),
                        "sd_log_fot": float(X[:, 1].std()), "sd_log_J": float(X[:, 0].std()),
                        "families": fam_tab}

    # 3. calibracao em codigo
    cal_root = HOLDOUT / "calib_code"
    cal = {}
    if cal_root.exists():
        dk0, dk1, dt1 = [], [], []
        for d in refs_in(HOLDOUT):
            ov = cal_root / d
            if not all((ov / f"{c}.pt").exists() for c in CALIBRATED):
                continue
            r0 = load_rows(HOLDOUT, d); r1 = load_rows(HOLDOUT, d, override=ov)
            k0, k1, t1 = deviations(r0, "kl"), deviations(r1, "kl"), deviations(r1, "tv")
            fams = ("gptq", "awq", "sparsegpt", "wanda")
            cal[d] = {"dev_kl_wikitext_calib": {f: k0[f] for f in fams}, "dev_kl_code_calib": {f: k1[f] for f in fams},
                      "dev_tv_code_calib": {f: t1[f] for f in fams},
                      "kl_ratio_code_vs_wikitext": {c: next(r["kl"] for r in r1 if r["cfg"] == c) /
                                                    next(r["kl"] for r in r0 if r["cfg"] == c) for c in CALIBRATED}}
            dk0 += [k0[f] for f in fams]; dk1 += [k1[f] for f in fams]; dt1 += [t1[f] for f in fams]
        if dk0:
            res["calib_code"] = {"n_refs": len(cal), "wikitext_calib_sqrt_kl": summary(dk0),
                                 "code_calib_sqrt_kl": summary(dk1), "code_calib_tv": summary(dt1),
                                 "kl_ratio_median": float(np.median([v for x in cal.values()
                                                                     for v in x["kl_ratio_code_vs_wikitext"].values()])),
                                 "per_ref": cal}
    Path(args.out).write_text(json.dumps(res, indent=1))
    for k in ("code", "published"):
        print(k, "sqrtKL", res[k]["sqrt_kl"], "| TV", res[k]["tv"])
    print("mecanismo", {k: v for k, v in res["mechanism"].items() if k != "families"})
    for fam, v in res["mechanism"]["families"].items():
        print(f"  {fam:10s} desvio {v['dev_pct']:+6.1f}%  J {v['J_ratio']:.2f}  flips/TV {v['fot_ratio']:.2f}  c {v['c_ratio']:.2f}")
    if "calib_code" in res:
        print("calibracao em codigo", {k: v for k, v in res["calib_code"].items() if k != "per_ref"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
