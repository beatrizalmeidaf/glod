#!/usr/bin/env python3
"""Relatorio das extensoes de escopo (MoE e QAT) contra os criterios do paper.

Varre GLOD_RESULTS (uma arvore SEPARADA da do paper, ex.: .../fid_ext) e, para cada
referencia `<ref>/<ref>/bf16.pt`, calcula por config flips, KL e TV medios e:

  * expoente do ajuste log flips = a log KL + b  (P1 publicado: 0,459-0,578)
  * mediana de flips/TV                         (P3 publicado: 0,76-1,29)
  * kappa = mediana de flips/sqrt(KL) em 1e-3 < KL < 0,05
  * desvio de cada familia da curva em KL ajustada SEM ela, e do flips/TV das outras
    familias (a comparacao post hoc do corpus de codigo: 2,4% em KL e 1,6% em TV nas
    referencias publicadas)

Diretorios de OUTRO modelo sob a mesma referencia (`<ref>/<outro>/raw.pt`, ex.: o
checkpoint QAT contra o Gemma denso) entram como pontos cruzados, lidos contra a curva e
o flips/TV da referencia. Resume tambem os JSON de `moe-routing`. Nada daqui entra no
numbers.json do paper.

    GLOD_RESULTS=/local/$USER/ews_results/fid_ext python -m glod ext-report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from glod.paths import OUT

PUBLISHED = {"exponent": (0.459, 0.578), "flip_over_tv": (0.76, 1.29),
             "family_dev_kl_median_pct": 2.4, "family_dev_tv_median_pct": 1.6}
FAMILY = [(r"u\d+", "rtn"), (r"q40", "q4_0"), (r"(s\d+)?g\d+", "gauss"), (r"mag\d+", "magnitude"),
          (r"skip[\d_]+", "skip"), (r"kvt?\d+", "kv"), (r"gptq\d+", "gptq"), (r"awq\d+", "awq"),
          (r"sgpt\d+", "sparsegpt"), (r"wanda\d+", "wanda"), (r"lq\d+_\d+", "single"), (r"raw", "raw")]


def family(cfg: str) -> str:
    for pat, fam in FAMILY:
        if re.fullmatch(pat, cfg):
            return fam
    return "other"


def point(ref: dict, path: Path) -> dict:
    r = torch.load(path, weights_only=False)
    flip = float((r["top1"] != ref["top1"]).float().mean())
    return {"config": path.stem, "family": family(path.stem), "flip": flip,
            "kl": float(r["kl"].double().mean()), "tv": float(r["tv"].double().mean())}


def fit(pts: list[dict]) -> tuple[float, float] | None:
    ok = [p for p in pts if 1e-3 < p["kl"] < 2 and p["flip"] > 0]
    if len(ok) < 3:
        return None
    a, b = np.polyfit(np.log([p["kl"] for p in ok]), np.log([p["flip"] for p in ok]), 1)
    return float(a), float(b)


def family_devs(pts: list[dict]) -> dict:
    """Desvio multiplicativo de cada familia: curva em KL sem ela, e flips/TV das outras."""
    out = {}
    for fam in sorted({p["family"] for p in pts}):
        own = [p for p in pts if p["family"] == fam and 1e-3 < p["kl"] < 2]
        rest = [p for p in pts if p["family"] != fam]
        f = fit(rest)
        if not own or f is None:
            continue
        a, b = f
        dev_kl = float(np.median([p["flip"] / np.exp(a * np.log(p["kl"]) + b) for p in own]))
        base = np.median([p["flip"] / p["tv"] for p in rest if p["tv"] > 0])
        dev_tv = float(np.median([p["flip"] / p["tv"] for p in own]) / base)
        out[fam] = {"n": len(own), "dev_kl": dev_kl, "dev_tv": dev_tv}
    return out


def summarize_ref(ref_dir: Path) -> dict:
    name = ref_dir.name
    ref = torch.load(ref_dir / name / "bf16.pt", weights_only=False)
    own = [point(ref, f) for f in sorted((ref_dir / name).glob("*.pt"))
           if f.stem not in ("bf16",) and not f.stem.startswith("ofc@")]
    fitted = [p for p in own if p["family"] not in ("raw", "single", "other")]
    f = fit(fitted)
    ratios = [p["flip"] / p["tv"] for p in fitted if p["tv"] > 0]
    win = [p["flip"] / np.sqrt(p["kl"]) for p in fitted if 1e-3 < p["kl"] < 0.05]
    devs = family_devs(fitted)
    s = {"ref": name, "n_configs": len(fitted), "points": own,
         "exponent": f[0] if f else None, "flip_over_tv_median": float(np.median(ratios)) if ratios else None,
         "kappa": float(np.median(win)) if win else None, "families": devs,
         "family_dev_kl_median_pct": float(np.median([100 * abs(v["dev_kl"] - 1) for v in devs.values()]))
         if devs else None,
         "family_dev_tv_median_pct": float(np.median([100 * abs(v["dev_tv"] - 1) for v in devs.values()]))
         if devs else None}
    lo, hi = PUBLISHED["exponent"]
    s["P1_exponent_in_range"] = None if f is None else bool(lo <= f[0] <= hi)
    lo, hi = PUBLISHED["flip_over_tv"]
    s["P3_ratio_in_range"] = None if not ratios else bool(lo <= s["flip_over_tv_median"] <= hi)
    # outros modelos lidos contra esta referencia (ex.: QAT contra o denso)
    cross = []
    base = s["flip_over_tv_median"]
    for d in sorted(p for p in ref_dir.iterdir() if p.is_dir() and p.name != name):
        for fpt in sorted(d.glob("*.pt")):
            pt = point(ref, fpt)
            pt["model_dir"] = d.name
            if f:
                pt["dev_kl"] = pt["flip"] / float(np.exp(f[0] * np.log(pt["kl"]) + f[1]))
            if base:
                pt["dev_tv"] = pt["flip"] / pt["tv"] / base
            cross.append(pt)
    s["cross"] = cross
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    root = Path(args.out)
    refs = sorted(d for d in root.iterdir() if d.is_dir() and (d / d.name / "bf16.pt").exists())
    report = {"root": str(root), "published": PUBLISHED, "refs": [], "moe_routing": {}}
    print(f"{'referencia':44s} {'n':>3s} {'expoente':>8s} {'flip/TV':>7s} {'kappa':>6s} "
          f"{'dKL%':>5s} {'dTV%':>5s}  P1 P3")
    for d in refs:
        s = summarize_ref(d)
        report["refs"].append(s)
        fmt = lambda v, w=".3f": "   -" if v is None else format(v, w)
        print(f"{s['ref']:44s} {s['n_configs']:3d} {fmt(s['exponent']):>8s} {fmt(s['flip_over_tv_median']):>7s} "
              f"{fmt(s['kappa']):>6s} {fmt(s['family_dev_kl_median_pct'], '.1f'):>5s} "
              f"{fmt(s['family_dev_tv_median_pct'], '.1f'):>5s}  "
              f"{'ok' if s['P1_exponent_in_range'] else '--'} {'ok' if s['P3_ratio_in_range'] else '--'}")
        for c in s["cross"]:
            print(f"   x {c['model_dir']:38s} {c['config']:6s} flip {c['flip']:.4f} KL {c['kl']:.4f} "
                  f"TV {c['tv']:.4f} flip/TV {c['flip'] / max(c['tv'], 1e-12):.2f} "
                  f"desvio KL {c.get('dev_kl', float('nan')):.2f} TV {c.get('dev_tv', float('nan')):.2f}")
    for f in sorted((root / "analysis").glob("moe_routing__*.json")):
        r = json.loads(f.read_text())
        report["moe_routing"][f.stem] = r
        print(f"\n{f.stem}: margem do roteador mediana {r['router_margin']['median']:.3f}, "
              f"{100 * r['router_margin']['frac_below_0.1']:.1f}% < 0,1")
        for c, v in r["configs"].items():
            print(f"   {c:8s} flip livre {v['free']['flip']:.4f} congelado {v['frozen']['flip']:.4f} "
                  f"(roteamento {100 * v['routing_share_of_flips']:.0f}% dos flips) | troca de rota "
                  f"{v['route_change_rate']:.3f} | flips/TV {v['free']['flip_over_tv']:.2f} -> "
                  f"{v['frozen']['flip_over_tv']:.2f}")
    out = root / "analysis" / "ext_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\ngravado: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
