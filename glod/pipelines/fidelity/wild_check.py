#!/usr/bin/env python3
"""Avalia o teste pre-registrado "in the wild" (results/inthewild_predictions.json).

Le as referencias `<modelo>__<corpus>__fp32` da arvore GLOD_RESULTS (separada da do paper)
e aplica os criterios W1-W5 exatamente como registrados, com as exclusoes registradas
(execucao com KL ou TV nao finito = falha do kernel, fora dos criterios).

    GLOD_RESULTS=/local/$USER/ews_results/fid_wild python -m glod wild-check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from glod.paths import OUT, ref_slug


def load_points(ref_dir: Path) -> dict[str, dict]:
    name = ref_dir.name
    ref = torch.load(ref_dir / name / "bf16.pt", weights_only=False)
    pts = {}
    for f in sorted((ref_dir / name).glob("*.pt")):
        if f.stem == "bf16":
            continue
        r = torch.load(f, weights_only=False)
        kl, tv = float(r["kl"].double().mean()), float(r["tv"].double().mean())
        pts[f.stem] = {"flip": float((r["top1"] != ref["top1"]).float().mean()), "kl": kl, "tv": tv,
                       "finite": bool(np.isfinite(kl) and np.isfinite(tv))}
    return pts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registration", default="results/inthewild_predictions.json")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    reg = json.loads(Path(args.registration).read_text())
    body = {k: v for k, v in reg.items() if k != "sha256_of_body"}
    ok_hash = hashlib.sha256(json.dumps(body, indent=1, sort_keys=True).encode()).hexdigest() == reg["sha256_of_body"]
    root = Path(args.out)
    sim, real = reg["configs_simulated"], reg["configs_real_kernel"]
    rel = [c.replace("/", "--") for c in reg["configs_released_weights"]]
    res = {"registration_hash_ok": ok_hash, "models": {}, "kernel_failures": []}
    lo3, hi3 = 0.89, 1.29
    for model in reg["models"]:
        refs = {}
        for c in reg["corpora"]:
            d = root / ref_slug(model, corpus=c, fp32=True)
            if (d / d.name / "bf16.pt").exists():
                refs[c] = load_points(d)
        m = {"corpora_scored": sorted(refs), "W1": {}, "W2": {}, "W3": {}, "W4": {}}
        for c, pts in refs.items():
            for k, v in pts.items():
                if not v["finite"]:
                    res["kernel_failures"].append(f"{model} {c} {k}")
            s = {k: v for k, v in pts.items() if k in sim and v["finite"] and v["flip"] > 0}
            fit = [v for v in s.values() if 1e-3 < v["kl"] < 2]
            if len(s) >= 8 and len(fit) >= 3:
                a = float(np.polyfit(np.log([v["kl"] for v in fit]), np.log([v["flip"] for v in fit]), 1)[0])
                m["W1"][c] = {"exponent": a, "pass": 0.459 <= a <= 0.578}
            if s:
                ratios = [v["flip"] / v["tv"] for v in s.values()]
                med = float(np.median(ratios))
                m["W2"][c] = {"median_flip_over_tv": med, "pass": 0.99 <= med <= 1.25}
                inside = float(np.mean([lo3 <= x <= hi3 for x in ratios]))
                m["W3"][c] = {"share_inside": inside, "n": len(ratios), "pass": inside >= 0.90}
            for k in real + rel:
                v = pts.get(k)
                if v and v["finite"]:
                    x = v["flip"] / v["tv"]
                    m["W4"][f"{c}:{k}"] = {"flip_over_tv": x, "pass": lo3 <= x <= hi3}
        # W5: transferencia entre corpora dentro do modelo
        err_tv, err_kl = [], []
        cs = sorted(refs)
        for A in cs:
            sa = [v for k, v in refs[A].items() if k in sim and v["finite"] and v["flip"] > 0]
            if not sa:
                continue
            r_tv = np.median([v["flip"] / v["tv"] for v in sa])
            r_kl = np.median([v["flip"] / np.sqrt(v["kl"]) for v in sa])
            for B in cs:
                if B == A:
                    continue
                for k, v in refs[B].items():
                    if k in sim and v["finite"] and v["flip"] > 0:
                        err_tv.append(abs(r_tv * v["tv"] / v["flip"] - 1))
                        err_kl.append(abs(r_kl * np.sqrt(v["kl"]) / v["flip"] - 1))
        if err_tv:
            mt, mk = float(np.median(err_tv)), float(np.median(err_kl))
            m["W5"] = {"median_err_tv": mt, "median_err_kl": mk, "n": len(err_tv), "pass": mt < mk and mt <= 0.10}
        for w in ("W1", "W2", "W3", "W4"):
            m[f"{w}_passed"] = f"{sum(x['pass'] for x in m[w].values())} of {len(m[w])}"
        res["models"][model] = m
    out = root / "analysis" / "wild_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"registro integro: {ok_hash}; falhas de kernel excluidas: {len(res['kernel_failures'])}")
    for model, m in res["models"].items():
        w5 = m.get("W5", {})
        print(f"{model}: corpora {m['corpora_scored']} | W1 {m['W1_passed']} | W2 {m['W2_passed']} | "
              f"W3 {m['W3_passed']} | W4 {m['W4_passed']} | W5 "
              + (f"TV {100 * w5['median_err_tv']:.1f}% x KL {100 * w5['median_err_kl']:.1f}% -> "
                 f"{'passa' if w5['pass'] else 'falha'}" if w5 else "-"))
    print(f"gravado: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
