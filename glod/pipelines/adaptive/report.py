#!/usr/bin/env python3
"""Secao 7: adaptatividade por token em geracao real, contra a fronteira estatica.

Para cada politica adaptativa (cascade/predictive a um limiar de entropia), interpola
a fronteira ESTATICA (precisao mista entre projecoes, sem troca por token) no MESMO
orcamento de bits e reporta a diferenca de acuracia. Contabiliza bits de dois modos:
  incremental  so o residual de bits dos tokens promovidos (memoria/IO)
  cascata      o token promovido paga base + alta (compute refeito)

    python -m glod adaptive-report
"""
from __future__ import annotations

import json
import math

import numpy as np

from glod.paths import OUT as ROOT
AN = ROOT / "analysis"


def main() -> int:
    out = {"models": {}}
    print(f"{'modelo':22s} {'politica':18s} {'bits':>6s} {'acc':>6s} | {'estatico no mesmo bits':>22s} "
          f"{'diferenca':>10s}")
    deltas = {"incremental": [], "cascade": []}
    for d in sorted((ROOT / "adaptive_closedloop").iterdir()):
        f = d / "b3_h8.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        pts: dict[float, float] = {}
        for k, v in res.items():
            if isinstance(v, dict) and (k.startswith("mix@") or k in ("static_base", "static_high")):
                # a fronteira tem de ser uma funcao: se dois pontos estaticos caem no mesmo
                # orcamento de bits, a interpolacao viraria lixo (acontece enquanto o job
                # roda, porque os bits dos mix@ so sao corrigidos ao final)
                pts[round(v["bits_incremental"], 3)] = max(v["acc"], pts.get(round(v["bits_incremental"], 3), 0))
        st = sorted(pts.items())
        if len(st) < 3 or len(pts) < sum(1 for k, v in res.items() if isinstance(v, dict)
                                         and (k.startswith("mix@") or k in ("static_base", "static_high"))):
            print(f"{d.name[:22]:22s} (fronteira estatica invalida: {len(st)} pontos distintos; "
                  f"job incompleto?)")
            continue
        if len(st) < 3:
            print(f"{d.name[:22]:22s} (fronteira estatica incompleta: {len(st)} pontos)")
            continue
        sx, sy = np.array([p[0] for p in st]), np.array([p[1] for p in st])
        rows = []
        for k, v in sorted(res.items(), key=lambda kv: kv[1]["bits_incremental"]
                           if isinstance(kv[1], dict) else 0):
            if not isinstance(v, dict) or not k.startswith(("cascade@", "predictive@")):
                continue
            for mode, key in (("incremental", "bits_incremental"), ("cascade", "bits_cascade")):
                b = v[key]
                if b < sx.min() or b > sx.max():
                    continue
                base = float(np.interp(b, sx, sy))
                dl = 100 * (v["acc"] - base)
                deltas[mode].append(dl)
                if mode == "incremental":
                    rows.append({"policy": k, "bits": b, "acc": v["acc"], "static": base, "delta_pp": dl})
                    print(f"{d.name[:22]:22s} {k:18s} {b:6.2f} {100*v['acc']:6.1f} | {100*base:22.1f} "
                          f"{dl:+9.1f}pp")
        out["models"][d.name] = {"static_frontier": st, "adaptive": rows}
    print()
    for mode, v in deltas.items():
        if v:
            se = np.std(v, ddof=1) / math.sqrt(len(v))
            print(f"  adaptativo - estatico ({mode:11s}): {np.mean(v):+.2f}pp  IC95 "
                  f"[{np.mean(v) - 1.96 * se:+.2f}, {np.mean(v) + 1.96 * se:+.2f}]  (n={len(v)})")
    out["deltas"] = deltas
    (AN / "adaptive_closedloop.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {AN / 'adaptive_closedloop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
