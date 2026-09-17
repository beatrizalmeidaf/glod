#!/usr/bin/env python3
"""Experimento de cerco 2: a variacao de kappa entre DOMINIOS e a mesma geometria
que explica a variacao entre MODELOS?

Para cada (modelo, dominio in {gsm8k, mmlu_pt}) calcula, so com o denso:
  p_gap_lt1   fracao de tokens com gap top1-top2 < 1 nat
  margin_mean media de p1 - p2
e, das configs comprimidas, kappa = mediana(flips / sqrt(KL)) no regime 1e-3 < KL < 0.2.
Ajusta kappa ~ preditor usando SO os pontos modelo-agregados e testa se o mesmo
ajuste preve os pontos por dominio (sem reajustar).

    python ews_domain.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from scipy.stats import pearsonr

from ews.paths import OUT as ROOT


def load(p):
    return torch.load(p, map_location="cpu", mmap=True, weights_only=False)


def main() -> int:
    law = json.loads((ROOT / "analysis" / "law.json").read_text())
    pts = []
    for rs in sorted(law["fits"]):
        ref_p = ROOT / rs / rs / "bf16.pt"
        cp = ROOT / "corpora" / f"{rs.replace('__fp32', '')}.json"
        if not ref_p.exists() or not cp.exists():
            continue
        ref = load(ref_p)
        corpus = json.loads(cp.read_text())
        src = np.array([r["source"] for r, g in zip(corpus["records"], corpus["gen_ids"]) for _ in g])
        v = ref["topv"].double()
        gap = (v[:, 0] - v[:, 1]).numpy()
        marg = (v[:, 0].exp() - v[:, 1].exp()).numpy()
        cfgs = [r for r in law["rows"] if r["ref"] == rs and r["model"] == rs
                and r["family"] not in ("outro modelo", "camada") and 1e-3 < r["kl"] < 0.2]
        files = {r["config"]: load(ROOT / rs / rs / (r["config"].replace("/", "--") + ".pt")) for r in cfgs}
        for dom in ("todos", "gsm8k", "mmlu_pt"):
            m = np.ones(len(src), bool) if dom == "todos" else (src == dom)
            mt = torch.from_numpy(m)
            ks = []
            for name, c in files.items():
                kl = c["kl"][mt].double().mean().item()
                fl = (c["top1"][mt] != ref["top1"][mt]).double().mean().item()
                if kl > 1e-4:
                    ks.append(fl / math.sqrt(kl))
            pts.append({"ref": rs, "dom": dom, "kappa": float(np.median(ks)),
                        "p_gap_lt1": float((gap[m] < 1).mean()), "margin_mean": float(marg[m].mean()),
                        "n_cfg": len(ks)})
    agg = [p for p in pts if p["dom"] == "todos"]
    dom = [p for p in pts if p["dom"] != "todos"]
    print(f"{'referencia':32s} {'dominio':8s} {'kappa':>6s} {'p_gap<1':>8s} {'margem':>7s}")
    for p in pts:
        print(f"{p['ref'][:32]:32s} {p['dom']:8s} {p['kappa']:6.3f} {p['p_gap_lt1']:8.3f} {p['margin_mean']:7.3f}")
    out = {}
    for pred in ("p_gap_lt1", "margin_mean"):
        x = np.array([p[pred] for p in agg]); y = np.array([p["kappa"] for p in agg])
        c = np.polyfit(x, y, 1)
        xd = np.array([p[pred] for p in dom]); yd = np.array([p["kappa"] for p in dom])
        yh = np.polyval(c, xd)
        r2_transfer = 1 - ((yd - yh) ** 2).sum() / ((yd - yd.mean()) ** 2).sum()
        # a diferenca mmlu - gsm8k dentro de cada modelo e prevista?
        pairs = {}
        for p in dom:
            pairs.setdefault(p["ref"], {})[p["dom"]] = p
        dk = [v["mmlu_pt"]["kappa"] - v["gsm8k"]["kappa"] for v in pairs.values() if len(v) == 2]
        dh = [np.polyval(c, v["mmlu_pt"][pred]) - np.polyval(c, v["gsm8k"][pred]) for v in pairs.values() if len(v) == 2]
        print(f"\n[{pred}] ajuste nos agregados: kappa = {c[0]:+.3f}*x {c[1]:+.3f} (r={pearsonr(x, y)[0]:+.2f}, n={len(x)})")
        print(f"   transferido aos pontos por dominio SEM reajuste: R2 = {r2_transfer:.3f} (n={len(dom)}), "
              f"r = {pearsonr(xd, yd)[0]:+.2f}")
        print(f"   diferenca dentro do modelo (MMLU - GSM8K): observada media {np.mean(dk):+.3f}, prevista "
              f"{np.mean(dh):+.3f}; r(obs, prev) = {pearsonr(dk, dh)[0]:+.2f} (n={len(dk)})")
        out[pred] = {"coef": c.tolist(), "r2_transfer": float(r2_transfer),
                     "delta_obs": float(np.mean(dk)), "delta_pred": float(np.mean(dh))}
    (ROOT / "analysis" / "domain.json").write_text(json.dumps({"points": pts, "fits": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
