#!/usr/bin/env python3
"""Modelo isotropico EXATO de flips / TV: sem linearizar e sem extrapolar rho(0).

A previsao de primeira ordem (tv-theory), flips/TV = sqrt(2) rho(0) / E[h(p)], erra por
+9% em mediana. Ela faz tres aproximacoes: (1) linearizacao (perturbacao infinitesimal),
(2) rho(0) extrapolado das margens, (3) deslocamento isotropico dos logits. Aqui as duas
primeiras saem: sobre as distribuicoes top-64 + cauda do modelo DENSO, soma-se ruido
gaussiano i.i.d. N(0, s^2) aos 64 logits (a cauda agregada fica fixa) e mede-se KL, TV e
flips exatos, numa grade de s. Cada configuracao real e lida no s que reproduz o KL
medido; a razao flips/TV prevista nao usa nenhum numero da configuracao alem do KL.

O que sobra de erro e da hipotese isotropica. A validacao fora da amostra usa o corpus de
codigo (raiz holdout), que nao entra em nenhum desenho.

    python -m glod tv-exact
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

HOLDOUT = Path(os.environ.get("GLOD_HOLDOUT", "/local/user_beatrizalmeida/ews_results/holdout"))
KL_WINDOW = (1e-3, 0.25)
SIGMAS = np.geomspace(0.02, 3.0, 28)


@torch.no_grad()
def isotropic_curve(topv: torch.Tensor, dev: str, seed: int = 0, chunk: int = 32768) -> dict:
    """KL, TV e flips medios sob ruido isotropico, para cada s da grade."""
    lp = topv.double()
    tail = (1 - lp.exp().sum(-1)).clamp_min(1e-12).log()
    z = torch.cat([lp, tail[:, None]], 1)                     # 65 "tokens": top-64 + cauda
    g = torch.Generator(device=dev).manual_seed(seed)
    out = {"kl": [], "tv": [], "flip": []}
    for s in SIGMAS:
        kl = tv = fl = 0.0
        for a in range(0, z.shape[0], chunk):
            zz = z[a:a + chunk].to(dev)
            noise = torch.randn(zz.shape[0], 64, generator=g, device=dev, dtype=zz.dtype) * s
            zq = zz.clone()
            zq[:, :64] += noise
            lpn, lqn = zz.log_softmax(-1), zq.log_softmax(-1)
            P, Q = lpn.exp(), lqn.exp()
            kl += (P * (lpn - lqn)).sum().item()
            tv += 0.5 * (P - Q).abs().sum().item()
            fl += (zq[:, :64].argmax(-1) != 0).sum().item()   # arg-max denso e o indice 0
        n = z.shape[0]
        out["kl"].append(kl / n); out["tv"].append(tv / n); out["flip"].append(fl / n)
    return {k: np.array(v) for k, v in out.items()}


def at_kl(curve: dict, kl: float) -> dict:
    """Interpola a curva (em log) no KL medido."""
    x = np.log(curve["kl"])
    return {k: float(np.exp(np.interp(math.log(kl), x, np.log(curve[k])))) for k in ("tv", "flip")}


def configs(root: Path, d: str) -> list[dict]:
    base = root / d / d
    ref = torch.load(base / "bf16.pt", weights_only=False)
    rows = []
    for f in sorted(base.glob("*.pt")):
        if f.stem == "bf16":
            continue
        c = torch.load(f, weights_only=False)
        if "kl" not in c:
            continue
        K = c["kl"].double().clamp_min(0).mean().item()
        if not (KL_WINDOW[0] < K < KL_WINDOW[1]):
            continue
        rows.append({"config": f.stem, "kl": K, "tv": c["tv"].double().mean().item(),
                     "flip": (c["top1"] != ref["top1"]).double().mean().item()})
    return rows, ref["topv"]


def evaluate(root: Path, refs: list[str], dev: str) -> dict:
    per_ref, per_cfg = {}, []
    for d in refs:
        rows, topv = configs(root, d)
        if not rows:
            continue
        curve = isotropic_curve(topv, dev)
        for r in rows:
            m = at_kl(curve, r["kl"])
            r.update(pred_tv=m["tv"], pred_flip=m["flip"], pred_ratio=m["flip"] / m["tv"],
                     meas_ratio=r["flip"] / r["tv"], ref=d)
            per_cfg.append(r)
        pr = float(np.median([r["pred_ratio"] for r in rows])); me = float(np.median([r["meas_ratio"] for r in rows]))
        per_ref[d] = {"pred": pr, "meas": me, "n": len(rows),
                      "tv_err": float(np.median([r["pred_tv"] / r["tv"] - 1 for r in rows])),
                      "flip_err": float(np.median([r["pred_flip"] / r["flip"] - 1 for r in rows]))}
        print(f"{d[:34]:34s} n={len(rows):2d} previsto {pr:.3f} medido {me:.3f} | TV {100 * per_ref[d]['tv_err']:+.1f}% flips {100 * per_ref[d]['flip_err']:+.1f}%", flush=True)
    p = np.array([v["pred"] for v in per_ref.values()]); m = np.array([v["meas"] for v in per_ref.values()])
    e = p / m - 1
    summ = {"n_refs": len(p), "n_configs": len(per_cfg), "err_median": float(np.median(e)),
            "mape": float(np.mean(np.abs(e))), "max_abs_err": float(np.abs(e).max()),
            "pearson_log": float(stats.pearsonr(np.log(p), np.log(m)).statistic) if len(p) > 2 else None,
            "pred_range": [float(p.min()), float(p.max())], "meas_range": [float(m.min()), float(m.max())],
            "tv_err_median": float(np.median([v["tv_err"] for v in per_ref.values()])),
            "flip_err_median": float(np.median([v["flip_err"] for v in per_ref.values()]))}
    return {"summary": summ, "refs": per_ref, "configs": per_cfg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "analysis" / "tv_exact.json"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    tvm = json.loads((OUT / "analysis" / "tv_mechanism.json").read_text())["refs"]
    tvm = tvm if isinstance(tvm, list) else list(tvm.values())
    res = {"published": evaluate(OUT, [r["ref"] for r in tvm], args.device)}
    ho = [d for d in sorted(os.listdir(HOLDOUT)) if d.endswith("__fp32")] if HOLDOUT.exists() else []
    if ho:
        res["holdout_code"] = evaluate(HOLDOUT, ho, args.device)
    Path(args.out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, json.dumps(v["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
