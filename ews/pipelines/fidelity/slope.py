#!/usr/bin/env python3
"""Experimento B: por que a sensibilidade ao KL varia entre modelos.

Para cada modelo de referencia calcula preditores geometricos e de tarefa a
partir APENAS do modelo denso, e regride contra:
  s_flip = kappa  (flips por sqrt(KL), inclinacao token a token)
  s_task = |d_acc| / sqrt(KL)  (inclinacao KL -> acuracia GSM8K, ajuste por modelo)

Preditores (todos do denso, sem rodar nada comprimido):
  margin_mean      media de p(top1) - p(top2)
  gap_mean         media do gap de logit top1-top2 (nats)
  p_gap_lt1        fracao de tokens com gap < 1 nat  (tokens "na fronteira")
  rho0             densidade de gaps em 0 por nat (so com logits fp32)
  curv             E[1 - sum p^2]  (curvatura media do softmax)
  entropy_mean     entropia media
  len_mean         comprimento medio da geracao no GSM8K (cadeia de raciocinio)
  crit_frac        fracao de tokens cujo flip muda a resposta final (proxy: tokens
                   numericos/operadores na resposta do GSM8K)

    python ews_slope.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from ews.paths import OUT as ROOT
AN = ROOT / "analysis"


def load(p: Path) -> dict:
    return torch.load(p, map_location="cpu", mmap=True, weights_only=False)


def predictors(rs: str) -> dict | None:
    ref_p = ROOT / rs / rs / "bf16.pt"
    corpus_p = ROOT / "corpora" / f"{rs.replace('__fp32','')}.json"
    if not ref_p.exists() or not corpus_p.exists():
        return None
    ref = load(ref_p)
    corpus = json.loads(corpus_p.read_text())
    v = ref["topv"].double()
    gap = v[:, 0] - v[:, 1]
    p = v.exp()
    is_gsm = torch.tensor([r["source"] == "gsm8k" for r, g in zip(corpus["records"], corpus["gen_ids"])
                           for _ in g])
    lens = [len(g) for r, g in zip(corpus["records"], corpus["gen_ids"]) if r["source"] == "gsm8k"]
    tie = gap < 1e-6
    hs = torch.tensor([0.02, 0.05, 0.1, 0.2])
    dens = torch.stack([((gap > 1e-6) & (gap < h)).double().mean() / h for h in hs])
    A = torch.stack([torch.ones_like(hs), hs], 1).double()
    rho0 = float(torch.linalg.lstsq(A, dens[:, None]).solution[0])
    # tokens "criticos" no GSM8K: digitos e operadores dentro da resposta
    crit = 0.0
    n_tok = 0
    for r, g in zip(corpus["records"], corpus["gen_ids"]):
        if r["source"] != "gsm8k":
            continue
        n_tok += len(g)
    txt = " ".join(r["completion"] for r in corpus["records"] if r["source"] == "gsm8k")
    crit = len(re.findall(r"[\d+\-*/=]", txt)) / max(len(txt), 1)
    return {
        "margin_mean": (p[:, 0] - p[:, 1]).mean().item(),
        "gap_mean": gap.mean().item(),
        "gap_mean_gsm": gap[is_gsm].mean().item(),
        "p_gap_lt1": (gap < 1).double().mean().item(),
        "p_gap_lt1_gsm": (gap[is_gsm] < 1).double().mean().item(),
        "rho0": rho0,
        "tie_mass": tie.double().mean().item(),
        "curv": (1 - (p ** 2).sum(-1)).mean().item(),
        "entropy_mean": ref["entropy"].double().mean().item(),
        "len_mean": float(np.mean(lens)) if lens else float("nan"),
        "crit_char_frac": crit,
    }


def main() -> int:
    law = json.loads((AN / "law.json").read_text())
    cl = json.loads((AN / "closedloop.json").read_text()) if (AN / "closedloop.json").exists() else {"gsm8k": []}
    rows = []
    for rs, fit in law["fits"].items():
        pr = predictors(rs)
        if pr is None or fit["top"] is None:
            continue
        pts = [p for p in law["rows"] if p["ref"] == rs and p["model"] == rs and p["family"] != "outro modelo"]
        small = [p["flip"] / math.sqrt(p["kl"]) for p in pts if 1e-3 < p["kl"] < 0.05]
        gsm = [r for r in cl["gsm8k"] if r["model"] == rs]
        s_task = float("nan")
        if len(gsm) >= 5:
            x = np.sqrt([r["kl_gsm"] for r in gsm]); y = np.array([r["d_acc"] for r in gsm])
            s_task = -float(np.polyfit(x, y, 1)[0])
        rows.append({"ref": rs, "kappa": float(np.median(small)) if small else float("nan"),
                     "slope_flip": fit["top"]["slope"], "s_task": s_task, **pr})
    print(f"{'referencia':32s} {'kappa':>6s} {'s_task':>7s} {'gap_gsm':>8s} {'p_gap<1':>8s} {'curv':>6s} {'len':>6s} {'H':>6s}")
    for r in sorted(rows, key=lambda r: -(r["s_task"] if r["s_task"] == r["s_task"] else -1)):
        print(f"{r['ref']:32s} {r['kappa']:6.3f} {r['s_task']:7.3f} {r['gap_mean_gsm']:8.2f} "
              f"{r['p_gap_lt1_gsm']:8.3f} {r['curv']:6.3f} {r['len_mean']:6.1f} {r['entropy_mean']:6.3f}")
    keys = ["margin_mean", "gap_mean", "gap_mean_gsm", "p_gap_lt1", "p_gap_lt1_gsm", "rho0", "curv",
            "entropy_mean", "len_mean", "crit_char_frac", "kappa"]
    for target in ("kappa", "s_task"):
        have = [r for r in rows if r[target] == r[target]]
        print(f"\ncorrelacoes com {target} (n={len(have)}):")
        for k in keys:
            if k == target:
                continue
            x = np.array([r[k] for r in have]); y = np.array([r[target] for r in have])
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 4:
                continue
            print(f"   {k:16s} pearson {pearsonr(x[ok], y[ok])[0]:+.2f} (p={pearsonr(x[ok], y[ok])[1]:.3f}) "
                  f"spearman {spearmanr(x[ok], y[ok])[0]:+.2f}")
        # melhor preditor unico e regressao com 2 preditores
        best = max((abs(pearsonr(np.array([r[k] for r in have]), np.array([r[target] for r in have]))[0]), k)
                   for k in keys if k != target and np.isfinite([r[k] for r in have]).all())
        print(f"   melhor preditor unico: {best[1]} (|r|={best[0]:.2f})")
    (AN / "slope.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
