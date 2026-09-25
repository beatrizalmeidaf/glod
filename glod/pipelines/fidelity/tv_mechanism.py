#!/usr/bin/env python3
"""Por que flips / TV fica perto de 1, e de onde vem a variacao de kappa?

1. Decomposicao de kappa (identidade exata por configuracao):

       kappa = flips / sqrt(KL) = (flips / TV) * c * J,
       c = E[TV_t] / E[sqrt(KL_t)]          (conversao por token, 1a ordem)
       J = E[sqrt(KL_t)] / sqrt(E[KL_t])    (Jensen: concentracao do KL em poucos tokens)

   KL e tirado da media ANTES da raiz; TV e flips sao de 1a ordem por token. Se c e
   flips/TV quase nao variam, a dependencia de kappa com o corpus e J.

2. Primeira ordem de flips / TV. Num token onde so o par top-2 compete, com massa
   m = p1 + p2 e margem g, TV_t ~ m s(g) |dg| com s(g) = sigmoid(g) sigmoid(-g), e
   int_0^inf s(g) dg = 1/2 exatamente. Com densidade de margens plana perto de zero,
   E[TV] = rho(0) E|dg| / 2 = a previsao de flips da Eq. 5, e a razao seria 1. Os
   desvios se fatoram (identidade, medida por configuracao):

       flips / TV = R_flat * P_pair * (obs/pred Eq. 5) * w_dg
       R_flat = rho_c(0) / (2 E[m s(g)])          so do modelo denso; > 1 se a densidade cai
       P_pair = E[m s(g) |dg|] / E[TV_t]          fracao do TV explicada pelo par top-2
       w_dg   = E_{g<1}|dg| / E_{m s}|dg|         dependencia de |dg| com a margem

3. Truncamento: TV no top-64 + cauda contra o TV exato no subconjunto de 4096 posicoes.

5. Temperatura. Escalar os logits densos e perturbados por 1/T nao muda nenhum arg-max
   (os flips sao os mesmos), mas muda o TV. Quando T -> 0 as duas distribuicoes viram
   one-hot e TV_t -> 1[flip_t] exatamente, logo flips/TV -> 1 para QUALQUER perturbacao;
   R_flat e P_pair tambem -> 1 (a densidade e plana na escala T e so o par top-2 carrega
   massa). O valor em T = 1 e a correcao de temperatura finita dessa identidade.
   Calculado sobre o top-64 renormalizado (at_ref so cobre o top-64 da referencia).

4. Sensibilidade do proxy phi = P(g < h) ao limiar h (0.5, 1, 2 nats), por referencia
   bf16 (o proxy da Sec. 5 usa as referencias bf16).

    python -m glod tv-mechanism
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from scipy import stats

from glod.paths import OUT, corpus_of

AN = OUT / "analysis"
H = (0.02, 0.05, 0.1, 0.2)          # janelas da extrapolacao de rho_c(0), como em analyze prop
PHI_H = (0.5, 1.0, 2.0)
TEMPS = (0.25, 0.5, 0.7, 1.0, 1.4, 2.0)
KL_WINDOW = (1e-3, 0.25)


def ref_dirs(fp32: bool) -> list[str]:
    out = []
    for d in sorted(os.listdir(OUT)):
        if not (OUT / d / d / "bf16.pt").exists():
            continue
        if ("__fp32" in d) == fp32:
            out.append(d)
    return out


def margins(ref: dict):
    lp = ref["topv"].double()
    g = lp[:, 0] - lp[:, 1]
    m = lp[:, 0].exp() + lp[:, 1].exp()
    s = torch.sigmoid(g) * torch.sigmoid(-g)
    return g, m, s


def rho_c(g: torch.Tensor) -> float:
    nt = g > 1e-6
    dens = [((g > 0) & (g < h) & nt).double().mean().item() / h for h in H]
    return float(np.polyval(np.polyfit(H, dens, 1), 0))


def analyze_ref(d: str) -> dict | None:
    ref = torch.load(OUT / d / d / "bf16.pt", weights_only=False)
    g, m, s = margins(ref)
    nt = g > 1e-6
    frag = nt & (g < 1.0)
    rho = rho_c(g)
    ms = (m * s * nt).mean().item()
    r_flat = rho / (2 * ms)
    rows = []
    for f in sorted((OUT / d / d).glob("*.pt")):
        if f.stem == "bf16":
            continue
        c = torch.load(f, weights_only=False)
        if "kl" not in c or "at_ref" not in c:
            continue
        kl = c["kl"].double().clamp_min(0)
        tv = c["tv"].double()
        K = kl.mean().item()
        if not (KL_WINDOW[0] < K < KL_WINDOW[1]):
            continue
        flip = (c["top1"] != ref["top1"]).double().mean().item()
        a = c["at_ref"].double()
        dg = ((a[:, 0] - a[:, 1]) - g).abs()
        TV = tv.mean().item()
        e5 = rho * dg[frag].mean().item() / 2
        w_ms = (m * s * nt)
        rows.append({
            "config": f.stem, "kl": K, "flip": flip, "tv": TV,
            "flip_over_tv": flip / TV, "kappa": flip / math.sqrt(K),
            "c": TV / kl.sqrt().mean().item(), "J": kl.sqrt().mean().item() / math.sqrt(K),
            "p_pair": (w_ms * dg).mean().item() / TV, "eq5_obs_pred": flip / e5,
            "w_dg": dg[frag].mean().item() / ((w_ms * dg).sum().item() / w_ms.sum().item()),
        })
    if not rows:
        return None
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "config"}
    temp = temperature_sweep(ref, d)
    return {"ref": d, "corpus": corpus_of(d.replace("__fp32", "")), "n_configs": len(rows),
            "rho_c": rho, "r_flat": r_flat, **{f"med_{k}": v for k, v in med.items()}, "temperature": temp}


def temperature_sweep(ref: dict, d: str) -> dict:
    """flips/TV, R_flat e P_pair com os logits densos e perturbados divididos por T."""
    z = ref["topv"].double()
    acc = {T: {"ratio": [], "r_flat": [], "p_pair": []} for T in TEMPS}
    for f in sorted((OUT / d / d).glob("*.pt")):
        if f.stem == "bf16":
            continue
        c = torch.load(f, weights_only=False)
        if "kl" not in c or "at_ref" not in c:
            continue
        if not (KL_WINDOW[0] < c["kl"].double().mean().item() < KL_WINDOW[1]):
            continue
        zq = c["at_ref"].double()
        flip = (c["top1"] != ref["top1"]).double().mean().item()
        for T in TEMPS:
            lp, lq = torch.log_softmax(z / T, -1), torch.log_softmax(zq / T, -1)
            tv = 0.5 * (lp.exp() - lq.exp()).abs().sum(-1)
            g = lp[:, 0] - lp[:, 1]
            m = lp[:, 0].exp() + lp[:, 1].exp()
            sg = torch.sigmoid(g) * torch.sigmoid(-g)
            dg = ((lq[:, 0] - lq[:, 1]) - g).abs()
            nt = g > 1e-6
            acc[T]["ratio"].append(flip / tv.mean().item())
            acc[T]["r_flat"].append(rho_c(g) / (2 * (m * sg * nt).mean().item()))
            acc[T]["p_pair"].append((m * sg * dg * nt).mean().item() / tv.mean().item())
    return {str(T): {k: float(np.median(v)) for k, v in a.items()} for T, a in acc.items()}


def truncation() -> dict:
    ratios, kls = [], []
    for d in ref_dirs(False) + ref_dirs(True):
        for f in sorted((OUT / d / d).glob("*.pt")):
            if f.stem == "bf16":
                continue
            c = torch.load(f, weights_only=False)
            if "tv_full" not in c:
                continue
            full = c["tv_full"].double()
            trunc = c["tv"][c["subset"]].double()
            ok = torch.isfinite(full)
            if ok.sum() == 0 or trunc[ok].mean() <= 0:
                continue
            ratios.append(full[ok].mean().item() / trunc[ok].mean().item())
            kls.append(c["kl"].double().mean().item())
    r, k = np.array(ratios), np.array(kls)
    return {"n": len(r), "median": float(np.median(r)), "p95": float(np.percentile(r, 95)),
            "max": float(r.max()), "min": float(r.min()),
            "spearman_vs_kl": float(stats.spearmanr(r, k)[0]),
            "max_above_1nat": float(r[k > 1].max()) if (k > 1).any() else None}


def phi_thresholds() -> list[dict]:
    out = []
    for d in ref_dirs(False):
        ref = torch.load(OUT / d / d / "bf16.pt", weights_only=False)
        g, _, _ = margins(ref)
        out.append({"ref": d, **{f"phi_{h}": float((g < h).double().mean().item()) for h in PHI_H}})
    return out


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    torch.set_grad_enabled(False)
    refs = []
    for d in ref_dirs(True):
        r = analyze_ref(d)
        if r is not None:
            refs.append(r)
            print(f"{d:46s} n={r['n_configs']:2d} flip/TV {r['med_flip_over_tv']:.3f} = R_flat {r['r_flat']:.2f}"
                  f" x P_pair {r['med_p_pair']:.3f} x Eq5 {r['med_eq5_obs_pred']:.3f} x w {r['med_w_dg']:.3f}"
                  f" | kappa {r['med_kappa']:.3f} c {r['med_c']:.3f} J {r['med_J']:.3f}", flush=True)
    lk = np.log([r["med_kappa"] for r in refs])
    share = {}
    for k in ("flip_over_tv", "c", "J"):
        x = np.log([r[f"med_{k}"] for r in refs])
        share[k] = float(np.cov(x, lk)[0, 1] / lk.var(ddof=1))
    closure = [r["med_kappa"] / (r["med_flip_over_tv"] * r["med_c"] * r["med_J"]) for r in refs]
    fot = np.array([r["med_flip_over_tv"] for r in refs])
    summary = {
        "n_refs": len(refs),
        "var_share_log_kappa": share,
        "kappa_closure_median": float(np.median(closure)),
        "range": {k: [float(min(r[k] for r in refs)), float(max(r[k] for r in refs))]
                  for k in ("med_flip_over_tv", "med_c", "med_J", "med_kappa", "r_flat", "med_p_pair")},
        "cv": {k: float(np.std([r[k] for r in refs]) / np.mean([r[k] for r in refs]))
               for k in ("med_flip_over_tv", "med_c", "med_J", "med_kappa")},
        "spearman_flip_tv_vs_r_flat": float(stats.spearmanr(fot, [r["r_flat"] for r in refs])[0]),
        "spearman_r_flat_vs_p_pair": float(stats.spearmanr([r["r_flat"] for r in refs],
                                                           [r["med_p_pair"] for r in refs])[0]),
    }
    tsum = {}
    for T in TEMPS:
        r = np.array([x["temperature"][str(T)]["ratio"] for x in refs])
        rf = np.array([x["temperature"][str(T)]["r_flat"] for x in refs])
        pp = np.array([x["temperature"][str(T)]["p_pair"] for x in refs])
        tsum[str(T)] = {"ratio_median": float(np.median(r)), "ratio_min": float(r.min()), "ratio_max": float(r.max()),
                        "r_flat_median": float(np.median(rf)), "r_flat_max": float(rf.max()),
                        "p_pair_median": float(np.median(pp)), "p_pair_min": float(pp.min())}
    # a razao cresce monotonamente com T em quantas referencias?
    mono = sum(all(np.diff([x["temperature"][str(T)]["ratio"] for T in TEMPS]) >= -0.01) for x in refs)
    summary["temperature"] = tsum
    summary["temperature_monotone_refs"] = int(mono)
    print(json.dumps(summary, indent=1))
    trunc = truncation()
    print("truncamento TV:", trunc)
    AN.mkdir(parents=True, exist_ok=True)
    (AN / "tv_mechanism.json").write_text(json.dumps(
        {"summary": summary, "refs": refs, "tv_truncation": trunc, "phi_thresholds": phi_thresholds()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
