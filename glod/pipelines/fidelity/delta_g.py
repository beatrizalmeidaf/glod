#!/usr/bin/env python3
"""Duas perguntas sobre o deslocamento de margem dg = (q_i1 - q_i2) - (p_i1 - p_i2).

1. Simetria de sinal. O argumento de primeira ordem supoe dg simetrico em sinal
   (flips ~ rho_c E|dg| / 2). Medimos, nas referencias com cabeca fp32, a fracao de
   posicoes frageis (0 < g < 1 nat) com dg < 0 e a assimetria E[dg] / E|dg|.
   Um compressor que empurrasse sistematicamente as margens para baixo (ou para cima)
   quebraria o fator 1/2.

2. Por que kappa falha em drafts cross-model? kappa = rho_c * A / sqrt(2), e rho_c e
   do alvo; se flips/sqrt(KL) cai para um draft de outro modelo, ou A cai (o draft
   desloca menos as margens por unidade de KL) ou a fracao de dg que vira flip muda.
   Comparamos A e flips / (rho_c E|dg| / 2) dos drafts cross-model com os das
   configuracoes comprimidas do mesmo alvo, nas referencias bf16 (os drafts so foram
   pontuados ai), fora de empates.

    python -m glod delta-g
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os

import numpy as np
import torch

from glod.paths import OUT

AN = OUT / "analysis"
FAMILY = (("gptq", "gptq"), ("awq", "awq"), ("sgpt", "sparsegpt"), ("wanda", "wanda"),
          ("mag", "magnitude"), ("kvt", "kv"), ("kv", "kv"), ("skip", "skip"), ("lq", "camada"),
          ("ofc", "oficial"))


def family_of(cfg: str) -> str:
    for pre, fam in FAMILY:
        if cfg.startswith(pre):
            return fam
    if cfg[0] == "u" and cfg[1:].isdigit():
        return "rtn"
    if cfg[0] == "g" and cfg[1:].isdigit() or cfg.startswith("s") and "g" in cfg:
        return "gauss"
    return "other"


def dg_stats(ref: dict, c: dict, mask: torch.Tensor | None = None) -> dict | None:
    if mask is None:
        mask = torch.ones(len(ref["top1"]), dtype=torch.bool)
    p = ref["topv"][:, :2].double()
    q = c["at_ref"][:, :2].double()
    g = p[:, 0] - p[:, 1]
    dg = (q[:, 0] - q[:, 1]) - g
    frag = (g > 0) & (g < 1.0) & mask     # fora de empates exatos e perto da fronteira
    if int(frag.sum()) < 200:
        return None
    d = dg[frag]
    kl = c["kl"][mask].double().mean().item()
    return {"kl": kl, "flip": float((c["top1"] != ref["top1"])[mask].double().mean()),
            "share_negative": float((d < 0).double().mean()),
            "asym": float(d.mean() / d.abs().mean()),
            "E_abs_dg": float(d.abs().mean()), "A": float(d.abs().mean() / math.sqrt(2 * kl))}


def rho_c(ref: dict, mask: torch.Tensor) -> float:
    g = (ref["topv"][:, 0] - ref["topv"][:, 1]).double()[mask]
    hs = np.array([0.02, 0.05, 0.1, 0.2])
    dens = np.array([float(((g > 0) & (g < h)).double().mean()) / h for h in hs])
    b, a = np.polyfit(hs, dens, 1)
    return float(a)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    # 1. simetria, referencias fp32
    sym = []
    for rp in sorted(glob.glob(str(OUT / "*__fp32" / "*__fp32" / "bf16.pt"))):
        d = os.path.dirname(rp)
        ref = torch.load(rp, map_location="cpu", mmap=True, weights_only=False)
        for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
            cfg = os.path.basename(f)[:-3]
            if cfg == "bf16" or cfg.startswith("fid_"):
                continue
            c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            if "at_ref" not in c or not (1e-3 < c["kl"].double().mean().item() < 0.25):
                continue
            s = dg_stats(ref, c)
            if s:
                sym.append({"ref": os.path.basename(d), "config": cfg, "family": family_of(cfg), **s})
    byf = collections.defaultdict(list)
    for r in sym:
        byf[r["family"]].append(r)
    out = {"symmetry": {
        "n_configs": len(sym), "n_refs": len({r["ref"] for r in sym}),
        "share_negative_median": float(np.median([r["share_negative"] for r in sym])),
        "share_negative_p5": float(np.percentile([r["share_negative"] for r in sym], 5)),
        "share_negative_p95": float(np.percentile([r["share_negative"] for r in sym], 95)),
        "asym_median": float(np.median([r["asym"] for r in sym])),
        "asym_max_abs": float(max(abs(r["asym"]) for r in sym)),
        "by_family": {f: {"n": len(v), "share_negative_median": float(np.median([r["share_negative"] for r in v])),
                          "asym_median": float(np.median([r["asym"] for r in v]))}
                      for f, v in sorted(byf.items()) if len(v) >= 5}}}

    # 2. drafts cross-model contra as configuracoes do proprio alvo (bf16): so os pares
    # da decodificacao especulativa e so as posicoes GSM8K, exatamente como o spec-law
    spec = json.loads((AN / "spec_law.json").read_text())["rows"]
    pairs = collections.defaultdict(list)
    for r in spec:
        if r["cross_model"]:
            pairs[(r["model"], r["draft_model"])].append(r["draft_config"])
    cross = []
    for (tgt, drf), cfgs in sorted(pairs.items()):
        d = os.path.join(OUT, tgt, drf)
        rfile = [f"{c}.pt" for c in cfgs if os.path.exists(os.path.join(d, f"{c}.pt"))]
        if not rfile:
            continue
        ref = torch.load(os.path.join(OUT, tgt, tgt, "bf16.pt"), map_location="cpu", mmap=True, weights_only=False)
        corpus = json.loads((OUT / "corpora" / f"{tgt}.json").read_text())
        gsm = torch.tensor([r["source"] == "gsm8k" for r, g in zip(corpus["records"], corpus["gen_ids"])
                            for _ in g])
        rc = rho_c(ref, gsm)
        own = []
        for f in sorted(glob.glob(os.path.join(OUT, tgt, tgt, "*.pt"))):
            cfg = os.path.basename(f)[:-3]
            if cfg == "bf16":
                continue
            c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            if "at_ref" in c and 1e-3 < c["kl"][gsm].double().mean().item() < 0.6:
                s = dg_stats(ref, c, gsm)
                if s:
                    own.append(s)
        if len(own) < 3:
            continue
        A_own = float(np.median([s["A"] for s in own]))
        conv_own = float(np.median([s["flip"] / (rc * s["E_abs_dg"] / 2) for s in own]))
        for f in rfile:
            c = torch.load(os.path.join(d, f), map_location="cpu", mmap=True, weights_only=False)
            s = dg_stats(ref, c, gsm)
            if s:
                cross.append({"target": tgt, "draft": f"{drf}_{f[:-3]}", **s,
                              "A_rel": s["A"] / A_own,
                              "conv_rel": (s["flip"] / (rc * s["E_abs_dg"] / 2)) / conv_own,
                              "kappa_rel": (s["flip"] / math.sqrt(s["kl"])) /
                                           float(np.median([o["flip"] / math.sqrt(o["kl"]) for o in own]))})
    out["cross"] = {"n": len(cross), "rows": cross,
                    "A_rel_median": float(np.median([r["A_rel"] for r in cross])),
                    "A_rel_min": float(min(r["A_rel"] for r in cross)),
                    "A_rel_max": float(max(r["A_rel"] for r in cross)),
                    "conv_rel_median": float(np.median([r["conv_rel"] for r in cross])),
                    "kappa_rel_median": float(np.median([r["kappa_rel"] for r in cross])),
                    "share_negative_median": float(np.median([r["share_negative"] for r in cross]))}
    (AN / "delta_g.json").write_text(json.dumps(out, indent=1))
    s = out["symmetry"]
    print(f"simetria: {s['n_configs']} configs, {s['n_refs']} refs | P(dg<0) mediana {s['share_negative_median']:.3f} "
          f"[p5 {s['share_negative_p5']:.3f}, p95 {s['share_negative_p95']:.3f}] | E[dg]/E|dg| mediana "
          f"{s['asym_median']:+.3f}, max |.| {s['asym_max_abs']:.3f}")
    for f, v in s["by_family"].items():
        print(f"   {f:10s} n={v['n']:<3} P(dg<0) {v['share_negative_median']:.3f} assimetria {v['asym_median']:+.3f}")
    c = out["cross"]
    print(f"cross-model: n={c['n']} | kappa rel. mediana {c['kappa_rel_median']:.2f} | A rel. mediana "
          f"{c['A_rel_median']:.2f} [{c['A_rel_min']:.2f}, {c['A_rel_max']:.2f}] | conversao dg->flip rel. "
          f"{c['conv_rel_median']:.2f} | P(dg<0) {c['share_negative_median']:.3f}")
    print(f"-> {AN / 'delta_g.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
