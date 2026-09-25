#!/usr/bin/env python3
"""Adaptatividade no sentido UTIL para compressao: flips no mesmo orcamento de BITS.

Usa a grade ja medida (RTN g128: 3/4/5/6/8 bits efetivos b+0.25, e bf16 = 16).
Cada token escolhe uma largura; custo = bits medios; objetivo = minimo de flips.

  estatico     cada largura uniforme, interpolado linearmente entre larguras vizinhas
               (mistura por camada/sequencia, o que precisao mista estatica consegue)
  sequencia    uma largura por sequencia (oraculo)
  entropia     gate REALIZAVEL: tokens em que o modelo de 3 bits tem maior entropia
               recebem mais bits (varre largura promovida e fracao de tokens)
  oraculo      por token, conhecendo os flips (limite superior)

Aproximacao declarada (Fase 1c): cada largura usa o proprio prefixo no teacher forcing.

    python -m glod adaptive-bits --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from glod.paths import OUT as ROOT
LEVELS = [("u3", 3.25), ("u4", 4.25), ("u5", 5.25), ("u6", 6.25), ("u8", 8.25), ("bf16", 16.0)]


def pick_min(pts, budgets):
    pts = sorted(pts)
    c = np.array([p[0] for p in pts]); f = np.array([p[1] for p in pts])
    return {str(b): float(f[c <= b + 1e-9].min()) if (c <= b + 1e-9).any() else float("nan") for b in budgets}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--budgets", type=float, nargs="+", default=[3.5, 3.75, 4.0, 4.25, 4.5, 5.0])
    args = p.parse_args(argv)
    rs = args.model.split("/")[-1]
    d = ROOT / rs / rs
    ref = torch.load(d / "bf16.pt", map_location="cpu", mmap=True, weights_only=False)
    corpus = json.loads((ROOT / "corpora" / f"{rs}.json").read_text())
    seq = np.concatenate([np.full(len(g), i) for i, g in enumerate(corpus["gen_ids"])])
    names, bits, F = [], [], []
    H3 = None
    for name, b in LEVELS:
        if name == "bf16":
            F.append(np.zeros(len(ref["top1"])))
        else:
            f = d / f"{name}.pt"
            if not f.exists():
                continue
            c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            F.append((c["top1"] != ref["top1"]).numpy().astype(float))
            if name == "u3":
                H3 = c["entropy"].double().numpy()
        names.append(name); bits.append(b)
    F = np.stack(F); bits = np.array(bits); T = F.shape[1]
    static_pts = [(bits[l], F[l].mean()) for l in range(len(bits))]
    # estatico com mistura linear entre larguras vizinhas (precisao mista estatica)
    mix = []
    for i in range(len(bits) - 1):
        for w in np.linspace(0, 1, 21):
            mix.append((w * bits[i] + (1 - w) * bits[i + 1], w * F[i].mean() + (1 - w) * F[i + 1].mean()))
    res = {"static": pick_min(static_pts + mix, args.budgets)}
    # oraculo por token (Lagrangiano)
    pts = []
    for lam in np.concatenate([[0.0], np.geomspace(1e-4, 10, 150)]):
        ch = np.argmin(F + lam * bits[:, None], axis=0)
        pts.append((bits[ch].mean(), F[ch, np.arange(T)].mean()))
    res["oracle_token"] = pick_min(pts, args.budgets)
    # oraculo por sequencia
    n_seq = seq.max() + 1
    Fs = np.stack([np.bincount(seq, weights=F[l], minlength=n_seq) for l in range(len(bits))])
    Ls = np.bincount(seq, minlength=n_seq)
    pts = []
    for lam in np.concatenate([[0.0], np.geomspace(1e-4, 10, 150)]):
        ch = np.argmin(Fs + lam * bits[:, None] * Ls[None], axis=0)
        pts.append(((bits[ch] * Ls).sum() / T, Fs[ch, np.arange(n_seq)].sum() / T))
    res["oracle_sequence"] = pick_min(pts, args.budgets)
    # gate realizavel por entropia do modelo de 3 bits: base u3, tokens de maior entropia promovidos a l
    order = np.argsort(-H3)
    pts = []
    base = F[0]
    for l in range(1, len(bits)):
        gain = (F[l] - base)[order]
        cum_f = base.mean() + np.cumsum(gain) / T
        cum_b = bits[0] + np.arange(1, T + 1) * (bits[l] - bits[0]) / T
        step = max(T // 500, 1)
        pts += list(zip(cum_b[::step], cum_f[::step]))
    res["gate_entropy_u3"] = pick_min(pts, args.budgets)
    # mesma politica com custo de CASCATA: roda a base de 3 bits em todo token E refaz os
    # promovidos na largura l (o sinal e ex-post). Custo incremental (acima) vale para
    # memoria/IO com residuos em planos de bits; cascata vale para computacao.
    pts = []
    for l in range(1, len(bits)):
        gain = (F[l] - base)[order]
        cum_f = base.mean() + np.cumsum(gain) / T
        cum_b = bits[0] + np.arange(1, T + 1) * bits[l] / T
        step = max(T // 500, 1)
        pts += list(zip(cum_b[::step], cum_f[::step]))
    res["gate_entropy_u3_cascade"] = pick_min(pts, args.budgets)
    print(f"{args.model}: flips (%) no mesmo orcamento de bits medios")
    cols = ["static", "gate_entropy_u3", "gate_entropy_u3_cascade", "oracle_sequence"]
    print(f"{'bits':>5s} " + " ".join(f"{c:>24s}" for c in cols) + f" {'ganho incr':>10s} {'ganho cascata':>13s}")
    for b in args.budgets:
        k = str(b)
        st = res["static"][k]
        g = lambda c: st / res[c][k] if res[c][k] == res[c][k] and res[c][k] > 0 else float("nan")
        print(f"{b:5.2f} " + " ".join(f"{100 * res[c][k]:24.2f}" for c in cols)
              + f" {g('gate_entropy_u3'):9.2f}x {g('gate_entropy_u3_cascade'):12.2f}x")
    out = ROOT / "analysis" / "adaptive_bits.json"
    allr = json.loads(out.read_text()) if out.exists() else {}
    allr[args.model] = res
    out.write_text(json.dumps(allr, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
