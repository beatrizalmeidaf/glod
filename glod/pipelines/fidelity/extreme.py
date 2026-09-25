#!/usr/bin/env python3
"""Onde a relacao flips x sqrt(KL) deixa de valer: configuracoes fora da janela medida.

`glod grid score --out-subdir extreme` grava configuracoes agressivas (2 bits, poda de
50-70%, KV em 2 bits) em OUT/extreme/<ref>/, fora do alcance das analises da lei, para
que nao entrem em nenhum ajuste. Aqui so extraimos (KL, TV, flips) de cada uma, para a
figura do regime.

    python -m glod extreme
"""
from __future__ import annotations

import argparse
import json

import torch

from glod.paths import OUT

AN = OUT / "analysis"


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    torch.set_grad_enabled(False)
    rows = []
    root = OUT / "extreme"
    for d in sorted(p.name for p in root.iterdir()) if root.exists() else []:
        ref = torch.load(OUT / d / d / "bf16.pt", weights_only=False)
        for f in sorted((root / d).glob("*.pt")):
            c = torch.load(f, weights_only=False)
            r = {"ref": d, "config": f.stem, "kl": c["kl"].double().mean().item(),
                 "tv": c["tv"].double().mean().item(),
                 "flip": (c["top1"] != ref["top1"]).double().mean().item()}
            rows.append(r)
            print(f"{d:32s} {f.stem:8s} KL {r['kl']:7.3f} TV {r['tv']:.3f} flip {r['flip']:.3f} "
                  f"flip/sqrtKL {r['flip'] / r['kl'] ** 0.5:.3f} flip/TV {r['flip'] / r['tv']:.3f}")
    AN.mkdir(parents=True, exist_ok=True)
    (AN / "extreme.json").write_text(json.dumps({"rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
