#!/usr/bin/env python3
"""Quantos flips o argumento de primeira ordem de fato conta?

A Eq. do argumento de primeira ordem conta um evento so: o top-2 da referencia (i2)
ultrapassar o top-1 (i1). A definicao operacional de flip e mais larga -- qualquer
mudanca de arg-max, inclusive um i3 passando i1 enquanto i2 fica para tras. Se essa
segunda classe nao for desprezivel, a boa concordancia agregada entre medido e
previsto esconde uma compensacao, e a decomposicao rho_c * A so vale "em agregado".

Para cada configuracao medimos, entre os flips:
  - cover:  fracao em que i2 ultrapassa i1 no modelo perturbado (o evento da Eq.);
  - win_i2: fracao em que o vencedor e o proprio i2.
`at_ref` guarda as log-probs perturbadas nos top-64 da referencia, entao nao e
preciso repontuar nada.

A conta principal usa as referencias com cabeca de saida fp32 (`__fp32`), as mesmas
em que a decomposicao e validada: com logits bf16 o modelo PERTURBADO tambem empata
em grade, e um empate exato i1 = i2 decidido pela ordem do indice vira "flip sem
cruzamento", o que derruba a cobertura para ~5% em algumas configuracoes de KL baixo.
As referencias bf16 entram so como comparacao.

    python -m glod flip-mechanism
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from glod.paths import OUT

AN = OUT / "analysis"
# faixas: a janela de kappa, a da tabela de rho_c * A, e o que fica acima dela
BANDS = {"kl_lt_0.05": (0.0, 0.05), "kl_lt_0.25": (0.0, 0.25), "kl_ge_0.25": (0.25, float("inf"))}


def scan(fp32: bool) -> list[dict]:
    rows = []
    refs = [p for p in sorted(glob.glob(str(OUT / "*" / "*" / "bf16.pt")))
            if ("__fp32" in p) == fp32 and f"{os.sep}adversarial{os.sep}" not in p]
    for rp in refs:
        d = os.path.dirname(rp)
        ref = torch.load(rp, map_location="cpu", mmap=True, weights_only=False)
        a, topi = ref["top1"].long(), ref["topi"].long()
        for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
            cfg = os.path.basename(f)[:-3]
            if cfg == "bf16" or cfg.startswith("fid_"):
                continue
            c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            if "at_ref" not in c:
                continue
            b = c["top1"].long()
            fl = a != b
            n = int(fl.sum())
            if n == 0:
                continue
            q = c["at_ref"]
            # desigualdade estrita: empate no perturbado nao e cruzamento
            cross = q[:, 1] > q[:, 0]
            rows.append({"ref": os.path.basename(d), "config": cfg,
                         "kl": float(c["kl"].double().mean()), "n_flips": n,
                         "cover": float((cross & fl).sum()) / n,
                         "win_i2": float((b[fl] == topi[fl, 1]).double().mean())})
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {"n_configs": len(rows), "n_refs": len({r["ref"] for r in rows})}
    for band, (lo, hi) in BANDS.items():
        sel = [r for r in rows if lo <= r["kl"] < hi]
        if not sel:
            continue
        cov = np.array([r["cover"] for r in sel])
        win = np.array([r["win_i2"] for r in sel])
        out[band] = {"n": len(sel),
                     "cover_median": float(np.median(cov)), "cover_p5": float(np.percentile(cov, 5)),
                     "cover_min": float(cov.min()),
                     "win_i2_median": float(np.median(win)), "win_i2_min": float(win.min())}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    fp32, bf16 = scan(True), scan(False)
    if not fp32:
        print("nenhuma referencia __fp32 encontrada")
        return 1
    out = {"fp32": summarize(fp32), "bf16": summarize(bf16), "rows_fp32": fp32}
    (AN / "flip_mechanism.json").write_text(json.dumps(out, indent=1))

    for head in ("fp32", "bf16"):
        s = out[head]
        print(f"\ncabeca {head}: {s['n_configs']} configuracoes, {s['n_refs']} referencias")
        for band in BANDS:
            if band in s:
                v = s[band]
                print(f"  {band:<11} n={v['n']:<4} i2 ultrapassa i1: mediana {v['cover_median']:.3f} "
                      f"p5 {v['cover_p5']:.3f} min {v['cover_min']:.3f} | vencedor = i2: "
                      f"mediana {v['win_i2_median']:.3f}")
    print(f"\n-> {AN / 'flip_mechanism.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
