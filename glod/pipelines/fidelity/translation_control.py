#!/usr/bin/env python3
"""O efeito de corpus sobre kappa e artefato da traducao automatica do MMLU?

O corpus primario (mix) e metade GSM8K e metade MMLU traduzido para o portugues.
Texto traduzido tem distribuicoes proprias (translationese), o que poderia mover a
densidade de margens por um motivo que nada tem a ver com dominio. Os 9 modelos
de gsm8k e mmlu-en tambem tem o mix, entao da para comparar, por modelo e na MESMA
configuracao:

  - kappa na metade MMLU-PT do mix  vs  kappa no corpus MMLU-en nativo;
  - kappa na metade GSM8K do mix    vs  kappa no corpus gsm8k (controle: mesmo
    idioma e mesma fonte, deve dar 1).

Se a primeira razao ficar perto de 1, nem a traducao nem o idioma movem kappa; o
teste nao separa os dois, e nao precisa, se nenhum deles tiver efeito. kappa e
medido na janela KL < 0.05 nos dois lados, como no resto do artigo.

    python -m glod translation-control
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

from glod.paths import OUT

AN = OUT / "analysis"
KL_MAX = 0.05
PAIRS = (("mmlu_pt", "mmlu_en"), ("gsm8k", "gsm8k"))


def src_mask(slug: str, src: str) -> torch.Tensor:
    c = json.loads((OUT / "corpora" / f"{slug}.json").read_text())
    return torch.tensor([s == src for r, g in zip(c["records"], c["gen_ids"])
                         for s in [r["source"]] * len(g)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    models = sorted(os.path.basename(x)[:-len("__mmlu_en")]
                    for x in glob.glob(str(OUT / "*__mmlu_en")))
    out = {}
    for part, other in PAIRS:
        per_model = {}
        for m in models:
            dm, do = OUT / m / m, OUT / f"{m}__{other}" / f"{m}__{other}"
            if not (dm / "bf16.pt").exists() or not (do / "bf16.pt").exists():
                continue
            am = torch.load(dm / "bf16.pt", map_location="cpu", mmap=True, weights_only=False)["top1"]
            ao = torch.load(do / "bf16.pt", map_location="cpu", mmap=True, weights_only=False)["top1"]
            k = src_mask(m, part)
            ka, kb = [], []
            for f in sorted(glob.glob(str(dm / "*.pt"))):
                cfg = os.path.basename(f)[:-3]
                fo = do / f"{cfg}.pt"
                if cfg == "bf16" or not fo.exists():
                    continue
                c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                o = torch.load(fo, map_location="cpu", mmap=True, weights_only=False)
                kl1 = c["kl"][k].double().mean().item()
                kl2 = o["kl"].double().mean().item()
                if not (kl1 < KL_MAX and kl2 < KL_MAX):
                    continue
                ka.append((c["top1"] != am)[k].double().mean().item() / math.sqrt(kl1))
                kb.append((o["top1"] != ao).double().mean().item() / math.sqrt(kl2))
            if ka:
                per_model[m] = {"n": len(ka), "kappa_mix_part": float(np.median(ka)),
                                "kappa_corpus": float(np.median(kb)),
                                "ratio": float(np.median(ka) / np.median(kb))}
        rat = [v["ratio"] for v in per_model.values()]
        out[f"{part}_vs_{other}"] = {
            "per_model": per_model, "n_models": len(rat),
            "n_configs_min": min(v["n"] for v in per_model.values()),
            "n_configs_max": max(v["n"] for v in per_model.values()),
            "ratio_median": float(np.median(rat)),
            "ratio_min": float(min(rat)), "ratio_max": float(max(rat))}
    (AN / "translation_control.json").write_text(json.dumps(out, indent=1))

    for key, v in out.items():
        print(f"\n{key}: {v['n_models']} modelos, razao mediana {v['ratio_median']:.2f} "
              f"[{v['ratio_min']:.2f}, {v['ratio_max']:.2f}]")
        for m, r in v["per_model"].items():
            print(f"  {m:<28} n={r['n']:<3} {r['kappa_mix_part']:.3f} / {r['kappa_corpus']:.3f} "
                  f"= {r['ratio']:.2f}")
    print(f"\n-> {AN / 'translation_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
