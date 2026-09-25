#!/usr/bin/env python3
"""Baixa para GLOD_HF_CACHE/datasets os datasets que os corpora usam.

    python scripts/download_datasets.py            # todos os que o registro conhece
    python scripts/download_datasets.py --only gsm8k mmlu_en

O CSV do MMLU PT-BR nao vem daqui: ele e versionado a mao em GLOD_DATA/mmlu_PT-BR.csv.
Falha de cota de disco e reportada por dataset e nao aborta os outros (foi o que
aconteceu com o c4 e o humaneval na DGX).
"""
from __future__ import annotations

import argparse
import os
import sys

# dataset por corpus: (path, config, split)
SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "mmlu_en": ("cais/mmlu", "all", "test"),
    "wikitext": ("Salesforce/wikitext", "wikitext-2-raw-v1", "train"),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    p.add_argument("--cache-dir", default=None,
                   help="default: $GLOD_HF_CACHE/datasets (ou ./data/datasets se nao houver)")
    args = p.parse_args(argv)

    from datasets import load_dataset

    root = args.cache_dir or os.path.join(
        os.environ.get("GLOD_HF_CACHE", os.path.join(os.getcwd(), "data")), "datasets")
    os.makedirs(root, exist_ok=True)
    print(f"cache de datasets: {root}")
    rc = 0
    for key in args.only:
        path, config, split = SOURCES[key]
        print(f"[{key}] {path} ({config}) split={split} ...", flush=True)
        try:
            ds = load_dataset(path, config, split=split, cache_dir=root)
            print(f"[{key}] ok: {len(ds)} linhas")
        except Exception as e:                      # cota, rede, dataset gated
            print(f"[{key}] FALHOU: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
    # wikitext tambem e usado como calibracao de GPTQ/AWQ/SparseGPT
    print("obs: a calibracao dos compressores usa o mesmo wikitext-2-raw-v1")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
