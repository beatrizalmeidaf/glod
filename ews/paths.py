"""Caminhos e nomes de referencia, num lugar so.

Tudo e configuravel por variavel de ambiente, porque o mesmo codigo roda na DGX
(resultados no disco local, cota de 500G no /raid) e dentro do container, onde os
diretorios entram por volume:

    EWS_RESULTS   raiz dos resultados do estudo de fidelidade (analysis/, corpora/, <ref>/)
    EWS_RAW       raiz dos resultados da Fase 1 (results/raw), usada pelos pipelines legados
    EWS_HF_CACHE  cache do Hugging Face (modelos e datasets)
    EWS_DATA      dados versionados a mao (CSV do MMLU PT-BR)
    EWS_FIGS      saida das figuras do paper
"""
from __future__ import annotations

import os
from pathlib import Path

OUT = Path(os.environ.get("EWS_RESULTS", "/local/user_beatrizalmeida/ews_results/fid"))
RAW = Path(os.environ.get("EWS_RAW", "results/raw"))
CACHE_DIR = os.environ.get("EWS_HF_CACHE", "/local/user_beatrizalmeida/hf_cache")
DATA_DIR = Path(os.environ.get("EWS_DATA", "data"))
FIGS = Path(os.environ.get("EWS_FIGS", "results/figs"))

MMLU_PT_CSV = str(DATA_DIR / "mmlu_PT-BR.csv")
#: corpus default: GSM8K + MMLU PT-BR, o mesmo dos resultados ja publicados no paper
DEFAULT_CORPUS = "mix"


def slug(model: str, revision: str | None = None) -> str:
    s = model.split("/")[-1]
    return f"{s}@{revision}" if revision else s


def ref_slug(model: str, revision: str | None = None, *, corpus: str = DEFAULT_CORPUS,
             fp32: bool = False) -> str:
    """Nome da referencia no disco.

    Os sufixos sao acumulativos e o corpus default NAO tem sufixo: e o que mantem
    os resultados antigos validos (`Qwen3-4B`, `Qwen3-4B__fp32`) e da um espaco
    separado para cada corpus novo (`Qwen3-4B__mmlu_en`).
    """
    s = slug(model, revision)
    if corpus and corpus != DEFAULT_CORPUS:
        s += f"__{corpus}"
    if fp32:
        s += "__fp32"
    return s


def corpus_of(ref: str) -> str:
    """Corpus de uma referencia no disco: `Qwen3-4B__mmlu_en__fp32` -> `mmlu_en`, sem sufixo -> `mix`."""
    from ews.corpora.registry import CORPORA

    for part in ref.split("__")[1:]:
        if part in CORPORA:
            return part
    return DEFAULT_CORPUS


def model_of(ref: str) -> str:
    """Modelo de uma referencia no disco: `Qwen3-4B__mmlu_en__fp32` -> `Qwen3-4B`."""
    return ref.split("__")[0]


def corpus_path(ref: str) -> Path:
    return OUT / "corpora" / f"{ref}.json"


def add_corpus_arg(parser) -> None:
    """--corpus, com os ids que `ews.corpora.registry` conhece."""
    from ews.corpora.registry import CORPORA

    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
                        help="corpus de referencia (default: mix = GSM8K + MMLU PT-BR)")
