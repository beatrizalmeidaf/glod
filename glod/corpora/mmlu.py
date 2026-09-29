"""Dataset MMLU (PT-BR) e formatacao de prompt para a Fase 1.

O scoring e por *logit da alternativa*: em vez de gerar texto, lemos a
distribuicao do proximo token e comparamos apenas os 4 ids de A/B/C/D.
Isso torna a avaliacao deterministica e barata (1 forward por exemplo),
que e o que permite varrer uma escada de capacidade inteira.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)

CHOICES: tuple[str, ...] = ("A", "B", "C", "D")
REQUIRED_COLUMNS = ("Question", "A", "B", "C", "D", "Answer", "Subject")


def format_mmlu_prompt(row: pd.Series) -> str:
    """Formato zero-shot em PT-BR (mantido igual ao da primeira rodada)."""
    return (
        f"Assunto: {row['Subject']}\n"
        f"Pergunta: {row['Question']}\n"
        f"A) {row['A']}\n"
        f"B) {row['B']}\n"
        f"C) {row['C']}\n"
        f"D) {row['D']}\n"
        f"Resposta:"
    )


def build_prompt(row: pd.Series, shots: Sequence[pd.Series] = ()) -> str:
    """Prompt n-shot: exemplares resolvidos + a questao alvo."""
    blocks = [
        f"{format_mmlu_prompt(s)} {str(s['Answer']).strip().upper()}" for s in shots
    ]
    blocks.append(format_mmlu_prompt(row))
    return "\n\n".join(blocks)


@dataclass
class MMLUSplit:
    """Conjunto de avaliacao ja materializado como prompts + gabarito."""

    prompts: list[str]
    targets: list[int]  # indice em CHOICES
    subjects: list[str]

    def __len__(self) -> int:
        return len(self.prompts)


def load_mmlu(
    csv_path: str,
    *,
    n_samples: int = 0,
    n_shot: int = 0,
    seed: int = 42,
) -> MMLUSplit:
    """Carrega o CSV, amostra `n_samples` e monta os prompts.

    Os exemplares few-shot saem de um pool **disjunto** do conjunto de
    avaliacao (evita contaminacao trivial).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes em {csv_path}: {missing}")

    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df[df["Answer"].astype(str).str.strip().str.upper().isin(CHOICES)]

    rng = df.sample(frac=1.0, random_state=seed)  # embaralha uma vez
    shot_pool = rng.iloc[:n_shot] if n_shot else rng.iloc[:0]
    evaluable = rng.iloc[n_shot:]

    if 0 < n_samples < len(evaluable):
        evaluable = evaluable.sample(n_samples, random_state=seed)

    shots = [shot_pool.iloc[i] for i in range(len(shot_pool))]
    prompts, targets, subjects = [], [], []
    for _, row in evaluable.iterrows():
        prompts.append(build_prompt(row, shots))
        targets.append(CHOICES.index(str(row["Answer"]).strip().upper()))
        subjects.append(str(row["Subject"]))

    LOGGER.info(
        "MMLU: %d questoes de avaliacao (%d-shot, %d assuntos) de %s",
        len(prompts), n_shot, len(set(subjects)), csv_path,
    )
    return MMLUSplit(prompts=prompts, targets=targets, subjects=subjects)


def resolve_choice_ids(
    tokenizer: PreTrainedTokenizerBase, *, leading_space: bool = True
) -> list[int]:
    """Ids dos tokens de alternativa.

    Critico: em tokenizers SentencePiece (Gemma), ``"A"`` e ``" A"`` sao ids
    DIFERENTES (236776 vs 562). Depois de ``"Resposta:"`` o modelo emite a
    variante com espaco - pontuar a variante sem espaco le confianca de uma
    regiao com ~0.02% da massa de probabilidade. Inofensivo para a acuracia
    (o ranking relativo vaza), mas fatal para o gate por entropia da Fase 2.
    """
    prefix = " " if leading_space else ""
    ids = []
    for c in CHOICES:
        enc = tokenizer.encode(prefix + c, add_special_tokens=False)
        if not enc:
            raise ValueError(f"tokenizer devolveu vazio para {prefix + c!r}")
        ids.append(enc[-1])
    if len(set(ids)) != len(CHOICES):
        raise ValueError(f"ids de alternativa colidiram: {ids}")
    return ids


# A massa de probabilidade nos 4 ids e calculada por batch em
# `glod.core.scoring.score_choices` (chave "choice_mass" dos diagnosticos), para
# nao materializar a distribuicao do vocabulario inteiro.
