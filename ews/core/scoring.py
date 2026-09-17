"""Scoring em batch da ultima posicao, com padding a esquerda.

A escada de capacidade da Fase 1 exige K x N forwards do 12B; sem batching
isso vira horas. Batchar prompts de comprimentos diferentes exige cuidado:

1. `padding_side="left"` para que a ultima posicao seja sempre token real.
2. `position_ids` calculados a partir da `attention_mask` - o transformers
   usa `arange(seq_len)` por padrao, o que atribui posicoes de RoPE erradas
   aos prompts que receberam padding.
3. `use_cache=False` - camada pulada nao escreve no seu slot de KV cache.
4. `logits_to_keep=1` - so a ultima posicao precisa de logits. Sem isso o
   `lm_head` materializa [batch, seq, 262208] e joga quase tudo fora: com
   batch 32 e seq 256 sao 4 GiB em bf16 (8 GiB apos `.float()`) por batch,
   desperdicio que estoura a VRAM assim que algo mais divide a GPU.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from tqdm.auto import tqdm

from ews.core.model_loader import LoadedModel

LOGGER = logging.getLogger(__name__)


def _left_pad_batch(
    loaded: LoadedModel, prompts: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tok = loaded.tokenizer
    previous_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(list(prompts), return_tensors="pt", padding=True)
    finally:
        tok.padding_side = previous_side

    input_ids = enc["input_ids"].to(loaded.device)
    attention_mask = enc["attention_mask"].to(loaded.device)
    # posicoes reais: o padding a esquerda nao pode consumir indices de RoPE
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids


@torch.inference_mode()
def score_choices(
    loaded: LoadedModel,
    prompts: Sequence[str],
    choice_ids: Sequence[int],
    *,
    batch_size: int = 16,
    desc: str | None = None,
    show_progress: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Devolve (logits das alternativas [N,4], diagnosticos por exemplo).

    Os diagnosticos sao resumos escalares - `choice_mass` (quanta massa de
    probabilidade caiu nos 4 ids) e `entropy` (entropia da distribuicao do
    vocabulario inteiro, em nats). Sao calculados por batch e descartados em
    seguida: devolver a distribuicao completa custaria ~2 GiB de RAM para
    N=2000. A entropia e o sinal barato que o gate da Fase 2 vai consumir.
    """
    choice_index = torch.tensor(list(choice_ids), device=loaded.device)
    all_choice_logits: list[torch.Tensor] = []
    all_mass: list[torch.Tensor] = []
    all_entropy: list[torch.Tensor] = []

    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc or "scoring", leave=False)

    for start in iterator:
        chunk = prompts[start : start + batch_size]
        input_ids, attention_mask, position_ids = _left_pad_batch(loaded, chunk)
        out = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        )
        last = out.logits[:, -1, :].float()
        all_choice_logits.append(last.index_select(dim=-1, index=choice_index).cpu())

        # Diagnosticos em fp64. O `log_softmax` sobre 262k classes em fp32
        # erra ~2e-3 nats - o bastante para a entropia ultrapassar log(V) e
        # violar a cota teorica. O custo e um tensor [batch, vocab] fp64
        # transitorio (~67 MiB com batch 32), desprezivel ao lado do forward.
        log_probs = last.double().log_softmax(dim=-1)
        probs = log_probs.exp()
        all_mass.append(
            probs.index_select(dim=-1, index=choice_index).sum(-1).float().cpu()
        )
        all_entropy.append((-(probs * log_probs).sum(dim=-1)).float().cpu())

    stats = {
        "choice_mass": torch.cat(all_mass),
        "entropy": torch.cat(all_entropy),
    }
    return torch.cat(all_choice_logits), stats


def predictions_and_correctness(
    choice_logits: torch.Tensor, targets: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax entre as 4 alternativas e vetor booleano de acerto."""
    preds = choice_logits.argmax(dim=-1)
    gold = torch.tensor(list(targets))
    return preds, preds.eq(gold)


def gold_nll(choice_logits: torch.Tensor, targets: Sequence[int]) -> torch.Tensor:
    """NLL da alternativa correta *renormalizada entre as 4 opcoes*.

    Sinal muito mais liso que a acuracia 0/1 - por isso e o que usamos no
    perfil de importancia por camada, onde N e pequeno e a acuracia seria
    ruido puro.
    """
    gold = torch.tensor(list(targets))
    return torch.nn.functional.cross_entropy(
        choice_logits, gold, reduction="none"
    )
