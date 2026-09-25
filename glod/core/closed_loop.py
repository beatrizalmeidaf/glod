"""Fase 1d - avaliacao em MALHA FECHADA do gate por token.

O teacher forcing mede fidelidade token a token sobre a trajetoria da
referencia. Mas a metrica do `ews_ideia.md` e acuracia de tarefa x latencia:
um token divergente pode ser inofensivo (sinonimo) ou descarrilar a resposta.
Aqui o gate decide de fato qual token entra na sequencia, e o resto da geracao
condiciona nessa escolha.

Politica em cascata: a cada passo a base propoe; se a entropia da base passa do
limiar, o token da referencia e usado. Ambos os modelos processam todo token
comprometido para manter seus KV caches coerentes - o custo computacional disso
nao importa, porque os pesos lidos por token sao contabilizados analiticamente
(base sempre + referencia nos passos promovidos).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from transformers import DynamicCache

from glod.core.model_loader import LoadedModel


@dataclass
class ClosedLoopResult:
    gen_ids: list[list[int]]
    promoted: list[list[bool]]  # por token gerado: a referencia foi usada?

    @property
    def promote_rate(self) -> float:
        flat = [p for row in self.promoted for p in row]
        return sum(flat) / max(len(flat), 1)

    @property
    def n_tokens(self) -> int:
        return sum(len(g) for g in self.gen_ids)


class _Stepper:
    """Decodificacao incremental com padding a esquerda para um modelo."""

    def __init__(self, loaded: LoadedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.loaded = loaded
        self.device = loaded.device
        self.attn = attention_mask.to(self.device)
        self.cache = DynamicCache(config=loaded.model.config.get_text_config())
        pos = (self.attn.cumsum(-1) - 1).clamp(min=0)
        out = loaded.model(
            input_ids=input_ids.to(self.device), attention_mask=self.attn, position_ids=pos,
            past_key_values=self.cache, use_cache=True,
            cache_position=torch.arange(self.attn.shape[1], device=self.device),
            logits_to_keep=1,
        )
        self.logits = out.logits[:, -1, :].float()

    def step(self, tokens: torch.Tensor) -> None:
        self.attn = torch.cat(
            [self.attn, torch.ones((self.attn.shape[0], 1), dtype=self.attn.dtype,
                                   device=self.device)], dim=1)
        pos = self.attn.sum(-1, keepdim=True) - 1
        out = self.loaded.model(
            input_ids=tokens.to(self.device)[:, None], attention_mask=self.attn,
            position_ids=pos, past_key_values=self.cache, use_cache=True,
            cache_position=torch.tensor([self.attn.shape[1] - 1], device=self.device),
            logits_to_keep=1,
        )
        self.logits = out.logits[:, -1, :].float()


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = logits.double().log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


@torch.inference_mode()
def cascade_generate(
    base: LoadedModel,
    ref: Optional[LoadedModel],
    prompts: Sequence[str],
    *,
    threshold: float,
    eos_ids: set[int],
    max_new_tokens: int = 256,
) -> ClosedLoopResult:
    """Greedy com gate por entropia da base. `threshold=inf` = so a base;
    `threshold=-inf` = sempre a referencia. `ref=None` exige threshold=inf."""
    tok = base.tokenizer
    previous = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = previous

    b = _Stepper(base, enc["input_ids"], enc["attention_mask"])
    r = _Stepper(ref, enc["input_ids"], enc["attention_mask"]) if ref is not None else None
    n = len(prompts)
    done = torch.zeros(n, dtype=torch.bool)
    gen: list[list[int]] = [[] for _ in range(n)]
    promoted: list[list[bool]] = [[] for _ in range(n)]
    eos = torch.tensor(sorted(eos_ids))

    for _ in range(max_new_tokens):
        base_tok = b.logits.argmax(-1).cpu()
        if r is not None:
            promote = (_entropy(b.logits) > threshold).cpu()
            ref_tok = r.logits.argmax(-1).cpu()
            nxt = torch.where(promote, ref_tok, base_tok)
        else:
            if threshold != float("inf"):
                raise ValueError("sem referencia, so e possivel threshold=inf")
            promote = torch.zeros(n, dtype=torch.bool)
            nxt = base_tok
        pad = tok.pad_token_id
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        for i in range(n):
            if not done[i]:
                gen[i].append(int(nxt[i]))
                promoted[i].append(bool(promote[i]))
        done |= torch.isin(nxt, eos)
        if bool(done.all()):
            break
        b.step(nxt)
        if r is not None:
            r.step(nxt)
    return ClosedLoopResult(gen_ids=gen, promoted=promoted)
