"""Registro de corpora de referencia.

Um corpus aqui e so uma lista de registros {prompt, gold, source}: o corpus greedy
de verdade sai de `generate_greedy` no pipeline de fidelidade. Trocar de dataset e
portanto trocar o id passado em `--corpus`, e nada mais.

  mix       GSM8K + MMLU PT-BR (o dos resultados do paper; sem sufixo no disco)
  gsm8k     so GSM8K (raciocinio, geracao longa)
  mmlu_pt   so MMLU PT-BR (multipla escolha em portugues, CSV em data/)
  mmlu_en   so MMLU em ingles (cais/mmlu, split de teste)
  wikitext  continuacao de texto livre (wikitext-2-raw-v1), gerada pelo modelo
  wikitext_nat  o texto REAL da wikitext como sequencia (sem geracao): unica opcao
            valida para modelos base, que entram em laco com greedy

`gold` fica vazio em wikitext: as metricas de tarefa (acuracia) nao se aplicam, mas
todas as metricas por token (flips, KL, TV) se aplicam do mesmo jeito.
"""
from __future__ import annotations

from ews.corpora.token_oracle import (
    build_prompts_gsm8k,
    build_prompts_wikitext_natural,
    build_prompts_mmlu_en,
    build_prompts_mmlu_pt,
    build_prompts_wikitext,
)
from ews.paths import MMLU_PT_CSV

CORPORA = {
    "mix": ("gsm8k", "mmlu_pt"),
    "gsm8k": ("gsm8k",),
    "mmlu_pt": ("mmlu_pt",),
    "mmlu_en": ("mmlu_en",),
    "wikitext": ("wikitext",),
    "wikitext_nat": ("wikitext_nat",),   # texto real, sem geracao
}


def build_records(corpus: str, tokenizer, n_prompts: int, seed: int, cache_dir: str,
                  mmlu_pt_csv: str = MMLU_PT_CSV) -> list[dict]:
    """Registros do corpus pedido. `cache_dir` e a raiz do cache de datasets."""
    if corpus not in CORPORA:
        raise ValueError(f"corpus desconhecido: {corpus} (conhecidos: {sorted(CORPORA)})")
    out: list[dict] = []
    for part in CORPORA[corpus]:
        if part == "gsm8k":
            out += build_prompts_gsm8k(tokenizer, n_prompts, seed, cache_dir)
        elif part == "mmlu_pt":
            out += build_prompts_mmlu_pt(tokenizer, mmlu_pt_csv, n_prompts, seed)
        elif part == "mmlu_en":
            out += build_prompts_mmlu_en(tokenizer, n_prompts, seed, cache_dir)
        elif part == "wikitext":
            out += build_prompts_wikitext(tokenizer, n_prompts, seed, cache_dir)
        elif part == "wikitext_nat":
            out += build_prompts_wikitext_natural(tokenizer, n_prompts, seed, cache_dir)
        else:                                     # pragma: no cover - registro errado
            raise AssertionError(part)
    return out
