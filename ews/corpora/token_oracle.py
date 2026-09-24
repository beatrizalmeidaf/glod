"""Fase 1c - heterogeneidade de capacidade POR TOKEN, em geracao.

A tese do `ews_ideia.md` e sobre tokens ("a grande maioria dos tokens e
altamente previsivel"), mas as Fases 1a/1b mediram por EXEMPLO num unico token
de resposta de multipla escolha - o pior testbed possivel para essa tese, com
piso de chute de 25% inflando qualquer oraculo.

Protocolo:
1. O modelo de referencia (12B bf16) gera greedy sobre prompts reais.
2. Cada configuracao barata e avaliada com *teacher forcing* sobre a mesma
   sequencia: para cada token gerado, registramos o top-1 da configuracao.
3. A configuracao "basta" no token t se seu top-1 == token da referencia -
   exatamente o criterio de aceitacao da decodificacao especulativa greedy.

Criterio contra a referencia (e nao contra um gabarito) elimina a inflacao por
sorte do MMLU: acertar por acaso um token num vocabulario de 262k e desprezivel.

Aproximacao declarada: cada configuracao processa o prefixo inteiro na propria
precisao (seu proprio KV). Num sistema EWS real o KV do prefixo viria de
configuracoes mistas. E a mesma aproximacao usada para estimar aceitacao em
decodificacao especulativa; a validacao em malha fechada fica para a Fase 1d.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from tqdm.auto import tqdm

from ews.core.model_loader import LoadedModel

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------ prompts
def _chat(tokenizer, text: str) -> str:
    if getattr(tokenizer, "chat_template", None) is None:  # modelo base: texto cru
        return text + "\n\n"
    # enable_thinking=False desliga o raciocinio longo do Qwen3; templates que nao
    # usam a variavel (Gemma) simplesmente a ignoram.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
        enable_thinking=False,
    )


def build_prompts_gsm8k(tokenizer, n: int, seed: int, cache_dir: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)["test"]
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idx:
        q = ds[i]["question"]
        text = (
            "Solve the following problem step by step. "
            "At the end, write 'Answer: <number>'.\n\n" + q
        )
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"prompt": _chat(tokenizer, text), "gold": gold, "source": "gsm8k"})
    return out


def build_prompts_mmlu_pt(tokenizer, csv_path: str, n: int, seed: int) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(csv_path).dropna(subset=["Question", "A", "B", "C", "D", "Answer"])
    df = df.sample(min(n, len(df)), random_state=seed)
    out = []
    for _, r in df.iterrows():
        text = (
            "Responda a questao abaixo explicando brevemente o raciocinio e "
            "termine com 'Resposta: <letra>'.\n\n"
            f"Assunto: {r['Subject']}\nPergunta: {r['Question']}\n"
            f"A) {r['A']}\nB) {r['B']}\nC) {r['C']}\nD) {r['D']}"
        )
        out.append({
            "prompt": _chat(tokenizer, text),
            "gold": str(r["Answer"]).strip().upper(),
            "source": "mmlu_pt",
        })
    return out


def build_prompts_mmlu_en(tokenizer, n: int, seed: int, cache_dir: str) -> list[dict]:
    """MMLU em ingles (cais/mmlu, split de teste), no mesmo formato do PT-BR.

    Serve de controle de idioma: o vies de prior de letra em MCQ que apareceu no
    Gemma em portugues nao replicou em ingles (ver o apendice de higiene).
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=cache_dir)
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    letters = "ABCD"
    out = []
    for i in idx:
        r = ds[i]
        ch = list(r["choices"])[:4]
        if len(ch) < 4:
            continue
        text = (
            "Answer the question below, explaining your reasoning briefly, and "
            "finish with 'Answer: <letter>'.\n\n"
            f"Subject: {r['subject']}\nQuestion: {r['question']}\n"
            + "\n".join(f"{letters[j]}) {c}" for j, c in enumerate(ch))
        )
        out.append({"prompt": _chat(tokenizer, text), "gold": letters[int(r["answer"])],
                    "source": "mmlu_en"})
    return out


def build_prompts_wikitext(tokenizer, n: int, seed: int, cache_dir: str,
                           prompt_tokens: int = 64) -> list[dict]:
    """Continuacao de texto livre: o prompt pede para continuar um trecho da wikitext.

    Sem gabarito (gold vazio): as metricas por token valem, as de tarefa nao. E o
    corpus que mede a lei fora de prompts de pergunta e resposta curta.

    O trecho vai DENTRO do chat template, como instrucao. A primeira versao passava o
    texto cru e, com greedy, os modelos instruct entravam em laco (99-100% das
    sequencias dos Gemma repetiam mais da metade dos 4-gramas): margens enormes, kappa
    10x menor, e um corpus que media o laco e nao a lei.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                      cache_dir=cache_dir)
    rng = random.Random(seed)
    long_rows = [i for i in range(len(ds)) if len(ds[i]["text"]) > 400]
    idx = rng.sample(long_rows, min(n, len(long_rows)))
    out = []
    for i in idx:
        ids = tokenizer(ds[i]["text"].strip(), add_special_tokens=False)["input_ids"]
        if len(ids) <= prompt_tokens:
            continue
        text = ("Continue the following encyclopedia text in the same style, with new "
                "information. Write one paragraph.\n\n" + tokenizer.decode(ids[:prompt_tokens]))
        out.append({"prompt": _chat(tokenizer, text), "gold": "", "source": "wikitext"})
    return out



def build_prompts_wikitext_natural(tokenizer, n: int, seed: int, cache_dir: str,
                                   prompt_tokens: int = 64, cont_tokens: int = 192) -> list[dict]:
    """Texto REAL da wikitext: prompt = primeiros tokens, sequencia = os seguintes.

    Nao ha geracao: a sequencia pontuada e o proprio texto do dataset. Serve a dois
    fins que a geracao greedy nao atende:
      * modelos BASE (sem SFT/RLHF) entram em laco com greedy (50-83% das sequencias),
        o que destruiria a comparacao base x instruct;
      * todos os modelos sao medidos exatamente nos MESMOS tokens, entao a diferenca de
        kappa e de margens, nao de texto gerado.
    A metrica continua sendo flip do top-1 sob teacher forcing contra a referencia.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                      cache_dir=cache_dir)
    rng = random.Random(seed)
    need = prompt_tokens + cont_tokens
    long_rows = [i for i in range(len(ds)) if len(ds[i]["text"]) > 4 * need]
    out = []
    for i in rng.sample(long_rows, min(4 * n, len(long_rows))):
        ids = tokenizer(ds[i]["text"].strip(), add_special_tokens=False)["input_ids"]
        if len(ids) < need:
            continue
        out.append({"prompt": tokenizer.decode(ids[:prompt_tokens]), "gold": "",
                    "source": "wikitext_nat",
                    "continuation": tokenizer.decode(ids[prompt_tokens:need])})
        if len(out) >= n:
            break
    return out


def natural_corpus(tokenizer, records: list[dict]) -> "GeneratedCorpus":
    """GeneratedCorpus a partir de texto dado (campo `continuation`), sem gerar nada."""
    prompt_ids, gen_ids, keep = [], [], []
    for r in records:
        p_ids = tokenizer(r["prompt"], add_special_tokens=True)["input_ids"]
        g_ids = tokenizer(r["continuation"], add_special_tokens=False)["input_ids"]
        if not g_ids:
            continue
        prompt_ids.append(p_ids); gen_ids.append(g_ids); keep.append(r)
    return GeneratedCorpus(records=keep, prompt_ids=prompt_ids, gen_ids=gen_ids)


def repetition_stats(gen_ids: list[list[int]], n: int = 4) -> dict:
    """Fracao de n-gramas repetidos por sequencia gerada.

    Um corpus greedy em que o modelo entra em laco mede o laco, nao a lei: as margens
    ficam enormes e kappa despenca. `degenerate_frac` e a fracao de sequencias com mais
    da metade dos n-gramas repetidos.
    """
    rates = []
    for g in gen_ids:
        ng = [tuple(g[i:i + n]) for i in range(len(g) - n + 1)]
        rates.append(1 - len(set(ng)) / len(ng) if ng else 0.0)
    rates.sort()
    return {"median_repeat": rates[len(rates) // 2] if rates else 0.0,
            "degenerate_frac": sum(r > 0.5 for r in rates) / max(len(rates), 1)}

# --------------------------------------------------------------- geracao
@dataclass
class GeneratedCorpus:
    """Sequencias de referencia: ids do prompt e ids gerados (ate o EOS, incluso)."""

    prompt_ids: list[list[int]]
    gen_ids: list[list[int]]
    records: list[dict]

    @property
    def n_tokens(self) -> int:
        return sum(len(g) for g in self.gen_ids)

    def token_sequence_index(self) -> torch.Tensor:
        """Para cada token gerado (achatado), o indice da sequencia de origem."""
        return torch.cat([
            torch.full((len(g),), i, dtype=torch.int32) for i, g in enumerate(self.gen_ids)
        ])

    def token_position(self) -> torch.Tensor:
        """Para cada token gerado (achatado), sua posicao dentro da geracao."""
        return torch.cat([torch.arange(len(g), dtype=torch.int32) for g in self.gen_ids])

    def to_dict(self) -> dict:
        return {"prompt_ids": self.prompt_ids, "gen_ids": self.gen_ids, "records": self.records}

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratedCorpus":
        return cls(prompt_ids=d["prompt_ids"], gen_ids=d["gen_ids"], records=d["records"])


def _eos_ids(loaded: LoadedModel) -> set[int]:
    ids: set[int] = set()
    for source in (loaded.model.generation_config.eos_token_id, loaded.tokenizer.eos_token_id):
        if source is None:
            continue
        ids.update(source if isinstance(source, (list, tuple)) else [source])
    end_of_turn = loaded.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn != loaded.tokenizer.unk_token_id:
        ids.add(end_of_turn)
    return ids


def truncate_at_eos(ids: Sequence[int], eos: set[int]) -> list[int]:
    """Corta apos o primeiro EOS (incluindo-o). Sem EOS, devolve tudo."""
    out = []
    for t in ids:
        out.append(int(t))
        if int(t) in eos:
            break
    return out


@torch.inference_mode()
def generate_greedy(
    loaded: LoadedModel,
    records: list[dict],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 32,
) -> GeneratedCorpus:
    tok = loaded.tokenizer
    eos = _eos_ids(loaded)
    prompt_ids_all, gen_ids_all = [], []
    previous_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for start in tqdm(range(0, len(records), batch_size), desc="geracao ref"):
            chunk = records[start : start + batch_size]
            enc = tok([r["prompt"] for r in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(loaded.device)
            out = loaded.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
            )
            width = enc["input_ids"].shape[1]
            for row in range(len(chunk)):
                real = enc["attention_mask"][row].bool()
                prompt_ids_all.append(enc["input_ids"][row][real].tolist())
                gen_ids_all.append(truncate_at_eos(out[row, width:].tolist(), eos))
    finally:
        tok.padding_side = previous_side
    for r, g in zip(records, gen_ids_all):
        r["completion"] = tok.decode(g, skip_special_tokens=True)
    return GeneratedCorpus(prompt_ids=prompt_ids_all, gen_ids=gen_ids_all, records=records)


# ---------------------------------------------------------- teacher forcing
@dataclass
class TokenScores:
    """Leituras por token gerado (achatado na ordem do corpus)."""

    top1: torch.Tensor          # int32  [T]
    ref_logprob: torch.Tensor   # float32 [T] log p_config(token da referencia)
    entropy: torch.Tensor       # float32 [T] entropia da distribuicao (nats)
    margin: torch.Tensor        # float32 [T] p(top1) - p(top2)
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)  # camada -> bf16 [T, H]

    def to_dict(self) -> dict:
        return {"top1": self.top1, "ref_logprob": self.ref_logprob,
                "entropy": self.entropy, "margin": self.margin, "hidden": self.hidden}

    @classmethod
    def from_dict(cls, d: dict) -> "TokenScores":
        return cls(**d)


def teacher_forcing_batch(
    prompt_ids: Sequence[Sequence[int]],
    gen_ids: Sequence[Sequence[int]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Monta [prompt + gen[:-1]] com padding a esquerda.

    Com padding a esquerda, os ultimos `len(gen)` logits de cada linha preveem
    exatamente `gen[0..len-1]`. Devolve (input_ids, attention_mask,
    position_ids, largura_da_janela_de_logits).
    """
    rows = [list(p) + list(g[:-1]) for p, g in zip(prompt_ids, gen_ids)]
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, width - len(r):] = torch.tensor(r, dtype=torch.long)
        mask[i, width - len(r):] = 1
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    keep = max(len(g) for g in gen_ids)
    return input_ids, mask, position_ids, keep


@torch.inference_mode()
def score_tokens(
    loaded: LoadedModel,
    corpus: GeneratedCorpus,
    *,
    batch_size: int = 8,
    capture_layers: Sequence[int] = (),
    desc: str = "scoring",
) -> TokenScores:
    """Teacher forcing de uma configuracao sobre o corpus de referencia."""
    decoder = loaded.decoder
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            captured[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook

    for layer_idx in capture_layers:
        hooks.append(decoder.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    top1_all, lp_all, ent_all, mar_all = [], [], [], []
    hid_all: dict[int, list[torch.Tensor]] = {k: [] for k in capture_layers}
    pad_id = loaded.tokenizer.pad_token_id
    # ordena por comprimento para reduzir padding (a ordem original e restaurada)
    order = sorted(range(len(corpus.gen_ids)),
                   key=lambda i: len(corpus.prompt_ids[i]) + len(corpus.gen_ids[i]))
    per_seq: dict[int, tuple] = {}
    try:
        for start in tqdm(range(0, len(order), batch_size), desc=desc, leave=False):
            idx = order[start : start + batch_size]
            gens = [corpus.gen_ids[i] for i in idx]
            ids, mask, pos, keep = teacher_forcing_batch(
                [corpus.prompt_ids[i] for i in idx], gens, pad_id)
            out = loaded.model(
                input_ids=ids.to(loaded.device), attention_mask=mask.to(loaded.device),
                position_ids=pos.to(loaded.device), use_cache=False, logits_to_keep=keep,
            )
            logits = out.logits  # [B, keep, V]
            for row, (seq_i, g) in enumerate(zip(idx, gens)):
                n = len(g)
                lg = logits[row, keep - n:, :].float()
                logp = lg.log_softmax(-1)
                p = logp.exp()
                target = torch.tensor(g, device=lg.device)
                top2 = p.topk(2, dim=-1).values
                # bfloat16, NUNCA float16: o fluxo residual do Gemma passa de 65504
                # nas camadas intermediarias (medido: 56% dos tokens com inf na
                # camada 32 em fp16), e um unico inf envenena a padronizacao do probe.
                hidden = {k: captured[k][row, -n:, :].to(torch.bfloat16).cpu()
                          for k in capture_layers}
                for k, h in hidden.items():
                    if not torch.isfinite(h).all():
                        raise FloatingPointError(
                            f"estado oculto nao-finito na camada {k} (sequencia {seq_i})")
                per_seq[seq_i] = (
                    lg.argmax(-1).to(torch.int32).cpu(),
                    logp.gather(-1, target[:, None]).squeeze(-1).cpu(),
                    (-(p * logp).sum(-1)).cpu(),
                    (top2[:, 0] - top2[:, 1]).cpu(),
                    hidden,
                )
            captured.clear()
    finally:
        for h in hooks:
            h.remove()

    for i in range(len(corpus.gen_ids)):
        t1, lp, en, mg, hid = per_seq[i]
        top1_all.append(t1); lp_all.append(lp); ent_all.append(en); mar_all.append(mg)
        for k in capture_layers:
            hid_all[k].append(hid[k])
    return TokenScores(
        top1=torch.cat(top1_all), ref_logprob=torch.cat(lp_all),
        entropy=torch.cat(ent_all), margin=torch.cat(mar_all),
        hidden={k: torch.cat(v) for k, v in hid_all.items()},
    )


# ------------------------------------------------------------------ analise
def consistent_token_oracle(
    agree: torch.Tensor, costs: Sequence[float], ref_cost: float
) -> dict[str, float]:
    """Custo medio por token do oraculo que preserva 100% do greedy da referencia.

    `agree[c, t]` = configuracao c reproduz o token t. Cada token paga o custo
    da configuracao mais barata que o reproduz; se nenhuma reproduz, paga a
    referencia. `costs` e ordenado como as linhas de `agree`.
    """
    costs_t = torch.tensor(list(costs), dtype=torch.float64)
    order = torch.argsort(costs_t)
    sorted_agree = agree[order]
    any_ok = sorted_agree.any(0)
    first = sorted_agree.float().argmax(0)
    per_token = torch.where(any_ok, costs_t[order][first], torch.tensor(float(ref_cost),
                                                                          dtype=torch.float64))
    return {"mean_cost": per_token.mean().item(),
            "saving": 1 - per_token.mean().item() / ref_cost,
            "per_token_cost": per_token}


def run_lengths(flags: torch.Tensor, seq_index: torch.Tensor) -> list[int]:
    """Comprimentos das rajadas de `True` consecutivos, sem cruzar sequencias."""
    runs, current = [], 0
    prev_seq = None
    for f, s in zip(flags.tolist(), seq_index.tolist()):
        if s != prev_seq:
            if current:
                runs.append(current)
            current = 0
            prev_seq = s
        if f:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def burstiness(flags: torch.Tensor, seq_index: torch.Tensor) -> dict[str, float]:
    """Autocorrelacao de um indicador binario (ex.: 'base insuficiente').

    lift = P(h_t | h_{t-1}) / P(h_t). lift ~ 1 -> tokens dificeis espalhados
    (cada um paga I/O). lift >> 1 -> rajadas (ΔW em cache amortiza o I/O).
    """
    f = flags.bool()
    same = seq_index[1:] == seq_index[:-1]
    prev, cur = f[:-1][same], f[1:][same]
    p = f.float().mean().item()
    p_given = cur[prev].float().mean().item() if prev.any() else float("nan")
    runs = run_lengths(flags, seq_index)
    return {
        "rate": p,
        "p_given_prev": p_given,
        "lift": p_given / p if p > 0 else float("nan"),
        "mean_run": sum(runs) / len(runs) if runs else 0.0,
        "n_runs": len(runs),
    }


def burstiness_null(
    flags: torch.Tensor, seq_index: torch.Tensor, *, n_reps: int = 50, seed: int = 0
) -> dict[str, float]:
    """Lift esperado se a ORDEM dos tokens dentro de cada sequencia fosse aleatoria.

    Preserva a taxa de tokens dificeis POR SEQUENCIA (algumas geracoes sao mais
    dificeis que outras) e destroi apenas a estrutura local. Lift observado
    acima deste nulo = rajadas locais reais, exploraveis por cache de ΔW.
    """
    g = torch.Generator().manual_seed(seed)
    f = flags.bool()
    bounds = torch.nonzero(torch.diff(seq_index, prepend=seq_index[:1] - 1)).flatten().tolist()
    bounds.append(len(f))
    lifts, runs = [], []
    for _ in range(n_reps):
        shuffled = f.clone()
        for a, b in zip(bounds[:-1], bounds[1:]):
            shuffled[a:b] = f[a:b][torch.randperm(b - a, generator=g)]
        stats = burstiness(shuffled, seq_index)
        lifts.append(stats["lift"])
        runs.append(stats["mean_run"])
    return {"lift": sum(lifts) / len(lifts), "mean_run": sum(runs) / len(runs)}


def cached_streaming_cost(
    hard: torch.Tensor, seq_index: torch.Tensor, *, window: int,
    base_cost: float, ref_cost: float, delta_io: float,
) -> dict[str, float]:
    """EWS com cache de ΔW: carrega no primeiro token dificil, descarta apos
    `window` tokens faceis seguidos.

    Enquanto o ΔW esta carregado, os tokens rodam na referencia (fidelidade
    total, custo de leitura `ref_cost`). Fora dele, rodam na base. `hard` e o
    indicador de que a base NAO reproduz a referencia: com o indicador verdadeiro
    isto e um limite superior (oraculo); com a decisao de um gate, e realizavel.

    Devolve pesos lidos/token (proxy de latencia em decode limitado por banda),
    I/O de disco/token (so recargas do ΔW) e taxa de tokens divergentes.
    """
    loaded, easy_streak, prev_seq = False, 0, None
    read, io, wrong = 0.0, 0.0, 0
    for h, s in zip(hard.bool().tolist(), seq_index.tolist()):
        if s != prev_seq:
            loaded, easy_streak, prev_seq = False, 0, s
        if h and not loaded:
            loaded, easy_streak = True, 0
            io += delta_io
        if loaded:
            read += ref_cost
            easy_streak = 0 if h else easy_streak + 1
            if easy_streak >= window:
                loaded = False
        else:
            read += base_cost
            wrong += int(h)
    n = len(hard)
    return {"read_per_token": read / n, "io_per_token": io / n, "divergence": wrong / n}


def speculative_expected_cost(
    agree_draft: torch.Tensor, seq_index: torch.Tensor, draft_cost: float,
    verify_cost: float, k: int,
) -> dict[str, float]:
    """Custo de pesos lidos por token da decodificacao especulativa greedy.

    Simula o laco exato sobre o indicador medido: a cada rodada o draft propoe
    ate k tokens (le `draft_cost` por token proposto), a verificacao paralela
    le `verify_cost` uma vez e aceita o prefixo concordante + 1 token corrigido
    (ou bonus). Saida identica a da referencia (sem perda) por construcao.

    Em decode limitado por banda de memoria, pesos lidos por token e o proxy de
    latencia; em regime limitado por disco, e o I/O.
    """
    total_cost, total_tokens = 0.0, 0
    flags = agree_draft.bool().tolist()
    seqs = seq_index.tolist()
    start = 0
    n = len(flags)
    while start < n:
        seq = seqs[start]
        end = start
        while end < n and seqs[end] == seq:
            end += 1
        t = start
        while t < end:
            proposed = min(k, end - t)
            accepted = 0
            while accepted < proposed and flags[t + accepted]:
                accepted += 1
            emitted = min(accepted + 1, end - t)
            total_cost += proposed * draft_cost + verify_cost
            total_tokens += emitted
            t += emitted
        start = end
    return {"cost_per_token": total_cost / total_tokens, "k": k}


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC por ranking (Mann-Whitney), sem dependencias externas."""
    s = scores.double()
    y = labels.bool()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.empty_like(s)
    order = torch.argsort(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    # empates: media dos ranks
    uniq, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = torch.zeros(len(uniq), dtype=torch.float64).index_add_(0, inv, ranks)
        ranks = (sums / counts.double())[inv]
    return ((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


# ---------------------------------------------------- granularidade e gates
def chunk_oracle_cost(
    agree: torch.Tensor, costs: Sequence[float], ref_cost: float,
    seq_index: torch.Tensor, positions: torch.Tensor, chunk: Optional[int],
) -> float:
    """Oraculo cuja unidade de decisao e um BLOCO de `chunk` tokens consecutivos.

    Uma configuracao so serve o bloco se reproduzir TODOS os tokens dele.
    `chunk=1` e o oraculo por token; `chunk=None` e a sequencia inteira (a
    unidade "por exemplo" das Fases 1a/1b, agora medida no mesmo corpus).
    """
    if chunk is None:
        group = seq_index.long()
    else:
        group = seq_index.long() * 1_000_000 + positions.long() // chunk
    _, gid = torch.unique(group, return_inverse=True)
    n_groups = int(gid.max()) + 1
    costs_t = torch.tensor(list(costs), dtype=torch.float64)
    group_ok = torch.ones((agree.shape[0], n_groups), dtype=torch.int8)
    group_ok = group_ok.scatter_reduce(
        1, gid.expand(agree.shape[0], -1), agree.to(torch.int8), reduce="amin")
    group_cost = torch.full((n_groups,), float(ref_cost), dtype=torch.float64)
    for c in torch.argsort(costs_t, descending=True).tolist():  # o mais barato sobrescreve por ultimo
        group_cost = torch.where(group_ok[c].bool(), costs_t[c], group_cost)
    return group_cost[gid].mean().item()


def gate_curve(
    score: torch.Tensor, hard: torch.Tensor, *, base_cost: float, ref_cost: float,
    mode: str, prefix_fraction: float = 0.0, n_points: int = 200,
    hard_if_promoted: Optional[torch.Tensor] = None,
) -> list[dict]:
    """Curva (fracao promovida, custo/token, divergencia) varrendo o limiar.

    Modelos de custo, porque o momento da decisao muda o que se paga:
      cascade     - sinal so existe DEPOIS de rodar a base no token: base + p*ref
      predictive  - decisao antes do token: (1-p)*base + p*ref
      early_probe - roda a fracao `prefix_fraction` das camadas na base, decide,
                    e roda o resto na base ou na referencia.

    `hard_if_promoted` e o indicador de divergencia da configuracao que roda
    quando o gate promove. Sem ele, supoe-se (otimista) que promover reproduz a
    referencia - valido para cascade/predictive, NAO para early_probe, cujo
    prefixo ja rodou na base: passe a divergencia MEDIDA do prefixo quantizado.
    """
    s = score.double()
    qs = torch.linspace(0, 1, n_points, dtype=torch.float64)
    thresholds = torch.quantile(s, qs).tolist() + [float("inf")]
    out = []
    for thr in thresholds:
        promote = s > thr
        p = promote.double().mean().item()
        div = (hard.bool() & ~promote).double().mean().item()
        if hard_if_promoted is not None:
            div += (hard_if_promoted.bool() & promote).double().mean().item()
        if mode == "cascade":
            cost = base_cost + p * ref_cost
        elif mode == "predictive":
            cost = (1 - p) * base_cost + p * ref_cost
        elif mode == "early_probe":
            f = prefix_fraction
            cost = f * base_cost + (1 - f) * ((1 - p) * base_cost + p * ref_cost)
        else:
            raise ValueError(mode)
        out.append({"promoted": p, "cost": cost, "divergence": div})
    return out


def cost_at_divergence(curve: list[dict], target: float) -> float:
    """Menor custo entre os pontos da curva com divergencia <= alvo."""
    ok = [pt["cost"] for pt in curve if pt["divergence"] <= target + 1e-12]
    return min(ok) if ok else float("nan")
