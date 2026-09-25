"""Fidelidade de decisao token a token contra uma referencia, com KL exato.

Para cada token do corpus greedy da referencia (teacher forcing):
  top1, logprob do token gerado, entropia,
  KL(p_ref || p_cfg) e TV nos top-K da referencia + cauda agregada,
  logprobs da config nos indices top-K da referencia (float16) - insumo da
  teoria (gap top1-top2, direcoes da perturbacao).
Num subconjunto fixo de tokens a referencia guarda a distribuicao COMPLETA
(float16) e cada config calcula KL e TV exatos no vocabulario inteiro - o que
valida (ou nao) a truncagem em top-K.
"""

from __future__ import annotations

from typing import Optional

import torch

from glod.core.model_loader import LoadedModel
from glod.corpora.token_oracle import GeneratedCorpus, teacher_forcing_batch


def subset_index(n_tokens: int, size: int = 4096, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.sort(torch.randperm(n_tokens, generator=g)[:size]).values


@torch.inference_mode()
def score_fidelity(loaded: LoadedModel, corpus: GeneratedCorpus, *, ref: Optional[dict], k: int = 64,
                   batch_size: int = 8, subset: Optional[torch.Tensor] = None) -> dict:
    """`ref=None` -> este modelo E a referencia (guarda top-K e distribuicao do subconjunto)."""
    pad = loaded.tokenizer.pad_token_id
    offsets = [0]
    for g in corpus.gen_ids:
        offsets.append(offsets[-1] + len(g))
    T = offsets[-1]
    is_ref = ref is None
    top1 = torch.empty(T, dtype=torch.int32)
    lp = torch.empty(T)
    ent = torch.empty(T)
    kl = torch.zeros(T)
    tv = torch.zeros(T)
    topv = torch.empty(T, k)
    topi = torch.empty(T, k, dtype=torch.int32)          # ref: indices; cfg: nao usado
    at_ref = None if is_ref else torch.empty(T, k, dtype=torch.float16)
    sub_pos = {} if subset is None else {int(t): j for j, t in enumerate(subset.tolist())}
    full_store = None
    if subset is not None and is_ref:
        full_store = {}
    kl_full = torch.full((len(sub_pos),), float("nan"))
    tv_full = torch.full((len(sub_pos),), float("nan"))

    order = sorted(range(len(corpus.gen_ids)),
                   key=lambda i: len(corpus.prompt_ids[i]) + len(corpus.gen_ids[i]))
    dev = loaded.device
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        gens = [corpus.gen_ids[i] for i in idx]
        ids, mask, pos, keep = teacher_forcing_batch([corpus.prompt_ids[i] for i in idx], gens, pad)
        logits = loaded.model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                              position_ids=pos.to(dev), use_cache=False, logits_to_keep=keep).logits
        for row, (si, g) in enumerate(zip(idx, gens)):
            n = len(g)
            a, b = offsets[si], offsets[si] + n
            logp = logits[row, keep - n:, :].double().log_softmax(-1)
            tgt = torch.tensor(g, device=dev)
            top1[a:b] = logp.argmax(-1).int().cpu()
            lp[a:b] = logp.gather(-1, tgt[:, None]).squeeze(-1).float().cpu()
            ent[a:b] = (-(logp.exp() * logp).sum(-1)).float().cpu()
            if is_ref:
                v, i_ = logp.topk(k, dim=-1)
                topv[a:b] = v.float().cpu()
                topi[a:b] = i_.int().cpu()
            else:
                rv = ref["topv"][a:b].to(dev).double()
                ri = ref["topi"][a:b].to(dev).long()
                vocab = logp.shape[-1]  # outro modelo da familia pode ter vocabulario menor
                qv = logp.gather(-1, ri.clamp(max=vocab - 1)).masked_fill(ri >= vocab, -1e4)
                P, Q = rv.exp(), qv.exp()
                Pt, Qt = P.sum(-1), Q.sum(-1)
                tailP, tailQ = (1 - Pt).clamp(min=1e-30), (1 - Qt).clamp(min=1e-30)
                kl[a:b] = ((P * (rv - qv)).sum(-1) + tailP * (tailP.log() - tailQ.log())).float().cpu()
                tv[a:b] = (0.5 * ((P - Q).abs().sum(-1) + (tailP - tailQ).abs())).float().cpu()
                at_ref[a:b] = qv.half().cpu()
            if sub_pos:
                rows = [(t - a, sub_pos[t]) for t in range(a, b) if t in sub_pos]
                if rows:
                    local = torch.tensor([r[0] for r in rows], device=dev)
                    js = [r[1] for r in rows]
                    if is_ref:
                        for (loc, j), v in zip(rows, logp[local].half().cpu()):
                            full_store[j] = v
                    else:
                        R = torch.stack([ref["full"][j] for j in js]).to(dev).double()
                        V = min(R.shape[-1], logp.shape[-1])
                        R = R[:, :V].log_softmax(-1)  # re-normaliza apos float16 / corte de vocabulario
                        Qf = logp[local][:, :V].log_softmax(-1)
                        kl_full[js] = ((R.exp() * (R - Qf)).sum(-1)).float().cpu()
                        tv_full[js] = (0.5 * (R.exp() - Qf.exp()).abs().sum(-1)).float().cpu()
    out = {"top1": top1, "logprob": lp, "entropy": ent}
    if is_ref:
        out.update(topv=topv, topi=topi)
        if full_store is not None:
            out["full"] = torch.stack([full_store[j] for j in range(len(sub_pos))])
            out["subset"] = subset
    else:
        out.update(kl=kl, tv=tv, at_ref=at_ref)
        if sub_pos:
            out.update(kl_full=kl_full, tv_full=tv_full, subset=subset)
    return out
