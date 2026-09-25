"""AMQ: quantizacao ciente do arg-max (arg-max-aware quantization).

Parte do GPTQ (mesmos codigos inteiros, mesmos zeros, mesmo formato: B bits, grupos
de 128 ao longo da entrada) e reajusta PONTA A PONTA apenas as escalas por grupo,
s = s0 * exp(theta). O formato final e identico ao do GPTQ - mesmos bits, mesmo
kernel -, so os fp16 das escalas mudam.

A perda e o que distingue as variantes, todas com o mesmo otimizador, os mesmos dados
e o mesmo criterio de parada (melhor valor do proprio objetivo na validacao):

  kl       KL(p_denso || q) por token (destilacao; o baseline honesto do reajuste)
  dg       KL + beta * E_frag |dg_t|: deslocamento da margem densa top-1/top-2 nos tokens
           frageis (margem densa < 1 nat). E a previsao de primeira ordem do paper,
           flips ~ rho(0) E|dg| / 2, usada como objetivo: suave e sem mirar tokens
           individuais
  flip     KL + beta * sigmoid(-u_t / tau): u_t e a margem, no comprimido, do arg-max
           denso sobre o melhor concorrente (substituto suave da taxa de flips do
           ataque, App. A.6, aqui minimizado)
  tvfrag   KL + beta * TV_t nos tokens frageis

A perda so conta nas posicoes marcadas em `lmask` (numa sequencia gerada, as posicoes
que preveem tokens da resposta). Os alvos densos (top-K + cauda) sao calculados uma
vez, antes de quantizar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from glod.core import compressors as C

LOGGER = logging.getLogger(__name__)
FRAGILE = 1.0   # nats


class QLinear(nn.Module):
    """W = s0 * exp(theta) * (codes - zero), por grupo de `group` colunas."""

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bias, group: int, rank: int = 0):
        super().__init__()
        self.register_buffer("codes", codes, persistent=False)           # uint8 [out, in]
        self.register_buffer("scale0", scale.float(), persistent=False)  # [out, ng]
        self.register_buffer("zero", zero.float(), persistent=False)     # [out, ng]
        self.theta = nn.Parameter(torch.zeros_like(self.scale0))
        self.bias = bias
        self.group = group
        self.in_features, self.out_features = codes.shape[1], codes.shape[0]
        self.rank = rank
        if rank > 0:   # correcao de baixo posto (liberdade de DIRECAO); A = 0 -> parte do GPTQ
            g = torch.Generator(device=codes.device).manual_seed(codes.shape[0] * 7919 + codes.shape[1])
            self.lr_A = nn.Parameter(torch.zeros(self.out_features, rank, device=codes.device))
            self.lr_B = nn.Parameter(torch.randn(rank, self.in_features, device=codes.device, generator=g)
                                     / self.in_features ** 0.5)

    def weight_fp32(self) -> torch.Tensor:
        s = self.scale0 * self.theta.exp()
        out, cols = self.codes.shape
        q = self.codes.float().view(out, -1, self.group)
        return ((q - self.zero[:, :, None]) * s[:, :, None]).view(out, cols)

    @property
    def weight(self) -> torch.Tensor:   # para codigo que le .weight (dtype/device)
        return self.weight_fp32().to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight_fp32().to(x.dtype), self.bias)
        if self.rank > 0:
            y = y + F.linear(F.linear(x, self.lr_B.to(x.dtype)), self.lr_A.to(x.dtype))
        return y

    def trainable(self) -> list[nn.Parameter]:
        return [self.theta] + ([self.lr_A, self.lr_B] if self.rank > 0 else [])


@torch.no_grad()
def gptq_to_qlinear(model: nn.Module, bank: C.WeightBank, layers: nn.ModuleList, calib: torch.Tensor,
                    bits: int = 4, group: int = 128, batch_size: int = 8) -> dict[str, QLinear]:
    """GPTQ camada a camada (entradas capturadas com as camadas anteriores ja
    quantizadas, como em `calibrated_compress`), trocando cada Linear por QLinear."""
    device = next(model.parameters()).device
    q_mods: dict[str, QLinear] = {}
    for idx in range(bank.num_layers):
        targets = bank.layer(idx)
        stats = C.LayerStats(targets, need_hessian=True, n_samples=0)
        with stats:
            C.run_until_layer(model, layers, idx, calib, device, batch_size)
        for t in targets:
            W = t.module.weight.float()
            Q, codes, s, z = C.gptq(W, stats.H[t.name] / stats.count[t.name] * 2, bits, group, return_params=True)
            t.module.weight.copy_(Q.to(t.module.weight.dtype))   # as proximas camadas veem pesos quantizados
            q_mods[t.name] = QLinear(codes, s, z, t.module.bias, group)
        del stats
        torch.cuda.empty_cache()
    install_qlinear(bank, q_mods)
    return q_mods


def install_qlinear(bank: C.WeightBank, q_mods: dict[str, QLinear]) -> None:
    for t in bank.targets:
        bank.decoder.set_submodule(t.name, q_mods[t.name])


# ------------------------------------------------------------------ dados
@dataclass
class TrainSet:
    ids: torch.Tensor     # [N, L] int64, preenchido a direita
    attn: torch.Tensor    # [N, L] bool
    lmask: torch.Tensor   # [N, L] bool: posicoes cuja previsao entra na perda
    topv: torch.Tensor | None = None   # [N, L, K] fp16, logprobs densos
    topi: torch.Tensor | None = None   # [N, L, K] int32

    def __len__(self) -> int:
        return self.ids.shape[0]

    def batch(self, idx: torch.Tensor, dev) -> dict:
        Lb = int(self.attn[idx].sum(1).max())
        out = {"ids": self.ids[idx, :Lb].to(dev), "attn": self.attn[idx, :Lb].to(dev),
               "lmask": self.lmask[idx, :Lb].to(dev)}
        if self.topv is not None:
            out["topv"] = self.topv[idx, :Lb].to(dev)
            out["topi"] = self.topi[idx, :Lb].to(dev)
        return out

    def split(self, n_val: int, seed: int = 0) -> tuple["TrainSet", "TrainSet"]:
        perm = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        sel = lambda ix: TrainSet(self.ids[ix], self.attn[ix], self.lmask[ix],  # noqa: E731
                                  None if self.topv is None else self.topv[ix],
                                  None if self.topi is None else self.topi[ix])
        return sel(perm[n_val:]), sel(perm[:n_val])


def windows_set(ids: torch.Tensor) -> TrainSet:
    """Janelas de texto (C4/WikiText): todas as posicoes menos a ultima contam."""
    attn = torch.ones_like(ids, dtype=torch.bool)
    lmask = attn.clone()
    lmask[:, -1] = False
    return TrainSet(ids.long(), attn, lmask)


def generated_set(prompt_ids: list[list[int]], gen_ids: list[list[int]], pad: int, max_len: int) -> TrainSet:
    """Prompt + resposta gerada: contam as posicoes que preveem tokens da resposta."""
    rows = [(p + g)[:max_len] for p, g in zip(prompt_ids, gen_ids)]
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    attn = torch.zeros((len(rows), L), dtype=torch.bool)
    lmask = torch.zeros((len(rows), L), dtype=torch.bool)
    for i, (r, p) in enumerate(zip(rows, prompt_ids)):
        ids[i, :len(r)] = torch.tensor(r)
        attn[i, :len(r)] = True
        lmask[i, len(p) - 1:len(r) - 1] = True
    return TrainSet(ids, attn, lmask)


@torch.no_grad()
def attach_dense_targets(model: nn.Module, ts: TrainSet, k: int = 64, batch_size: int = 8) -> None:
    dev = next(model.parameters()).device
    N, L = ts.ids.shape
    ts.topv = torch.zeros(N, L, k, dtype=torch.float16)
    ts.topi = torch.zeros(N, L, k, dtype=torch.int32)
    for s in range(0, N, batch_size):
        idx = torch.arange(s, min(s + batch_size, N))
        b = ts.batch(idx, dev)
        lp = model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False).logits.float().log_softmax(-1)
        v, i = lp.topk(k, dim=-1)
        Lb = b["ids"].shape[1]
        ts.topv[idx, :Lb] = v.half().cpu()
        ts.topi[idx, :Lb] = i.int().cpu()


# ------------------------------------------------------------------ perdas
def token_losses(logits: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor) -> dict[str, torch.Tensor]:
    """Por posicao: KL e TV (top-K denso + cauda, como na avaliacao), a margem u do
    arg-max denso no comprimido, o deslocamento dg da margem densa top-1/top-2 e a
    margem densa g."""
    lq = logits.float().log_softmax(-1)
    rv = topv.float()
    ri = topi.long()
    qv = lq.gather(-1, ri)
    P, Q = rv.exp(), qv.exp()
    tailP = (1 - P.sum(-1)).clamp(min=1e-30)
    tailQ = (1 - Q.sum(-1)).clamp(min=1e-30)
    kl = (P * (rv - qv)).sum(-1) + tailP * (tailP.log() - tailQ.log())
    tv = 0.5 * ((P - Q).abs().sum(-1) + (tailP - tailQ).abs())
    z = logits.float()
    star = ri[..., 0:1]
    u = z.gather(-1, star).squeeze(-1) - z.scatter(-1, star, float("-inf")).amax(-1)
    g = rv[..., 0] - rv[..., 1]
    dg = (qv[..., 0] - qv[..., 1]) - g
    return {"kl": kl, "tv": tv, "u": u, "dg": dg, "g": g}


def _mmean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp(min=1)


def objective(kind: str, L: dict[str, torch.Tensor], m: torch.Tensor, *, beta: float, tau: float) -> torch.Tensor:
    mf = m.float()
    kl = _mmean(L["kl"], mf)
    frag = mf * (L["g"] < FRAGILE).float()
    if kind == "kl":
        return kl
    if kind == "dg":
        return kl + beta * _mmean(L["dg"].abs(), frag)
    if kind == "flip":
        return kl + beta * _mmean(torch.sigmoid(-L["u"] / tau), mf)
    if kind == "tvfrag":
        return kl + beta * _mmean(L["tv"], frag)
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model: nn.Module, ts: TrainSet, kind: str, *, beta: float, tau: float, batch_size: int = 8) -> dict:
    dev = next(model.parameters()).device
    acc = {"kl": 0.0, "tv": 0.0, "flip": 0.0, "dg_frag": 0.0, "obj": 0.0, "n": 0.0, "nf": 0.0}
    for s in range(0, len(ts), batch_size):
        b = ts.batch(torch.arange(s, min(s + batch_size, len(ts))), dev)
        logits = model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False).logits
        L = token_losses(logits, b["topv"], b["topi"])
        m = b["lmask"].float()
        n = m.sum().item()
        fr = m * (L["g"] < FRAGILE).float()
        acc["kl"] += (L["kl"] * m).sum().item(); acc["tv"] += (L["tv"] * m).sum().item()
        acc["flip"] += ((L["u"] < 0).float() * m).sum().item()
        acc["dg_frag"] += (L["dg"].abs() * fr).sum().item(); acc["nf"] += fr.sum().item()
        acc["obj"] += objective(kind, L, b["lmask"], beta=beta, tau=tau).item() * n
        acc["n"] += n
    n, nf = acc.pop("n"), acc.pop("nf")
    out = {k: v / n for k, v in acc.items() if k != "dg_frag"}
    out["dg_frag"] = acc["dg_frag"] / max(nf, 1)
    return out


def train_scales(model: nn.Module, q_mods: dict[str, QLinear], train: TrainSet, val: TrainSet, *,
                 kind: str, steps: int, batch_size: int = 8, lr: float = 1e-3, lr_lowrank: float = 1e-5, beta: float = 1.0,
                 tau: float = 0.3, seed: int = 0, eval_every: int = 100) -> tuple[list[dict], dict]:
    """Devolve (historico, theta do melhor ponto pelo objetivo da propria variante na validacao)."""
    dev = next(model.parameters()).device
    params = [p_ for m in q_mods.values() for p_ in m.trainable()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    # theta e uma escala em log (lr ~1e-3); A e B somam direto nos pesos (~1e-2), entao precisam
    # de passos ~100x menores - com o mesmo lr a correcao de baixo posto diverge (medido)
    thetas = [m.theta for m in q_mods.values()]
    lowrank = [p_ for m in q_mods.values() if m.rank > 0 for p_ in (m.lr_A, m.lr_B)]
    groups = [{"params": thetas, "lr": lr}] + ([{"params": lowrank, "lr": lr_lowrank}] if lowrank else [])
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    g = torch.Generator().manual_seed(seed)
    hist = []
    best = {"obj": float("inf"), "step": -1, "theta": state(q_mods)}

    def check(step):
        model.eval()
        v = evaluate(model, val, kind, beta=beta, tau=tau)
        model.train()
        rec = {"step": step, **{f"val_{k}": x for k, x in v.items()}}
        hist.append(rec)
        LOGGER.info("[%s] step %d | val obj %.4f kl %.4f tv %.4f flip %.4f |dg|frag %.4f", kind, step,
                    v["obj"], v["kl"], v["tv"], v["flip"], v["dg_frag"])
        if v["obj"] < best["obj"]:
            best.update(obj=v["obj"], step=step, theta=state(q_mods))

    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    try:
        with torch.enable_grad():
            check(0)
            for step in range(1, steps + 1):
                b = train.batch(torch.randint(0, len(train), (batch_size,), generator=g), dev)
                logits = model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False).logits
                L = token_losses(logits, b["topv"], b["topi"])
                loss = objective(kind, L, b["lmask"], beta=beta, tau=tau)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
                if step % eval_every == 0 or step == steps:
                    check(step)
    finally:
        model.gradient_checkpointing_disable()
        model.eval()
        for p in params:
            p.requires_grad_(False)
    LOGGER.info("[%s] melhor ponto na validacao: step %d", kind, best["step"])
    return hist, best


def state(q_mods: dict[str, QLinear]) -> dict:
    """theta por modulo; com baixo posto, tambem (A, B) sob a chave '<nome>|lr'."""
    out = {k: m.theta.detach().cpu().clone() for k, m in q_mods.items()}
    for k, m in q_mods.items():
        if m.rank > 0:
            out[k + "|lr"] = (m.lr_A.detach().cpu().clone(), m.lr_B.detach().cpu().clone())
    return out


def load_state(q_mods: dict[str, QLinear], st: dict) -> None:
    with torch.no_grad():
        for k, m in q_mods.items():
            m.theta.copy_(st[k].to(m.theta.device))
            if m.rank > 0 and k + "|lr" in st:
                A_, B_ = st[k + "|lr"]
                m.lr_A.copy_(A_.to(m.lr_A.device)); m.lr_B.copy_(B_.to(m.lr_B.device))


def reset(q_mods: dict[str, QLinear]) -> None:
    """Volta ao GPTQ: theta = 0 e A = 0 (B mantem a inicializacao fixa)."""
    with torch.no_grad():
        for m in q_mods.values():
            m.theta.zero_()
            if m.rank > 0:
                m.lr_A.zero_()
