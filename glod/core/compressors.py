"""Compressores reais (simulados em bf16) para o estudo de fidelidade de decisao.

Tudo aqui e fake-quant/fake-prune: o peso comprimido e escrito de volta no
`nn.Linear` em bf16 e o custo em bytes nao importa - a pergunta e como a
DISTRIBUICAO de saida muda. Implementacoes seguem os algoritmos de referencia:

* GPTQ (Frantar et al., 2023): OBQ em lotes com Cholesky da inversa de H,
  quantizacao assimetrica por grupo, parametros do grupo calculados ao chegar
  na primeira coluna do grupo, damp 1%.
* SparseGPT (Frantar & Alistarh, 2023): mesma maquinaria, mascara por bloco
  pelo criterio w^2 / diag(Hinv)^2; nao estruturado ou N:M.
* Wanda (Sun et al., 2024): |W| * ||X_j||, poda por linha de saida ou N:M.
* AWQ (Lin et al., 2024): busca de s = mean|x|^alpha COMPARTILHADO entre as
  projecoes que leem a mesma entrada, minimizando o erro de saida em
  ativacoes de calibracao; sem a busca de clipping do artigo.

Todos os metodos calibrados sao SEQUENCIAIS: as entradas da camada l sao
capturadas com as camadas < l ja comprimidas, como nos artigos. A captura nao
depende da arquitetura: roda o modelo inteiro e aborta logo apos a camada l.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------- alvos
@dataclass
class Target:
    name: str
    layer_idx: int
    module: nn.Linear
    original: torch.Tensor  # copia bf16 em CPU


#: roteador de um bloco MoE (Qwen3-MoE, OLMoE, Mixtral: `mlp.gate`/`block_sparse_moe.gate`).
#: Nenhum modelo denso do estudo tem uma Linear com esse nome exato (os densos usam
#: `gate_proj`/`gate_up_proj`), entao a exclusao nao muda nenhum resultado publicado.
ROUTER_RE = re.compile(r"\.(?:mlp|block_sparse_moe)\.gate$|\.router$")


class WeightBank:
    """Todas as `nn.Linear` dentro das camadas do decoder, com copia original.

    O roteador de blocos MoE fica FORA por padrao, como nos quantizadores usados na
    pratica; `quant_router=True` (ou GLOD_QUANT_ROUTER=1) o inclui.
    """

    def __init__(self, decoder: nn.Module, quant_router: Optional[bool] = None) -> None:
        import os
        if quant_router is None:
            quant_router = os.environ.get("GLOD_QUANT_ROUTER", "0") == "1"
        self.decoder = decoder
        self.targets: list[Target] = []
        self.routers: list[str] = []
        for name, module in decoder.named_modules():
            m = re.search(r"layers\.(\d+)\.", name + ".")
            if isinstance(module, nn.Linear) and ROUTER_RE.search(name):
                self.routers.append(name)
                if not quant_router:
                    continue
            if isinstance(module, nn.Linear) and m and name.startswith("layers."):
                self.targets.append(Target(name, int(m.group(1)), module,
                                           module.weight.detach().to("cpu", copy=True)))
        if not self.targets:
            raise RuntimeError("nenhuma nn.Linear encontrada nas camadas do decoder")
        self.num_layers = 1 + max(t.layer_idx for t in self.targets)
        self.extra: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}  # nome -> (tensor vivo, copia original)

    def layer(self, idx: int) -> list[Target]:
        return [t for t in self.targets if t.layer_idx == idx]

    @torch.no_grad()
    def restore(self) -> None:
        for t in self.targets:
            t.module.weight.copy_(t.original.to(t.module.weight.device))
        for live, orig in self.extra.values():
            live.copy_(orig.to(live.device))
        self.extra.clear()

    @torch.no_grad()
    def map(self, fn: Callable[[torch.Tensor, Target], torch.Tensor],
            layers: Optional[Iterable[int]] = None) -> None:
        wanted = None if layers is None else set(layers)
        for t in self.targets:
            if wanted is not None and t.layer_idx not in wanted:
                continue
            w = t.original.to(t.module.weight.device)
            t.module.weight.copy_(fn(w, t).to(t.module.weight.dtype))

    def total_params(self) -> int:
        return sum(t.original.numel() for t in self.targets)


# ------------------------------------------------------- operadores simples
def rtn(w: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Afim assimetrica por grupo ao longo da entrada (igual a glod.quantize)."""
    from glod.core.quantize import quantize_dequantize
    return quantize_dequantize(w, bits, group)


def rtn_q40(w: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Formato Q4_0 do llama.cpp: simetrico, grupo 32, d = (valor de maior |w|) / -8,
    q = clamp(round(w/d), -8, 7). E o formato alvo dos checkpoints QAT do Gemma 3."""
    flat = w.float().reshape(-1, group)
    idx = flat.abs().argmax(-1, keepdim=True)
    d = flat.gather(-1, idx) / -8
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.clamp(torch.round(flat / d), -8, 7)
    return (q * d).reshape(w.shape)


def gaussian_like_rtn(w: torch.Tensor, bits: int, gen: torch.Generator, group: int = 128) -> torch.Tensor:
    """Ruido N(0, step^2/12) por grupo: mesmo MSE do RTN, sem estrutura de arredondamento."""
    flat = w.float().reshape(-1, w.shape[-1])
    out = torch.empty_like(flat)
    for s in range(0, flat.shape[-1], group):
        blk = flat[:, s:s + group]
        step = (blk.amax(-1, keepdim=True) - blk.amin(-1, keepdim=True)) / (2 ** bits - 1)
        noise = torch.randn(blk.shape, generator=gen, device=blk.device, dtype=blk.dtype)
        out[:, s:s + group] = blk + noise * step / math.sqrt(12)
    return out.reshape(w.shape)


def magnitude_prune(w: torch.Tensor, frac: float) -> torch.Tensor:
    flat = w.float().reshape(-1, w.shape[-1])
    k = int(frac * flat.shape[-1])
    if k == 0:
        return w.float()
    thr = flat.abs().kthvalue(k, dim=-1, keepdim=True).values
    return torch.where(flat.abs() <= thr, torch.zeros_like(flat), flat).reshape(w.shape)


# ------------------------------------------------------------- calibracao
class _StopForward(Exception):
    pass


def wikitext_calibration(tokenizer, n_seq: int = 128, seq_len: int = 512, seed: int = 0,
                         cache_dir: Optional[str] = None) -> torch.Tensor:
    """n_seq janelas aleatorias de seq_len tokens do treino do WikiText-2 (sem padding)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(ids) - seq_len - 1, (n_seq,), generator=g)
    batch = torch.stack([ids[s:s + seq_len] for s in starts])
    if tokenizer.bos_token_id is not None:
        batch = torch.cat([torch.full((n_seq, 1), tokenizer.bos_token_id), batch[:, :-1]], dim=1)
    return batch


def c4_calibration(tokenizer, n_seq: int = 128, seq_len: int = 512, seed: int = 0,
                   cache_dir: Optional[str] = None) -> torch.Tensor:
    """Como wikitext_calibration, mas do 1o shard de validacao do C4 (en): texto disjunto
    dos corpora wikitext/wikitext_nat, que saem do treino do WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                      split="validation", cache_dir=cache_dir)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    rows = []
    for i in order:                      # um documento por janela, so os que tem seq_len tokens
        ids = tokenizer(ds[i]["text"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(ids) > seq_len:
            s = int(torch.randint(0, len(ids) - seq_len, (1,), generator=g))
            rows.append(ids[s:s + seq_len])
        if len(rows) == n_seq:
            break
    batch = torch.stack(rows)
    if tokenizer.bos_token_id is not None:
        batch = torch.cat([torch.full((n_seq, 1), tokenizer.bos_token_id), batch[:, :-1]], dim=1)
    return batch


def code_calibration(tokenizer, n_seq: int = 128, seq_len: int = 512, seed: int = 0,
                     cache_dir: Optional[str] = None) -> torch.Tensor:
    """Como c4_calibration, mas de arquivos Python do codeparrot-clean-valid (GitHub),
    disjunto das tarefas do MBPP usadas no corpus de codigo."""
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True, cache_dir=cache_dir)
    g = torch.Generator().manual_seed(seed)
    rows = []
    for r in ds.shuffle(seed=seed, buffer_size=2000):
        ids = tokenizer(r["content"], return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(ids) > seq_len:
            s = int(torch.randint(0, len(ids) - seq_len, (1,), generator=g))
            rows.append(ids[s:s + seq_len])
        if len(rows) == n_seq:
            break
    batch = torch.stack(rows)
    if tokenizer.bos_token_id is not None:
        batch = torch.cat([torch.full((n_seq, 1), tokenizer.bos_token_id), batch[:, :-1]], dim=1)
    return batch


CALIBRATION = {"wikitext": wikitext_calibration, "c4": c4_calibration, "code": code_calibration}


class LayerStats:
    """Estatisticas das entradas das Linear de UMA camada."""

    def __init__(self, targets: Sequence[Target], *, need_hessian: bool, n_samples: int) -> None:
        self.targets = list(targets)
        self.need_hessian = need_hessian
        self.n_samples = n_samples
        self.H: dict[str, torch.Tensor] = {}
        self.sq: dict[str, torch.Tensor] = {}      # soma de x^2 por coluna (Wanda)
        self.absmean: dict[str, torch.Tensor] = {}  # soma de |x| por coluna (AWQ)
        self.samples: dict[str, list[torch.Tensor]] = {}
        self.count: dict[str, int] = {}
        self.input_key: dict[str, int] = {}         # identifica Linear que leem o mesmo tensor
        # referencias vivas as entradas do 1o lote: sem isso o alocador reutiliza enderecos
        # de tensores liberados e data_ptr iguais agrupariam Linear que NAO leem a mesma
        # entrada (bug real: grupos AWQ fundidos em silencio quando as dimensoes coincidem)
        self._alive: list[torch.Tensor] = []
        self._hooks = []

    def _hook(self, t: Target):
        def fn(_m, inputs):
            x = inputs[0].detach()
            key = t.name
            if key not in self.input_key:
                self.input_key[key] = x.data_ptr()
                self._alive.append(inputs[0])
            x = x.reshape(-1, x.shape[-1]).float()
            n = x.shape[0]
            if n == 0:
                return
            self.count[key] = self.count.get(key, 0) + n
            if self.need_hessian:
                if key not in self.H:
                    self.H[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                self.H[key].addmm_(x.T, x)
            self.sq[key] = self.sq.get(key, 0) + (x * x).sum(0)
            self.absmean[key] = self.absmean.get(key, 0) + x.abs().sum(0)
            if self.n_samples:
                keep = self.samples.setdefault(key, [])
                have = sum(s.shape[0] for s in keep)
                if have < self.n_samples:
                    idx = torch.randperm(n, device=x.device)[: min(n, self.n_samples - have)]
                    keep.append(x[idx].to(torch.bfloat16))
        return fn

    def __enter__(self):
        for t in self.targets:
            self._hooks.append(t.module.register_forward_pre_hook(self._hook(t)))
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        return False

    def groups(self) -> list[list[Target]]:
        """Linear agrupadas pela mesma entrada (q/k/v; gate/up) - o grupo do AWQ."""
        by: dict[int, list[Target]] = {}
        for t in self.targets:
            if not self.count.get(t.name):
                continue            # expert MoE sem token: fica para o fallback
            by.setdefault(self.input_key[t.name], []).append(t)
        self._alive.clear()
        return list(by.values())


@torch.no_grad()
def run_until_layer(model: nn.Module, layers: nn.ModuleList, idx: int, calib: torch.Tensor,
                    device, batch_size: int = 8) -> None:
    """Forward do calib inteiro, abortando logo depois da camada `idx`."""
    def stop(*_):
        raise _StopForward
    handle = layers[idx].register_forward_hook(stop)
    try:
        for s in range(0, len(calib), batch_size):
            try:
                model(input_ids=calib[s:s + batch_size].to(device), use_cache=False)
            except _StopForward:
                pass
    finally:
        handle.remove()


# ------------------------------------------------------------------ GPTQ
def _quant_params(x: torch.Tensor, maxq: int) -> tuple[torch.Tensor, torch.Tensor]:
    xmin = torch.minimum(x.min(1).values, torch.zeros(1, device=x.device))
    xmax = torch.maximum(x.max(1).values, torch.zeros(1, device=x.device))
    both0 = (xmin == 0) & (xmax == 0)
    xmin[both0], xmax[both0] = -1, 1
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    return scale.unsqueeze(1), zero.unsqueeze(1)


def _qdq(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def _hinv_upper(H: torch.Tensor, W: torch.Tensor, percdamp: float) -> torch.Tensor:
    """Cholesky da inversa de H com amortecimento adaptativo, no proprio tensor.

    Com 1% de damp o Hessiano de camadas grandes (medido no Mistral-Small-24B,
    d_in=32768) as vezes nao e positivo-definido em float32; subir o damp e o
    remedio padrao das implementacoes de GPTQ/SparseGPT. A soma vai direto na
    diagonal: uma identidade 32768x32768 custaria 4 GiB de VRAM.
    """
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    mean_diag = torch.mean(torch.diag(H)).item()
    added = 0.0
    for factor in (1, 3, 10, 30, 100):
        target = percdamp * factor * mean_diag
        H.diagonal().add_(target - added)
        added = target
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            del L
            out = torch.linalg.cholesky(Hinv, upper=True)
            del Hinv
            return out
        except torch._C._LinAlgError:
            if factor == 100:
                raise
            LOGGER.warning("Cholesky falhou com damp %.3f; subindo", percdamp * factor)
    raise RuntimeError("inalcancavel")


@torch.no_grad()
def gptq(W: torch.Tensor, H: torch.Tensor, bits: int, group: int = 128, blocksize: int = 128,
         percdamp: float = 0.01, return_params: bool = False):
    """GPTQ por grupo. Com `return_params`, devolve tambem os codigos inteiros e as
    escalas/zeros por grupo, tais que Q == scale * (codes - zero) (usado pelo AMQ)."""
    W = W.clone().float()
    cols = W.shape[1]
    maxq = 2 ** bits - 1
    Hinv = _hinv_upper(H, W, percdamp)
    Q = torch.zeros_like(W)
    codes = torch.zeros(W.shape, dtype=torch.uint8, device=W.device) if return_params else None
    n_groups = (cols + group - 1) // group
    scales = torch.zeros(W.shape[0], n_groups, device=W.device) if return_params else None
    zeros = torch.zeros(W.shape[0], n_groups, device=W.device) if return_params else None
    scale = zero = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                scale, zero = _quant_params(W[:, i1 + i:i1 + i + group], maxq)
                if return_params:
                    scales[:, (i1 + i) // group] = scale.squeeze(1)
                    zeros[:, (i1 + i) // group] = zero.squeeze(1)
            w = W1[:, i]
            d = Hinv1[i, i]
            q = _qdq(w.unsqueeze(1), scale, zero, maxq).flatten()
            if return_params:
                codes[:, i1 + i] = torch.clamp(torch.round(w / scale.squeeze(1)) + zero.squeeze(1), 0, maxq).to(torch.uint8)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    if return_params:
        return Q, codes, scales, zeros
    return Q


@torch.no_grad()
def sparsegpt(W: torch.Tensor, H: torch.Tensor, sparsity: float, prunen: int = 0, prunem: int = 0,
              blocksize: int = 128, percdamp: float = 0.01) -> torch.Tensor:
    W = W.clone().float()
    cols = W.shape[1]
    Hinv = _hinv_upper(H, W, percdamp)
    mask = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if prunen == 0:
            tmp = W1 ** 2 / (torch.diag(Hinv1).reshape(1, -1)) ** 2
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            mask1 = tmp <= thresh
        else:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            if prunen != 0 and i % prunem == 0:
                tmp = W1[:, i:i + prunem] ** 2 / (torch.diag(Hinv1)[i:i + prunem].reshape(1, -1)) ** 2
                mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return W


@torch.no_grad()
def wanda(W: torch.Tensor, sq: torch.Tensor, count: int, sparsity: float,
          prunen: int = 0, prunem: int = 0) -> torch.Tensor:
    W = W.clone().float()
    metric = W.abs() * torch.sqrt(sq / count).reshape(1, -1)
    mask = torch.zeros_like(W, dtype=torch.bool)
    if prunen:
        for i in range(0, W.shape[1], prunem):
            tmp = metric[:, i:i + prunem]
            mask.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
    else:
        idx = torch.sort(metric, dim=-1, stable=True)[1][:, : int(W.shape[1] * sparsity)]
        mask.scatter_(1, idx, True)
    W[mask] = 0
    return W


@torch.no_grad()
def awq_group(ws: Sequence[torch.Tensor], absmean: torch.Tensor, X: torch.Tensor, bits: int,
              group: int = 128, grid: int = 20) -> tuple[list[torch.Tensor], float]:
    """Escala compartilhada s = mean|x|^alpha para as Linear que leem a mesma entrada."""
    X = X.float()
    refs = [X @ w.float().T for w in ws]
    best, best_alpha, best_ws = float("inf"), 0.0, [w.float() for w in ws]
    for a in range(grid):
        alpha = a / grid
        s = absmean.clamp(min=1e-4).pow(alpha)
        s = s / (s.max() * s.min()).sqrt()
        cand = [rtn(w.float() * s.view(1, -1), bits, group).float() / s.view(1, -1) for w in ws]
        err = sum(((X @ c.T) - r).pow(2).mean().item() for c, r in zip(cand, refs))
        if err < best:
            best, best_alpha, best_ws = err, alpha, cand
    return best_ws, best_alpha


@torch.no_grad()
def calibrated_compress(model: nn.Module, bank: WeightBank, layers: nn.ModuleList, calib: torch.Tensor,
                        method: str, *, bits: int = 4, sparsity: float = 0.5, prunen: int = 0,
                        prunem: int = 0, group: int = 128, batch_size: int = 8) -> dict:
    """Comprime camada a camada, capturando entradas com as anteriores ja comprimidas."""
    bank.restore()
    device = next(model.parameters()).device
    info: dict = {"method": method, "alphas": []}
    for idx in range(bank.num_layers):
        targets = bank.layer(idx)
        stats = LayerStats(targets, need_hessian=method in ("gptq", "sparsegpt"),
                           n_samples=2048 if method == "awq" else 0)
        with stats:
            run_until_layer(model, layers, idx, calib, device, batch_size)
        # experts MoE que nao receberam nenhum token de calibracao nao tem estatistica:
        # recebem o operador nao calibrado do mesmo formato (RTN / magnitude)
        unseen = [t for t in targets if not stats.count.get(t.name)]
        for t in unseen:
            W = t.module.weight.float()
            if method in ("gptq", "awq"):
                new = rtn(W, bits, group)
            elif prunem:
                new = W        # N:M sem estatistica: mantem denso e conta no fallback
            else:
                new = magnitude_prune(W, sparsity)
            t.module.weight.copy_(new.to(t.module.weight.dtype))
        info["fallback"] = info.get("fallback", 0) + len(unseen)
        info["n_targets"] = info.get("n_targets", 0) + len(targets)
        seen = {t.name for t in targets} - {t.name for t in unseen}
        if method == "awq":
            for grp in stats.groups():
                key = grp[0].name
                X = torch.cat(stats.samples[key])
                new, alpha = awq_group([t.module.weight for t in grp], stats.absmean[key] / stats.count[key],
                                       X, bits, group)
                info["alphas"].append(alpha)
                for t, w in zip(grp, new):
                    t.module.weight.copy_(w.to(t.module.weight.dtype))
        else:
            for t in targets:
                if t.name not in seen:
                    continue
                W = t.module.weight.float()
                if method == "gptq":
                    new = gptq(W, stats.H[t.name] / stats.count[t.name] * 2, bits, group)
                elif method == "sparsegpt":
                    new = sparsegpt(W, stats.H[t.name] / stats.count[t.name] * 2, sparsity, prunen, prunem)
                elif method == "wanda":
                    new = wanda(W, stats.sq[t.name], stats.count[t.name], sparsity, prunen, prunem)
                else:
                    raise ValueError(method)
                t.module.weight.copy_(new.to(t.module.weight.dtype))
        del stats
        torch.cuda.empty_cache()
    return info


# ----------------------------------------------------------- KV cache fake-quant
def _qdq_lastdim(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Afim assimetrica com um grupo por vetor (ultima dimensao) - por token e por cabeca."""
    xf = x.float()
    lo = xf.amin(-1, keepdim=True)
    hi = xf.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    q = torch.clamp(torch.round((xf - lo) / scale), 0, 2 ** bits - 1)
    return (q * scale + lo).to(x.dtype)


def _qdq_keys_per_channel(k: torch.Tensor, bits: int, token_group: int = 32) -> torch.Tensor:
    """KIVI: chaves [B, H, T, d] quantizadas POR CANAL, em grupos de `token_group` tokens."""
    B, H, T, d = k.shape
    pad = (-T) % token_group
    kf = torch.nn.functional.pad(k.float(), (0, 0, 0, pad), value=float("nan"))
    g = kf.view(B, H, -1, token_group, d)
    lo = torch.nan_to_num(g, nan=float("inf")).amin(3, keepdim=True)
    hi = torch.nan_to_num(g, nan=float("-inf")).amax(3, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-8)
    q = torch.clamp(torch.round((g - lo) / scale), 0, 2 ** bits - 1) * scale + lo
    return q.view(B, H, -1, d)[:, :, :T].to(k.dtype)


@contextlib.contextmanager
def kv_fake_quant(model: nn.Module, decoder: nn.Module, bits: int, per_channel_keys: bool = True):
    """Quantiza chaves POS-RoPE (o que vai para o cache) e valores.

    Esquema KIVI (padrao): chaves por canal em grupos de 32 tokens, valores por
    token e cabeca. `per_channel_keys=False`: chaves tambem por token (ingenuo;
    colapsa a 4 bits porque as chaves tem canais outlier, como o KIVI reporta).
    Chaves: embrulha `apply_rotary_pos_emb` do modulo da arquitetura.
    Valores: hook na saida de v_proj (ou na fatia v de qkv_proj no Phi3).
    """
    attn = decoder.layers[0].self_attn if hasattr(decoder.layers[0], "self_attn") else None
    if attn is None:
        raise RuntimeError("camada sem self_attn")
    inner = getattr(attn, "layer", attn)
    mod = sys.modules[type(inner).__module__]
    original = mod.apply_rotary_pos_emb

    def patched(q, k, *args, **kwargs):
        qe, ke = original(q, k, *args, **kwargs)
        return qe, (_qdq_keys_per_channel(ke, bits) if per_channel_keys else _qdq_lastdim(ke, bits))

    cfg = decoder.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    hooks = []
    for layer in decoder.layers:
        sa = layer.self_attn
        if hasattr(sa, "v_proj"):
            def vhook(_m, _i, out):
                shp = out.shape
                return _qdq_lastdim(out.view(*shp[:-1], -1, head_dim), bits).view(shp)
            hooks.append(sa.v_proj.register_forward_hook(vhook))
        elif hasattr(sa, "qkv_proj"):
            q_dim = cfg.num_attention_heads * head_dim
            kv_dim = cfg.num_key_value_heads * head_dim

            def qkvhook(_m, _i, out, q_dim=q_dim, kv_dim=kv_dim):
                v = out[..., q_dim + kv_dim:]
                vq = _qdq_lastdim(v.reshape(*v.shape[:-1], -1, head_dim), bits).reshape(v.shape)
                return torch.cat([out[..., :q_dim + kv_dim], vq], dim=-1)
            hooks.append(sa.qkv_proj.register_forward_hook(qkvhook))
        else:
            raise RuntimeError("projecao de valor nao encontrada")
    mod.apply_rotary_pos_emb = patched
    try:
        yield
    finally:
        mod.apply_rotary_pos_emb = original
        for h in hooks:
            h.remove()


# ------------------------------------------------- checkpoints oficiais GPTQ/AWQ
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _unpack_cols(q: torch.Tensor, bits: int) -> torch.Tensor:
    """int32 [a, b] -> int [a, b*32/bits], ordem de bits menos significativo primeiro."""
    shifts = torch.arange(0, 32, bits, device=q.device)
    return ((q.unsqueeze(-1) >> shifts) & (2 ** bits - 1)).reshape(q.shape[0], -1)


def dequant_gptq(qweight, qzeros, scales, g_idx, bits: int, out_features: int, in_features: int,
                 zero_offset: int) -> torch.Tensor:
    # qweight [in*bits/32, out] empacotado ao longo das LINHAS (entradas)
    iw = _unpack_cols(qweight.T.contiguous(), bits)[:, :in_features]          # [out, in]
    iz = _unpack_cols(qzeros, bits)[:, :out_features] + zero_offset           # [groups, out]
    g = g_idx.long() if g_idx is not None else torch.arange(in_features) // (in_features // scales.shape[0])
    return scales[g].T.float() * (iw.float() - iz[g].T.float())


def dequant_awq(qweight, qzeros, scales, bits: int, out_features: int, in_features: int, group: int,
                transposed: bool = False, reorder: bool = True) -> torch.Tensor:
    """AWQ GEMM. `transposed`: qweight [in, out*bits/32] (visto em Qwen3-*-AWQ) em vez de [out, in*bits/32]."""
    iw = _unpack_cols(qweight, bits)
    iz = _unpack_cols(qzeros, bits)                                            # [groups, out]
    if reorder:
        iw = iw[:, torch.arange(iw.shape[1]).view(-1, 32 // bits)[:, AWQ_REVERSE_ORDER].reshape(-1)]
        iz = iz[:, torch.arange(iz.shape[1]).view(-1, 32 // bits)[:, AWQ_REVERSE_ORDER].reshape(-1)]
    iz = iz[:, :out_features]
    iw = iw.T[:, :in_features] if transposed else iw[:, :in_features]            # [out, in]
    g = torch.arange(in_features) // group
    return scales[g].T.float() * (iw.float() - iz[g].T.float())


def dequant_awq_auto(qweight, qzeros, scales, bits, out_features, in_features, group, W0) -> torch.Tensor:
    """Escolhe orientacao/ordem pelo menor erro relativo contra o bf16 (as escalas fundidas do AWQ
    deixam o erro correto em ~0.1-0.6; layouts errados ficam acima de 1)."""
    pack = 32 // bits
    cands = []
    for transposed in (False, True):
        rows = in_features if transposed else out_features
        if qweight.shape[0] != rows or qweight.shape[1] * pack < (out_features if transposed else in_features):
            continue
        for reorder in (True, False):
            W = dequant_awq(qweight, qzeros, scales, bits, out_features, in_features, group, transposed, reorder)
            col = torch.nn.functional.cosine_similarity(W, W0, dim=0).mean().item()
            row = torch.nn.functional.cosine_similarity(W, W0, dim=1).mean().item()
            cands.append((-max(col, row), W))  # invariante as escalas por canal fundidas pelo AWQ
    return min(cands, key=lambda c: c[0])[1]


@torch.no_grad()
def load_official_quantized(bank: WeightBank, repo: str, cache_dir: str) -> dict:
    """Carrega um checkpoint GPTQ/AWQ publicado: Linear desquantizadas + TODOS os demais tensores.

    O AWQ funde as escalas de ativacao nas camadas anteriores (normas divididas por s,
    q/k/v/gate/up multiplicados por s; v_proj/up_proj absorvem as escalas de o/down),
    entao comparar peso a peso com o bf16 nao valida nada e carregar so as Linear
    quebraria o modelo. Normas e demais tensores do checkpoint sao copiados e
    guardados em `bank.extra` para o `restore`. Validacao: o erro relativo mediano das
    Linear (so informativo no AWQ) e o KL medido depois.
    """
    import json
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(snapshot_download(repo, cache_dir=cache_dir))
    cfg = json.loads((path / "config.json").read_text()).get("quantization_config", {})
    method = cfg.get("quant_method", "gptq")
    bits = int(cfg.get("bits", 4))
    group = int(cfg.get("group_size", 128))
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                tensors[k] = fh.get_tensor(k)
    rel_errs = []
    linear_prefixes = set()
    for t in bank.targets:
        prefix = next((p for p in (f"model.{t.name}", t.name, f"model.language_model.{t.name}")
                       if f"{p}.qweight" in tensors), None)
        if prefix is None:
            raise KeyError(f"{t.name} ausente em {repo}")
        linear_prefixes.add(prefix)
        W0 = t.original.float()
        out_f, in_f = W0.shape
        qw, qz, sc = (tensors[f"{prefix}.{s_}"] for s_ in ("qweight", "qzeros", "scales"))
        if method == "awq":
            W = dequant_awq_auto(qw, qz, sc, bits, out_f, in_f, group, W0)
        else:
            gi = tensors.get(f"{prefix}.g_idx")
            cands = [dequant_gptq(qw, qz, sc, gi, bits, out_f, in_f, off) for off in (1, 0)]
            W = min(cands, key=lambda c: ((c - W0).norm() / W0.norm()).item())
        rel_errs.append(1 - max(torch.nn.functional.cosine_similarity(W, W0, dim=0).mean().item(),
                                torch.nn.functional.cosine_similarity(W, W0, dim=1).mean().item()))
        t.module.weight.copy_(W.to(t.module.weight.dtype).to(t.module.weight.device))
    # demais tensores (normas, embeddings, bias): copia do checkpoint, guardando o original
    # o wrapper de skip insere ".layer." nos nomes (layers.N.layer.input_layernorm...)
    model_params = {re.sub(r"(layers\.\d+)\.layer\.", r"\1.", n): p for n, p in bank.decoder.named_parameters()}
    n_extra = 0
    for k, v in tensors.items():
        if k.endswith((".qweight", ".qzeros", ".scales", ".g_idx")):
            continue
        name = k[len("model."):] if k.startswith("model.") else k
        if name not in model_params:
            continue
        live = model_params[name]
        if live.shape != v.shape:
            continue
        if name not in bank.extra:
            bank.extra[name] = (live.data, live.data.detach().to("cpu", copy=True))
        live.data.copy_(v.to(live.dtype).to(live.device))
        n_extra += 1
    med = sorted(rel_errs)[len(rel_errs) // 2]
    if med > 0.2:
        raise RuntimeError(f"layout de {repo} parece errado: 1 - cosseno mediano {med:.3f}")
    return {"method": method, "bits": bits, "group": group, "median_1_minus_cos": med,
            "max_1_minus_cos": max(rel_errs), "extra_tensors_loaded": n_extra}
