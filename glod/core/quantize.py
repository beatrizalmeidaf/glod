"""Fase 1b - Elastic Capacity via quantizacao SIMULADA (fake-quant / QDQ).

Versao 1 do documento raiz: `W_l = W_l^base + dW_l`, onde `W_l^base` e uma
representacao quantizada residente e `dW_l` e a correcao recuperada sob
demanda. Aqui nada e realmente comprimido em memoria: quantizamos e
desquantizamos mantendo armazenamento bf16, e contabilizamos os bytes de
forma analitica. Mesma "mentira" do layer-skipping - isola a matematica do
gargalo de I/O, sem bitsandbytes e sem kernel.

Esquema: quantizacao afim por grupo ao longo da dimensao de entrada (group
size 128 por padrao), que e a granularidade de GPTQ/AWQ/NF4. Bits efetivos
por peso = `bits + 32/group_size` (escala e zero-point em fp16 por grupo).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

# Projecoes lineares do bloco decoder que sao alvo de quantizacao. Embeddings,
# normalizacoes e lm_head ficam em bf16 - e o que GPTQ/AWQ/bitsandbytes fazem,
# e mante-los fora evita atribuir ao metodo uma degradacao que nao e dele.
TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

FP16_BITS = 16


def effective_bits(bits: int, group_size: int) -> float:
    """Bits por peso incluindo o overhead de escala + zero-point por grupo."""
    if bits >= FP16_BITS:
        return float(FP16_BITS)
    return bits + 2 * FP16_BITS / group_size


def quantize_dequantize(
    weight: torch.Tensor, bits: int, group_size: int = 128
) -> torch.Tensor:
    """Quantizacao afim por grupo seguida de desquantizacao (QDQ).

    Grupos sao formados ao longo da ULTIMA dimensao (entrada da projecao).
    Quando a dimensao nao e divisivel pelo grupo, a sobra vira um grupo
    proprio menor - nunca preenchemos com valores artificiais, que poluiriam
    o min/max do grupo.
    """
    if bits >= FP16_BITS:
        return weight.clone()
    if bits < 1:
        raise ValueError(f"bits deve ser >= 1, recebido {bits}")

    levels = 2**bits - 1
    original_shape = weight.shape
    flat = weight.reshape(-1, original_shape[-1]).float()
    n_in = flat.shape[-1]

    if n_in % group_size == 0:
        grouped = flat.reshape(flat.shape[0], n_in // group_size, group_size)
        out = _qdq_blocks(grouped, levels).reshape(flat.shape)
    else:
        out = torch.empty_like(flat)
        for start in range(0, n_in, group_size):
            end = min(start + group_size, n_in)
            block = flat[:, start:end].unsqueeze(1)
            out[:, start:end] = _qdq_blocks(block, levels).squeeze(1)

    return out.reshape(original_shape).to(weight.dtype)


def _qdq_blocks(blocks: torch.Tensor, levels: int) -> torch.Tensor:
    """QDQ afim em tensores ja agrupados [..., n_groups, group_size]."""
    lo = blocks.amin(dim=-1, keepdim=True)
    hi = blocks.amax(dim=-1, keepdim=True)
    scale = (hi - lo) / levels
    # grupo constante (hi == lo): escala 0 levaria a divisao por zero; o
    # valor e representado exatamente pelo proprio zero-point.
    safe = scale.clamp(min=torch.finfo(blocks.dtype).tiny)
    codes = ((blocks - lo) / safe).round_().clamp_(0, levels)
    return torch.where(scale > 0, codes * scale + lo, lo.expand_as(blocks))


@dataclass
class QuantTarget:
    """Uma projecao linear quantizavel, com seu peso original preservado."""

    name: str
    module: nn.Linear
    layer_idx: int
    original: torch.Tensor  # copia intocada, usada para restaurar/re-quantizar

    @property
    def numel(self) -> int:
        return self.original.numel()


class QuantizationSimulator:
    """Aplica e desfaz QDQ nas projecoes do decoder, por camada.

    Mantem uma copia dos pesos originais para que toda aplicacao parta do
    bf16 exato - quantizar em cima de um peso ja quantizado acumularia erro
    e produziria uma curva bits x qualidade sistematicamente pessimista.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        group_size: int = 128,
        storage_device: Optional[torch.device | str] = None,
        target_suffixes: Sequence[str] = TARGET_SUFFIXES,
    ) -> None:
        self.group_size = group_size
        self.targets: list[QuantTarget] = []
        pattern = re.compile(r"layers\.(\d+)\.")
        suffixes = tuple(target_suffixes)

        for name, module in decoder.named_modules():
            if not isinstance(module, nn.Linear) or not name.endswith(suffixes):
                continue
            match = pattern.search(name)
            if match is None:  # projecao fora do stack de camadas
                continue
            device = (
                torch.device(storage_device)
                if storage_device is not None
                else module.weight.device
            )
            self.targets.append(
                QuantTarget(
                    name=name,
                    module=module,
                    layer_idx=int(match.group(1)),
                    original=module.weight.detach().to(device).clone(),
                )
            )

        if not self.targets:
            raise RuntimeError("nenhuma projecao quantizavel encontrada no decoder")
        self.num_layers = 1 + max(t.layer_idx for t in self.targets)
        # mapa COMPLETO (toda camada presente), para permitir aplicacao
        # incremental: numa escada, degraus vizinhos diferem por poucas
        # camadas, e requantizar as 48 a cada degrau custa a transferencia
        # inteira da copia original.
        self._bits_map: dict[int, int] = {i: FP16_BITS for i in range(self.num_layers)}
        LOGGER.info(
            "QuantizationSimulator: %d projecoes em %d camadas (%.2f GiB de copia)",
            len(self.targets), self.num_layers,
            sum(t.original.numel() * t.original.element_size() for t in self.targets) / 1024**3,
        )

    # ----------------------------------------------------------------- estado
    @property
    def current_bits(self) -> dict[int, int]:
        """Precisao das camadas que NAO estao em bf16."""
        return {k: v for k, v in self._bits_map.items() if v < FP16_BITS}

    @property
    def bits_map(self) -> dict[int, int]:
        """Precisao de todas as camadas, bf16 inclusive."""
        return dict(self._bits_map)

    def layer_numel(self) -> list[int]:
        counts = [0] * self.num_layers
        for target in self.targets:
            counts[target.layer_idx] += target.numel
        return counts

    # -------------------------------------------------------------- aplicacao
    @torch.no_grad()
    def apply(self, bits_per_layer: dict[int, int], *, incremental: bool = True) -> None:
        """Define a precisao por camada; camadas ausentes voltam a bf16.

        Cada peso e sempre recalculado A PARTIR DO ORIGINAL, nunca em cima do
        peso ja quantizado - requantizar em cascata acumularia erro.

        `incremental=True` pula as camadas cuja precisao nao mudou. So e valido
        se ninguem alterou os pesos por fora; `incremental=False` reaplica tudo.
        """
        desired = {
            i: bits_per_layer.get(i, FP16_BITS) for i in range(self.num_layers)
        }
        for target in self.targets:
            bits = desired[target.layer_idx]
            if incremental and self._bits_map[target.layer_idx] == bits:
                continue
            source = target.original.to(target.module.weight.device)
            new = (
                source
                if bits >= FP16_BITS
                else quantize_dequantize(source, bits, self.group_size)
            )
            target.module.weight.copy_(new)
        self._bits_map = desired

    def apply_uniform(self, bits: int) -> None:
        self.apply({i: bits for i in range(self.num_layers)})

    @torch.no_grad()
    def restore(self) -> None:
        """Devolve todos os pesos ao bf16 original, bit a bit."""
        for target in self.targets:
            target.module.weight.copy_(target.original.to(target.module.weight.device))
        self._bits_map = {i: FP16_BITS for i in range(self.num_layers)}

    # ---------------------------------------------------------- contabilidade
    def weight_bytes(
        self, bits_per_layer: dict[int, int], *, non_target_bytes: int = 0
    ) -> float:
        """Bytes lidos por exemplo sob esta configuracao de precisao.

        `non_target_bytes` cobre o que nunca e quantizado (embeddings, normas,
        lm_head) e portanto sempre precisa ser lido.
        """
        total = float(non_target_bytes)
        for target in self.targets:
            bits = bits_per_layer.get(target.layer_idx, FP16_BITS)
            total += target.numel * effective_bits(bits, self.group_size) / 8
        return total

    def delta_bytes(self, base_bits: int, layer_ids: Iterable[int]) -> float:
        """Bytes de `dW` para promover `layer_ids` de `base_bits` para bf16.

        E a diferenca entre ler o peso cheio e ler so a base - ou seja, o
        custo de I/O que o gate do GLOD decide pagar ou nao.
        """
        per_weight = (FP16_BITS - effective_bits(base_bits, self.group_size)) / 8
        wanted = set(layer_ids)
        return sum(t.numel for t in self.targets if t.layer_idx in wanted) * per_weight
