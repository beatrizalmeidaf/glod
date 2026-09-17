"""Fase 1b, passo 2 - escada de capacidade por PRECISAO (Versao 1 do documento).

`W_l = W_l^base + dW_l`, onde `W_l^base` e a camada quantizada num bitrate
agressivo e `dW_l` e a informacao que promove aquela camada a um bitrate
maior. O gate decide, por input, quais camadas promover.

A diferenca em relacao ao Elastic Depth da Fase 1a: nenhuma camada e
amputada. O fluxo residual continua completo, e o eixo de bytes tem
granularidade fina - que foi exatamente o que faltou ao layer-skipping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import torch
from tqdm.auto import tqdm

from ews.corpora.mmlu import MMLUSplit
from ews.core.model_loader import LoadedModel
from ews.core.quantize import FP16_BITS, QuantizationSimulator
from ews.core.scoring import gold_nll, predictions_and_correctness, score_choices

LOGGER = logging.getLogger(__name__)


@dataclass
class QuantSensitivity:
    """Ganho de promover cada camada de `base_bits` para `target_bits`.

    `delta_nll[l]` = NLL(base inteira) - NLL(base com a camada l promovida).
    Positivo significa que promover aquela camada ajuda.
    """

    delta_nll: list[float]
    delta_se: list[float]  # erro padrao PAREADO do ganho de cada camada
    base_nll: float
    base_bits: int
    target_bits: int
    order_descending: list[int] = field(init=False)  # mais valiosa primeiro

    def __post_init__(self) -> None:
        self.order_descending = sorted(
            range(len(self.delta_nll)), key=lambda i: -self.delta_nll[i]
        )

    @property
    def n_significant(self) -> int:
        """Camadas cujo ganho e distinguivel de zero (|media| > 2 EP)."""
        return sum(
            1 for d, e in zip(self.delta_nll, self.delta_se) if abs(d) > 2 * e
        )

    def signal_to_noise(self) -> float:
        """Dispersao entre camadas dividida pelo ruido tipico de medicao.

        Abaixo de ~1 a ordenacao das camadas e majoritariamente ruido, e a
        escada construida a partir dela nao significa nada.
        """
        import statistics

        if len(self.delta_nll) < 2:
            return 0.0
        spread = statistics.stdev(self.delta_nll)
        noise = statistics.mean(self.delta_se) if self.delta_se else 0.0
        return spread / noise if noise > 0 else float("inf")


def quant_sensitivity_profile(
    loaded: LoadedModel,
    sim: QuantizationSimulator,
    calib: MMLUSplit,
    choice_ids: Sequence[int],
    *,
    base_bits: int,
    target_bits: int = FP16_BITS,
    batch_size: int = 16,
) -> QuantSensitivity:
    """Perfil por PROMOCAO: parte da base inteira e promove UMA camada por vez.

    A direcao importa. Rebaixar uma unica camada partindo do modelo bf16
    saudavel quase nao move a NLL - os outros 47 blocos compensam, e o efeito
    afoga no ruido de calibracao (medimos mediana de dNLL ~ -0.001 a 3 bits).
    Partindo da base degradada, a contribuicao de cada promocao fica visivel,
    e e exatamente a operacao que a escada e o gate executam.
    """
    base_map = {i: base_bits for i in range(sim.num_layers)}
    sim.apply(base_map)
    logits, _ = score_choices(
        loaded, calib.prompts, choice_ids, batch_size=batch_size, show_progress=False
    )
    base_per_example = gold_nll(logits, calib.targets)
    base = base_per_example.mean().item()

    deltas: list[float] = []
    errors: list[float] = []
    n = len(calib)
    for layer_idx in tqdm(
        range(sim.num_layers), desc=f"promocao {base_bits}b->{target_bits}b"
    ):
        sim.apply({**base_map, layer_idx: target_bits})
        logits, _ = score_choices(
            loaded, calib.prompts, choice_ids, batch_size=batch_size,
            show_progress=False,
        )
        # diferenca PAREADA por exemplo: o mesmo item e avaliado nas duas
        # configuracoes, entao a variancia entre itens (que domina a NLL
        # marginal) cancela e o erro padrao fica muito menor.
        paired = base_per_example - gold_nll(logits, calib.targets)
        deltas.append(paired.mean().item())
        errors.append((paired.std(unbiased=True) / (n**0.5)).item())
    sim.restore()
    return QuantSensitivity(
        delta_nll=deltas, delta_se=errors, base_nll=base,
        base_bits=base_bits, target_bits=target_bits,
    )


@dataclass
class QuantRung:
    """Um degrau: quais camadas foram promovidas de `base_bits` a `target_bits`."""

    index: int
    promoted: list[int]
    base_bits: int
    target_bits: int
    weight_bytes: float

    @property
    def n_promoted(self) -> int:
        return len(self.promoted)

    @property
    def label(self) -> str:
        return f"{self.n_promoted:2d}x{self.target_bits}b"

    def bits_map(self, num_layers: int) -> dict[int, int]:
        promoted = set(self.promoted)
        return {
            i: (self.target_bits if i in promoted else self.base_bits)
            for i in range(num_layers)
        }


def build_quant_ladder(
    sensitivity: QuantSensitivity,
    sim: QuantizationSimulator,
    *,
    base_bits: int,
    target_bits: int,
    promote_counts: Sequence[int],
    non_target_bytes: int = 0,
) -> list[QuantRung]:
    """Degraus ANINHADOS por numero de camadas promovidas, custo crescente.

    As camadas entram na ordem de sensibilidade: a mais prejudicada pela
    quantizacao e a primeira a ser promovida. E o mesmo criterio guloso do
    Elastic Depth, e herda a mesma ressalva - sensibilidade medida uma a uma
    nao e aditiva.
    """
    ladder: list[QuantRung] = []
    for i, k in enumerate(sorted(set(promote_counts))):
        promoted = sorted(sensitivity.order_descending[:k])
        rung = QuantRung(
            index=i, promoted=promoted, base_bits=base_bits,
            target_bits=target_bits, weight_bytes=0.0,
        )
        rung.weight_bytes = sim.weight_bytes(
            rung.bits_map(sim.num_layers), non_target_bytes=non_target_bytes
        )
        ladder.append(rung)
    return ladder


def evaluate_quant_ladder(
    loaded: LoadedModel,
    sim: QuantizationSimulator,
    ladder: Sequence[QuantRung],
    split: MMLUSplit,
    choice_ids: Sequence[int],
    *,
    batch_size: int = 16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Matriz de acerto [K, N] e diagnosticos do degrau mais barato.

    A entropia do degrau base e guardada porque e a feature que o gate real
    da Fase 2 vai consumir - ter isso agora evita reavaliar tudo depois.
    """
    rows: list[torch.Tensor] = []
    base_stats: dict[str, torch.Tensor] = {}
    for rung in tqdm(list(ladder), desc="escada de precisao"):
        sim.apply(rung.bits_map(sim.num_layers))
        logits, stats = score_choices(
            loaded, split.prompts, choice_ids, batch_size=batch_size,
            show_progress=False,
        )
        _, correct = predictions_and_correctness(logits, split.targets)
        rows.append(correct)
        if rung.index == 0:
            base_stats = {k: v.clone() for k, v in stats.items()}
            base_stats["choice_logits"] = logits.clone()
    sim.restore()
    return torch.stack(rows), base_stats
