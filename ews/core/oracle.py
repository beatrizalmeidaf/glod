"""Passo 4 - Gate Oraculo sobre uma escada de capacidade do proprio 12B.

Diferenca central em relacao a uma cascata 4B->12B: aqui existe UM modelo.
A "base" e o esqueleto do 12B com camadas desligadas (`g_l = 0`), conforme
`W_l = W_l^base + dW_l` do documento raiz. Uma cascata de dois modelos mede
*model cascading* (FrugalGPT, Big Little Decoder), que e prior art densa e
paga 2x nos exemplos dificeis - o EWS paga 1x.

Duas definicoes de oraculo sao reportadas, e a distincao importa:

* **consistente** - so "economiza", nunca "ganha". Nos exemplos em que o 12B
  denso acerta, escolhe o degrau mais barato que tambem acerta; nos demais,
  usa o denso. Por construcao a acuracia e IGUAL a do denso, e a metrica
  vira bytes economizados. E o numero honesto para o paper.
* **irrestrito** - melhor degrau qualquer. Pode superar o denso, explorando
  casos em que uma versao mutilada acerta por sorte. E um teto valido para a
  classe de politica, mas inflado por um componente nao-aprendivel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from math import comb, sqrt
from pathlib import Path
from typing import Optional, Sequence

import torch

from ews.core.elastic_depth import ElasticDepthController
from ews.corpora.mmlu import MMLUSplit
from ews.core.model_loader import LoadedModel
from ews.core.scoring import gold_nll, predictions_and_correctness, score_choices

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------- estatistica
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson - preferivel ao normal com n pequeno ou p perto de 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (centre - half), 100 * (centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """p-valor bilateral do teste exato de McNemar para dados pareados.

    `only_a` = acertos exclusivos do sistema A, `only_b` = do sistema B.
    Comparacoes no mesmo conjunto de itens SAO pareadas; usar dois ICs
    independentes e comparar sobreposicao e conservador demais e errado.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2**n))


# ------------------------------------------------------- perfil de importancia
@dataclass
class LayerImportance:
    """Sensibilidade leave-one-out de cada camada do decoder."""

    delta_nll: list[float]  # aumento da NLL da alternativa correta ao pular a camada
    dense_nll: float
    order_ascending: list[int] = field(init=False)  # menos importante primeiro

    def __post_init__(self) -> None:
        self.order_ascending = sorted(
            range(len(self.delta_nll)), key=lambda i: self.delta_nll[i]
        )


def layer_importance_profile(
    full: LoadedModel,
    ctrl: ElasticDepthController,
    calib: MMLUSplit,
    choice_ids: Sequence[int],
    *,
    batch_size: int = 16,
) -> LayerImportance:
    """Mede quanto cada camada, sozinha, importa (leave-one-out).

    Usa NLL da alternativa correta em vez de acuracia: com o N pequeno de um
    conjunto de calibracao, acuracia 0/1 e ruido. O smoke test do passo 3 ja
    tinha mostrado que *quais* camadas se pula domina *quantas* - pular 10
    uniformes deu NLL melhor que pular 5 -, entao a escada nao pode ser
    uniforme; ela sai desta ordenacao.
    """
    with ctrl.dense():
        logits, _ = score_choices(
            full, calib.prompts, choice_ids, batch_size=batch_size,
            desc="calib denso", show_progress=False,
        )
    dense = gold_nll(logits, calib.targets).mean().item()

    deltas: list[float] = []
    from tqdm.auto import tqdm

    for layer_idx in tqdm(range(ctrl.num_layers), desc="perfil de importancia"):
        with ctrl.skipping([layer_idx]):
            logits, _ = score_choices(
                full, calib.prompts, choice_ids, batch_size=batch_size,
                show_progress=False,
            )
        deltas.append(gold_nll(logits, calib.targets).mean().item() - dense)

    return LayerImportance(delta_nll=deltas, dense_nll=dense)


# --------------------------------------------------------- escada de capacidade
@dataclass
class Rung:
    """Um degrau da escada: quais camadas ficam desligadas e quanto custa."""

    index: int
    skipped: list[int]
    weight_bytes: int  # pesos efetivamente lidos por exemplo neste degrau

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)


def select_skipped(order_ascending: Sequence[int], k: int, min_gap: int = 1) -> list[int]:
    """Escolhe as k camadas menos importantes, opcionalmente sem encostar.

    `min_gap=1` e a selecao gulosa pura. `min_gap=2` proibe pular duas camadas
    adjacentes. A distincao importa porque o leave-one-out mede importancia
    INDIVIDUAL, e importancia de camada nao e aditiva: remover um bloco
    contiguo do stack corta o fluxo residual de um jeito que remover as mesmas
    camadas espalhadas nao corta.
    """
    chosen: list[int] = []
    for layer in order_ascending:
        if len(chosen) >= k:
            break
        if all(abs(layer - c) >= min_gap for c in chosen):
            chosen.append(layer)
    return sorted(chosen)


def build_capacity_ladder(
    importance: LayerImportance,
    ctrl: ElasticDepthController,
    full: LoadedModel,
    skip_counts: Sequence[int],
    *,
    min_gap: int = 1,
) -> list[Rung]:
    """Degraus ANINHADOS C_0 subset C_1 ... subset C_K (denso por ultimo).

    O aninhamento nao e cosmetico: e o que permite interpretar o oraculo como
    "quanta capacidade recuperar", e o que a Fase 3 vai realmente streamar de
    forma incremental.
    """
    non_decoder_bytes = full.param_bytes - ctrl.total_param_bytes
    ladder: list[Rung] = []
    for i, k in enumerate(sorted(set(skip_counts), reverse=True)):
        skipped = select_skipped(importance.order_ascending, k, min_gap=min_gap)
        active_bytes = ctrl.total_param_bytes - sum(
            ctrl.layer_param_bytes[l] for l in skipped
        )
        ladder.append(
            Rung(index=i, skipped=skipped, weight_bytes=non_decoder_bytes + active_bytes)
        )
    return ladder  # custo crescente; o ultimo degrau e o denso


# ------------------------------------------------------------------- oraculo
@dataclass
class OracleResult:
    n: int
    dense_accuracy: float
    dense_bytes: int
    consistent_accuracy: float
    consistent_bytes: float
    unconstrained_accuracy: float
    unconstrained_bytes: float
    rung_accuracy: list[float]
    rung_bytes: list[int]
    correct: torch.Tensor  # [K+1, N] booleano por degrau


def evaluate_ladder(
    full: LoadedModel,
    ctrl: ElasticDepthController,
    ladder: Sequence[Rung],
    split: MMLUSplit,
    choice_ids: Sequence[int],
    *,
    batch_size: int = 16,
) -> torch.Tensor:
    """Matriz de acerto [K+1, N]: cada degrau avaliado em todo o conjunto."""
    from tqdm.auto import tqdm

    rows: list[torch.Tensor] = []
    for rung in tqdm(list(ladder), desc="escada de capacidade"):
        with ctrl.skipping(rung.skipped):
            logits, _ = score_choices(
                full, split.prompts, choice_ids, batch_size=batch_size,
                show_progress=False,
            )
        _, correct = predictions_and_correctness(logits, split.targets)
        rows.append(correct)
    return torch.stack(rows)


def reference_index(correct: torch.Tensor) -> int:
    """Degrau de referencia = o de MAIOR acuracia, nao o mais caro.

    Escadas de precisao nao sao monotonas: em Gemma-3-12B, promover 36 das 48
    camadas de 3b para 4b da 62.30%, enquanto promover as 48 da 59.60% - erros
    de quantizacao se cancelam parcialmente, e consertar certas camadas quebra
    esse equilibrio. Ancorar o oraculo no degrau mais caro o obrigaria a
    preservar uma acuracia pior do que a escada ja alcanca.
    """
    return int(correct.float().mean(dim=1).argmax().item())


def compute_oracle(
    correct: torch.Tensor, ladder: Sequence, *, reference: Optional[int] = None
) -> OracleResult:
    """Aplica as duas politicas de oraculo sobre a matriz de acerto."""
    n = correct.shape[1]
    dense_idx = reference_index(correct) if reference is None else reference
    dense_correct = correct[dense_idx]
    costs = torch.tensor([r.weight_bytes for r in ladder], dtype=torch.float64)

    # primeiro degrau (mais barato) que acerta; len(ladder) se nenhum acerta
    any_correct = correct.any(dim=0)
    cheapest = torch.where(
        any_correct, correct.float().argmax(dim=0), torch.tensor(dense_idx)
    )

    # consistente: so economiza onde o denso ja acertava
    consistent_idx = torch.where(dense_correct, cheapest, torch.tensor(dense_idx))
    consistent_bytes = costs[consistent_idx].mean().item()

    # irrestrito: melhor degrau qualquer
    unconstrained_bytes = costs[cheapest].mean().item()

    return OracleResult(
        n=n,
        dense_accuracy=dense_correct.float().mean().item(),
        dense_bytes=ladder[dense_idx].weight_bytes,
        consistent_accuracy=dense_correct.float().mean().item(),  # igual por construcao
        consistent_bytes=consistent_bytes,
        unconstrained_accuracy=any_correct.float().mean().item(),
        unconstrained_bytes=unconstrained_bytes,
        rung_accuracy=[correct[i].float().mean().item() for i in range(len(ladder))],
        rung_bytes=[r.weight_bytes for r in ladder],
        correct=correct,
    )


def permutation_null_test(
    correct: torch.Tensor,
    ladder: Sequence,
    *,
    n_reps: int = 2000,
    seed: int = 0,
    reference: Optional[int] = None,
) -> dict[str, float]:
    """Quanto da economia do oraculo e heterogeneidade real, e quanto e sorte?

    Um oraculo que escolhe entre K degraus sempre economiza ALGO, mesmo sem
    nenhum sinal: um degrau no nivel do chute acerta 25% das vezes e as vezes
    e o mais barato disponivel. O nulo precisa medir exatamente essa parcela.

    Construcao: **permuta as colunas** do bloco de degraus baratos, mantendo
    a linha densa fixa. Isso preserva a acuracia marginal de cada degrau E a
    correlacao entre degraus (degraus colapsados sao quase o mesmo modelo
    degenerado, e ignorar isso infla o nulo), destruindo apenas a associacao
    entre o resultado do degrau e QUAL exemplo e. Sortear cada degrau de forma
    independente seria errado: quebra tambem a correlacao entre degraus e
    superestima P(algum degrau acerta).
    """
    generator = torch.Generator().manual_seed(seed)
    dense_idx = reference_index(correct) if reference is None else reference
    n = correct.shape[1]
    costs = torch.tensor([r.weight_bytes for r in ladder], dtype=torch.float64)
    dense = correct[dense_idx]

    def consistent_bytes(matrix: torch.Tensor) -> float:
        cheapest = torch.where(
            matrix.any(dim=0), matrix.float().argmax(dim=0), torch.tensor(dense_idx)
        )
        return costs[torch.where(dense, cheapest, torch.tensor(dense_idx))].mean().item()

    observed = consistent_bytes(correct)
    samples = torch.empty(n_reps, dtype=torch.float64)
    others = [i for i in range(correct.shape[0]) if i != dense_idx]
    for i in range(n_reps):
        perm = torch.randperm(n, generator=generator)
        shuffled = correct.clone()
        shuffled[others] = correct[others][:, perm]
        samples[i] = consistent_bytes(shuffled)

    full = costs[dense_idx].item()
    std = samples.std().item()
    return {
        "observed_saving": 1 - observed / full,
        "null_saving_mean": 1 - samples.mean().item() / full,
        "null_saving_std": std / full,
        "z": (samples.mean().item() - observed) / std if std > 0 else 0.0,
        "p_empirical": (samples <= observed).float().mean().item(),
        "attributable_saving": (samples.mean().item() - observed) / full,
    }


def conditional_dependence(
    correct: torch.Tensor, ladder: Sequence
) -> list[tuple[str, float, float]]:
    """(n_skip, P(degrau acerta), P(degrau acerta | denso acerta)) por degrau.

    Razao > 1 e a hipotese do EWS em forma crua: os exemplos que o degrau
    barato resolve sao os mesmos que o modelo completo resolve, ou seja,
    existe um subconjunto "facil" identificavel.
    """
    dense = correct[reference_index(correct)]
    return [
        (getattr(r, "label", str(getattr(r, "n_skipped", i))),
         correct[i].float().mean().item(),
         correct[i][dense].float().mean().item())
        for i, r in enumerate(ladder)
    ]


def load_importance(path: str | Path) -> LayerImportance:
    """Reaproveita um perfil leave-one-out ja calculado (custa ~13 min no 12B)."""
    payload = json.loads(Path(path).read_text())["layer_importance"]
    return LayerImportance(
        delta_nll=payload["delta_nll"], dense_nll=payload["dense_nll"]
    )


def dump_raw(
    path: str | Path,
    *,
    importance: LayerImportance,
    ladder: Sequence[Rung],
    result: OracleResult,
    baselines: dict[str, float],
    meta: dict,
    null_test: Optional[dict] = None,
) -> Path:
    """Grava o resultado cru em JSON - insumo do passo 5 e do paper."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "baselines": baselines,
        "layer_importance": {
            "dense_nll": importance.dense_nll,
            "delta_nll": importance.delta_nll,
            "order_ascending": importance.order_ascending,
        },
        "ladder": [
            {"index": r.index, "n_skipped": r.n_skipped,
             "skipped": r.skipped, "weight_bytes": r.weight_bytes}
            for r in ladder
        ],
        "oracle": {
            "n": result.n,
            "dense_accuracy": result.dense_accuracy,
            "dense_bytes": result.dense_bytes,
            "consistent_accuracy": result.consistent_accuracy,
            "consistent_bytes": result.consistent_bytes,
            "unconstrained_accuracy": result.unconstrained_accuracy,
            "unconstrained_bytes": result.unconstrained_bytes,
            "rung_accuracy": result.rung_accuracy,
            "rung_bytes": result.rung_bytes,
        },
        "null_test": null_test,
        "correct_matrix": result.correct.to(torch.uint8).tolist(),
    }
    out.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Resultado cru gravado em %s", out)
    return out
