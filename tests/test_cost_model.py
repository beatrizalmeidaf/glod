"""Validacoes do modelo de custo fechado contra simulacao.

    python tests/test_cost_model.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ews.core.cost_model import (
    dprime_from_auroc,
    gate_cost,
    promote_rate_at_divergence,
    required_auroc,
    speculative_cost,
)
from ews.corpora.token_oracle import (
    auroc,
    cost_at_divergence,
    gate_curve,
    speculative_expected_cost,
)

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FALHA'}] {label}{(' | ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


print("\n=== binormal: d' <-> AUROC ===")
torch.manual_seed(0)
for target in (0.6, 0.75, 0.9):
    dp = dprime_from_auroc(target)
    y = torch.rand(200000) < 0.15
    s = torch.randn(200000) + dp * y.float()
    check(abs(auroc(s, y) - target) < 0.005, f"AUROC simulada com d'({target}) ~ {target}",
          f"{auroc(s, y):.4f}")

print("\n=== custo do gate: formula fechada vs curva simulada ===")
for A in (0.7, 0.9):
    pi, n = 0.14, 400000
    y = torch.rand(n) < pi
    s = torch.randn(n) + dprime_from_auroc(A) * y.float()
    for mode in ("predictive", "cascade"):
        curve = gate_curve(s, y, base_cost=6.73, ref_cost=22.70, mode=mode, n_points=2000)
        for eps in (0.01, 0.05):
            sim = cost_at_divergence(curve, eps)
            closed = gate_cost(pi, A, eps, d=6.73, v=22.70, mode=mode)
            check(abs(sim - closed) / closed < 0.02, f"A={A} {mode} eps={eps}",
                  f"simulado {sim:.3f} vs fechado {closed:.3f}")
check(promote_rate_at_divergence(0.1, 0.8, 0.2) == 0.0, "eps >= pi: nao precisa promover nada")
check(abs(promote_rate_at_divergence(0.1, 1 - 1e-9, 1e-4) - 0.1) < 1e-3,
      "gate quase perfeito com eps pequeno promove ~pi",
      f"{promote_rate_at_divergence(0.1, 1 - 1e-9, 1e-4):.5f}")
check(promote_rate_at_divergence(0.1, 0.9, 0.0) == 1.0,
      "eps=0 sob caudas gaussianas exige promover tudo")

print("\n=== especulativa: formula iid vs simulacao do laco ===")
for alpha in (0.6, 0.86, 0.95):
    flags = torch.rand(300000) < alpha
    seqs = torch.zeros(300000, dtype=torch.long)
    for k in (2, 4, 6):
        sim = speculative_expected_cost(flags, seqs, draft_cost=3.62, verify_cost=22.70, k=k)
        per_round = (1 - alpha ** (k + 1)) / (1 - alpha)
        closed = (k * 3.62 + 22.70) / per_round
        check(abs(sim["cost_per_token"] - closed) / closed < 0.01, f"alpha={alpha} k={k}",
              f"simulado {sim['cost_per_token']:.3f} vs fechado {closed:.3f}")
best, k = speculative_cost(0.0, d=1.0, v=10.0)
check(k == 0 and best == 10.0, "draft inutil (alpha=0): especulativa degenera na referencia")
best1, _ = speculative_cost(1.0, d=1.0, v=10.0, k_max=4)
check(abs(best1 - (4 + 10) / 5) < 1e-9, "alpha=1: (k d + v)/(k+1) no k maximo")

print("\n=== AUROC requerida ===")
target = speculative_cost(0.86, d=6.73, v=22.70)[0]
req = required_auroc(0.14, 0.01, d=6.73, v=22.70, target_cost=target)
check(0.5 < req < 1.0, "AUROC requerida em (0.5, 1)", f"{req:.4f}")
check(gate_cost(0.14, req, 0.01, d=6.73, v=22.70, mode="predictive") <= target + 1e-6,
      "na AUROC requerida o gate iguala a especulativa")
check(gate_cost(0.14, req - 0.01, 0.01, d=6.73, v=22.70, mode="predictive") > target,
      "abaixo dela o gate perde")
impossible = required_auroc(0.5, 0.0, d=9.0, v=10.0, target_cost=5.0)
check(impossible != impossible, "alvo inalcancavel ate pelo oraculo -> nan")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FALHA(S): {FAILURES}")
    sys.exit(1)
print("todos os testes passaram")
