"""Modelo de custo fechado: quando um gate ex-ante vence a verificacao ex-post?

Grandezas de entrada (todas medidas na Fase 1c):
  pi     - prevalencia de tokens em que a base diverge da referencia
  A      - AUROC do sinal do gate para prever essa divergencia
  d, v   - pesos lidos por token pela base (draft) e pela referencia
  alpha  - taxa de aceitacao do draft (= 1 - pi quando draft = base)

Gate (ROC binormal de variancia igual, d' = sqrt(2) * Phi^-1(A)):
  para um orcamento de divergencia eps, o gate precisa de TPR = 1 - eps/pi;
  negativos ~ N(0,1), positivos ~ N(d',1), promove se s > t:
    TPR = 1 - Phi(t - d')  =>  t = d' + Phi^-1(1 - TPR)
    FPR = 1 - Phi(t)       =  Phi(Phi^-1(TPR) - d')
  e  p_promove = pi*TPR + (1-pi)*FPR.  (Com caudas gaussianas TPR = 1 exige
  FPR = 1 para qualquer d' finito: eps = 0 so e atingivel promovendo tudo.)
  custo preditivo = (1-p) d + p v ;  custo cascata = d + p v.

Especulativa greedy (aceitacoes i.i.d.):
  E[tokens por rodada] = (1 - alpha^(k+1)) / (1 - alpha)
  custo/token = (k d + v) / E[tokens por rodada], minimizado em k.

O modelo e deliberadamente simples: o objetivo e prever a ORDEM entre os
paradigmas e a AUROC minima que um gate precisaria, e validar essa previsao
contra as curvas medidas.
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist

_N = NormalDist()


def dprime_from_auroc(auroc: float) -> float:
    auroc = min(max(auroc, 1e-9), 1 - 1e-9)
    return sqrt(2.0) * _N.inv_cdf(auroc)


def promote_rate_at_divergence(pi: float, auroc: float, eps: float) -> float:
    """Fracao de tokens promovidos para deixar no maximo `eps` de divergencia."""
    if eps >= pi:
        return 0.0
    tpr = 1.0 - eps / pi
    if tpr >= 1.0:
        return 1.0
    fpr = _N.cdf(_N.inv_cdf(tpr) - dprime_from_auroc(auroc))
    return pi * tpr + (1.0 - pi) * fpr


def gate_cost(pi: float, auroc: float, eps: float, *, d: float, v: float, mode: str) -> float:
    p = promote_rate_at_divergence(pi, auroc, eps)
    if mode == "predictive":
        return (1.0 - p) * d + p * v
    if mode == "cascade":
        return d + p * v
    raise ValueError(mode)


def speculative_cost(alpha: float, *, d: float, v: float, k_max: int = 16) -> tuple[float, int]:
    """Menor custo/token da especulativa greedy e o k que o atinge."""
    best = (v, 0)  # k = 0 equivale a rodar so a referencia
    for k in range(1, k_max + 1):
        if alpha >= 1.0:
            per_round = k + 1
        else:
            per_round = (1.0 - alpha ** (k + 1)) / (1.0 - alpha)
        cost = (k * d + v) / per_round
        if cost < best[0]:
            best = (cost, k)
    return best


def required_auroc(pi: float, eps: float, *, d: float, v: float, target_cost: float,
                   mode: str = "predictive", tol: float = 1e-4) -> float:
    """Menor AUROC para o gate igualar `target_cost` (nan se nem o oraculo iguala)."""
    if gate_cost(pi, 1 - 1e-9, eps, d=d, v=v, mode=mode) > target_cost:
        return float("nan")
    if gate_cost(pi, 0.5, eps, d=d, v=v, mode=mode) <= target_cost:
        return 0.5
    lo, hi = 0.5, 1 - 1e-9
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if gate_cost(pi, mid, eps, d=d, v=v, mode=mode) <= target_cost:
            hi = mid
        else:
            lo = mid
    return hi
