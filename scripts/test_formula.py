#!/usr/bin/env python3
"""GLOD: testar a lei flips = kappa*sqrt(KL) contra as medicoes reais do paper.

Nao ha nada simulado aqui. O script le `data/measurements.csv`, que traz as 742
configuracoes medidas (modelo, corpus, familia, configuracao, KL, flips) usadas no
artigo, e confronta a previsao com o que foi de fato observado. Nao precisa de GPU.

    # a lei numa referencia: previsto x medido, configuracao por configuracao
    python3 scripts/test_formula.py --model gemma-3-4b-it --corpus gsm8k

    # quanto a familia do metodo ainda importa, a KL casado (a pergunta central)
    python3 scripts/test_formula.py --families

    # o que existe para escolher
    python3 scripts/test_formula.py --list

    # so a previsao, para um kappa e um KL que voce fornece
    python3 scripts/test_formula.py --kappa 0.35 --kl 0.10
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "data" / "measurements.csv"
# janela em que kappa e definido no artigo (Eq. 3): o regime de compressao implantada
KAPPA_LO, KAPPA_HI = 1e-3, 0.05
RULE = "=" * 72


def load() -> list[dict]:
    if not CSV.exists():
        raise SystemExit(f"nao encontrei {CSV}\n"
                         "Regenere com: python3 docs/paper/scripts/make_figures.py")
    with CSV.open() as f:
        return [{**r, "kl": float(r["kl"]), "flip": float(r["flip"])}
                for r in csv.DictReader(f)]


def kappa_of(rows: list[dict]) -> float | None:
    """Eq. (3) do artigo: mediana de flips/sqrt(KL) na janela de baixa divergencia."""
    ks = [r["flip"] / math.sqrt(r["kl"]) for r in rows if KAPPA_LO < r["kl"] < KAPPA_HI]
    return st.median(ks) if ks else None


def fit_loglog(rows: list[dict]) -> tuple[float, float]:
    """Inclinacao e R^2 do ajuste log-log, sem numpy."""
    xs = [math.log(r["kl"]) for r in rows]
    ys = [math.log(r["flip"]) for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return slope, 1 - ss_res / ss_tot


def cmd_list(rows) -> None:
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["corpus"])].append(r)
    print(f"\n{len(by)} referencias medidas (modelo x corpus):\n")
    print(f"{'modelo':<34} {'corpus':<14} {'configs':>7} {'kappa':>7}")
    print("-" * 66)
    for (m, c), rs in sorted(by.items()):
        k = kappa_of(rs)
        print(f"{m:<34} {c:<14} {len(rs):>7} {k if k is None else round(k, 3):>7}")
    print("\nUse --model <modelo> --corpus <corpus> para testar a lei numa delas.\n")


def cmd_reference(rows, model: str, corpus: str) -> int:
    sel = [r for r in rows if r["model"] == model and r["corpus"] == corpus]
    if not sel:
        print(f"\nsem medicoes para ({model}, {corpus}). Rode --list para ver o que existe.\n")
        return 1
    kappa = kappa_of(sel)
    slope, r2 = fit_loglog(sel)
    print(f"\n{RULE}\n  GLOD | {model} | corpus {corpus}\n{RULE}")
    print(f"\nkappa medido (Eq. 3, janela {KAPPA_LO:g} < KL < {KAPPA_HI:g}) : {kappa:.4f}")
    print(f"expoente ajustado em log-log                        : {slope:.4f}  (R2 {r2:.4f})")
    print("\nA previsao abaixo usa APENAS kappa e o KL medido de cada configuracao.")
    print("A coluna 'erro' e (previsto - medido) / medido.\n")
    print(f"{'config':<12} {'familia':<12} {'KL':>10} {'flips medido':>13} "
          f"{'previsto':>10} {'erro':>8}")
    print("-" * 70)
    errs = []
    for r in sorted(sel, key=lambda r: r["kl"]):
        pred = kappa * math.sqrt(r["kl"])
        err = (pred - r["flip"]) / r["flip"]
        errs.append(abs(err))
        print(f"{r['config']:<12} {r['family']:<12} {r['kl']:>10.5f} {r['flip']:>13.5f} "
              f"{pred:>10.5f} {err:>+7.1%}")
    inwin = [abs((kappa * math.sqrt(r["kl"]) - r["flip"]) / r["flip"])
             for r in sel if KAPPA_LO < r["kl"] < KAPPA_HI]
    print("-" * 70)
    print(f"erro absoluto mediano, todas as {len(errs)} configuracoes : {st.median(errs):.1%}")
    if inwin:
        print(f"erro absoluto mediano dentro da janela de kappa   : {st.median(inwin):.1%}")
    print("\nO erro cresce nas pontas porque um unico kappa nao absorve o desvio do")
    print("expoente em relacao a 1/2; dentro da janela onde kappa e definido ele e pequeno.\n")
    return 0


def cmd_families(rows) -> int:
    """A pergunta central: a KL casado, o metodo ainda importa?

    Para cada referencia ajustamos a lei de potencia DELA e medimos o quanto cada
    familia se desvia. Agregamos uma observacao por referencia (a unidade
    independente), como no artigo.
    """
    per_fam = defaultdict(list)
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["corpus"])].append(r)
    for rs in by.values():
        if len(rs) < 8:
            continue
        slope, _ = fit_loglog(rs)
        inter = (sum(math.log(r["flip"]) for r in rs) / len(rs)
                 - slope * sum(math.log(r["kl"]) for r in rs) / len(rs))
        d = defaultdict(list)
        for r in rs:
            d[r["family"]].append(math.log10(r["flip"]) - (inter + slope * math.log(r["kl"])) / math.log(10))
        for f, v in d.items():
            per_fam[f].append(st.mean(v))
    print(f"\n{RULE}\n  A KL casado, quanto a familia do metodo ainda importa?\n{RULE}\n")
    print("Desvio multiplicativo sobre a taxa de flips, em relacao a lei da propria")
    print("referencia. Uma observacao por referencia; IC 95% de Student entre referencias.\n")
    print(f"{'familia':<14} {'refs':>5} {'flips x':>9} {'IC 95%':>18} {'signif.':>8}")
    print("-" * 60)
    worst = 0.0
    nsig = 0
    for f, v in sorted(per_fam.items(), key=lambda kv: st.mean(kv[1])):
        n = len(v)
        if n < 3:
            continue
        m = st.mean(v)
        se = st.stdev(v) / math.sqrt(n)
        # t de Student 97.5% aproximado, suficiente para o proposito do demo
        t = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(n, 2.0 + 6.0 / n)
        lo, hi = 10 ** (m - t * se), 10 ** (m + t * se)
        sig = lo > 1 or hi < 1
        nsig += sig
        worst = max(worst, abs(10 ** m - 1))
        print(f"{f:<14} {n:>5} {10 ** m:>9.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} "
              f"{'sim' if sig else '':>8}")
    print("-" * 60)
    print(f"\nDesvio maximo: {worst:.1%}. {nsig} familias tem intervalo excluindo 1.")
    print("Ou seja: fungibilidade a menos de ~5%, NAO igualdade. Remover camadas")
    print("inteiras produz menos flips por unidade de divergencia; poda calibrada, mais.\n")
    return 0


def cmd_predict(kappa: float, kl: float) -> int:
    print(f"\n{RULE}\n  GLOD | previsao isolada (sem dados medidos)\n{RULE}\n")
    print(f"kappa fornecido : {kappa:.4f}")
    print(f"KL fornecido    : {kl:.5f} nat")
    print(f"\nflips = {kappa:.4f} * sqrt({kl:.5f}) = {kappa * math.sqrt(kl):.5f} "
          f"({100 * kappa * math.sqrt(kl):.2f}%)\n")
    if not (KAPPA_LO < kl < KAPPA_HI):
        print(f"AVISO: KL={kl:g} esta fora da janela {KAPPA_LO:g}-{KAPPA_HI:g} em que kappa")
        print("e definido. A lei foi medida ate ~2 nats; acima disso o regime quadratico")
        print("que a sustenta quebra (App. B do artigo).\n")
    print("kappa NAO e constante do modelo: depende do par (modelo, corpus) e varia")
    print("1,6-2,6x entre corpora do mesmo modelo. Use --list para ver os medidos.\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="modelo medido (ver --list)")
    p.add_argument("--corpus", help="corpus medido (ver --list)")
    p.add_argument("--list", action="store_true", help="lista as referencias disponiveis")
    p.add_argument("--families", action="store_true",
                   help="desvio por familia de compressor a KL casado")
    p.add_argument("--kappa", type=float, help="previsao isolada: kappa")
    p.add_argument("--kl", type=float, help="previsao isolada: KL em nats")
    a = p.parse_args()

    if a.kappa is not None and a.kl is not None:
        return cmd_predict(a.kappa, a.kl)
    rows = load()
    if a.list:
        cmd_list(rows)
        return 0
    if a.families:
        return cmd_families(rows)
    if a.model and a.corpus:
        return cmd_reference(rows, a.model, a.corpus)
    # sem argumentos: a demonstracao mais direta da lei
    print("\n(sem argumentos: mostrando uma referencia de exemplo; veja --help)")
    return cmd_reference(rows, "gemma-3-4b-it", "gsm8k")


if __name__ == "__main__":
    raise SystemExit(main())
