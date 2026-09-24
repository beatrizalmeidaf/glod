"""O CSV publico e o script de demonstracao precisam concordar com o paper.

`data/measurements.csv` e o unico artefato pelo qual alguem de fora consegue
conferir a lei sem GPU. Se ele divergir de `docs/paper/numbers.json`, o repo
passa a contradizer o artigo -- e foi exatamente isso que aconteceu antes, quando
`scripts/test_formula.py` gerava numeros sinteticos e os rotulava de empiricos.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "measurements.csv"
NUMBERS = ROOT / "docs" / "paper" / "numbers.json"


def rows():
    with CSV.open() as f:
        return [{**r, "kl": float(r["kl"]), "flip": float(r["flip"])}
                for r in csv.DictReader(f)]


def test_csv_matches_paper_scale():
    N = json.loads(NUMBERS.read_text())
    rs = rows()
    assert len(rs) == N["configs_total"]
    assert len({r["model"] for r in rs}) == N["models_total"]
    assert len({(r["model"], r["corpus"]) for r in rs}) == N["coverage"]["n_refs"]
    assert sorted({r["family"] for r in rs}) == N["families"]


def test_kappa_from_csv_matches_paper():
    """kappa recomputado do CSV bate com o range de cada corpus em numbers.json."""
    N = json.loads(NUMBERS.read_text())
    by = defaultdict(list)
    for r in rows():
        by[(r["model"], r["corpus"])].append(r)
    per_corpus = defaultdict(list)
    for (_, c), rs in by.items():
        ks = [r["flip"] / math.sqrt(r["kl"]) for r in rs if 1e-3 < r["kl"] < 0.05]
        if ks:
            per_corpus[c].append(st.median(ks))
    for c, ks in per_corpus.items():
        exp = N["by_corpus"][c]
        assert abs(min(ks) - exp["kappa_min"]) < 1e-6, c
        assert abs(max(ks) - exp["kappa_max"]) < 1e-6, c


def test_family_offsets_reproduce_table5():
    """A tabela de fungibilidade do artigo tem de sair do CSV publico sozinha."""
    N = json.loads(NUMBERS.read_text())
    by = defaultdict(list)
    for r in rows():
        by[(r["model"], r["corpus"])].append(r)
    per_fam = defaultdict(list)
    for rs in by.values():
        if len(rs) < 8:
            continue
        xs = [math.log(r["kl"]) for r in rs]
        ys = [math.log(r["flip"]) for r in rs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                 / sum((x - mx) ** 2 for x in xs))
        inter = my - slope * mx
        d = defaultdict(list)
        for r, x, y in zip(rs, xs, ys):
            d[r["family"]].append((y - (inter + slope * x)) / math.log(10))
        for f, v in d.items():
            per_fam[f].append(st.mean(v))
    for f, v in per_fam.items():
        if len(v) < 3:
            continue
        assert abs(10 ** st.mean(v) - N["family_offsets"][f]["factor"]) < 1e-6, f


def test_no_synthetic_data_in_demo():
    """O script de demonstracao nao pode voltar a inventar dados."""
    src = (ROOT / "scripts" / "test_formula.py").read_text()
    assert "random" not in src, "test_formula.py nao deve gerar numeros aleatorios"
    assert "simulate" not in src.lower(), "test_formula.py nao deve simular medicoes"


if __name__ == "__main__":
    # `make test` executa cada arquivo como script, nao via pytest: sem isto o teste
    # "passaria" sem rodar nada.
    import sys
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"  ok   {n}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL {n}: {e}")
    print(f"{len(fns) - bad}/{len(fns)} checks passaram")
    sys.exit(1 if bad else 0)
