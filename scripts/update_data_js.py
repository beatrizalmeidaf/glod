#!/usr/bin/env python3
"""Regenera web/data.js a partir de data/measurements.csv (as 802 medicoes do paper).

    python3 scripts/update_data_js.py

Nada e copiado de uma versao anterior do data.js: o kappa de cada referencia e
recalculado aqui com a mesma definicao do paper (Eq. 4: mediana de flips/sqrt(KL)
nas configuracoes com 1e-3 < KL < 0.05), e cada medicao leva KL, flips e TV, para
que o simulador compare as duas previsoes do paper (kappa*sqrt(KL) e TV) com o
valor medido.
"""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "measurements.csv"
JS = ROOT / "web" / "data.js"
KAPPA_LO, KAPPA_HI = 1e-3, 0.05   # janela da Eq. 4


def main() -> None:
    measurements: dict = {}
    with CSV.open() as f:
        for row in csv.DictReader(f):
            measurements.setdefault(row["corpus"], {}).setdefault(row["model"], []).append({
                "family": row["family"], "config": row["config"],
                "kl": float(row["kl"]), "flip": float(row["flip"]), "tv": float(row["tv"])})
    kappas = {}
    for corpus, models in measurements.items():
        for model, rows in models.items():
            w = [r["flip"] / math.sqrt(r["kl"]) for r in rows if KAPPA_LO < r["kl"] < KAPPA_HI]
            if w:
                kappas[f"{corpus}|{model}"] = statistics.median(w)
    data = {"kappas": kappas, "measurements": measurements}
    JS.write_text("const GLOD_DATA = " + json.dumps(data, indent=2) + ";\n")
    n = sum(len(v) for c in measurements.values() for v in c.values())
    print(f"{JS}: {n} configuracoes, {len(kappas)} referencias com kappa")


if __name__ == "__main__":
    main()
