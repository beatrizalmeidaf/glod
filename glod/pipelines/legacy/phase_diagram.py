#!/usr/bin/env python3
"""GLOD | Fase 1c - valida o modelo de custo fechado e calcula a AUROC exigida.

1. Validacao: para cada (modelo, base, sinal) compara o custo MEDIDO a uma
   divergencia fixa com o PREVISTO pelo modelo binormal a partir de apenas
   (prevalencia, AUROC, d, v). Idem para a especulativa (aceitacao iid).
2. Diagrama de fases: qual AUROC ex-ante um gate precisaria para empatar com a
   decodificacao especulativa sem perda, dado o que foi medido em cada modelo.

    python -m glod legacy-phase-diagram --models gemma-3-12b-it Qwen3-14B gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch

from glod.core.cost_model import gate_cost, required_auroc, speculative_cost
from glod.corpora.token_oracle import speculative_expected_cost

LOGGER = logging.getLogger("glod.phase")
GIB = 1024**3
from glod.paths import RAW
TOK = RAW / "tokens"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemma-3-12b-it", "Qwen3-14B"])
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    summary: dict = {}

    for model in args.models:
        path = TOK / f"{model}__analysis.json"
        if not path.exists():
            LOGGER.info("[%s] sem analise, pulando", model)
            continue
        rep = json.loads(path.read_text())
        own = TOK / f"{model}__corpus" / model
        ref = torch.load(own / "bf16.pt")
        corpus = json.loads((TOK / f"{model}__corpus.json").read_text())
        seq = torch.cat([torch.full((len(g),), i, dtype=torch.int32)
                         for i, g in enumerate(corpus["gen_ids"])])
        v = ref["weight_bytes"] / GIB
        LOGGER.info("=" * 96)
        LOGGER.info("%s | referencia %.2f GiB/tok", model, v)
        summary[model] = {}

        for bname, b in rep["bases"].items():
            d = b["cost_gib"]
            pi = 1 - b["p_sufficient"]
            flags = torch.load(own / f"{bname}.pt")["top1"] == ref["top1"]
            sd_meas = min(speculative_expected_cost(flags, seq, draft_cost=d, verify_cost=v, k=k)
                          ["cost_per_token"] for k in range(1, 9))
            sd_pred, k_pred = speculative_cost(1 - pi, d=d, v=v, k_max=8)
            LOGGER.info("-" * 96)
            LOGGER.info("base %s: d=%.2f GiB | pi=%.3f | especulativa medida %.2f vs prevista(iid) %.2f "
                        "(erro %+.1f%%)", bname, d, pi, sd_meas, sd_pred, 100 * (sd_pred / sd_meas - 1))
            LOGGER.info("  %-38s | %5s | %s", "sinal", "AUROC",
                        " | ".join(f"eps={e:.0%} med/prev" for e in args.eps))
            rows = {}
            for sname, a in b["anticipation"].items():
                if sname.startswith("oraculo") or sname.startswith("early-probe"):
                    continue  # early-probe tem divergencia do prefixo, fora do modelo binormal
                cells, errs = [], []
                for e in args.eps:
                    meas = a.get(f"gib@{e}")
                    pred = gate_cost(pi, a["auroc"], e, d=d, v=v, mode=a["mode"])
                    cells.append(f"{meas:6.2f}/{pred:6.2f}")
                    if meas == meas:
                        errs.append(pred / meas - 1)
                rows[sname] = {"auroc": a["auroc"], "mode": a["mode"], "rel_err": errs}
                LOGGER.info("  %-38s | %.3f | %s", sname, a["auroc"], " | ".join(cells))

            LOGGER.info("  AUROC EX-ANTE (preditivo) necessaria para empatar com a especulativa medida "
                        "(%.2f GiB/tok):", sd_meas)
            req = {}
            for e in args.eps:
                r = required_auroc(pi, e, d=d, v=v, target_cost=sd_meas, mode="predictive")
                req[e] = r
                best_exante = max((x["auroc"] for n, x in rows.items() if x["mode"] == "predictive"),
                                  default=float("nan"))
                LOGGER.info("    eps=%4.1f%% -> AUROC >= %s   (melhor ex-ante medido: %.3f)",
                            100 * e, "inalcancavel" if r != r else f"{r:.3f}", best_exante)
            summary[model][bname] = {"pi": pi, "d": d, "v": v, "sd_measured": sd_meas,
                                     "sd_predicted": sd_pred, "signals": rows,
                                     "required_auroc": req}

    out = TOK / "phase_diagram.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    LOGGER.info("gravado em %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
