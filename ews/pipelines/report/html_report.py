#!/usr/bin/env python3
"""Monta o relatorio HTML da Fase 1c a partir dos resultados medidos.

Nenhum numero e digitado a mao: tudo sai de results/raw/tokens/*.json,
results/raw/closedloop/*.json e do modelo de custo.

    python ews_report_build.py --template <template.html> --data-tokens <data_tokens.json> --out <report.html>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ews.core.cost_model import gate_cost

from ews.paths import RAW
TOK = RAW / "tokens"
CL = RAW / "closedloop"
MODELS = ["gemma-3-12b-it", "Qwen3-14B", "gemma-3-27b-it"]


def model_validation(tokens: dict) -> dict:
    """Melhor sinal ex-ante por (modelo, base): custo medido vs previsto a 1%."""
    out = {}
    for key in MODELS:
        out[key] = {}
        for b in ("u3", "u4"):
            base = tokens["models"][key]["bases"][b]
            ante = {n: a for n, a in base["anticipation"].items() if n.startswith("ex-ante")}
            name, sig = max(ante.items(), key=lambda kv: kv[1]["auroc"])
            pred = gate_cost(1 - base["p_sufficient"], sig["auroc"], 0.01,
                             d=base["gib"], v=tokens["models"][key]["ref_gib"], mode="predictive")
            out[key][b] = {"signal": name, "meas": sig["gib@0.01"], "pred": pred}
    return out


def closed_loop_points(tokens: dict) -> dict:
    points, criteria = [], set()

    def acc_of(d: dict, src: str) -> tuple[float, list, str]:
        s = d[src]
        if "acc_flexible" in s:
            return s["acc_flexible"], s["ci_flexible"], "flexível"
        return s["acc"], s["ci"], "estrito"

    files = {p.stem: json.loads(p.read_text()) for p in sorted(CL.glob("*.json"))}
    for stem, d in files.items():
        acc, ci, crit = acc_of(d, "gsm8k")
        mmlu = acc_of(d, "mmlu_pt")[0]
        criteria.add(crit)
        model, cfg = stem.split("__")
        pt = {"gib": d["gib"], "acc": acc, "ci": ci, "mmlu": mmlu, "criterion": crit,
              "promote": d.get("promote_rate") if "cascade" in cfg else None}
        if model == "gemma-3-4b-it":
            bits = "bf16" if cfg == "bf16" else cfg.replace("u", "") + "-bit"
            pt.update(group="base", label=f"Gemma-3-4B {bits}", short=f"4B {bits}", direct=True)
        elif cfg.startswith("cascade_"):
            _, base, rate = cfg.split("_")
            pt.update(group="cascade", base=base,
                      label=f"12B gate cascata base {base.replace('u', '')}-bit, promove {100 * d['promote_rate']:.1f}%",
                      short=f"promove {100 * d['promote_rate']:.0f}%", direct=base == "u3")
        else:
            bits = "bf16" if cfg == "bf16" else cfg.replace("u", "") + "-bit"
            pt.update(group="static", label=f"Gemma-3-12B {bits}", short=f"12B {bits}", direct=True)
        points.append(pt)

    ref = files.get("gemma-3-12b-it__bf16")
    if ref is not None:
        acc, ci, crit = acc_of(ref, "gsm8k")
        mmlu = acc_of(ref, "mmlu_pt")[0]
        g12 = tokens["models"]["gemma-3-12b-it"]
        spec = [("draft 12B 4-bit do mesmo modelo", g12["bases"]["u4"]["sd_measured"], "spec. draft 4-bit"),
                ("draft Gemma-3-4B 4-bit", tokens["sd_cross_12b"]["gemma-3-4b-it:u4"]["gib"], "spec. draft 4B")]
        for label, gib, short in spec:
            points.append({"group": "spec", "gib": gib, "acc": acc, "ci": ci, "mmlu": mmlu,
                           "criterion": crit, "promote": None, "label": f"12B especulativa sem perda, {label}",
                           "short": short, "direct": True, "dy": -10, "anchor": "middle", "dx": 0})
    note = ""
    if len(criteria) > 1:
        note = ("Atenção: pontos com critérios de extração diferentes (estrito e flexível) enquanto as "
                "rodadas refeitas não terminam.")
    return {"points": points, "note": note}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--data-tokens", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    tokens = json.loads(Path(args.data_tokens).read_text())
    counts = [sum(len(g) for g in json.loads((TOK / f"{k}__corpus.json").read_text())["gen_ids"])
              for k in MODELS]
    data = {"tokens": tokens, "token_counts": counts,
            "model_validation": model_validation(tokens), "closed": closed_loop_points(tokens)}
    html = Path(args.template).read_text().replace("/*DATA*/null", json.dumps(data, default=float))
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1024:.0f} KB | {len(data['closed']['points'])} pontos de malha fechada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
