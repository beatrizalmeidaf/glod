#!/usr/bin/env python3
"""Figura de speedup medido no vLLM para o site, no estilo das figuras do paper.

Le os relatorios `glod draft-select report` (os mesmos que geram a Tabela tab:specreal)
e desenha:
  (a) o speedup medido de cada draft nos quatro cenarios (2 alvos x GSM8K/codigo),
      marcando o draft que TV e KL escolhem pelo modelo de custo em bytes;
  (b) a aceitacao gulosa prevista (1 - TV e kappa*sqrt(KL)) contra a medida.

    /usr/bin/python3 scripts/site_speedup_figure.py [--out web/assets/paper/fig_speedup.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ANALYSIS = Path("/local/user_beatrizalmeida/ews_results/fid_ds/analysis")
SCENARIOS = [("Qwen3-14B", "", "Qwen3-14B\nGSM8K"), ("Qwen3-14B", "__code", "Qwen3-14B\ncode"),
             ("Qwen3-8B", "", "Qwen3-8B\nGSM8K"), ("Qwen3-8B", "__code", "Qwen3-8B\ncode")]
ORDER = ["q0.6b", "q1.7b", "q1.7b-gptq8", "q4b", "q4b-awq", "q8b", "q8b-awq"]
LABEL = {"q0.6b": "0.6B", "q1.7b": "1.7B", "q1.7b-gptq8": "1.7B-G8", "q4b": "4B",
         "q4b-awq": "4B-AWQ", "q8b": "8B", "q8b-awq": "8B-AWQ"}
SIZE_COLOR = {"0.6": "#9ecae1", "1.7": "#4292c6", "4": "#2171b5", "8": "#08306b"}
TV_C, KL_C = "#1f77b4", "#d62728"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True,
    "grid.alpha": 0.25, "figure.dpi": 200, "savefig.bbox": "tight",
})


def size_of(draft: str) -> str:
    return draft[1:].split("b")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/assets/paper/fig_speedup.png")
    args = ap.parse_args()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [2.4, 1]})

    x, ticks, ticklabels, centers = 0.0, [], [], []
    for target, suffix, title in SCENARIOS:
        rep = json.loads((ANALYSIS / f"draft_select__{target}{suffix}.json").read_text())
        rows = {d["draft"]: d for d in rep["drafts"]}
        pick_tv = max(rows.values(), key=lambda d: d["pred_speedup_tv"])["draft"]
        pick_kl = max(rows.values(), key=lambda d: d["pred_speedup_kl"])["draft"]
        start = x
        for name in [n for n in ORDER if n in rows]:
            d = rows[name]
            quant = "gptq" in name or "awq" in name
            ax.bar(x, d["greedy_speedup"], width=0.8, color=SIZE_COLOR[size_of(name)],
                   hatch="////" if quant else None, edgecolor="white", linewidth=0.4)
            top = d["greedy_speedup"]
            if name == pick_tv:
                ax.plot(x, top + 0.12, marker="v", color=TV_C, ms=5, zorder=4)
            if name == pick_kl:
                ax.plot(x, top + (0.30 if name == pick_tv else 0.12), marker="v", mfc="white",
                        mec=KL_C, mew=1.1, ms=5, zorder=4)
            ticks.append(x)
            ticklabels.append(LABEL[name])
            x += 1
        centers.append(((start + x - 1) / 2, title))
        x += 1.2

    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=60, ha="right", fontsize=6)
    ax.set_xlim(-0.8, x - 1.4)
    ax.set_ylim(0, 3.25)
    ax.set_ylabel("measured speedup (×)")
    ax.grid(axis="x", visible=False)
    for cx, title in centers:
        ax.text(cx, 3.18, title, ha="center", va="top", fontsize=6.5, linespacing=1.1)
    ax.set_title("(a) vLLM, greedy, k = 4: speedup of each draft", fontsize=8)
    ax.legend(handles=[
        Patch(facecolor=SIZE_COLOR["0.6"], label="0.6B"), Patch(facecolor=SIZE_COLOR["1.7"], label="1.7B"),
        Patch(facecolor=SIZE_COLOR["4"], label="4B"), Patch(facecolor=SIZE_COLOR["8"], label="8B"),
        Patch(facecolor="#bbbbbb", hatch="////", edgecolor="white", label="released GPTQ-8 / AWQ (hatched)"),
        Line2D([], [], marker="v", ls="", color=TV_C, label="picked by TV"),
        Line2D([], [], marker="v", ls="", mfc="white", mec=KL_C, label="picked by KL"),
        Line2D([], [], ls="--", color="black", lw=0.8, label="no speculation"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.52), ncol=4, frameon=False)

    lo, hi = 1.0, 0.0
    for target, suffix, _ in SCENARIOS:
        rep = json.loads((ANALYSIS / f"draft_select__{target}{suffix}.json").read_text())
        code = bool(suffix)
        for d in rep["drafts"]:
            m = d["greedy_accept"]
            bx.plot(m, d["pred_accept_tv"], marker="o", ls="", ms=3.5, color=TV_C,
                    mfc="white" if code else TV_C)
            bx.plot(m, d["pred_accept_kl"], marker="D", ls="", ms=3.2, color=KL_C,
                    mfc="white" if code else KL_C)
            lo = min(lo, m, d["pred_accept_tv"], d["pred_accept_kl"])
            hi = max(hi, m, d["pred_accept_tv"], d["pred_accept_kl"])
    lo, hi = lo - 0.02, hi + 0.02
    bx.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
    bx.set_xlim(lo, hi)
    bx.set_ylim(lo, hi)
    bx.set_xlabel("measured acceptance")
    bx.set_ylabel("predicted acceptance")
    bx.set_title("(b) acceptance per proposed token", fontsize=8)
    bx.legend(handles=[
        Line2D([], [], marker="o", ls="", color=TV_C, label="1 − TV"),
        Line2D([], [], marker="D", ls="", color=KL_C, label="κ·√KL"),
        Line2D([], [], marker="o", ls="", color="gray", label="GSM8K"),
        Line2D([], [], marker="o", ls="", color="gray", mfc="white", label="code"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
