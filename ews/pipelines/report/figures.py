#!/usr/bin/env python3
"""P4: figuras do paper, a partir dos JSONs de analise.

  fig1  lei flips x KL (377 configs, 18 referencias) + ataques + teto do oraculo
  fig2  kappa x geometria do modelo denso (fracao de gaps < 1 nat), com dominios
  fig3  fracao do teto informacional x d_model
  fig4  especulativa: previsto (so KL) x medido
  fig5  adaptatividade por token em geracao real (resultado negativo)
  fig6  trajetoria do ataque multicamada: maligno x benigno x faixa honesta

    python ews_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ews.paths import OUT as ROOT
from ews.paths import corpus_of

#: corpora mostrados nas figuras (default: so o do paper). `--corpus all` mostra todos.
SHOWN: set[str] | None = {"mix"}


def shown(ref: str) -> bool:
    return "__fp32" not in ref and (SHOWN is None or corpus_of(ref) in SHOWN)

AN = ROOT / "analysis"
from ews.paths import FIGS as FIG  # noqa: E402  (saida via EWS_FIGS)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, 
    "font.family": "serif", 
    "axes.grid": True,
    "grid.alpha": 0.25, 
    "figure.dpi": 200, 
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False
})

FAM_COLORS = {"rtn": "#1f77b4", "gaussiano": "#aec7e8", "gptq": "#d62728", "awq": "#ff7f0e",
              "sparsegpt": "#2ca02c", "wanda": "#98df8a", "magnitude": "#8c564b",
              "kv": "#9467bd", "skip": "#7f7f7f", "oficial": "#e377c2"}


def save(fig, name: str) -> None:
    # bbox_inches='tight' garante que as legendas externas nao sejam cortadas
    fig.savefig(FIG / f"{name}.pdf", bbox_inches='tight')
    fig.savefig(FIG / f"{name}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  -> {FIG / name}.pdf")


def fig1() -> None:
    law = json.loads((AN / "law.json").read_text())
    rows = [r for r in law["rows"] if r["ref"] == r["model"] and r["family"] != "outro modelo"
            and shown(r["ref"])]
    
    # Aumentando a largura para acomodar a legenda enorme fora do grafico
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    
    seen = set()
    for r in rows:
        fam = r["family"]
        c = FAM_COLORS.get(fam, "#333333")
        ax.scatter(r["kl"], r["flip"], s=9, color=c, alpha=0.7, linewidths=0,
                   label=fam if fam not in seen else None)
        seen.add(fam)
        
    x = np.logspace(-3.4, 0.35, 60)
    for ref, f in law["fits"].items():
        if not shown(ref):
            continue
        ax.plot(x, np.exp(f["top"]["intercept"] + f["top"]["slope"] * np.log(x)),
                color="k", lw=0.4, alpha=0.2)
                
    # ajuste conjunto do corpus mostrado (o do paper por default); com varios, o do mix
    one = next(iter(SHOWN)) if SHOWN is not None and len(SHOWN) == 1 else "mix"
    p = law.get("pooled_by_corpus", {}).get(one, law["pooled"])
    ax.plot(x, np.exp(p["intercept"] + p["slope"] * np.log(x)), color="k", lw=1.6,
            label=f"ajuste conjunto: inclinação {p['slope']:.2f}")
            
    adv = json.loads((AN / "adversarial.json").read_text()) if (AN / "adversarial.json").exists() else {"models": {}}
    mk = {"single_layer": ("*", "1 camada (ataque)", 70),
          "multi_malign": ("P", "multicamada (ataque)", 45)}
    for key, (m, lab, sz) in mk.items():
        pts = [(x_[1], x_[2]) for ref, r in adv["models"].items() if shown(ref)
               for x_ in r.get(key, [])]
        if pts:
            ax.scatter(*zip(*pts), marker=m, s=sz, facecolor="none", edgecolor="crimson",
                       linewidths=1.0, label=lab, zorder=5)
                       
    tp = [(0.0036, 0.0028), (0.0317, 0.0028), (0.0018, 0.0056), (0.0098, 0.0081)]
    ax.scatter(*zip(*tp), marker="v", s=26, color="navy", label="temperatura (benigno)", zorder=5)
    
    fd = json.loads((AN / "flipdirs.json").read_text())
    oc = {}
    for v in fd.values():
        for k, val in v["oracle_ceiling"].items():
            oc.setdefault(float(k), []).append(val)
    ks = sorted(oc)
    ax.plot(ks, [np.median(oc[k]) for k in ks], "--", color="darkorange", lw=1.5,
            label="teto do oráculo (token)")
            
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("KL por token (nats)")
    ax.set_ylabel("fração de decisões trocadas")
    ax.set_ylim(5e-4, 0.8)
    
    # Colocando a legenda COMPLETAMENTE FORA do grafico a direita
    ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig1_lei")


def fig2() -> None:
    sl = json.loads((AN / "slope.json").read_text())
    dm = json.loads((AN / "domain.json").read_text())
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    
    pts = [r for r in sl if shown(r["ref"])]
    xs = [r["p_gap_lt1"] for r in pts]
    ys = [r["kappa"] for r in pts]
    # com mais de um corpus na figura, a cor diz qual e: o ponto da afirmacao e que
    # corpora diferentes caem na MESMA reta, entao eles precisam ser distinguiveis
    groups: dict[str, list] = {}
    for r in pts:
        groups.setdefault(corpus_of(r["ref"]), []).append(r)
    if len(groups) == 1:
        ax.scatter(xs, ys, s=16, color="#1f77b4", label=f"modelos densos (n={len(xs)})")
    else:
        cols = {"mix": "#1f77b4", "gsm8k": "#ff7f0e", "mmlu_en": "#9467bd", "wikitext": "#17becf"}
        for c, g in sorted(groups.items()):
            ax.scatter([r["p_gap_lt1"] for r in g], [r["kappa"] for r in g], s=18,
                       color=cols.get(c, "#333333"), label=f"corpus {c} (n={len(g)})")
    
    c = np.polyfit(xs, ys, 1)
    xx = np.linspace(min(xs) * 0.9, max(xs) * 1.05, 20)
    r = np.corrcoef(xs, ys)[0, 1]
    ax.plot(xx, np.polyval(c, xx), "k-", lw=1.2, label=f"ajuste (r = {r:.2f})")
    
    for dom, mk, col in (("gsm8k", "^", "#2ca02c"), ("mmlu_pt", "s", "#d62728")):
        p = [(q["p_gap_lt1"], q["kappa"]) for q in dm["points"]
             if q["dom"] == dom and shown(q["ref"])]
        ax.scatter(*zip(*p), s=14, marker=mk, color=col, alpha=0.85, label=f"por domínio: {dom}")
                   
    ax.set_xlabel("fração de tokens com gap top1−top2 < 1 nat")
    ax.set_ylabel("κ = flips / √KL")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig2_kappa_geometria")


def fig3() -> None:
    fd = json.loads((AN / "flipdirs.json").read_text())
    law = json.loads((AN / "law.json").read_text())
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    
    xs, ys, names = [], [], []
    for mid, v in fd.items():
        ms = mid.split("/")[-1]
        pts = [r for r in law["rows"] if r["ref"] == ms and r["model"] == ms
               and r["family"] != "outro modelo"]
        for b, ceil in v["oracle_ceiling"].items():
            near = sorted(pts, key=lambda r: abs(r["kl"] - float(b)))[:3]
            if not near or abs(near[0]["kl"] - float(b)) > 0.5 * float(b):
                continue
            xs.append(v["H"])
            ys.append(float(np.mean([r["flip"] for r in near])) / ceil)
            names.append(ms)
            
    ax.scatter(xs, ys, s=14, color="#8c564b")
    ax.axhline(np.mean(ys), color="k", lw=1.2, ls="--",
               label=f"média observada: {np.mean(ys):.2f} (n={len(ys)})")
               
    ax.set_ylim(0, max(0.35, max(ys) * 1.15))
    ax.set_xlabel("dimensão do modelo (d_model)")
    ax.set_ylabel("flips observados / teto do oráculo")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig3_teto")


def fig4() -> None:
    sp = json.loads((AN / "spec_law.json").read_text())
    rows = sp["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    
    for ax, (key, lab) in zip(axes, (("accept_rate", "aceitação por token proposto"),
                                     ("tokens_per_round", "tokens por rodada"))):
        for pred, mk, col, name in (("L_greedy", "o", "#d62728", "previsão só pelo KL (lei)"),
                                    ("S_greedy", "s", "#1f77b4", "previsão pela sequência TF")):
            xs = [r[pred][key] for r in rows]
            ys = [r[f"meas_greedy_{'accept' if key == 'accept_rate' else 'tpr'}"] for r in rows]
            r2 = 1 - np.sum((np.array(ys) - np.array(xs)) ** 2) / np.sum((np.array(ys) - np.mean(ys)) ** 2)
            ax.scatter(xs, ys, s=18, marker=mk, color=col, alpha=0.8, label=f"{name}: R² {r2:.3f}")
                       
        lim = [min(min(ax.get_xlim()), min(ax.get_ylim())), max(max(ax.get_xlim()), max(ax.get_ylim()))]
        ax.plot(lim, lim, "k-", lw=0.6, alpha=0.5, zorder=0)
        ax.set_xlabel(f"previsto: {lab}")
        ax.set_ylabel(f"medido na decodificação real")
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=7, loc="upper left")
        
    fig.tight_layout()
    save(fig, "fig4_especulativa")


def fig5() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), sharey=True)
    models = ("Qwen3-4B", "gemma-3-4b-it", "Mistral-7B-Instruct-v0.3")
    
    lines = []
    labels = []
    
    for ax, ms in zip(axes, models):
        f = ROOT / "adaptive_closedloop" / ms / "b3_h8.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        pts = {}
        for k, v in d.items():
            if isinstance(v, dict) and (k.startswith("mix@") or k in ("static_base", "static_high")):
                b = round(v["bits_incremental"], 3)
                pts[b] = max(v["acc"], pts.get(b, 0))
        st = sorted(pts.items())
        l1, = ax.plot([x[0] for x in st], [100 * x[1] for x in st], "o-", color="#1f77b4", ms=3.5,
                lw=1.2, label="precisão mista estática")
                
        for pol, mk, col in (("cascade@", "s", "#d62728"), ("predictive@", "^", "#ff7f0e")):
            pts = sorted((v["bits_incremental"], v["acc"]) for k, v in d.items()
                         if isinstance(v, dict) and k.startswith(pol))
            if pts:
                l2, = ax.plot([x[0] for x in pts], [100 * x[1] for x in pts], mk, color=col, ms=4,
                        label=f"adaptativo por token ({pol[:-1]})")
                        
        if len(lines) == 0:
            h, l = ax.get_legend_handles_labels()
            lines.extend(h)
            labels.extend(l)
            
        ax.set_title(ms, fontsize=9)
        ax.set_xlabel("bits por peso (contagem incremental)")
        
    axes[0].set_ylabel("acurácia GSM8K (%)")
    
    # Legenda fora do plot para nao repetir, no topo, mas sem esmagar
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False, fontsize=8)
    
    fig.subplots_adjust(top=0.75, bottom=0.15, left=0.08, right=0.98) # Ajuste manual do layout para caber legenda
    save(fig, "fig5_adaptativo")


def fig6() -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    any_row = False
    for d in sorted((ROOT / "adversarial").iterdir()):
        f = d / "results_multi.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        kh = next((r["kappa_honest"] for r in res.values() if "kappa_honest" in r), None)
        for key, r in res.items():
            if "traj" not in r or not r["traj"]:
                continue
            col = "#d62728" if r["mode"].startswith("malign") else "#1f77b4"
            ax.plot([t["step"] for t in r["traj"]], [t["kappa_val"] / kh for t in r["traj"]],
                    color=col, lw=1.2, alpha=0.8,
                    label=("ataque maligno (maximiza flips)" if col == "#d62728" else
                           "ataque benigno (minimiza flips)") if not any_row or True else None)
            any_row = True
            
    if not any_row:
        plt.close(fig)
        print("  (fig6 sem dados ainda)")
        return
        
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    
    ax.axhline(1.0, color="k", lw=1.2, ls="--")
    ax.text(0.02, 1.03, "taxa média dos compressores honestos", transform=ax.get_yaxis_transform(),
            fontsize=7, va="bottom")
            
    ax.set_xlabel("passos de otimização do ataque")
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1.0, 1.5])
    ax.set_yticklabels(["0,25", "0,5", "1,0", "1,5"])
    ax.set_ylabel("κ do ataque / κ honesto (EM AMOSTRA)")
    
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="center right")
    fig.tight_layout()
    save(fig, "fig6_ataque")


def main(argv=None) -> int:
    """Gera as figuras pedidas (default: todas). Uma figura que falha nao derruba as outras."""
    import argparse

    figs = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=sorted(figs), default=sorted(figs))
    p.add_argument("--corpus", nargs="+", default=["mix"],
                   help="corpora nas figuras: mix (paper, default), outros ids, ou 'all'")
    args = p.parse_args(argv)
    global SHOWN
    SHOWN = None if "all" in args.corpus else set(args.corpus)
    rc = 0
    for name in args.only:
        try:
            figs[name]()
        except Exception as exc:  # pragma: no cover - dado faltando no disco
            print(f"  !! {name}: {type(exc).__name__}: {exc}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
