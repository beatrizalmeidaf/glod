#!/usr/bin/env python3
"""Orcamento do expoente: de onde vem o desvio de 1/2.

A contagem de primeira ordem do paper escreve, por configuracao,

    flip = eta * rho_c * A * sqrt(2 KL) / 2,      eta = flip / pred  (obs/pred)

com rho_c fixo na referencia. Tomando logs e regredindo em log KL SOBRE AS MESMAS
configuracoes, a linearidade do OLS da uma identidade exata:

    alpha = 1/2 + beta_A + beta_eta

beta_A    : o deslocamento de margem cresce mais (ou menos) que sqrt(KL) - anisotropia
            que muda com a intensidade, i.e. termos de ordem superior entre Delta g e KL
beta_eta  : a contagem de primeira ordem erra de forma dependente da intensidade - forma
            da densidade de margens longe de zero (a hipotese do apendice do paper:
            densidade caindo => beta_eta < 0 => alpha < 1/2)

O paper registra que a hipotese da forma da densidade acerta a ordem mas nao o nivel
(26 de 42 expoentes acima de 1/2). Se beta_eta < 0 e beta_A > 0 em quase todas as
referencias, o nivel vem da anisotropia crescente, e a lacuna fica explicada.

Tambem decompoe beta_A em duas partes medidas:  A = E|dg| / sqrt(2 KL), com
E|dg| = sqrt(E dg^2) * (E|dg| / sqrt(E dg^2)); a segunda razao (forma da distribuicao
de dg) so e reportada se o JSON trouxer E dg^2.

    python -m glod exponent-budget --src /local/$USER/ews_results/fid/analysis/prop.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from glod.paths import OUT, corpus_of


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def budget(cfgs: list[dict], kl_max: float, cover: dict | None = None) -> dict | None:
    ok = [c for c in cfgs if 1e-3 < c["kl"] < kl_max and c["flip"] > 0 and c.get("ratio")
          and c.get("anisotropy_A")]
    if len(ok) < 5:
        return None
    x = np.log([c["kl"] for c in ok])
    a = slope(x, np.log([c["flip"] for c in ok]))
    bA = slope(x, np.log([c["anisotropy_A"] for c in ok]))
    bE = slope(x, np.log([c["ratio"] for c in ok]))
    out = {"n": len(ok), "alpha": a, "beta_A": bA, "beta_eta": bE,
           "closure": a - (0.5 + bA + bE)}           # zero ate arredondamento: e identidade
    # eta = eta_top2 / cover, com cover a fracao dos flips que e o cruzamento i1<->i2
    # (flip-mechanism): separa o erro da contagem do par do que vem de fora dele
    if cover and all(c["config"] in cover for c in ok):
        cv = np.array([cover[c["config"]] for c in ok])
        out["beta_outside_pair"] = slope(x, -np.log(cv))
        out["beta_eta_top2"] = bE - out["beta_outside_pair"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(OUT / "analysis" / "prop.json"))
    p.add_argument("--out", default=str(OUT / "analysis" / "exponent_budget.json"))
    p.add_argument("--kl-max", type=float, nargs="+", default=[0.25, 2.0])
    args = p.parse_args(argv)
    prop = json.loads(Path(args.src).read_text())
    shape = {}
    ms = Path(args.src).with_name("margin_shape.json")
    if ms.exists():
        # rho1/rho0 por referencia, se o estagio margin-shape gravou nesse formato
        try:
            raw = json.loads(ms.read_text())
            rows = raw.get("rows", raw) if isinstance(raw, dict) else raw
            for r in rows if isinstance(rows, list) else []:
                if "ref" in r and "rho1_over_rho0" in r:
                    shape[r["ref"].replace("__fp32", "")] = r["rho1_over_rho0"]
        except Exception:                      # formato diferente: segue sem a coluna
            shape = {}
    covers: dict[str, dict[str, float]] = {}
    fm = Path(args.src).with_name("flip_mechanism.json")
    if fm.exists():
        for r in json.loads(fm.read_text()).get("rows_fp32", []):
            if r.get("cover"):
                covers.setdefault(r["ref"], {})[r["config"]] = r["cover"]
    report = {"src": args.src, "by_window": {}}
    for kmax in args.kl_max:
        rows = []
        for ref, v in prop.items():
            if "__fp32" not in ref or not isinstance(v, dict) or not isinstance(v.get("configs"), list):
                continue                        # so fp32: sem empates do bf16
            b = budget(v["configs"], kmax, covers.get(ref))
            if b:
                base = ref.replace("__fp32", "")
                b.update(ref=base, corpus=corpus_of(base), rho1_over_rho0=shape.get(base))
                rows.append(b)
        if not rows:
            continue
        al = np.array([r["alpha"] for r in rows]); bA = np.array([r["beta_A"] for r in rows])
        bE = np.array([r["beta_eta"] for r in rows])
        above = al > 0.5
        summ = {
            "n_refs": len(rows),
            "alpha_above_half": int(above.sum()),
            "median": {"alpha": float(np.median(al)), "beta_A": float(np.median(bA)), "beta_eta": float(np.median(bE))},
            "beta_A_positive": int((bA > 0).sum()), "beta_eta_negative": int((bE < 0).sum()),
            # entre as referencias acima de 1/2, que termo carrega o excesso?
            "above_half_mean": {"beta_A": float(bA[above].mean()) if above.any() else None,
                                "beta_eta": float(bE[above].mean()) if above.any() else None},
            # variancia entre referencias de (alpha - 1/2) repartida pelos dois termos
            "var_share": {"beta_A": float(np.cov(bA, al)[0, 1] / al.var(ddof=1)),
                          "beta_eta": float(np.cov(bE, al)[0, 1] / al.var(ddof=1))},
            "max_abs_closure": float(max(abs(r["closure"]) for r in rows)),
            "by_corpus": {},
        }
        for c in sorted({r["corpus"] for r in rows}):
            g = [r for r in rows if r["corpus"] == c]
            summ["by_corpus"][c] = {"n": len(g), **{k: float(np.median([r[k] for r in g]))
                                                    for k in ("alpha", "beta_A", "beta_eta")}}
        report["by_window"][str(kmax)] = {"summary": summ, "rows": rows}
        print(f"\n== janela 1e-3 < KL < {kmax}: {len(rows)} referencias fp32 "
              f"(fechamento max {summ['max_abs_closure']:.1e})")
        print(f"   alpha > 1/2 em {summ['alpha_above_half']}; beta_A > 0 em {summ['beta_A_positive']}; "
              f"beta_eta < 0 em {summ['beta_eta_negative']}")
        m = summ["median"]
        print(f"   medianas: alpha {m['alpha']:.3f} = 0.5 + beta_A {m['beta_A']:+.3f} + beta_eta {m['beta_eta']:+.3f}")
        print(f"   parte da variancia de alpha: beta_A {summ['var_share']['beta_A']:.2f}, "
              f"beta_eta {summ['var_share']['beta_eta']:.2f}")
        wc = [r for r in rows if "beta_outside_pair" in r]
        if wc:
            bo = np.array([r["beta_outside_pair"] for r in wc]); b2 = np.array([r["beta_eta_top2"] for r in wc])
            aw = np.array([r["alpha"] for r in wc])
            summ["outside_pair"] = {
                "n_refs": len(wc), "median_beta_outside_pair": float(np.median(bo)),
                "median_beta_eta_top2": float(np.median(b2)), "beta_outside_pair_positive": int((bo > 0).sum()),
                "beta_eta_top2_negative": int((b2 < 0).sum()),
                "var_share": {"outside_pair": float(np.cov(bo, aw)[0, 1] / aw.var(ddof=1)),
                              "eta_top2": float(np.cov(b2, aw)[0, 1] / aw.var(ddof=1))},
                "by_corpus": {c: {"beta_outside_pair": float(np.median([r["beta_outside_pair"] for r in wc
                                                                         if r["corpus"] == c])),
                                  "beta_eta_top2": float(np.median([r["beta_eta_top2"] for r in wc
                                                                     if r["corpus"] == c]))}
                              for c in sorted({r["corpus"] for r in wc})}}
            o = summ["outside_pair"]
            print(f"   beta_eta = beta_top2 {o['median_beta_eta_top2']:+.3f} + fora do par "
                  f"{o['median_beta_outside_pair']:+.3f} (medianas, {o['n_refs']} refs); fora do par > 0 em "
                  f"{o['beta_outside_pair_positive']}, top2 < 0 em {o['beta_eta_top2_negative']}")
            print(f"   parte da variancia de alpha: fora do par {o['var_share']['outside_pair']:.2f}, "
                  f"top2 {o['var_share']['eta_top2']:.2f}")
            for c, v in o["by_corpus"].items():
                print(f"   {c:14s} top2 {v['beta_eta_top2']:+.3f} fora do par {v['beta_outside_pair']:+.3f}")
        for c, v in summ["by_corpus"].items():
            print(f"   {c:14s} n={v['n']:2d} alpha {v['alpha']:.3f} beta_A {v['beta_A']:+.3f} "
                  f"beta_eta {v['beta_eta']:+.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\ngravado: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
