#!/usr/bin/env python3
"""Que tipo de token muda num flip, e qual tipo move a resposta final.

O artigo mostra que a KL prevê bem a taxa de flips e mal a acuracia downstream
(R2 0.39 fora do modelo). Uma explicacao candidata: a KL pesa todos os flips
igualmente, mas so uma fracao deles carrega a resposta. Aqui classificamos cada
flip pelo tipo de token trocado e testamos se a taxa de flips NUMERICOS prevê a
fracao de respostas alteradas melhor que a taxa total.

A classificacao e automatica e por regra -- nao ha anotacao humana. Isso e uma
limitacao (um flip de palavra pode ser fatal e um numerico inofensivo) e uma
vantagem (e reproduzivel e aplicavel a 10^5 posicoes).

    python -m glod flip-kinds --corpus mix
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
import torch

from glod.paths import CACHE_DIR, OUT

AN = OUT / "analysis"
DIGIT = re.compile(r"\d")
WORD = re.compile(r"^[A-Za-z]+$")
PUNCT_WS = re.compile(r"^[\s\W_]*$")


def classify(a: str, b: str) -> str:
    """Tipo de um flip, do mais consequente ao menos.

    A ordem importa: um par ('5', '3') e numerico mesmo que ambos sejam curtos,
    e ('Find', 'Determine') e lexical mesmo tendo tamanhos diferentes.
    """
    if DIGIT.search(a) or DIGIT.search(b):
        return "numeric"
    sa, sb = a.strip(), b.strip()
    if sa.lower() == sb.lower():          # so espacamento ou caixa
        return "format"
    if PUNCT_WS.match(a) and PUNCT_WS.match(b):
        return "punct"
    if PUNCT_WS.match(a) != PUNCT_WS.match(b):
        return "word_punct"               # palavra <-> pontuacao: muda a estrutura
    if WORD.match(sa) and WORD.match(sb):
        return "lexical"
    return "other"


KINDS = ["numeric", "lexical", "word_punct", "punct", "format", "other"]


def gsm_mask(corpus: dict) -> torch.Tensor:
    src = [r["source"] == "gsm8k" for r in corpus["records"]]
    return torch.tensor([s for s, g in zip(src, corpus["gen_ids"]) for s in [s] * len(g)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="mix")
    p.add_argument("--max-examples", type=int, default=12,
                   help="exemplos decodificados por tipo, para inspecao")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    closed = json.loads((AN / "closedloop.json").read_text())["gsm8k"]
    by_model = defaultdict(list)
    for r in closed:
        by_model[r["model"]].append(r)

    rows, examples = [], defaultdict(list)
    for model, recs in sorted(by_model.items()):
        slug = model if args.corpus == "mix" else f"{model}__{args.corpus}"
        d = OUT / slug / slug
        cf = OUT / "corpora" / f"{slug}.json"
        if not (d / "bf16.pt").exists() or not cf.exists():
            print(f"(pulando {slug}: sem referencia)")
            continue
        ref = torch.load(d / "bf16.pt", map_location="cpu", mmap=True, weights_only=False)
        corpus = json.loads(cf.read_text())
        mask = gsm_mask(corpus)
        # posicao relativa dentro da sequencia: um flip cedo contamina tudo o que vem
        # depois, entao "onde" o flip cai e a hipotese natural para o descasamento
        # entre taxa de flips e dano de acuracia
        lens = [len(g) for g in corpus["gen_ids"]]
        relpos = torch.tensor(np.concatenate(
            [np.arange(n) / max(n - 1, 1) for n in lens]), dtype=torch.float64)
        base_margin = (ref["topv"][:, 0] - ref["topv"][:, 1]).double()
        hf = corpus.get("model") or model
        tok = AutoTokenizer.from_pretrained(hf, cache_dir=str(CACHE_DIR))
        a = ref["top1"]
        for rec in recs:
            f = d / f"{rec['config']}.pt"
            if not f.exists():
                continue
            b = torch.load(f, map_location="cpu", mmap=True, weights_only=False)["top1"]
            fl = (a != b) & mask
            n_fl = int(fl.sum())
            if n_fl == 0:
                continue
            idx = fl.nonzero().flatten()
            # decodificar em lote: um decode por token e lento em 10^4 posicoes
            ta = tok.batch_decode(a[idx].unsqueeze(1))
            tb = tok.batch_decode(b[idx].unsqueeze(1))
            kinds = [classify(x, y) for x, y in zip(ta, tb)]
            c = Counter(kinds)
            n_pos = int(mask.sum())
            pos = relpos[fl]; marg = base_margin[fl]
            row = {"rel_pos_mean": float(pos.mean()),
                   "rel_pos_first_third": float((pos < 1 / 3).double().mean()),
                   "flip_margin_mean": float(marg.mean()),
                   "model": model, "config": rec["config"],
                   "kl_gsm": rec["kl_gsm"], "flip_gsm": rec["flip_gsm"],
                   "answer_changed": rec["answer_changed"], "d_acc": rec["d_acc"],
                   "n_positions": n_pos, "n_flips": n_fl}
            for k in KINDS:
                row[f"rate_{k}"] = c[k] / n_pos
                row[f"share_{k}"] = c[k] / n_fl
            rows.append(row)
            for x, y, k in zip(ta, tb, kinds):
                if len(examples[k]) < args.max_examples:
                    examples[k].append([x, y])
        print(f"  {model}: {len([r for r in rows if r['model'] == model])} configuracoes")

    if not rows:
        print("nenhum par (modelo, config) pontuavel")
        return 1

    def r2(pred, meas):
        pred, meas = np.asarray(pred), np.asarray(meas)
        A = np.vstack([pred, np.ones_like(pred)]).T
        coef, *_ = np.linalg.lstsq(A, meas, rcond=None)
        res = meas - A @ coef
        return float(1 - (res ** 2).sum() / ((meas - meas.mean()) ** 2).sum())

    y = [r["answer_changed"] for r in rows]

    def lomo(vals):
        """Leave-one-model-out: e a transferencia entre modelos que o artigo mede."""
        num = den = 0.0
        ybar = float(np.mean(y))
        for m in sorted({r["model"] for r in rows}):
            tr = [i for i, r in enumerate(rows) if r["model"] != m]
            te = [i for i, r in enumerate(rows) if r["model"] == m]
            if len(tr) < 3 or len(te) < 2:
                continue
            x = np.array([vals[i] for i in tr]); yy = np.array([y[i] for i in tr])
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            xt = np.array([vals[i] for i in te]); yt = np.array([y[i] for i in te])
            num += ((yt - (coef[0] * xt + coef[1])) ** 2).sum()
            den += ((yt - ybar) ** 2).sum()
        return float(1 - num / den) if den else float("nan")

    preds = {"sqrt_kl": [math.sqrt(r["kl_gsm"]) for r in rows],
             "flip_total": [r["flip_gsm"] for r in rows]}
    for k in KINDS:
        preds["flip_" + k] = [r["rate_" + k] for r in rows]
    fits = {k: {"r2_in_sample": r2(v, y), "r2_lomo": lomo(v)} for k, v in preds.items()}

    # A composicao dos flips depende da familia do metodo? Se poda de magnitude
    # concentrasse flips em identidades referenciais e quantizacao nao, o paradoxo
    # de "mesmo kappa, dano downstream diferente" estaria explicado. Testamos.
    def family_of(cfg: str) -> str:
        for pre, f in (("gptq", "gptq"), ("awq", "awq"), ("sgpt", "sparsegpt"),
                       ("wanda", "wanda"), ("mag", "magnitude"), ("kv", "kv"),
                       ("skip", "skip")):
            if cfg.startswith(pre):
                return f
        if re.fullmatch(r"u\d+", cfg):
            return "rtn"
        if re.fullmatch(r"g\d+", cfg):
            return "gauss"
        return "other"

    for r in rows:
        r["family"] = family_of(r["config"])
    win = [r for r in rows if 0.03 <= r["kl_gsm"] <= 0.20]   # faixa em que as familias coexistem
    byf = defaultdict(list)
    for r in win:
        byf[r["family"]].append(r)
    big = {f: v for f, v in byf.items() if len(v) >= 3}
    comp = {f: {"n": len(v), "kl_median": float(np.median([r["kl_gsm"] for r in v])),
                **{f"share_{k}": float(np.median([r[f"share_{k}"] for r in v])) for k in KINDS}}
            for f, v in sorted(big.items())}
    anova = None
    if len(big) >= 3:
        from scipy import stats as _st
        x = np.log([r["kl_gsm"] for r in win]); yy = np.array([r["share_lexical"] for r in win])
        b_, a_ = np.polyfit(x, yy, 1)
        resid = yy - (a_ + b_ * x)
        groups = [[resid[i] for i, r in enumerate(win) if r["family"] == f] for f in big]
        F, pv = _st.f_oneway(*groups)
        anova = {"F": float(F), "p": float(pv), "n": len(win), "n_families": len(big),
                 "residual_spread_pp": float(100 * (max(np.mean(g) for g in groups)
                                                    - min(np.mean(g) for g in groups))),
                 "kl_window": [0.03, 0.20]}
    posf = {f: {"n": len(v),
                "rel_pos_mean": float(np.mean([r["rel_pos_mean"] for r in v])),
                "first_third": float(np.mean([r["rel_pos_first_third"] for r in v])),
                "flip_margin": float(np.mean([r["flip_margin_mean"] for r in v]))}
            for f, v in sorted(big.items())}
    out_family = {"by_family": comp, "anova_lexical_share": anova, "position": posf}
    if len(big) >= 3:
        from scipy import stats as _s2
        x = np.log([r["kl_gsm"] for r in win]); yy = np.array([r["rel_pos_mean"] for r in win])
        b2, a2 = np.polyfit(x, yy, 1)
        rr = yy - (a2 + b2 * x)
        gs = [[rr[i] for i, r in enumerate(win) if r["family"] == f] for f in big]
        F2, p2 = _s2.f_oneway(*gs)
        out_family["anova_position"] = {"F": float(F2), "p": float(p2), "n": len(win),
                                        "spread": float(max(np.mean(g) for g in gs)
                                                        - min(np.mean(g) for g in gs))}

    shares = {k: float(np.median([r[f"share_{k}"] for r in rows])) for k in KINDS}
    out = {"corpus": args.corpus, "n_rows": len(rows),
           "n_models": len({r["model"] for r in rows}),
           "median_share": shares, "fits": fits, "rows": rows,
           "composition": out_family,
           "examples": {k: v for k, v in examples.items()}}
    (AN / "flip_kinds.json").write_text(json.dumps(out, indent=1))

    print(f"\nn = {len(rows)} (modelo, config) em {out['n_models']} modelos\n")
    print("fracao mediana dos flips, por tipo de token:")
    for k in KINDS:
        print(f"  {k:<11} {100 * shares[k]:5.1f}%")
    print("\nprevendo a fracao de respostas GSM8K alteradas:")
    print(f"  {'previsor':<14} {'R2 na amostra':>14} {'R2 leave-one-model-out':>24}")
    for k, v in sorted(fits.items(), key=lambda kv: -kv[1]["r2_lomo"]):
        print(f"  {k:<14} {v['r2_in_sample']:>14.3f} {v['r2_lomo']:>24.3f}")
    if anova:
        print("\ncomposicao dos flips por familia (KL casado, 0.03-0.20):")
        for f, v in sorted(comp.items(), key=lambda kv: -kv[1]["share_lexical"]):
            print(f"  {f:<12} n={v['n']:<3} KL {v['kl_median']:.3f}  "
                  f"lexical {100 * v['share_lexical']:.1f}%  numeric {100 * v['share_numeric']:.1f}%")
        print(f"  ANOVA sobre o residuo em log(KL): F={anova['F']:.2f} p={anova['p']:.3f} "
              f"({anova['n_families']} familias, amplitude {anova['residual_spread_pp']:.1f} pp)")
    print(f"\n-> {AN / 'flip_kinds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
