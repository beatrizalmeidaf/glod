#!/usr/bin/env python3
"""A lei de um passo compõe em geração livre? (resposta à crítica do teacher forcing)

A lei mede `flip` = P(o top-1 mudar) num passo, sob teacher forcing: o prefixo e o
mesmo para os dois modelos. A objecao obvia e que LLMs geram de forma autoregressiva,
onde cada flip muda o prefixo dos passos seguintes.

Este estagio testa a composicao SEM rodar nada novo: usa as geracoes livres reais que
ja existem em `closedloop/<modelo>/gsm8k_<config>.json` (400 prompts por configuracao,
com o bf16 como referencia) e mede, por par (prompt, config):

    L = comprimento do prefixo de tokens identico ao do bf16 (posicao do 1o desvio)

Com isso vem o risco empirico por token (estimador de taxa de falha):

    h_obs = (numero de trajetorias que divergiram) / (soma dos tokens sob risco)

e a previsao da lei, que supoe passos independentes:

    h_prev = flip medido em teacher forcing no mesmo (modelo, config)
    S(n) = (1 - flip)^n      (sobrevivencia prevista)

A razao h_obs / flip e o numero que interessa:
  ~1  a lei de um passo compoe como processo sem memoria - prever trajetoria e so
      elevar (1 - flip) a n;
  >1  a geracao livre diverge MAIS rapido que o passo isolado (erro se acumula nos
      estados, e o que a critica preve);
  <1  o modelo comprimido reconverge para a trajetoria da referencia (auto-correcao).

    python -m ews compound
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ews.paths import OUT as ROOT
from ews.paths import CACHE_DIR

AN = ROOT / "analysis"


def _model_id(slug: str) -> str | None:
    """Id completo do HF a partir do nome no disco (o corpus guarda o campo `model`)."""
    f = ROOT / "corpora" / f"{slug}.json"
    if not f.exists():
        return None
    with f.open() as fh:
        for line in fh:            # o arquivo e grande: o campo `model` vem no comeco
            i = line.find('"model"')
            if i >= 0:
                return json.loads("{" + line[i:].split(",", 1)[0] + "}")["model"]
    return None


def first_divergence(tok, a: str, b: str, max_tokens: int = 512) -> tuple[int, bool]:
    """(tokens iguais no inicio, divergiu?) entre duas geracoes."""
    ia = tok(a, add_special_tokens=False)["input_ids"][:max_tokens]
    ib = tok(b, add_special_tokens=False)["input_ids"][:max_tokens]
    n = min(len(ia), len(ib))
    for i in range(n):
        if ia[i] != ib[i]:
            return i, True
    # um e prefixo do outro: se tem comprimentos diferentes, divergiu no fim
    return n, len(ia) != len(ib)


def hazard_by_position(slug: str, config: str, n_pos: int = 512):
    """Taxa de flip POR POSICAO do token gerado, medida em teacher forcing.

    O flip medio trata todos os passos como iguais, mas os primeiros tokens de uma
    resposta sao os mais ambiguos (o modelo decide como comecar). A previsao correta de
    trajetoria nao e (1-flip)^n e sim o produto dos riscos por posicao.
    """
    import torch

    ref_p = ROOT / slug / slug / "bf16.pt"
    cfg_p = ROOT / slug / slug / f"{config}.pt"
    corp = ROOT / "corpora" / f"{slug}.json"
    if not (ref_p.exists() and cfg_p.exists() and corp.exists()):
        return None
    ref = torch.load(ref_p, map_location="cpu", mmap=True, weights_only=False)
    cfg = torch.load(cfg_p, map_location="cpu", mmap=True, weights_only=False)
    gen = json.loads(corp.read_text())["gen_ids"]
    pos = torch.cat([torch.arange(len(g)) for g in gen])
    fl = (cfg["top1"] != ref["top1"])
    if pos.numel() != fl.numel():
        return None
    h = np.full(n_pos, np.nan)
    for i in range(n_pos):
        m = pos == i
        if int(m.sum()) >= 20:
            h[i] = float(fl[m].double().mean())
    # posicoes sem amostra suficiente herdam a ultima taxa conhecida
    last = np.nan
    for i in range(n_pos):
        if np.isnan(h[i]):
            h[i] = last
        else:
            last = h[i]
    return h


def median_survival(h: np.ndarray) -> float:
    """Menor n com produto(1 - h_i) < 0.5, ou seja a mediana prevista do 1o desvio."""
    s = 1.0
    for i, hi in enumerate(h):
        if np.isnan(hi):
            break
        s *= (1 - hi)
        if s < 0.5:
            return float(i + 1)
    return float(len(h))


def flip_of(law_rows: list[dict], ref: str, config: str) -> tuple[float, float] | None:
    for r in law_rows:
        if r["ref"] == ref and r["model"] == ref and r["config"] == config:
            return r["flip"], r["kl"]
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--min-pairs", type=int, default=50)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    law = json.loads((AN / "law.json").read_text())["rows"]
    out: dict = {"pairs": [], "by_model": {}}
    print(f"{'modelo':24s} {'config':10s} {'flip TF':>8s} {'h_obs':>8s} {'razao':>6s} "
          f"{'L obs':>10s} {'prev unif':>9s} {'prev pos':>9s} {'n':>4s}")
    for d in sorted((ROOT / "closedloop").iterdir()):
        slug = d.name
        base = d / "gsm8k_bf16.json"
        if not base.exists():
            continue
        mid = _model_id(slug)
        if mid is None:
            continue
        tok = AutoTokenizer.from_pretrained(mid, cache_dir=CACHE_DIR)
        ref = json.loads(base.read_text())
        ref_by_idx = {r["idx"]: c for r, c in zip(ref["rows"], ref["completions"])}
        for f in sorted(d.glob("gsm8k_*.json")):
            cfg = f.stem[len("gsm8k_"):]
            if cfg == "bf16":
                continue
            fl = flip_of(law, slug, cfg)
            if fl is None:
                continue
            flip, kl = fl
            cur = json.loads(f.read_text())
            Ls, div = [], []
            for r, c in zip(cur["rows"], cur["completions"]):
                rc = ref_by_idx.get(r["idx"])
                if rc is None:
                    continue
                L, d_ = first_divergence(tok, rc, c, args.max_tokens)
                Ls.append(L)
                div.append(d_)
            if len(Ls) < args.min_pairs:
                continue
            exposure = float(sum(Ls)) + sum(1 for x in div if x)   # tokens sob risco
            h_obs = sum(div) / max(exposure, 1.0)
            ratio = h_obs / flip if flip > 0 else float("nan")
            med = float(np.median(Ls))
            med_pred = math.log(2) / flip if flip > 0 else float("inf")
            hz = hazard_by_position(slug, cfg, args.max_tokens)
            med_pos = median_survival(hz) if hz is not None else float("nan")
            row = {"model": slug, "config": cfg, "flip_tf": flip, "kl": kl,
                   "h_obs": h_obs, "ratio": ratio, "median_L": med,
                   "median_L_pred": med_pred, "median_L_pred_pos": med_pos,
                   "hazard_first_token": float(hz[0]) if hz is not None else float("nan"),
                   "n_pairs": len(Ls), "diverged_frac": float(np.mean(div))}
            out["pairs"].append(row)
            out["by_model"].setdefault(slug, []).append(ratio)
            print(f"{slug[:24]:24s} {cfg[:10]:10s} {flip:8.4f} {h_obs:8.4f} {ratio:6.2f} "
                  f"{med:10.0f} {med_pred:9.0f} {med_pos:9.0f} {len(Ls):4d}")

    # erro relativo das duas previsoes da mediana do 1o desvio
    for key, lab in (("median_L_pred", "risco uniforme (1-flip)^n"),
                     ("median_L_pred_pos", "risco por posicao")):
        v = [(r["median_L"], r[key]) for r in out["pairs"]
             if np.isfinite(r.get(key, float("nan"))) and r["median_L"] > 0]
        if v:
            err = [abs(a_ - b_) / a_ for a_, b_ in v]
            print(f"  previsao da mediana do 1o desvio, {lab:28s}: erro relativo mediano "
                  f"{100*np.median(err):5.1f}% (n={len(v)})")
    rs = [r["ratio"] for r in out["pairs"] if np.isfinite(r["ratio"])]
    if rs:
        se = np.std(rs, ddof=1) / math.sqrt(len(rs))
        print(f"\n  razao h_obs / flip(teacher forcing): {np.mean(rs):.2f} "
              f"IC95 [{np.mean(rs) - 1.96*se:.2f}, {np.mean(rs) + 1.96*se:.2f}] "
              f"| mediana {np.median(rs):.2f} | faixa {min(rs):.2f}-{max(rs):.2f} (n={len(rs)})")
        print("  >1 = geracao livre diverge mais rapido que o passo isolado; ~1 = compoe sem memoria")
        out["summary"] = {"mean": float(np.mean(rs)), "median": float(np.median(rs)),
                          "ci95": [float(np.mean(rs) - 1.96 * se), float(np.mean(rs) + 1.96 * se)],
                          "min": float(min(rs)), "max": float(max(rs)), "n": len(rs)}
    AN.mkdir(exist_ok=True)
    (AN / "compound.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {AN / 'compound.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
