#!/usr/bin/env python3
"""Escolher o draft da decodificacao especulativa por TV ou por KL, e medir no vLLM.

Cada draft candidato (modelos menores da familia do alvo e versoes quantizadas publicadas)
e pontuado em teacher forcing contra a referencia GSM8K do alvo (`glod grid score
--ref-model <alvo> --configs raw|ofc@...`). Dois seletores preveem a aceitacao greedy
por token proposto, sem ver nenhuma decodificacao especulativa:

  TV   a = 1 - TV                             (sem calibracao)
  KL   a = 1 - kappa_alvo * sqrt(KL)          (kappa publicado do alvo no GSM8K; senao no mix)

e dai o speedup previsto E / (k c + 1), com E = (1 - a^(k+1)) / (1 - a) tokens por rodada e
c = bytes de peso do draft / bytes do alvo. O vLLM (scripts/vllm_spec_bench.py, kernels reais,
prompts GSM8K disjuntos da calibracao) mede aceitacao e tokens/s. O relatorio compara:
erro da aceitacao prevista, Spearman previsto x medido e o arrependimento de cada seletor
(speedup realizado do draft escolhido sobre o do melhor draft).

    python -m glod draft-select prompts --target Qwen/Qwen3-14B --n 64 --out prompts.json
    python -m glod draft-select report --target Qwen/Qwen3-14B --vllm vllm.json \\
        --drafts q0.6=Qwen/Qwen3-0.6B:raw q4awq=Qwen/Qwen3-4B:ofc@Qwen/Qwen3-4B-AWQ ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, ref_slug, slug

FID = Path("/local/user_beatrizalmeida/ews_results/fid")   # kappa publicado dos alvos


def weight_bytes(repo: str, cache_dir: str) -> int:
    """Bytes dos .safetensors do repo no cache (o custo de ler o draft por token)."""
    base = Path(cache_dir) / f"models--{repo.replace('/', '--')}" / "snapshots"
    snaps = sorted(base.glob("*"))
    if not snaps:
        raise SystemExit(f"{repo} nao esta no cache {cache_dir}")
    return sum(f.stat().st_size for f in snaps[-1].rglob("*.safetensors"))


def published_kappa(target: str, corpus: str = "gsm8k") -> tuple[float, str]:
    """kappa publicado do alvo na tarefa; sem referencia da tarefa, o do mix (calibracao em outro corpus)."""
    rows = json.loads((FID / "analysis" / "slope.json").read_text())
    k = {r["ref"]: r["kappa"] for r in rows}
    s = slug(target)
    for ref, label in ((f"{s}__{corpus}__fp32", corpus), (f"{s}__{corpus}", corpus), (f"{s}__fp32", "mix"), (s, "mix")):
        if ref in k and np.isfinite(k[ref]):
            return float(k[ref]), label
    raise SystemExit(f"sem kappa publicado para {target}")


def cmd_prompts(args) -> int:
    from transformers import AutoTokenizer
    from glod.pipelines.tasks.closedloop import gsm8k_disjoint
    tok = AutoTokenizer.from_pretrained(args.target, cache_dir=args.cache_dir)
    rs = ref_slug(args.target, corpus=args.corpus)
    corpus = json.loads((OUT / "corpora" / f"{rs}.json").read_text())
    used = {tok.decode(p, skip_special_tokens=False) for p in corpus["prompt_ids"]}
    calib = {tuple(p) for p in corpus["prompt_ids"]}
    if args.corpus == "gsm8k":
        recs = gsm8k_disjoint(tok, args.n, used, args.cache_dir)
    else:   # code: tarefas do MBPP que nao estao no corpus de calibracao
        from glod.corpora.token_oracle import build_prompts_code
        recs = [r for r in build_prompts_code(tok, 500, 1234, f"{args.cache_dir}/datasets")
                if tuple(tok(r["prompt"], add_special_tokens=False).input_ids) not in calib][:args.n]
    ids = [tok(r["prompt"], add_special_tokens=False).input_ids for r in recs]
    overlap = sum(tuple(i) in calib for i in ids)
    Path(args.out).write_text(json.dumps(ids))
    print(f"{len(ids)} prompts disjuntos da calibracao (sobreposicao por ids: {overlap}) -> {args.out}")
    return 0


def accept_rate_per_draft_token(a: float, k: int) -> float:
    """O accept_rate do vLLM e aceitos / tokens propostos numa rodada de k: depois da 1a
    rejeicao os demais contam como nao aceitos. Com rodadas independentes e concordancia a
    por posicao, aceitos por rodada = a + a^2 + ... + a^k."""
    a = min(max(a, 0.0), 0.999999)
    return a * (1 - a ** k) / ((1 - a) * k)


def accept_to_speedup(a: float, k: int, c: float) -> float:
    a = min(max(a, 0.0), 0.999999)
    e = (1 - a ** (k + 1)) / (1 - a)
    return e / (k * c + 1)


def cmd_report(args) -> int:
    rs = ref_slug(args.target, corpus=args.corpus, fp32=True)
    ref = torch.load(OUT / rs / rs / "bf16.pt", weights_only=False)
    kappa, kappa_src = published_kappa(args.target, args.corpus)
    vl = json.loads(Path(args.vllm).read_text())
    t_bytes = weight_bytes(args.target, args.cache_dir)
    rows = []
    for spec in args.drafts:
        name, rest = spec.split("=", 1)
        model, cfg = rest.split(":", 1)
        f = OUT / rs / ref_slug(model, corpus=args.corpus, fp32=True) / f"{cfg.replace('/', '--')}.pt"
        if not f.exists() or name not in vl:
            print(f"pulando {name}: falta {'pontuacao' if not f.exists() else 'medicao vLLM'}")
            continue
        r = torch.load(f, weights_only=False)
        flip = float((r["top1"] != ref["top1"]).float().mean())
        tv, kl = float(r["tv"].double().mean()), float(r["kl"].double().mean())
        repo = cfg[4:] if cfg.startswith("ofc@") else model
        c = weight_bytes(repo, args.cache_dir) / t_bytes
        a_tv, a_kl = 1 - tv, 1 - kappa * np.sqrt(kl)
        row = {"draft": name, "model": model, "config": cfg, "flip": flip, "tv": tv, "kl": kl, "cost": c,
               "agree_tv": a_tv, "agree_kl": a_kl,
               "pred_accept_tv": accept_rate_per_draft_token(a_tv, args.k),
               "pred_accept_kl": accept_rate_per_draft_token(a_kl, args.k),
               "pred_speedup_tv": accept_to_speedup(a_tv, args.k, c),
               "pred_speedup_kl": accept_to_speedup(a_kl, args.k, c)}
        for mode in ("greedy", "sample"):
            row[f"{mode}_accept"] = vl[name][mode]["accept_rate"]
            row[f"{mode}_speedup"] = vl[name][mode]["speedup"]
        rows.append(row)
    if len(rows) < 3:
        raise SystemExit("menos de 3 drafts completos")
    out = {"target": args.target, "corpus": args.corpus, "k": args.k, "kappa": kappa, "kappa_corpus": kappa_src,
           "drafts": rows}
    for mode in ("greedy", "sample"):
        acc = np.array([x[f"{mode}_accept"] for x in rows]); spd = np.array([x[f"{mode}_speedup"] for x in rows])
        best = spd.max()
        m = {}
        for sel in ("tv", "kl"):
            pa = np.array([x[f"pred_accept_{sel}"] for x in rows]); ps = np.array([x[f"pred_speedup_{sel}"] for x in rows])
            pick = int(ps.argmax())
            m[sel] = {"accept_mape": float(np.mean(np.abs(pa - acc) / acc)),
                      "spearman_accept": float(spearmanr(pa, acc)[0]),
                      "spearman_speedup": float(spearmanr(ps, spd)[0]),
                      "pick": rows[pick]["draft"], "regret": float(spd[pick] / best)}
        m["best_draft"] = rows[int(spd.argmax())]["draft"]
        out[mode] = m
    suffix = "" if args.corpus == "gsm8k" else f"__{args.corpus}"
    target = Path(args.out) if args.out else OUT / "analysis" / f"draft_select__{slug(args.target)}{suffix}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1))
    for mode in ("greedy", "sample"):
        m = out[mode]
        print(f"[{mode}] melhor draft medido: {m['best_draft']}")
        for sel in ("tv", "kl"):
            s = m[sel]
            print(f"   {sel.upper()}: erro da aceitacao {100 * s['accept_mape']:.1f}% | Spearman aceitacao "
                  f"{s['spearman_accept']:+.2f}, speedup {s['spearman_speedup']:+.2f} | escolhe {s['pick']} "
                  f"({100 * s['regret']:.0f}% do melhor speedup)")
    print(f"gravado: {target}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts"); a.add_argument("--target", required=True); a.add_argument("--n", type=int, default=64)
    a.add_argument("--out", required=True); a.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    a.add_argument("--corpus", choices=["gsm8k", "code"], default="gsm8k")
    b = sub.add_parser("report"); b.add_argument("--target", required=True); b.add_argument("--vllm", required=True)
    b.add_argument("--drafts", nargs="+", required=True); b.add_argument("--k", type=int, default=4)
    b.add_argument("--out", default=None); b.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    b.add_argument("--corpus", choices=["gsm8k", "code"], default="gsm8k")
    args = p.parse_args(argv)
    return {"prompts": cmd_prompts, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
