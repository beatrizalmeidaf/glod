#!/usr/bin/env python3
"""O teto do ataque e um limite do modelo ou do otimizador que escolhemos?

O ataque do artigo usa perturbacoes aditivas de rank 16 por modulo. Como as
direcoes de flip tem dimensao efetiva de 50 a 221, e razoavel suspeitar que o
gargalo seja o proprio rank, e nao a geometria do modelo. Este estagio le as
variantes gravadas com `adv-multi --rank R --tag rankR` e compara a razao
kappa_ataque / kappa_honesto entre ranks, com o mesmo pareamento por KL que
`adv-report` usa (compressores honestos medidos a +-35% do mesmo KL).

    python -m glod rank-ablation
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
import torch

from glod.paths import OUT
from glod.pipelines.adversarial.report import honest_near_kl, test_mask

AN = OUT / "analysis"
TAGS = {"rank16": 16, "rank32": 32, "rank128": 128, "rankfull": "full"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["gemma-3-4b-it", "Qwen3-4B", "gemma-3-1b-it"])
    args = p.parse_args(argv)

    rows = []
    for m in args.models:
        msk = test_mask(m)
        ref_p = OUT / m / m / "bf16.pt"
        if msk is None or not ref_p.exists():
            print(f"(pulando {m}: sem referencia ou mascara de teste)")
            continue
        ref = torch.load(ref_p, map_location="cpu", mmap=True, weights_only=False)
        for f in sorted(glob.glob(str(OUT / "adversarial" / m / "fid_*malign_0.0*.pt"))):
            mm = re.match(r"fid_(?:(\w+)_)?malign_([\d.]+)\.pt", os.path.basename(f))
            if not mm:
                continue
            # a execucao original nao leva tag no nome; ela e o rank 16 do artigo
            tag = mm.group(1) or "rank16"
            if tag not in TAGS:
                continue
            c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            kl = c["kl"][msk].double().mean().item()
            kappa_atk = (c["top1"] != ref["top1"])[msk].double().mean().item() / math.sqrt(kl)
            kh, nh = honest_near_kl(m, kl)
            rows.append({"model": m, "tag": tag, "rank": TAGS[tag],
                         "budget": float(mm.group(2)), "kl": kl,
                         "kappa_attack": kappa_atk, "kappa_honest": kh,
                         "n_honest": nh, "ratio": kappa_atk / kh if kh == kh else float("nan")})
    # Piso de ruido: a execucao original do artigo e a replica de hoje usam o MESMO
    # rank e o mesmo orcamento. A diferenca entre elas e a escala contra a qual
    # qualquer efeito de rank precisa ser julgado.
    repl = {}
    for m in args.models:
        msk = test_mask(m)
        ref_p = OUT / m / m / "bf16.pt"
        if msk is None or not ref_p.exists():
            continue
        ref = torch.load(ref_p, map_location="cpu", mmap=True, weights_only=False)
        for bud in ("0.02", "0.05"):
            a = OUT / "adversarial" / m / f"fid_malign_{bud}.pt"
            b = OUT / "adversarial" / m / f"fid_rank16_malign_{bud}.pt"
            if not (a.exists() and b.exists()):
                continue
            def _k(f):
                c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                kl = c["kl"][msk].double().mean().item()
                return (c["top1"] != ref["top1"])[msk].double().mean().item() / math.sqrt(kl)
            repl[f"{m}@{bud}"] = _k(b) / _k(a)

    if not rows:
        print("nenhuma variante de rank encontrada; rode adv-multi --rank R --tag rankR")
        return 1

    by_rank = {}
    for r in rows:
        if r["ratio"] == r["ratio"]:
            by_rank.setdefault(str(r["rank"]), []).append(r["ratio"])
    summary = {k: {"n": len(v), "mean": float(np.mean(v)),
                   "min": float(np.min(v)), "max": float(np.max(v))}
               for k, v in by_rank.items()}
    # Comparacao INTERNA: mesmo modelo, mesmo orcamento, ranks diferentes. Dispensa
    # baseline honesto, que em modelos pequenos pode nao ter compressor pareavel
    # perto do orcamento (o gemma-3-1b nao tem nenhum a +-35% de KL 0.02).
    within = {}
    for m in {r["model"] for r in rows}:
        for bud in {r["budget"] for r in rows if r["model"] == m}:
            cell = {str(r["rank"]): r["kappa_attack"] for r in rows
                    if r["model"] == m and r["budget"] == bud}
            if len(cell) >= 2:
                base = cell.get("16")
                within[f"{m}@{bud}"] = {
                    "kappa_by_rank": cell,
                    "rel_to_rank16": ({k: v / base for k, v in cell.items()} if base else None)}
    # Convergencia: o ataque saturou dentro dos 400 passos, ou o teto e do orcamento de
    # passos? Lemos a trajetoria no lote de VALIDACAO fixo (`traj`), que e a mesma
    # usada para escolher o melhor iterado; `history` e de minilote e tem ~40% de ruido.
    # Medimos o quanto o melhor-ate-agora sobe no ultimo quarto (passo 300 -> fim). O
    # controle benigno entra para mostrar que o mesmo otimizador satura quando ha
    # para onde descer.
    conv = []
    for m in args.models:
        for tag, rk in (("multi", 16), ("rank32", 32), ("rank128", 128)):
            f = OUT / "adversarial" / m / f"results_{tag}.json"
            if not f.exists():
                continue
            for v in json.loads(f.read_text()).values():
                if not isinstance(v, dict) or "traj" not in v or v["budget"] not in (0.02, 0.05):
                    continue
                sign = 1 if v["mode"].startswith("malign") else -1
                t = [x for x in v["traj"] if 0.5 * v["budget"] < x["kl_val"] < 2 * v["budget"]]

                def best_upto(s, t=t, sign=sign):
                    k = [x["kappa_val"] for x in t if x["step"] <= s]
                    return (max(k) if sign > 0 else min(k)) if k else float("nan")
                b300, bend = best_upto(300), best_upto(10 ** 9)
                conv.append({"model": m, "rank": rk, "mode": v["mode"], "budget": v["budget"],
                             "kappa_init": v["traj"][0]["kappa_val"], "best_300": b300,
                             "best_end": bend, "gain_last_quarter": bend / b300 - 1,
                             "change_vs_init": bend / v["traj"][0]["kappa_val"] - 1})
    mal = [c for c in conv if c["mode"].startswith("malign")]
    ben = [c for c in conv if c["mode"].startswith("benign")]
    convergence = None
    if mal:
        g = np.array([c["gain_last_quarter"] for c in mal])
        convergence = {
            "rows": conv, "n_malign": len(mal),
            "n_malign_gain_le_5pct": int((g <= 0.05).sum()),
            "malign_gain_median": float(np.median(g)), "malign_gain_max": float(g.max()),
            "still_rising": [f"{c['model']}@r{c['rank']}@{c['budget']}: {c['gain_last_quarter']:+.3f}"
                             for c in mal if c["gain_last_quarter"] > 0.05],
            "benign_n": len(ben),
            "benign_gain_max_abs": (float(max(abs(c["gain_last_quarter"]) for c in ben))
                                    if ben else None),
            "benign_change_min": float(min(c["change_vs_init"] for c in ben)) if ben else None,
            "benign_change_max": float(max(c["change_vs_init"] for c in ben)) if ben else None}

    allr = [r["ratio"] for r in rows if r["ratio"] == r["ratio"]]
    out = {"rows": rows, "by_rank": summary, "within_model": within, "convergence": convergence,
           "same_config_replication": (
               {"ratios": repl, "max_abs_dev": float(max(abs(v - 1) for v in repl.values()))}
               if repl else None),
           "ratio_min": float(np.min(allr)), "ratio_max": float(np.max(allr)),
           "n_models": len({r["model"] for r in rows}),
           "ranks_tested": sorted(summary, key=lambda k: (k == "full", k))}
    (AN / "rank_ablation.json").write_text(json.dumps(out, indent=1))

    print(f"\n{'modelo':<14} {'rank':>5} {'budget':>7} {'KL':>8} {'kappa atk':>10} "
          f"{'kappa hon':>10} {'razao':>7}")
    print("-" * 68)
    for r in sorted(rows, key=lambda r: (r["model"], r["budget"], str(r["rank"]))):
        print(f"{r['model']:<14} {str(r['rank']):>5} {r['budget']:>7.2f} {r['kl']:>8.4f} "
              f"{r['kappa_attack']:>10.4f} {r['kappa_honest']:>10.4f} {r['ratio']:>7.3f}")
    print("\nmedia da razao por rank:")
    for k in out["ranks_tested"]:
        v = summary[k]
        print(f"  rank {k:>4}: {v['mean']:.3f}  [{v['min']:.3f}, {v['max']:.3f}]  (n={v['n']})")
    if repl:
        print("\npiso de ruido (mesma configuracao, duas execucoes):")
        for k_, v in sorted(repl.items()):
            print(f"  {k_:<28} razao {v:.3f}")
    print("\ncomparacao interna (kappa do ataque, mesmo modelo e orcamento):")
    for k, v in sorted(within.items()):
        rel = v["rel_to_rank16"]
        if rel:
            print(f"  {k:<28} " + "  ".join(f"r{r}={x:.3f}" for r, x in sorted(
                rel.items(), key=lambda kv: (kv[0] == "full", kv[0]))))
    if convergence:
        print(f"\nconvergencia (melhor kappa na validacao, passo 300 -> fim): "
              f"{convergence['n_malign_gain_le_5pct']}/{convergence['n_malign']} celulas malignas "
              f"sobem <= 5%; max {convergence['malign_gain_max']:+.1%}")
        for s in convergence["still_rising"]:
            print(f"  ainda subindo: {s}")
    print(f"\nfaixa geral das razoes: {out['ratio_min']:.3f}--{out['ratio_max']:.3f}")
    print(f"-> {AN / 'rank_ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
