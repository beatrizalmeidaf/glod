#!/usr/bin/env python3
"""EWS | Fase 1c - analise (somente CPU/GPU leve) dos scores por token.

Perguntas, na ordem em que decidem a tese:
  A. Fronteira estatica: concordancia com a referencia x pesos lidos por token.
  B. Oraculo por token: quanto um gate perfeito economiza (vs. o por-exemplo).
  C. Teto analitico por token: P(base basta) e fracao elastica.
  D. Rajadas: os tokens dificeis vem em sequencia? (define se cache de ΔW paga)
  E. Gate realizavel: entropia/margem (cascata), token anterior (preditivo),
     probe em camada inicial (Early-Probe do documento). AUROC e curvas.
  F. Baseline obrigatorio: decodificacao especulativa greedy (sem perda) com
     draft = configuracao barata do mesmo modelo ou o 4B.

Uso:
    python ews_tokens_analyze.py --model gemma-3-12b-it
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ews.corpora.token_oracle import (
    auroc,
    burstiness,
    burstiness_null,
    cached_streaming_cost,
    chunk_oracle_cost,
    consistent_token_oracle,
    cost_at_divergence,
    gate_curve,
    speculative_expected_cost,
)

LOGGER = logging.getLogger("ews.tokens.analyze")
GIB = 1024**3


def load_run(base: Path, model: str) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    corpus = json.loads((base / f"{model}__corpus.json").read_text())
    own_dir = base / f"{model}__corpus" / model
    own = {p.stem: torch.load(p) for p in sorted(own_dir.glob("*.pt"))}
    cross = {}
    for d in (base / f"{model}__corpus").iterdir():
        if d.is_dir() and d.name != model:
            for p in sorted(d.glob("*.pt")):
                cross[f"{d.name}:{p.stem}"] = torch.load(p)
    return corpus, own, cross


def crossfit_probe_scores(x: torch.Tensor, y: torch.Tensor, seq: torch.Tensor,
                          valid: torch.Tensor, *, device: str) -> torch.Tensor:
    """Scores fora-da-amostra para todos os tokens, com 2 dobras POR SEQUENCIA.

    Tokens invalidos (ex.: primeiro token da sequencia num probe do token
    anterior) recebem +inf = sempre promover, a escolha conservadora de custo.
    """
    n_seq = int(seq.max()) + 1
    g = torch.Generator().manual_seed(0)
    fold_of_seq = torch.zeros(n_seq, dtype=torch.long)
    fold_of_seq[torch.randperm(n_seq, generator=g)[: n_seq // 2]] = 1
    fold = fold_of_seq[seq.long()]
    scores = torch.full((len(y),), float("inf"))
    for k in (0, 1):
        tr, te = (fold != k) & valid, (fold == k) & valid
        scores[te] = logistic_probe_scores(x[tr], y[tr], x[te], device=device)
    return scores


def logistic_probe_scores(x_train, y_train, x_test, *, device: str, epochs: int = 300) -> torch.Tensor:
    """Regressao logistica com padronizacao; devolve os logits no conjunto de teste."""
    if not (torch.isfinite(x_train).all() and torch.isfinite(x_test).all()):
        raise FloatingPointError("features nao-finitas no probe: AUROC seria lixo")
    mu, sd = x_train.mean(0), x_train.std(0).clamp(min=1e-4)
    xt = ((x_train - mu) / sd).to(device)
    xv = ((x_test - mu) / sd).to(device)
    yt = y_train.float().to(device)
    w = torch.zeros(xt.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    pos_weight = ((1 - yt.mean()) / yt.mean().clamp(min=1e-4)).detach()
    opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                xt @ w + b, yt, pos_weight=pos_weight)
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (xv @ w + b).cpu()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-3-12b-it")
    p.add_argument("--base-dir", default="results/raw/tokens")
    p.add_argument("--device", default="cpu")
    p.add_argument("--bases", nargs="+", default=["u3", "u4"],
                   help="configuracoes base para as analises de gate/rajada/especulativa")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    base = Path(args.base_dir)
    corpus, own, cross = load_run(base, args.model)
    report: dict = {"model": args.model}

    gen = torch.tensor([t for g in corpus["gen_ids"] for t in g], dtype=torch.int32)
    seq = torch.cat([torch.full((len(g),), i, dtype=torch.int32)
                     for i, g in enumerate(corpus["gen_ids"])])
    src = [corpus["records"][int(i)]["source"] for i in seq]
    is_gsm = torch.tensor([s == "gsm8k" for s in src])
    ref = own["bf16"]
    ref_top1, ref_cost = ref["top1"], ref["weight_bytes"]
    T = len(gen)

    LOGGER.info("=" * 78)
    LOGGER.info("%s | %d sequencias | %d tokens (gsm8k %d, mmlu_pt %d)", args.model,
                len(corpus["gen_ids"]), T, int(is_gsm.sum()), int((~is_gsm).sum()))
    consistency = (ref_top1 == gen).float().mean().item()
    LOGGER.info("piso de ruido: bf16 teacher-forced reproduz %.2f%% da propria geracao",
                100 * consistency)
    report["noise_floor"] = consistency

    # --------------------------------------------------------------- A
    LOGGER.info("-" * 78)
    LOGGER.info("A. FRONTEIRA ESTATICA (referencia = top-1 do bf16, mesmo caminho de calculo)")
    LOGGER.info("%-18s | %7s | %8s | %8s | %8s", "config", "GiB/tok", "concord", "gsm8k", "mmlu_pt")
    rows = []
    allcfg = {**own, **cross}
    for name, d in sorted(allcfg.items(), key=lambda kv: kv[1]["weight_bytes"]):
        ag = d["top1"] == ref_top1
        rows.append((name, d["weight_bytes"], ag))
        LOGGER.info("%-18s | %7.2f | %7.2f%% | %7.2f%% | %7.2f%%", name, d["weight_bytes"] / GIB,
                    100 * ag.float().mean(), 100 * ag[is_gsm].float().mean(),
                    100 * ag[~is_gsm].float().mean())
    report["static"] = [{"config": n, "gib": b / GIB, "agreement": ag.float().mean().item()}
                        for n, b, ag in rows]

    # --------------------------------------------------------------- B
    LOGGER.info("-" * 78)
    LOGGER.info("B. ORACULO POR TOKEN (preserva 100%% do greedy da referencia)")
    families = {
        "uniforme": [n for n in own if n.startswith("u")],
        "3b+x4": [n for n in own if n.startswith("q3+") and n.endswith("x4")] + ["u3"],
        "3b+x16": [n for n in own if n.startswith("q3+") and n.endswith("x16")] + ["u3"],
        "skip": [n for n in own if n.startswith("skip")],
        "todas (mesmo modelo)": [n for n in own if n != "bf16"],
    }
    report["oracle"] = {}
    for fam, names in families.items():
        names = [n for n in names if n in own]
        if not names:
            continue
        agree = torch.stack([own[n]["top1"] == ref_top1 for n in names])
        costs = [own[n]["weight_bytes"] for n in names]
        res = consistent_token_oracle(agree, costs, ref_cost)
        LOGGER.info("  %-22s | %6.2f GiB/tok | economia %5.1f%% vs bf16",
                    fam, res["mean_cost"] / GIB, 100 * res["saving"])
        report["oracle"][fam] = {"gib": res["mean_cost"] / GIB, "saving": res["saving"]}

    # --------------------------------------------------------------- G
    LOGGER.info("-" * 78)
    LOGGER.info("G. GRANULARIDADE DA DECISAO (mesmo corpus, mesmas configs, oraculo consistente)")
    pos = torch.cat([torch.arange(len(g), dtype=torch.int32) for g in corpus["gen_ids"]])
    report["granularity"] = {}
    for fam in ("todas (mesmo modelo)", "uniforme"):
        names = [n for n in families[fam] if n in own]
        agree = torch.stack([own[n]["top1"] == ref_top1 for n in names])
        costs = [own[n]["weight_bytes"] for n in names]
        cells = []
        for chunk in (1, 2, 4, 8, 16, 32, 64, None):
            c = chunk_oracle_cost(agree, costs, ref_cost, seq, pos, chunk)
            cells.append((chunk, c))
            report["granularity"].setdefault(fam, {})[str(chunk)] = {
                "gib": c / GIB, "saving": 1 - c / ref_cost}
        LOGGER.info("  %-22s | %s", fam, " | ".join(
            f"{'seq' if ch is None else ch}:{100 * (1 - c / ref_cost):4.1f}%" for ch, c in cells))

    # --------------------------------------------------- C, D, E, F por base
    report["bases"] = {}
    for bname in args.bases:
        if bname not in own:
            continue
        bd = own[bname]
        hard = bd["top1"] != ref_top1
        bcost = bd["weight_bytes"]
        out: dict = {"cost_gib": bcost / GIB}
        LOGGER.info("=" * 78)
        LOGGER.info("BASE %s (%.2f GiB/tok) -> referencia bf16 (%.2f GiB/tok)",
                    bname, bcost / GIB, ref_cost / GIB)

        # C
        p_suff = 1 - hard.float().mean().item()
        ceiling = p_suff * (1 - bcost / ref_cost)
        LOGGER.info("C. P(base basta) = %.2f%% | fracao elastica = %.1f%% | teto binario = %.1f%%",
                    100 * p_suff, 100 * (1 - bcost / ref_cost), 100 * ceiling)
        out.update(p_sufficient=p_suff, ceiling_binary=ceiling)

        # D
        bst = burstiness(hard, seq)
        nul = burstiness_null(hard, seq, n_reps=20)
        LOGGER.info("D. rajadas: taxa %.3f | P(dificil|anterior dificil) %.3f | lift %.2f "
                    "(nulo intra-seq %.2f) | rajada media %.2f (nulo %.2f)",
                    bst["rate"], bst["p_given_prev"], bst["lift"], nul["lift"],
                    bst["mean_run"], nul["mean_run"])
        out.update(burst=bst, burst_null=nul)

        # E - sinais realizaveis e o preco da antecipacao
        LOGGER.info("E. gate realizavel (AUROC fora-da-amostra para prever 'base diverge'):")
        n_layers = 1 + max(bd.get("hidden", {0: None}).keys()) if bd.get("hidden") else 48
        first_tok = torch.cat([torch.tensor([True]), seq[1:] != seq[:-1]])
        ent, mar = bd["entropy"].float(), bd["margin"].float()
        prev_ent = torch.roll(ent, 1)
        prev_ent[first_tok] = float("inf")
        signals: dict[str, tuple] = {
            "ex-post: entropia do token": (ent, "cascade", 0.0, None),
            "ex-post: -margem do token": (-mar, "cascade", 0.0, None),
            "ex-ante: entropia do anterior": (prev_ent, "predictive", 0.0, None),
        }
        hidden = bd.get("hidden", {})
        last_layer = max(hidden) if hidden else None
        for layer in sorted(hidden):
            h = hidden[layer].float()
            if layer != last_layer:
                sc = crossfit_probe_scores(h, hard, seq, torch.ones_like(hard), device=args.device)
                prefix_cfg = f"p{bname[1:]}to{layer + 1}"
                measured = (own[prefix_cfg]["top1"] != ref_top1) if prefix_cfg in own else None
                tag = "" if measured is not None else " (fidelidade SUPOSTA)"
                signals[f"early-probe: camada {layer}{tag}"] = (
                    sc, "early_probe", (layer + 1) / n_layers, measured)
            else:
                hp = torch.roll(h, 1, dims=0)
                sc = crossfit_probe_scores(hp, hard, seq, ~first_tok, device=args.device)
                signals[f"ex-ante: probe camada {layer} do anterior"] = (sc, "predictive", 0.0, None)
        signals["oraculo (gate perfeito)"] = (hard.float(), "predictive", 0.0, None)

        out["auroc"], out["anticipation"] = {}, {}
        targets = (0.005, 0.01, 0.02, 0.05)
        LOGGER.info("   %-36s | %5s | %s", "sinal", "AUROC",
                    " | ".join(f"GiB@div<={100*t:.1f}%" for t in targets))
        for sname, (sval, mode, frac, measured) in signals.items():
            finite = torch.isfinite(sval)
            a = auroc(sval[finite], hard[finite])
            curve = gate_curve(sval.nan_to_num(posinf=1e30), hard, base_cost=bcost,
                               ref_cost=ref_cost, mode=mode, prefix_fraction=frac,
                               hard_if_promoted=measured)
            costs_t = [cost_at_divergence(curve, t) for t in targets]
            out["auroc"][sname] = a
            out["anticipation"][sname] = {"mode": mode, "auroc": a,
                                          **{f"gib@{t}": c / GIB for t, c in zip(targets, costs_t)}}
            LOGGER.info("   %-36s | %.3f | %s", sname, a,
                        " | ".join(f"{c / GIB:12.2f}" for c in costs_t))
        LOGGER.info("   (cascade = base + refazer; predictive = decide antes; early_probe = "
                    "prefixo na base, fidelidade do prefixo MEDIDA quando disponivel)")
        for layer in sorted(hidden):
            cfg = f"p{bname[1:]}to{layer + 1}"
            if cfg in own:
                LOGGER.info("   fidelidade medida de %-9s (camadas <%2d em %s bits): %.2f%%",
                            cfg, layer + 1, bname[1:],
                            100 * (own[cfg]["top1"] == ref_top1).float().mean())

        delta_io = ref_cost - bcost
        LOGGER.info("   cache de ΔW com ORACULO (limite superior), por janela:")
        out["cache_oracle"] = []
        for w in (0, 1, 2, 4, 8, 16, 32):
            c = cached_streaming_cost(hard, seq, window=w, base_cost=bcost,
                                      ref_cost=ref_cost, delta_io=delta_io)
            out["cache_oracle"].append({"window": w, **{k: v / GIB if k != "divergence" else v
                                                         for k, v in c.items()}})
            LOGGER.info("     janela %2d | leitura %6.2f GiB/tok | I/O %6.3f GiB/tok",
                        w, c["read_per_token"] / GIB, c["io_per_token"] / GIB)

        # F - especulativa
        LOGGER.info("F. decodificacao especulativa (sem perda), draft=%s, verificador=bf16:", bname)
        out["speculative"] = []
        for k in (1, 2, 3, 4, 6, 8):
            s = speculative_expected_cost(~hard, seq, draft_cost=bcost, verify_cost=ref_cost, k=k)
            out["speculative"].append({"k": k, "gib": s["cost_per_token"] / GIB})
            LOGGER.info("     k=%d | %6.2f GiB/tok", k, s["cost_per_token"] / GIB)
        report["bases"][bname] = out

    for cname, d in cross.items():
        hard = d["top1"] != ref_top1
        best = min((speculative_expected_cost(~hard, seq, draft_cost=d["weight_bytes"],
                                              verify_cost=ref_cost, k=k)["cost_per_token"], k)
                   for k in (1, 2, 3, 4, 6, 8))
        LOGGER.info("F. especulativa com draft %s (%.2f GiB): melhor %.2f GiB/tok (k=%d) | "
                    "concordancia %.2f%%", cname, d["weight_bytes"] / GIB, best[0] / GIB, best[1],
                    100 * (~hard).float().mean())
        report.setdefault("speculative_cross", {})[cname] = {"gib": best[0] / GIB, "k": best[1]}

    out_path = base / f"{args.model}__analysis.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    LOGGER.info("analise gravada em %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
