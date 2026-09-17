#!/usr/bin/env python3
"""EWS | Fase 1c - heterogeneidade de capacidade por token em geracao.

Estagios:
    gen    - o modelo de referencia (bf16) gera greedy sobre GSM8K + MMLU-PT.
    score  - cada configuracao barata e avaliada com teacher forcing no corpus.
    (analise: ews_tokens_analyze.py, somente CPU)

Configuracoes (`--configs`):
    bf16                    referencia
    u8 u4 u3 u2             quantizacao uniforme (RTN afim, grupo 128)
    q3+Nx4 / q3+Nx16        base 3-bit com N camadas promovidas (ordem MMLU)
    skipN                   elastic depth, N camadas desligadas (ordem espacada)

Exemplos:
    python ews_tokens.py gen   --model google/gemma-3-12b-it --device cuda:0
    python ews_tokens.py score --model google/gemma-3-12b-it --device cuda:0 --configs bf16 u4 u3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch

from ews.core.elastic_depth import make_elastic_depth
from ews.core.model_loader import DEFAULT_CACHE_DIR, load_model
from ews.core.oracle import select_skipped
from ews.core.quantize import QuantizationSimulator
from ews.corpora.token_oracle import (
    GeneratedCorpus,
    build_prompts_gsm8k,
    build_prompts_mmlu_pt,
    generate_greedy,
    score_tokens,
)

LOGGER = logging.getLogger("ews.tokens")
from ews.paths import RAW
OUT = RAW / "tokens"


def slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def corpus_path(model_id: str) -> Path:
    return OUT / f"{slug(model_id)}__corpus.json"


def set_out_dir(path: str) -> None:
    global OUT
    OUT = Path(path)


def extract_answer(record: dict) -> bool | None:
    text = record.get("completion", "")
    if record["source"] == "gsm8k":
        m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
        if not m:
            return None
        try:
            return abs(float(m[-1].replace(",", "")) - float(record["gold"])) < 1e-6
        except ValueError:
            return None
    m = re.findall(r"Resposta:\s*\**\(?([ABCD])", text)
    return (m[-1] == record["gold"]) if m else None


# --------------------------------------------------------------------- gen
def stage_gen(args: argparse.Namespace) -> None:
    loaded = load_model(args.model, role="ref", device=args.device, dtype="bfloat16",
                        cache_dir=args.cache_dir)
    records = (
        build_prompts_gsm8k(loaded.tokenizer, args.n_prompts, args.seed,
                            f"{args.cache_dir}/datasets")
        + build_prompts_mmlu_pt(loaded.tokenizer, args.mmlu_path, args.n_prompts, args.seed)
    )
    corpus = generate_greedy(loaded, records, max_new_tokens=args.max_new_tokens,
                             batch_size=args.gen_batch)
    for r in corpus.records:
        r["correct"] = extract_answer(r)

    by_src: dict[str, list] = {}
    for r, g in zip(corpus.records, corpus.gen_ids):
        by_src.setdefault(r["source"], []).append((r, g))
    for src, items in by_src.items():
        parsed = [r["correct"] for r, _ in items if r["correct"] is not None]
        n_tok = sum(len(g) for _, g in items)
        truncated = sum(1 for _, g in items if len(g) >= args.max_new_tokens)
        LOGGER.info("%s: %d seqs | %d tokens (media %.0f) | %d truncadas | acc %.1f%% (%d parseadas)",
                    src, len(items), n_tok, n_tok / len(items), truncated,
                    100 * sum(parsed) / max(len(parsed), 1), len(parsed))

    OUT.mkdir(parents=True, exist_ok=True)
    corpus_path(args.model).write_text(json.dumps(
        {"model": args.model, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
         **corpus.to_dict()}))
    LOGGER.info("corpus gravado em %s (%d tokens)", corpus_path(args.model), corpus.n_tokens)


# ------------------------------------------------------------------- score
def ladder_orders(model_id: str) -> dict[str, list[int]]:
    """Ordens estaticas medidas no MMLU (Fases 1a/1b), so existem para o 12B."""
    orders: dict[str, list[int]] = {}
    if "12b" not in model_id:
        return orders
    raw = Path("results/raw")
    for key, fname, field in (("x4", "qoracle_3b_4b.json", "sensitivity"),
                              ("x16", "qoracle_3b_16b.json", "sensitivity")):
        if (raw / fname).exists():
            orders[key] = json.loads((raw / fname).read_text())[field]["order_descending"]
    if (raw / "oracle_phase1.json").exists():
        orders["skip"] = json.loads((raw / "oracle_phase1.json").read_text())[
            "layer_importance"]["order_ascending"]
    return orders


def stage_score(args: argparse.Namespace) -> None:
    corpus_src = args.corpus_of or args.model
    payload = json.loads(corpus_path(corpus_src).read_text())
    corpus = GeneratedCorpus.from_dict(payload)
    LOGGER.info("corpus de %s: %d seqs, %d tokens", corpus_src, len(corpus.gen_ids), corpus.n_tokens)

    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16",
                        cache_dir=args.cache_dir)
    sim = QuantizationSimulator(loaded.decoder, group_size=128, storage_device="cpu")
    ctrl = make_elastic_depth(loaded.model)
    non_target = loaded.param_bytes - sum(t.numel * 2 for t in sim.targets)
    orders = ladder_orders(args.model)
    n_layers = sim.num_layers
    out_dir = OUT / f"{slug(corpus_src)}__corpus" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.configs:
        target = out_dir / f"{name}.pt"
        if target.exists() and not args.overwrite:
            LOGGER.info("[%s] ja existe, pulando", name)
            continue
        skipped: list[int] = []
        if name == "bf16":
            bits = {}
        elif m := re.fullmatch(r"u(\d+)", name):
            bits = {i: int(m.group(1)) for i in range(n_layers)}
        elif m := re.fullmatch(r"q(\d+)\+(\d+)x(\d+)", name):
            base, k, tgt = map(int, m.groups())
            order = orders[f"x{tgt}"]
            promoted = set(order[:k])
            bits = {i: (tgt if i in promoted else base) for i in range(n_layers)}
        elif m := re.fullmatch(r"p(\d+)to(\d+)", name):
            # prefixo quantizado: camadas [0, K) em `bits`, o resto em bf16.
            # Mede a fidelidade REAL de um early-probe que decide na camada K.
            pbits, k = int(m.group(1)), int(m.group(2))
            bits = {i: pbits for i in range(k)}
        elif m := re.fullmatch(r"skip(\d+)", name):
            bits = {}
            skipped = select_skipped(orders["skip"], int(m.group(1)), min_gap=3)
        else:
            raise ValueError(f"configuracao desconhecida: {name}")

        sim.apply(bits)
        if skipped:  # configuracoes de skip sao bf16; camadas desligadas nao sao lidas
            weight_bytes = loaded.param_bytes - sum(ctrl.layer_param_bytes[l] for l in skipped)
        else:
            weight_bytes = sim.weight_bytes(bits, non_target_bytes=non_target)

        capture = args.capture_layers if name in args.capture_for else []
        with ctrl.skipping(skipped):
            scores = score_tokens(loaded, corpus, batch_size=args.batch_size,
                                  capture_layers=capture, desc=name)
        torch.save({"name": name, "model": args.model, "corpus_of": corpus_src,
                    "weight_bytes": float(weight_bytes), "bits": bits, "skipped": skipped,
                    **scores.to_dict()}, target)
        LOGGER.info("[%s] %.2f GiB/token | salvo em %s", name, weight_bytes / 1024**3, target)
    sim.restore()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "score"])
    p.add_argument("--model", required=True)
    p.add_argument("--corpus-of", default=None,
                   help="avaliar sobre o corpus gerado por OUTRO modelo (ex.: 4B sobre o do 12B).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--mmlu-path", default=MMLU_PT_CSV)
    p.add_argument("--n-prompts", type=int, default=256, help="prompts POR fonte")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--configs", nargs="+", default=["bf16"])
    p.add_argument("--capture-layers", type=int, nargs="*", default=[])
    p.add_argument("--capture-for", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-dir", default="results/raw/tokens")
    args = p.parse_args(argv)
    set_out_dir(args.out_dir)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    {"gen": stage_gen, "score": stage_score}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
