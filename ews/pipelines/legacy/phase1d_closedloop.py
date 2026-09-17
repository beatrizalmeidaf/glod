#!/usr/bin/env python3
"""EWS | Fase 1d - acuracia de tarefa em malha fechada x pesos lidos por token.

Produz o grafico central do `ews_ideia.md`: acuracia x custo de decode para
  * 4B denso (bf16 e 4-bit)           - o baseline "modelo pequeno"
  * 12B denso e 12B quantizado estatico
  * 12B + gate EWS em cascata (base u3/u4, promove pela entropia)
A decodificacao especulativa e sem perda: sua acuracia e a do 12B bf16, e seu
custo vem da analise por token (ews_tokens_analyze.py).

Os limiares do gate sao quantis da entropia medida por teacher forcing no
corpus da Fase 1c; os prompts avaliados aqui sao DISJUNTOS daquele corpus.

    python ews_closedloop.py static  --model google/gemma-3-12b-it --config u3 --device cuda:0
    python ews_closedloop.py cascade --base u3 --rates 0.05 0.1 0.2 --device cuda:0 --ref-device cuda:3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys

import torch

from ews.core.closed_loop import cascade_generate
from ews.core.model_loader import DEFAULT_CACHE_DIR, load_model
from ews.core.oracle import wilson_interval
from ews.core.quantize import QuantizationSimulator
from ews.corpora.token_oracle import _chat, _eos_ids, build_prompts_mmlu_pt
from ews.pipelines.legacy.phase1c_tokens import extract_answer, slug

LOGGER = logging.getLogger("ews.closedloop")
MAX_NEW_TOKENS = 384
from ews.paths import RAW
OUT = RAW / "closedloop"
TOKENS = RAW / "tokens"


def disjoint_prompts(tokenizer, n: int, cache_dir: str, mmlu_path: str, used: set[str]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", cache_dir=f"{cache_dir}/datasets")["test"]
    order = list(range(len(ds)))
    random.Random(1234).shuffle(order)
    gsm = []
    for i in order:
        q = ds[i]["question"]
        text = ("Solve the following problem step by step. "
                "At the end, write 'Answer: <number>'.\n\n" + q)
        prompt = _chat(tokenizer, text)
        if prompt in used:
            continue
        gsm.append({"prompt": prompt, "source": "gsm8k",
                    "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", "")})
        if len(gsm) == n:
            break
    mmlu = [r for r in build_prompts_mmlu_pt(tokenizer, mmlu_path, 4 * n, seed=1234)
            if r["prompt"] not in used][:n]
    return gsm + mmlu


def flexible_correct(record: dict) -> bool:
    """Extracao flexivel: formato estrito se houver; senao, ultimo numero/letra.

    O criterio estrito mistura 'nao seguiu o formato' com 'errou', e isso pesa
    desigualmente entre modelos (o 4B descumpre o formato ~3x mais que o 12B).
    Reportamos os dois; comparacoes entre modelos usam o flexivel.
    """
    strict = extract_answer(record)
    if strict is not None:
        return strict
    text = record.get("completion", "")
    if record["source"] == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if not nums:
            return False
        try:
            return abs(float(nums[-1].replace(",", "").rstrip(".")) - float(record["gold"])) < 1e-6
        except ValueError:
            return False
    letters = re.findall(r"\b([ABCD])\b", text)
    return bool(letters) and letters[-1] == record["gold"]


def score_run(records: list[dict], gen_ids: list[list[int]], tokenizer) -> dict:
    out = {}
    for r, g in zip(records, gen_ids):
        r["completion"] = tokenizer.decode(g, skip_special_tokens=True)
        r["correct"] = extract_answer(r)
        r["correct_flexible"] = flexible_correct(r)
        r["n_tokens"] = len(g)
    for src in ("gsm8k", "mmlu_pt"):
        rows = [r for r in records if r["source"] == src]
        k = sum(1 for r in rows if r["correct"])  # nao parseada conta como erro
        kf = sum(1 for r in rows if r["correct_flexible"])
        lo, hi = wilson_interval(k, len(rows))
        flo, fhi = wilson_interval(kf, len(rows))
        out[src] = {"acc": k / len(rows), "ci": [lo / 100, hi / 100], "n": len(rows),
                    "unparsed": sum(1 for r in rows if r["correct"] is None),
                    "acc_flexible": kf / len(rows), "ci_flexible": [flo / 100, fhi / 100],
                    "truncated": sum(1 for r in rows if r["n_tokens"] >= MAX_NEW_TOKENS)}
    out["records"] = [{k: r[k] for k in ("source", "gold", "completion", "correct",
                                         "correct_flexible", "n_tokens")} for r in records]
    return out


def build_bits(name: str, n_layers: int) -> dict[int, int]:
    return {} if name == "bf16" else {i: int(name[1:]) for i in range(n_layers)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["static", "cascade"])
    p.add_argument("--model", default="google/gemma-3-12b-it")
    p.add_argument("--corpus-of", default="google/gemma-3-12b-it")
    p.add_argument("--config", default="bf16", help="static: bf16 | u8 | u4 | u3")
    p.add_argument("--base", default="u3", help="cascade: base quantizada do mesmo modelo")
    p.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ref-device", default="cuda:3")
    p.add_argument("--n", type=int, default=256, help="prompts por fonte")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--mmlu-path", default=MMLU_PT_CSV)
    args = p.parse_args(argv)
    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = args.max_new_tokens
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)

    used = {r["prompt"] for r in json.loads(
        (TOKENS / f"{slug(args.corpus_of)}__corpus.json").read_text())["records"]}

    base = load_model(args.model, role="base", device=args.device, dtype="bfloat16",
                      cache_dir=args.cache_dir)
    sim = QuantizationSimulator(base.decoder, group_size=128, storage_device="cpu")
    non_target = base.param_bytes - sum(t.numel * 2 for t in sim.targets)
    eos = _eos_ids(base)
    records = disjoint_prompts(base.tokenizer, args.n, args.cache_dir, args.mmlu_path, used)
    LOGGER.info("%d prompts disjuntos do corpus de calibracao", len(records))

    def run(threshold: float, ref) -> tuple[list[list[int]], float]:
        gens, prom = [], []
        for s in range(0, len(records), args.batch):
            chunk = [r["prompt"] for r in records[s : s + args.batch]]
            res = cascade_generate(base, ref, chunk, threshold=threshold, eos_ids=eos,
                                   max_new_tokens=args.max_new_tokens)
            gens += res.gen_ids
            prom += [x for row in res.promoted for x in row]
            LOGGER.info("  lote %d/%d", s // args.batch + 1, (len(records) + args.batch - 1) // args.batch)
        return gens, (sum(prom) / max(len(prom), 1))

    if args.mode == "static":
        bits = build_bits(args.config, sim.num_layers)
        sim.apply(bits)
        cost = sim.weight_bytes(bits, non_target_bytes=non_target)
        gens, _ = run(float("inf"), None)
        acc = score_run([dict(r) for r in records], gens, base.tokenizer)
        tag = f"{slug(args.model)}__{args.config}"
        payload = {"tag": tag, "gib": cost / 1024**3, "promote_rate": 0.0, **acc}
        (OUT / f"{tag}.json").write_text(json.dumps(payload, indent=2))
        LOGGER.info("%s | %.2f GiB/tok | gsm8k %.1f%% | mmlu_pt %.1f%%", tag, payload["gib"],
                    100 * acc["gsm8k"]["acc"], 100 * acc["mmlu_pt"]["acc"])
        return 0

    bits = build_bits(args.base, sim.num_layers)
    sim.apply(bits)
    base_cost = sim.weight_bytes(bits, non_target_bytes=non_target)
    ref = load_model(args.model, role="ref", device=args.ref_device, dtype="bfloat16",
                     cache_dir=args.cache_dir)
    ref_cost = ref.param_bytes
    ent = torch.load(TOKENS / f"{slug(args.corpus_of)}__corpus" / slug(args.model)
                     / f"{args.base}.pt")["entropy"].double()
    for rate in args.rates:
        thr = torch.quantile(ent, 1 - rate).item()
        gens, realized = run(thr, ref)
        acc = score_run([dict(r) for r in records], gens, base.tokenizer)
        cost = base_cost + realized * ref_cost
        tag = f"{slug(args.model)}__cascade_{args.base}_r{rate:.2f}"
        payload = {"tag": tag, "gib": cost / 1024**3, "promote_rate": realized,
                   "threshold": thr, "target_rate": rate, **acc}
        (OUT / f"{tag}.json").write_text(json.dumps(payload, indent=2))
        LOGGER.info("%s | promove %.1f%% | %.2f GiB/tok | gsm8k %.1f%% | mmlu_pt %.1f%%", tag,
                    100 * realized, payload["gib"], 100 * acc["gsm8k"]["acc"],
                    100 * acc["mmlu_pt"]["acc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
