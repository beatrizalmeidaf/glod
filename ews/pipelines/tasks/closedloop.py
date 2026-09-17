#!/usr/bin/env python3
"""Consequencias em malha fechada da lei flips ~ sqrt(KL).

  gsm8k  geracao greedy REAL de cada configuracao em prompts GSM8K disjuntos do
         corpus de calibracao; acuracia, mudanca de resposta vs bf16.
  spec   decodificacao especulativa REAL (lote 1): alvo bf16, draft = modelo
         comprimido; greedy e amostragem (T=1, regra de rejeicao de Leviathan).
         Mede taxa de aceitacao por token proposto e tokens por rodada.

    python ews_fid_closedloop.py gsm8k --model Qwen/Qwen3-4B --configs bf16 u4 gptq3 --device cuda:0
    python ews_fid_closedloop.py spec  --model Qwen/Qwen3-4B --configs u4 gptq3 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time

import torch
from transformers import DynamicCache

from ews.core import compressors as C
from ews.core.elastic_depth import make_elastic_depth
from ews.core.model_loader import load_model
from ews.corpora.token_oracle import _chat, _eos_ids, generate_greedy
from ews.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from ews.paths import OUT, slug
from ews.pipelines.fidelity.build_grid import apply_config

LOGGER = logging.getLogger("ews.closedloop2")


def gsm8k_disjoint(tokenizer, n: int, used: set[str], cache_dir: str, seed: int = 1234) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", cache_dir=f"{cache_dir}/datasets")["test"]
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    out = []
    for i in order:
        text = ("Solve the following problem step by step. "
                "At the end, write 'Answer: <number>'.\n\n" + ds[i]["question"])
        prompt = _chat(tokenizer, text)
        if prompt in used:
            continue
        out.append({"prompt": prompt, "source": "gsm8k", "idx": i,
                    "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", "")})
        if len(out) == n:
            break
    return out


def parse_answer(text: str):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def stage_gsm8k(args) -> None:
    out_dir = OUT / "closedloop" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in args.configs if not (out_dir / f"gsm8k_{c.replace('/', '--')}.json").exists()]
    if not todo:
        return
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank = C.WeightBank(loaded.decoder)
    ctrl = make_elastic_depth(loaded.model)
    ctrl.allow_kv_cache = True  # mascara fixa durante toda a geracao
    used = {r["prompt"] for r in json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text())["records"]}
    records = gsm8k_disjoint(loaded.tokenizer, args.n, used, args.cache_dir)
    calib: dict = {}

    def calib_fn():
        if "x" not in calib:
            calib["x"] = C.wikitext_calibration(loaded.tokenizer, 128, 512, cache_dir=f"{args.cache_dir}/datasets")
        return calib["x"]

    for name in todo:
        t0 = time.time()
        recs = [dict(r) for r in records]
        with apply_config(name, loaded, bank, ctrl, calib_fn, args):
            corpus = generate_greedy(loaded, recs, max_new_tokens=args.max_new_tokens, batch_size=args.batch)
        rows = []
        for r, g in zip(corpus.records, corpus.gen_ids):
            ans = parse_answer(r["completion"])
            rows.append({"idx": r["idx"], "answer": ans, "n_tokens": len(g),
                         "correct": ans is not None and abs(ans - float(r["gold"])) < 1e-6})
        acc = sum(x["correct"] for x in rows) / len(rows)
        (out_dir / f"gsm8k_{name.replace('/', '--')}.json").write_text(json.dumps(
            {"config": name, "acc": acc, "rows": rows, "completions": [r["completion"] for r in corpus.records]}))
        LOGGER.info("[%s] GSM8K acc %.4f (n=%d) %.0fs", name, acc, len(rows), time.time() - t0)


# --------------------------------------------------------------- especulativa
@torch.inference_mode()
def _feed(model, cache: DynamicCache, tokens: list[int], device) -> torch.Tensor:
    start = cache.get_seq_length()
    ids = torch.tensor([tokens], device=device)
    pos = torch.arange(start, start + len(tokens), device=device)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=pos,
                position_ids=pos[None])
    return out.logits[0].double()


@torch.inference_mode()
def speculative(target, draft, prompt_ids: list[int], *, k: int, max_new: int, eos: set[int],
                sampling: bool, gen: torch.Generator) -> dict:
    dev = target.device
    tc, dc = DynamicCache(), DynamicCache()
    seq = list(prompt_ids)
    new = 0
    proposed = accepted = rounds = 0
    accept_probs: list[float] = []
    while new < max_new:
        # draft propoe k tokens
        drafts, qs = [], []
        logits = _feed(draft.model, dc, seq[dc.get_seq_length():], dev)[-1]
        for j in range(k):
            q = logits.softmax(-1)
            d = int(torch.multinomial(q.float(), 1, generator=gen)) if sampling else int(q.argmax())
            drafts.append(d); qs.append(q)
            if j < k - 1:
                logits = _feed(draft.model, dc, [d], dev)[-1]
        # alvo verifica em um forward
        tl = _feed(target.model, tc, seq[tc.get_seq_length():] + drafts, dev)[-(k + 1):]
        ps = tl.softmax(-1)
        V = ps.shape[-1]
        # draft de outro tamanho da familia pode ter vocabulario menor (Gemma 1B: 262144 vs 262208)
        qs = [q if q.shape[-1] == V else torch.nn.functional.pad(q, (0, V - q.shape[-1])) for q in qs]
        i = 0
        bonus = None
        while i < k:
            if sampling:
                p_x, q_x = ps[i, drafts[i]].item(), qs[i][drafts[i]].item()
                a = min(1.0, p_x / max(q_x, 1e-300))
                accept_probs.append(a)
                if torch.rand(1, generator=gen, device=dev).item() < a:
                    i += 1
                    continue
                resid = (ps[i] - qs[i]).clamp(min=0)
                bonus = int(torch.multinomial((resid / resid.sum()).float(), 1, generator=gen))
                break
            if int(ps[i].argmax()) == drafts[i]:
                i += 1
                continue
            bonus = int(ps[i].argmax())
            break
        if bonus is None:
            bonus = int(torch.multinomial(ps[k].float(), 1, generator=gen)) if sampling else int(ps[k].argmax())
        commit = drafts[:i] + [bonus]
        proposed += k
        accepted += i
        rounds += 1
        stop = False
        for t in commit:
            seq.append(t)
            new += 1
            if t in eos or new >= max_new:
                stop = True
                break
        tc.crop(len(seq) - 1)
        dc.crop(min(dc.get_seq_length(), len(seq) - 1))
        if stop:
            break
    return {"proposed": proposed, "accepted": accepted, "rounds": rounds, "new_tokens": new,
            "gen": seq[len(prompt_ids):], "accept_probs": accept_probs}


def stage_spec(args) -> None:
    out_dir = OUT / "closedloop" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = load_model(args.model, role="ref", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    draft = load_model(args.draft_model or args.model, role="draft", device=args.device, dtype="bfloat16",
                       cache_dir=args.cache_dir)
    bank = C.WeightBank(draft.decoder)
    ctrl = make_elastic_depth(draft.model)
    eos = _eos_ids(target)
    used = {r["prompt"] for r in json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text())["records"]}
    records = gsm8k_disjoint(target.tokenizer, args.n, used, args.cache_dir, seed=4321)
    calib: dict = {}

    def calib_fn():
        if "x" not in calib:
            calib["x"] = C.wikitext_calibration(draft.tokenizer, 128, 512, cache_dir=f"{args.cache_dir}/datasets")
        return calib["x"]

    for name in args.configs:
        tag = f"spec_{slug(args.draft_model) + '_' if args.draft_model else ''}{name.replace('/', '--')}.json"
        if (out_dir / tag).exists():
            continue
        res = {}
        with apply_config(name, draft, bank, ctrl, calib_fn, args):
            for mode in ("greedy", "sample"):
                gen = torch.Generator(device=args.device).manual_seed(0)
                tot = {"proposed": 0, "accepted": 0, "rounds": 0, "new_tokens": 0, "accept_probs": []}
                gens = []
                for r in records:
                    ids = target.tokenizer(r["prompt"], add_special_tokens=False).input_ids
                    o = speculative(target, draft, ids, k=args.k, max_new=args.max_new_tokens, eos=eos,
                                    sampling=mode == "sample", gen=gen)
                    for key in ("proposed", "accepted", "rounds", "new_tokens"):
                        tot[key] += o[key]
                    tot["accept_probs"] += o["accept_probs"]
                    gens.append(o["gen"])
                res[mode] = {"accept_rate": tot["accepted"] / tot["proposed"],
                             "tokens_per_round": tot["new_tokens"] / tot["rounds"],
                             "mean_accept_prob": (sum(tot["accept_probs"]) / len(tot["accept_probs"]))
                             if tot["accept_probs"] else None,
                             "gens": gens, **{k_: tot[k_] for k_ in ("proposed", "accepted", "rounds", "new_tokens")}}
                LOGGER.info("[%s|%s] aceitacao %.4f | tokens/rodada %.3f", name, mode,
                            res[mode]["accept_rate"], res[mode]["tokens_per_round"])
        (out_dir / tag).write_text(json.dumps({"config": name, "k": args.k, **res}))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gsm8k", "spec"])
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", default=None)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--calib-batch", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    {"gsm8k": stage_gsm8k, "spec": stage_spec}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
