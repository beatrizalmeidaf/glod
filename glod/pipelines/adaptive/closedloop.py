#!/usr/bin/env python3
"""Validacao em GERACAO REAL da precisao adaptativa por token, com KV cache misto e consistente.

O resultado adaptativo de `glod adaptive-bits` e teacher forcing (cada precisao com o
proprio prefixo). Aqui a precisao troca por token durante a geracao greedy e o KV de cada
posicao vem da precisao que de fato a processou - as duas copias do modelo compartilham,
posicao a posicao, o mesmo cache misto, como num sistema GLOD.

Copias: base (RTN --base-bits) e alta (RTN --high-bits, 16 = bf16). A cada passo ambas
processam o token (simulacao); a politica escolhe qual saida vale e qual KV fica.

Politicas (limiares = quantis da entropia da base no corpus de calibracao):
  static_base / static_high      tudo numa precisao
  mix@f                          precisao mista ESTATICA: fracao f das projecoes em alta
  cascade@r                      entropia da saida da base > limiar -> refaz em alta
                                 (custo de computacao: base + promovidos)
  predictive@r                   entropia do passo ANTERIOR > limiar -> roda em alta,
                                 decidido antes do forward (custo so incremental)
Prefill do prompt na base.

    python -m glod adaptive-closedloop --model Qwen/Qwen3-4B --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time

import torch

from glod.core import compressors as C
from glod.core.closed_loop import _Stepper, _entropy
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import _eos_ids
from glod.pipelines.tasks.closedloop import gsm8k_disjoint, parse_answer
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, slug

LOGGER = logging.getLogger("glod.adapt_cl")


def eff_bits(b: int) -> float:
    return 16.0 if b >= 16 else b + 0.25


@torch.no_grad()
def merge_last(dst: _Stepper, src: _Stepper, mask: torch.Tensor) -> None:
    """Na ultima posicao do cache, onde mask=True copia o KV de src para dst."""
    m = mask.to(dst.device)[:, None, None]
    for ld, ls in zip(dst.cache.layers, src.cache.layers):
        ld.keys[:, :, -1, :] = torch.where(m, ls.keys[:, :, -1, :], ld.keys[:, :, -1, :])
        ld.values[:, :, -1, :] = torch.where(m, ls.values[:, :, -1, :], ld.values[:, :, -1, :])


@torch.no_grad()
def generate_policy(base, high, prompts, *, policy: str, thr: float, eos, max_new: int,
                    prefill: str = "base") -> dict:
    tok = base.tokenizer
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = prev
    b = _Stepper(base, enc["input_ids"], enc["attention_mask"])
    h = _Stepper(high, enc["input_ids"], enc["attention_mask"]) if high is not None else None
    if h is not None:  # prompt processado numa precisao so; a outra copia herda esse KV
        src, dst = (b, h) if prefill == "base" else (h, b)
        for ld, ls in zip(dst.cache.layers, src.cache.layers):
            ld.keys.copy_(ls.keys)
            ld.values.copy_(ls.values)
        if prefill == "high":
            b.logits = h.logits.clone()
    n = len(prompts)
    done = torch.zeros(n, dtype=torch.bool)
    gen = [[] for _ in range(n)]
    promoted = [[] for _ in range(n)]
    eos_t = torch.tensor(sorted(eos))
    prev_H = _entropy(b.logits).cpu()
    for _ in range(max_new):
        Hb = _entropy(b.logits).cpu()
        if policy == "cascade":
            promote = Hb > thr
        elif policy == "predictive":
            promote = prev_H > thr
        else:
            promote = torch.zeros(n, dtype=torch.bool)
        logits = b.logits if h is None else torch.where(promote.to(b.device)[:, None], h.logits, b.logits)
        nxt = logits.argmax(-1).cpu()
        nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
        for i in range(n):
            if not done[i]:
                gen[i].append(int(nxt[i]))
                promoted[i].append(bool(promote[i]))
        prev_H = torch.where(promote, _entropy(h.logits).cpu(), Hb) if h is not None else Hb
        done |= torch.isin(nxt, eos_t)
        if bool(done.all()):
            break
        # o KV do token que ACABOU de ser processado fica da precisao escolhida neste passo
        if h is not None:
            merge_last(b, h, promote)
            merge_last(h, b, ~promote)
        b.step(nxt)
        if h is not None:
            h.step(nxt)
    return {"gen": gen, "promoted": promoted}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--base-bits", type=int, default=3)
    p.add_argument("--high-bits", type=int, default=8)
    p.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35])
    p.add_argument("--mixes", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.5])
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=320)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    torch.cuda.set_device(torch.device(args.device))
    tag = f"b{args.base_bits}_h{args.high_bits}"
    out_f = OUT / "adaptive_closedloop" / slug(args.model) / f"{tag}.json"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_f.read_text()) if out_f.exists() else {}

    base = load_model(args.model, role="base", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    high = load_model(args.model, role="high", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank_b, bank_h = C.WeightBank(base.decoder), C.WeightBank(high.decoder)
    eos = _eos_ids(base)
    corpus = json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text())
    used = {r["prompt"] for r in corpus["records"]}
    recs = gsm8k_disjoint(base.tokenizer, args.n, used, args.cache_dir, seed=2718)
    # limiares: quantis da entropia da base (RTN base_bits) no corpus de calibracao (teacher forcing)
    cal = torch.load(OUT / slug(args.model) / slug(args.model) / f"u{args.base_bits}.pt",
                     map_location="cpu", mmap=True, weights_only=False)["entropy"].double()
    thr = {r: torch.quantile(cal, 1 - r).item() for r in args.rates}

    def set_bits(bank, b):
        bank.restore()
        if b < 16:
            bank.map(lambda w, t: C.rtn(w, b))

    def set_mix(bank, frac):
        bank.restore()
        order = list(range(len(bank.targets)))
        random.Random(0).shuffle(order)
        hi = set(order[: int(round(frac * len(order)))])
        for i, t in enumerate(bank.targets):
            w = t.original.to(t.module.weight.device)
            bb = args.high_bits if i in hi else args.base_bits
            t.module.weight.copy_((w if bb >= 16 else C.rtn(w, bb)).to(t.module.weight.dtype))

    def run(name, policy, threshold, use_high, bits_fixed=None) -> None:
        if name in res:
            return
        t0 = time.time()
        gens, proms = [], []
        for s in range(0, len(recs), args.batch):
            chunk = [r["prompt"] for r in recs[s:s + args.batch]]
            o = generate_policy(base, high if use_high else None, chunk, policy=policy, thr=threshold, eos=eos,
                                max_new=args.max_new_tokens)
            gens += o["gen"]; proms += o["promoted"]
        ok, n_tok, n_prom = [], 0, 0
        for r, g, pr in zip(recs, gens, proms):
            a = parse_answer(base.tokenizer.decode(g, skip_special_tokens=True))
            ok.append(a is not None and abs(a - float(r["gold"])) < 1e-6)
            n_tok += len(pr); n_prom += sum(pr)
        rate = n_prom / max(n_tok, 1)
        bb, hb = eff_bits(args.base_bits), eff_bits(args.high_bits)
        bi = bb + rate * (hb - bb) if bits_fixed is None else bits_fixed
        bc = bb + rate * hb if bits_fixed is None else bits_fixed
        res[name] = {"policy": policy, "acc": sum(ok) / len(ok), "promote_rate": rate,
                     "bits_incremental": bi, "bits_cascade": bc,
                     "correct": ok, "seconds": time.time() - t0}
        out_f.write_text(json.dumps(res, indent=1))
        LOGGER.info("[%s] acc %.4f | promove %.3f | bits incr %.2f | bits cascata %.2f | %.0fs", name,
                    res[name]["acc"], rate, res[name]["bits_incremental"], res[name]["bits_cascade"], time.time() - t0)

    set_bits(bank_b, args.base_bits)
    run("static_base", "static", float("inf"), False)
    set_bits(bank_h, args.high_bits)
    for r in args.rates:
        run(f"cascade@{r}", "cascade", thr[r], True)
        run(f"predictive@{r}", "predictive", thr[r], True)
    bank_h.restore()
    for f in args.mixes:  # precisao mista estatica na copia base (sem troca por token)
        set_mix(bank_b, f)
        bb, hb = eff_bits(args.base_bits), eff_bits(args.high_bits)
        run(f"mix@{f}", "static", float("inf"), False, bits_fixed=bb + f * (hb - bb))
    set_bits(bank_b, args.high_bits)
    run("static_high", "static", float("inf"), False, bits_fixed=eff_bits(args.high_bits))
    out_f.write_text(json.dumps(res, indent=1))
    print(f"\n{args.model} {tag}: acuracia GSM8K (n={args.n}) x bits medios")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["bits_incremental"]):
        print(f"   {k:18s} acc {100*v['acc']:5.1f} | bits incr {v['bits_incremental']:5.2f} | cascata {v['bits_cascade']:5.2f} "
              f"| promove {100*v['promote_rate']:4.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
