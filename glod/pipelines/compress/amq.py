#!/usr/bin/env python3
"""AMQ: treina e avalia a quantizacao ciente do arg-max contra GPTQ e contra o
reajuste de escalas por KL (mesmo otimizador, mesmos dados, mesma parada).

Fluxo, para um modelo:
  1. dados de treino (--train-source):
       selfgen  respostas greedy do proprio modelo denso a prompts DISJUNTOS da
                avaliacao: GSM8K train, MMLU auxiliary_train e Alpaca (o codigo fica de
                fora, como dominio nunca visto). E o regime da decodificacao
                especulativa, que rascunha as saidas do proprio modelo.
       c4       janelas de texto do C4 (validacao)
     com uma fatia de validacao separada; alvos densos (top-64) calculados antes de
     quantizar;
  2. GPTQ com a calibracao do paper (WikiText-2, 128 x 512, semente 0) -> QLinear com
     theta = 0, que E o baseline "gptq4" (os codigos sao salvos e reutilizados);
  3. para cada perda em --kinds: theta <- 0, treina, guarda o melhor ponto pelo
     objetivo da propria variante na validacao e pontua;
  4. pontuacao: o mesmo `score_fidelity` do grid, contra as referencias fp32 do modelo
     em todos os corpora disponiveis (e o holdout de codigo). Grava em
     OUT/amq/<ref>/<tag>.pt, fora do alcance das analises do paper.

    python -m glod amq --model Qwen/Qwen3-4B --device cuda:0 --kinds kl dg flip tvfrag
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

from glod.core import amq as A
from glod.core import compressors as C
from glod.core.fidelity import score_fidelity, subset_index
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import GeneratedCorpus, _chat, generate_greedy
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, ref_slug
from glod.pipelines.fidelity.build_grid import use_fp32_logits

LOGGER = logging.getLogger("glod.amq")
HOLDOUT = Path(os.environ.get("GLOD_HOLDOUT", "/local/user_beatrizalmeida/ews_results/holdout"))
CKPT = Path(os.environ.get("GLOD_AMQ_CKPT", "/local/user_beatrizalmeida/ews_results/amq_ckpt"))


def eval_refs(model_id: str) -> list[tuple[str, Path]]:
    """(slug da referencia fp32, raiz) para todos os corpora do modelo."""
    base = ref_slug(model_id, None)
    out = []
    for root in (OUT, HOLDOUT):
        if not root.exists():
            continue
        for d in sorted(os.listdir(root)):
            if d.endswith("__fp32") and (d == f"{base}__fp32" or d.startswith(f"{base}__")) \
                    and (root / d / d / "bf16.pt").exists() and (root / "corpora" / f"{d[:-6]}.json").exists():
                out.append((d, root))
    return out


def score_all(loaded, refs, tag: str, out_root: Path, batch_size: int) -> dict:
    res_all = {}
    for rs, root in refs:
        target = out_root / rs / f"{tag}.pt"
        if target.exists():
            r = torch.load(target, weights_only=False)
        else:
            corpus = GeneratedCorpus.from_dict(json.loads((root / "corpora" / f"{rs[:-6]}.json").read_text()))
            ref = torch.load(root / rs / rs / "bf16.pt", weights_only=False)
            r = score_fidelity(loaded, corpus, ref=ref, batch_size=batch_size, subset=subset_index(corpus.n_tokens, 4096))
            r["flip_rate"] = (r["top1"] != ref["top1"]).double().mean().item()
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(r, target)
        res_all[rs] = {"flip": r["flip_rate"], "kl": r["kl"].double().mean().item(), "tv": r["tv"].double().mean().item()}
        LOGGER.info("  %-18s %-34s flip %.4f  KL %.4f  TV %.4f", tag, rs, *res_all[rs].values())
    return res_all


def selfgen_prompts(tok, n_gsm: int, n_mmlu: int, n_alpaca: int, cache_dir: str, seed: int = 0) -> list[dict]:
    """Prompts disjuntos de toda avaliacao: GSM8K train, MMLU auxiliary_train, Alpaca."""
    from datasets import load_dataset
    rng = random.Random(seed)
    out = []
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    for i in rng.sample(range(len(ds)), n_gsm):
        text = ("Solve the following problem step by step. At the end, write 'Answer: <number>'.\n\n"
                + ds[i]["question"])
        out.append({"prompt": _chat(tok, text), "source": "gsm8k_train"})
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train", cache_dir=cache_dir)
    letters = "ABCD"
    for i in rng.sample(range(len(ds)), n_mmlu):
        r = ds[i]
        ch = list(r["choices"])[:4]
        if len(ch) < 4:
            continue
        text = ("Answer the question below, explaining your reasoning briefly, and finish with 'Answer: <letter>'.\n\n"
                f"Subject: {r['subject'] or 'general'}\nQuestion: {r['question']}\n"
                + "\n".join(f"{letters[j]}) {c}" for j, c in enumerate(ch)))
        out.append({"prompt": _chat(tok, text), "source": "mmlu_aux"})
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
    for i in rng.sample(range(len(ds)), n_alpaca):
        r = ds[i]
        text = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
        out.append({"prompt": _chat(tok, text), "source": "alpaca"})
    return out


def build_trainset(args, loaded) -> A.TrainSet:
    tok = loaded.tokenizer
    if args.train_source == "c4":
        return A.windows_set(C.CALIBRATION["c4"](tok, args.train_n, args.train_len, seed=args.train_seed,
                                                 cache_dir=f"{args.cache_dir}/datasets"))
    path = CKPT / ref_slug(args.model, None) / "selfgen.json"
    if path.exists():
        corpus = GeneratedCorpus.from_dict(json.loads(path.read_text()))
    else:
        recs = selfgen_prompts(tok, args.n_gsm, args.n_mmlu, args.n_alpaca, f"{args.cache_dir}/datasets")
        t0 = time.time()
        corpus = generate_greedy(loaded, recs, max_new_tokens=args.gen_tokens, batch_size=args.gen_batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(corpus.to_dict()))
        LOGGER.info("selfgen: %d respostas, %d tokens (%.0fs)", len(corpus.gen_ids), corpus.n_tokens, time.time() - t0)
    keep = [i for i, g in enumerate(corpus.gen_ids) if len(g) >= 8]
    return A.generated_set([corpus.prompt_ids[i] for i in keep], [corpus.gen_ids[i] for i in keep],
                           tok.pad_token_id if tok.pad_token_id is not None else 0, args.max_len)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--kinds", nargs="+", default=["kl", "dg", "flip", "tvfrag"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--train-source", choices=["selfgen", "c4"], default="selfgen")
    p.add_argument("--n-gsm", type=int, default=2000)
    p.add_argument("--n-mmlu", type=int, default=1500)
    p.add_argument("--n-alpaca", type=int, default=2500)
    p.add_argument("--gen-tokens", type=int, default=256)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--train-n", type=int, default=4096, help="c4: numero de janelas")
    p.add_argument("--train-len", type=int, default=512, help="c4: tamanho da janela")
    p.add_argument("--train-seed", type=int, default=1)
    p.add_argument("--score-batch", type=int, default=8)
    p.add_argument("--calib-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--rank", type=int, default=0, help="correcao de baixo posto por projecao (0 = so escalas)")
    p.add_argument("--lr-lowrank", type=float, default=1e-5, help="lr de A e B (somam direto nos pesos)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    refs = eval_refs(args.model)
    LOGGER.info("referencias de avaliacao: %s", [r[0] for r in refs])
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    use_fp32_logits(loaded.model)
    model, tok = loaded.model, loaded.tokenizer
    bank = C.WeightBank(loaded.decoder)
    out_root = OUT / "amq"
    slug = ref_slug(args.model, None)
    ck = CKPT / slug
    ck.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f"{slug}__summary{args.tag_suffix}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    torch.set_grad_enabled(False)

    t0 = time.time()
    data = build_trainset(args, loaded)
    tfile = CKPT / slug / f"dense_targets_{args.train_source}.pt"
    if tfile.exists():                            # alvos do modelo DENSO: iguais em toda rodada
        cached = torch.load(tfile, weights_only=False)
        assert cached["ids"].shape == data.ids.shape and torch.equal(cached["ids"], data.ids)
        data.topv, data.topi = cached["topv"], cached["topi"]
    else:
        A.attach_dense_targets(model, data)
        torch.save({"ids": data.ids, "topv": data.topv, "topi": data.topi}, tfile)
    train, val = data.split(args.n_val, seed=args.seed)
    LOGGER.info("treino %d seqs (%d posicoes na perda), validacao %d seqs (%.0fs)", len(train),
                int(train.lmask.sum()), len(val), time.time() - t0)

    pfile = ck / f"gptq{args.bits}_params.pt"
    if pfile.exists():
        params = torch.load(pfile, weights_only=False)
        q_mods = {t.name: A.QLinear(*[x.to(loaded.device) for x in params[t.name]], t.module.bias, 128, rank=args.rank)
                  for t in bank.targets}
        A.install_qlinear(bank, q_mods)
        LOGGER.info("codigos GPTQ reutilizados de %s", pfile)
    else:
        calib = C.CALIBRATION["wikitext"](tok, 128, 512, seed=0, cache_dir=f"{args.cache_dir}/datasets")
        q_mods = A.gptq_to_qlinear(model, bank, loaded.decoder.layers, calib, bits=args.bits, batch_size=args.calib_batch)
        torch.save({k: (m.codes.cpu(), m.scale0.cpu(), m.zero.cpu()) for k, m in q_mods.items()}, pfile)

    tag0 = f"gptq{args.bits}"
    if tag0 not in summary:
        summary[tag0] = {"eval": score_all(loaded, refs, tag0, out_root, args.score_batch),
                         "val": A.evaluate(model, val, "kl", beta=args.beta, tau=args.tau)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1))

    for spec in args.kinds:
        kind, _, b = spec.partition("@")            # "dg@0.1": beta proprio da variante
        beta = float(b) if b else args.beta
        tag = f"amq{args.bits}{'r%d' % args.rank if args.rank else ''}_{kind}{('b' + b) if b else ''}{args.tag_suffix}"
        if tag in summary and all(rs in summary[tag]["eval"] for rs, _ in refs):
            LOGGER.info("%s ja feito", tag)
            continue
        A.reset(q_mods)
        t0 = time.time()
        hist, best = A.train_scales(model, q_mods, train, val, kind=kind, steps=args.steps, batch_size=args.batch,
                                    lr=args.lr, lr_lowrank=args.lr_lowrank, beta=beta, tau=args.tau, seed=args.seed,
                                    eval_every=args.eval_every)
        secs = time.time() - t0
        A.load_state(q_mods, best["theta"])
        torch.save(best["theta"], ck / f"{tag}_theta.pt")
        summary[tag] = {"kind": kind, "beta": beta, "train_seconds": secs, "best_step": best["step"], "hist": hist,
                        "args": {k: v for k, v in vars(args).items()},
                        "eval": score_all(loaded, refs, tag, out_root, args.score_batch)}
        summary_path.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
