#!/usr/bin/env python3
"""Estudo de fidelidade de decisao: flips de top-1 x KL para compressores reais.

Estagios:
  gen    corpus greedy da referencia (GSM8K + MMLU-PT, 256 prompts cada)
  score  teacher forcing de cada configuracao contra a referencia

Gramatica de configuracoes:
  bf16                 referencia (so no proprio modelo de referencia)
  raw                  modelo sem perturbacao contra OUTRA referencia (--ref-model)
  uB  gB  sNgB         RTN g128 / ruido gaussiano de mesma variancia (seed N)
  magP                 poda por magnitude P% por linha
  skipA_B              remove camadas
  lqB_L                so a camada L em RTN B bits
  kvB / kvtB           KV cache fake-quant B bits: KIVI (chaves por canal) / ingenuo (chaves por token)
  gptqB awqB           GPTQ / AWQ, B bits, g128, calibracao WikiText-2 (128x512)
  sgptP sgpt24         SparseGPT P% / 2:4
  wandaP wanda24       Wanda P% / 2:4
  ofc@<repo>           checkpoint GPTQ/AWQ oficial desquantizado (ex.: ofc@Qwen/Qwen2.5-7B-Instruct-AWQ)

    python ews_fidelity.py gen   --model Qwen/Qwen3-4B --device cuda:0
    python ews_fidelity.py score --model Qwen/Qwen3-4B --device cuda:0 --configs bf16 u4 gptq4
    python ews_fidelity.py score --model Qwen/Qwen3-4B --ref-model Qwen/Qwen3-14B --configs raw u4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch

from ews.core import compressors as C
from ews.core.elastic_depth import make_elastic_depth
from ews.core.fidelity import score_fidelity, subset_index
from ews.core.model_loader import load_model
from ews.corpora.registry import build_records
from ews.corpora.token_oracle import GeneratedCorpus, generate_greedy
from ews.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from ews.paths import OUT, add_corpus_arg, ref_slug

LOGGER = logging.getLogger("ews.fidelity")


def stage_gen(args) -> None:
    path = OUT / "corpora" / f"{ref_slug(args.model, args.revision, corpus=args.corpus)}.json"
    if path.exists():
        LOGGER.info("corpus existe: %s", path)
        return
    loaded = load_model(args.model, role="ref", device=args.device, dtype="bfloat16",
                        cache_dir=args.cache_dir, revision=args.revision)
    records = build_records(args.corpus, loaded.tokenizer, args.n_prompts, args.seed,
                            f"{args.cache_dir}/datasets")
    corpus = generate_greedy(loaded, records, max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": args.model, "revision": args.revision,
                                "corpus": args.corpus, **corpus.to_dict()}))
    LOGGER.info("corpus %s: %d seqs, %d tokens", path, len(corpus.gen_ids), corpus.n_tokens)


@contextlib.contextmanager
def apply_config(name: str, loaded, bank: C.WeightBank, ctrl, calib_fn, args):
    """Aplica a configuracao, cede, e restaura. Devolve metadados em dict."""
    meta: dict = {}
    kv_ctx = contextlib.nullcontext()
    skipped: list[int] = []
    bank.restore()
    dec = loaded.decoder
    if name in ("bf16", "raw"):
        pass
    elif m := re.fullmatch(r"u(\d+)", name):
        b = int(m.group(1)); bank.map(lambda w, t: C.rtn(w, b))
    elif m := re.fullmatch(r"(?:s(\d+))?g(\d+)", name):
        gen = torch.Generator(device=loaded.device).manual_seed(int(m.group(1) or 0))
        b = int(m.group(2)); bank.map(lambda w, t: C.gaussian_like_rtn(w, b, gen))
    elif m := re.fullmatch(r"mag(\d+)", name):
        f = int(m.group(1)) / 100; bank.map(lambda w, t: C.magnitude_prune(w, f))
    elif m := re.fullmatch(r"skip([\d_]+)", name):
        skipped = [int(x) for x in m.group(1).split("_")]
    elif m := re.fullmatch(r"lq(\d+)_(\d+)", name):
        b, L = int(m.group(1)), int(m.group(2)); bank.map(lambda w, t: C.rtn(w, b), layers=[L])
    elif m := re.fullmatch(r"kv(t?)(\d+)", name):
        kv_ctx = C.kv_fake_quant(loaded.model, dec, int(m.group(2)), per_channel_keys=not m.group(1))
    elif m := re.fullmatch(r"(gptq|awq)(\d+)", name):
        t0 = time.time()
        meta = C.calibrated_compress(loaded.model, bank, dec.layers, calib_fn(), m.group(1),
                                     bits=int(m.group(2)), batch_size=args.calib_batch)
        meta["seconds"] = time.time() - t0
    elif m := re.fullmatch(r"(sgpt|wanda)(\d+)", name):
        method = "sparsegpt" if m.group(1) == "sgpt" else "wanda"
        kw = dict(prunen=2, prunem=4) if m.group(2) == "24" else dict(sparsity=int(m.group(2)) / 100)
        t0 = time.time()
        meta = C.calibrated_compress(loaded.model, bank, dec.layers, calib_fn(), method,
                                     batch_size=args.calib_batch, **kw)
        meta["seconds"] = time.time() - t0
    elif name.startswith("ofc@"):
        meta = C.load_official_quantized(bank, name[4:], args.cache_dir)
    else:
        raise ValueError(f"configuracao desconhecida: {name}")
    try:
        with kv_ctx, ctrl.skipping(skipped):
            yield meta
    finally:
        bank.restore()


def use_fp32_logits(model) -> None:
    """lm_head em float32: logits continuos. Em bf16 os logits caem numa grade de
    0.0625-0.25 nat (magnitudes 10-40) e ~0.85% dos passos greedy tem EMPATE exato
    entre top-1 e top-2 - desempates dominam os flips em perturbacoes pequenas."""
    head = model.get_output_embeddings()
    if head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        head = torch.nn.Linear(head.in_features, head.out_features, bias=False, device=head.weight.device)
        head.weight = torch.nn.Parameter(model.get_input_embeddings().weight.detach().float(), requires_grad=False)
        model.set_output_embeddings(head)
    else:
        head.float()
    head.register_forward_pre_hook(lambda _m, inp: (inp[0].float(),))


def stage_score(args) -> None:
    ref_model = args.ref_model or args.model
    ref_rev = args.ref_revision if args.ref_model else args.revision
    rs = ref_slug(ref_model, ref_rev, corpus=args.corpus)
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{rs}.json").read_text()))
    if args.logits_fp32:
        rs += "__fp32"
    out_dir = OUT / rs / (ref_slug(args.model, args.revision, corpus=args.corpus,
                                   fp32=args.logits_fp32))
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in args.configs if not (out_dir / f"{c.replace('/', '--')}.pt").exists()]
    if not todo:
        LOGGER.info("nada a fazer em %s", out_dir)
        return
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16",
                        cache_dir=args.cache_dir, revision=args.revision)
    if args.logits_fp32:
        use_fp32_logits(loaded.model)
    bank = C.WeightBank(loaded.decoder)
    ctrl = make_elastic_depth(loaded.model)
    ref_file = OUT / rs / rs / "bf16.pt"
    ref = torch.load(ref_file) if ref_file.exists() else None
    subset = subset_index(corpus.n_tokens, args.subset)
    calib_cache: dict = {}

    def calib_fn():
        if "x" not in calib_cache:
            calib_cache["x"] = C.wikitext_calibration(loaded.tokenizer, args.calib_n, args.calib_len,
                                                      cache_dir=f"{args.cache_dir}/datasets")
        return calib_cache["x"]

    for name in todo:
        is_ref = name == "bf16" and not args.ref_model
        if not is_ref and ref is None:
            raise RuntimeError(f"referencia ausente: rode 'bf16' em {rs} primeiro")
        t0 = time.time()
        with apply_config(name, loaded, bank, ctrl, calib_fn, args) as meta:
            res = score_fidelity(loaded, corpus, ref=None if is_ref else ref, batch_size=args.batch_size,
                                 subset=subset)
        res["meta"] = {**meta, "config": name, "model": args.model, "revision": args.revision,
                       "ref": rs, "seconds_total": time.time() - t0}
        target = out_dir / f"{name.replace('/', '--')}.pt"
        torch.save(res, target)
        if is_ref:
            ref = res
            LOGGER.info("[%s] referencia salva (auto-consistencia %.4f)", name,
                        (res["top1"] == torch.tensor([t for g in corpus.gen_ids for t in g])).float().mean())
        else:
            flip = (res["top1"] != ref["top1"]).float().mean().item()
            klm = res["kl"].double().mean().item()
            klf = res["kl_full"].double().nanmean().item()
            LOGGER.info("[%s] flip %.4f | KL_top %.4f | KL_full(sub) %.4f | flip/sqrtKL %.3f | %.0fs | %s",
                        name, flip, klm, klf, flip / max(klm, 1e-12) ** 0.5, time.time() - t0,
                        {k: v for k, v in meta.items() if k != "alphas"})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "score"])
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--ref-model", default=None)
    p.add_argument("--ref-revision", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--configs", nargs="+", default=["bf16"])
    p.add_argument("--n-prompts", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--subset", type=int, default=4096)
    p.add_argument("--calib-n", type=int, default=128)
    p.add_argument("--calib-len", type=int, default=512)
    p.add_argument("--calib-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--logits-fp32", action="store_true", help="lm_head em float32 (sem grade/empates do bf16)")
    add_corpus_arg(p)
    args = p.parse_args(argv)
    globals()["OUT"] = Path(args.out)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    {"gen": stage_gen, "score": stage_score}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
