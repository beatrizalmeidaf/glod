#!/usr/bin/env python3
"""Avaliacao do AMQ fora do teacher forcing: especulativa, acuracia e exportacao.

  spec    decodificacao especulativa real (lote 1, k rascunhos) com o modelo denso como
          alvo e cada variante quantizada como draft: aceitacao greedy e por amostragem,
          probabilidade media de aceitacao, tokens por rodada. Prompts GSM8K disjuntos
          dos corpora (mesma semente do spec-bench).
  acc     acuracia GSM8K (greedy, 400 problemas disjuntos) do denso e de cada variante,
          com as respostas por problema para comparar perdas e reparos.
  export  grava a variante no formato AWQ (4 bits, grupo 128, com zeros): e a mesma
          parametrizacao W = s * (q - z), servida pelos kernels Marlin do vLLM.

Variantes: gptq4 (theta = 0, os codigos do GPTQ), amq4_<perda> (theta treinado) e
cfg:<nome> (qualquer configuracao do grid, ex. cfg:awq4).

    python -m glod amq-eval spec --model Qwen/Qwen3-4B --variants gptq4 amq4_kl amq4_flip
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from glod.core import compressors as C
from glod.core.elastic_depth import make_elastic_depth
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import _eos_ids
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, slug
from glod.pipelines.compress.amq import CKPT
from glod.pipelines.fidelity.build_grid import apply_config
from glod.pipelines.tasks.closedloop import gsm8k_disjoint, parse_answer, speculative

LOGGER = logging.getLogger("glod.amq_eval")


@torch.no_grad()
def materialize(loaded, bank: C.WeightBank, model_id: str, variant: str, bits: int = 4, group: int = 128) -> None:
    """Escreve nos pesos bf16 do modelo a variante AMQ/GPTQ salva pelo estagio amq."""
    ck = CKPT / slug(model_id)
    params = torch.load(ck / f"gptq{bits}_params.pt", weights_only=False)
    theta = None if variant == f"gptq{bits}" else torch.load(ck / f"{variant}_theta.pt", weights_only=False)
    for t in bank.targets:
        codes, s0, z = params[t.name]
        s = s0 if theta is None else s0 * theta[t.name].exp()
        dev = t.module.weight.device
        out, cols = codes.shape
        W = ((codes.to(dev).float().view(out, -1, group) - z.to(dev)[:, :, None]) * s.to(dev)[:, :, None]).view(out, cols)
        if theta is not None and t.name + "|lr" in theta:      # correcao de baixo posto, fundida em bf16
            A_, B_ = theta[t.name + "|lr"]
            W = W + A_.to(dev) @ B_.to(dev)
        t.module.weight.copy_(W.to(t.module.weight.dtype))


class Variant:
    """Contexto que aplica uma variante ao modelo carregado e restaura depois."""

    def __init__(self, loaded, bank, ctrl, model_id: str, variant: str, cache_dir: str):
        self.loaded, self.bank, self.ctrl, self.model_id, self.variant = loaded, bank, ctrl, model_id, variant
        self.cache_dir = cache_dir
        self._cfg = None

    def __enter__(self):
        if self.variant.startswith("cfg:"):
            cache: dict = {}

            def calib_fn():
                if "x" not in cache:
                    cache["x"] = C.wikitext_calibration(self.loaded.tokenizer, 128, 512, cache_dir=f"{self.cache_dir}/datasets")
                return cache["x"]
            self._cfg = apply_config(self.variant[4:], self.loaded, self.bank, self.ctrl, calib_fn,
                                     SimpleNamespace(calib_batch=8, cache_dir=self.cache_dir))
            self._cfg.__enter__()
        elif self.variant != "dense":
            materialize(self.loaded, self.bank, self.model_id, self.variant)
        return self

    def __exit__(self, *exc):
        if self._cfg is not None:
            self._cfg.__exit__(*exc)
        self.bank.restore()


def stage_spec(args) -> None:
    torch.cuda.set_device(torch.device(args.device))
    target = load_model(args.model, role="ref", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    draft = load_model(args.model, role="draft", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank, ctrl = C.WeightBank(draft.decoder), make_elastic_depth(draft.model)
    eos = _eos_ids(target)
    corpus = json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text())
    recs = gsm8k_disjoint(target.tokenizer, args.n, {r["prompt"] for r in corpus["records"]}, args.cache_dir, seed=9001)
    prompts = [target.tokenizer(r["prompt"], add_special_tokens=False).input_ids for r in recs]
    path = OUT / "amq" / f"{slug(args.model)}__spec.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for v in args.variants:
        if v in res:
            continue
        row = {}
        with Variant(draft, bank, ctrl, args.model, v, args.cache_dir):
            for mode in ("greedy", "sample"):
                gen = torch.Generator(device=args.device).manual_seed(0)
                tot = {"proposed": 0, "accepted": 0, "rounds": 0, "new_tokens": 0, "probs": []}
                per_prompt = []
                for ids in prompts:
                    o = speculative(target, draft, ids, k=args.k, max_new=args.max_new_tokens, eos=eos,
                                    sampling=mode == "sample", gen=gen)
                    for key in ("proposed", "accepted", "rounds", "new_tokens"):
                        tot[key] += o[key]
                    tot["probs"] += o["accept_probs"]
                    per_prompt.append(o["accepted"] / max(o["proposed"], 1))
                row[mode] = {"accept_rate": tot["accepted"] / tot["proposed"],
                             "tokens_per_round": tot["new_tokens"] / tot["rounds"],
                             "mean_accept_prob": float(np.mean(tot["probs"])) if tot["probs"] else None,
                             "per_prompt_accept": per_prompt}
                LOGGER.info("[%s | %s] aceitacao %.4f  tokens/rodada %.3f  E[min(1,p/q)] %s", v, mode,
                            row[mode]["accept_rate"], row[mode]["tokens_per_round"], row[mode]["mean_accept_prob"])
        res[v] = row
        path.write_text(json.dumps(res, indent=1))


@torch.no_grad()
def stage_acc(args) -> None:
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank, ctrl = C.WeightBank(loaded.decoder), make_elastic_depth(loaded.model)
    tok = loaded.tokenizer
    eos = _eos_ids(loaded)
    corpus = json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text())
    recs = gsm8k_disjoint(tok, args.n, {r["prompt"] for r in corpus["records"]}, args.cache_dir, seed=31337)
    path = OUT / "amq" / f"{slug(args.model)}__acc.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    tok.padding_side = "left"
    for v in ["dense"] + [x for x in args.variants if x != "dense"]:
        if v in res:
            continue
        correct = []
        t0 = time.time()
        with Variant(loaded, bank, ctrl, args.model, v, args.cache_dir):
            for b in range(0, len(recs), args.batch):
                chunk = recs[b:b + args.batch]
                enc = tok([r["prompt"] for r in chunk], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(loaded.device)
                out = loaded.model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                            top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
                for r, row in zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist()):
                    cut = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
                    ans = parse_answer(tok.decode(row[:cut], skip_special_tokens=True))
                    try:
                        correct.append(ans is not None and abs(float(ans) - float(r["gold"])) < 1e-6)
                    except ValueError:
                        correct.append(False)
        res[v] = {"acc": float(np.mean(correct)), "correct": correct, "seconds": time.time() - t0}
        LOGGER.info("[%s] GSM8K %.3f (%d problemas, %.0fs)", v, res[v]["acc"], len(correct), time.time() - t0)
        path.write_text(json.dumps(res, indent=1))


AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_awq(x: torch.Tensor) -> torch.Tensor:
    """[rows, cols] inteiros 0..15 -> [rows, cols/8] int32 na ordem intercalada do AWQ."""
    x = x.to(torch.int32).view(x.shape[0], -1, 8)[:, :, AWQ_ORDER]
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.int32)
    for i in range(8):
        out |= (x[:, :, i] & 0xF) << (4 * i)
    return out


def stage_export(args) -> None:
    """Formato AWQ-GEMM: qweight [in, out/8], qzeros [in/g, out/8], scales [in/g, out] fp16."""
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    ck = CKPT / slug(args.model)
    params = torch.load(ck / "gptq4_params.pt", weights_only=False)
    theta = None if args.variant == "gptq4" else torch.load(ck / f"{args.variant}_theta.pt", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)
    sd = model.state_dict()
    prefix = next(k for k in sd if ".layers.0." in k).split("layers.0.")[0]      # ex.: "model."
    out = {}
    quantized = set()
    for name, (codes, s0, z) in params.items():
        full = prefix + name
        s = (s0 if theta is None else s0 * theta[name].exp()).half()
        out[full + ".qweight"] = _pack_awq(codes.T.contiguous())
        out[full + ".qzeros"] = _pack_awq(z.round().to(torch.int32).T.contiguous())
        out[full + ".scales"] = s.T.contiguous()
        quantized.add(full + ".weight")
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    for k, v in sd.items():
        if k in quantized or (tied and k.endswith("lm_head.weight")):   # amarrado: o runtime reconstroi
            continue
        out[k] = v.clone().contiguous()
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    cfg = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
    cfg.quantization_config = {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True,
                               "version": "gemm", "modules_to_not_convert": None}
    cfg.save_pretrained(dst)
    AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir).save_pretrained(dst)
    try:
        model.generation_config.save_pretrained(dst)
    except Exception:  # noqa: BLE001 - alguns modelos nao tem generation_config
        pass
    LOGGER.info("exportado %s (%s) em %s: %d tensores quantizados", args.model, args.variant, dst, len(quantized))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["spec", "acc", "export"])
    p.add_argument("--model", required=True)
    p.add_argument("--variants", nargs="+", default=["gptq4", "amq4_kl", "amq4_flip", "amq4_tvfrag"])
    p.add_argument("--variant", default="amq4_flip", help="export: variante a gravar")
    p.add_argument("--out", default=None, help="export: pasta de saida")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    {"spec": stage_spec, "acc": stage_acc, "export": stage_export}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
