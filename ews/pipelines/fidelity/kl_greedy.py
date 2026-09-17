#!/usr/bin/env python3
"""Experimento C: quantizador KL-guloso (o algoritmo mais bobo que so olha o KL).

Se a tese e "a identidade do metodo nao importa, so o KL que ele produz", entao
um quantizador que ignora Hessiana, ignora ativacoes por canal e simplesmente
escolhe, por projecao, o recorte (clipping) que minimiza o KL de SAIDA do modelo
inteiro num lote de calibracao deve cair na mesma reta - e, no mesmo KL, ter os
mesmos flips que GPTQ/AWQ.

Algoritmo (sequencial por camada, guloso por projecao):
  para cada camada l, para cada projecao P em l:
      para cada c em grid de recorte {0.55 ... 1.0}:
          W_P <- RTN(clip(W_P, c * max|W| por grupo), bits)
          mede KL(p_denso || p_atual) num lote pequeno de calibracao
      fica com o melhor c
Custo: n_camadas * n_proj * |grid| forwards de um lote pequeno.

    python ews_klgreedy.py --model Qwen/Qwen3-4B --device cuda:0 --bits 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch

from ews.core import compressors as C
from ews.core.elastic_depth import make_elastic_depth
from ews.core.fidelity import score_fidelity, subset_index
from ews.core.model_loader import load_model
from ews.core.quantize import quantize_dequantize
from ews.corpora.token_oracle import GeneratedCorpus
from ews.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from ews.paths import OUT, slug

LOGGER = logging.getLogger("ews.klgreedy")


def clip_rtn(w: torch.Tensor, bits: int, ratio: float, group: int = 128) -> torch.Tensor:
    """RTN com recorte simetrico dos extremos do grupo (ratio=1.0 -> RTN puro)."""
    if ratio >= 1.0:
        return quantize_dequantize(w, bits, group)
    flat = w.float().reshape(-1, w.shape[-1])
    out = torch.empty_like(flat)
    for s in range(0, flat.shape[-1], group):
        blk = flat[:, s:s + group]
        lo, hi = blk.amin(-1, keepdim=True), blk.amax(-1, keepdim=True)
        mid = (lo + hi) / 2
        half = (hi - lo) / 2 * ratio
        out[:, s:s + group] = blk.clamp(mid - half, mid + half)
    return quantize_dequantize(out.reshape(w.shape), bits, group)


@torch.no_grad()
def kl_on_batch(model, ids, ref_logprobs, chunk: int = 2) -> float:
    kls = []
    for i in range(0, ids.shape[0], chunk):
        out = model(input_ids=ids[i:i + chunk], use_cache=False).logits.float().log_softmax(-1)
        r = ref_logprobs[i:i + chunk].to(out.device).float()
        kls.append((r.exp() * (r - out)).sum(-1).mean().item())
    return sum(kls) / len(kls)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    p.add_argument("--grid", type=float, nargs="+", default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6])
    p.add_argument("--calib-seqs", type=int, default=8)
    p.add_argument("--calib-len", type=int, default=256)
    p.add_argument("--score-batch", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)

    out_dir = OUT / "klgreedy" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text()))
    ref_full = torch.load(OUT / slug(args.model) / slug(args.model) / "bf16.pt")
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank = C.WeightBank(loaded.decoder)
    make_elastic_depth(loaded.model)
    calib = C.wikitext_calibration(loaded.tokenizer, args.calib_seqs, args.calib_len,
                                   cache_dir=f"{args.cache_dir}/datasets").to(args.device)
    ref_lp = torch.cat([loaded.model(input_ids=calib[i:i + 2], use_cache=False).logits.float().log_softmax(-1).cpu()
                        for i in range(0, len(calib), 2)])
    results = {}
    for bits in args.bits:
        bank.restore()
        t0 = time.time()
        chosen = {}
        for layer in range(bank.num_layers):
            for t in bank.layer(layer):
                best = (float("inf"), 1.0)
                w0 = t.original.to(t.module.weight.device)
                for ratio in args.grid:
                    t.module.weight.copy_(clip_rtn(w0, bits, ratio).to(t.module.weight.dtype))
                    kl = kl_on_batch(loaded.model, calib, ref_lp)
                    if kl < best[0]:
                        best = (kl, ratio)
                t.module.weight.copy_(clip_rtn(w0, bits, best[1]).to(t.module.weight.dtype))
                chosen[t.name] = best[1]
            if layer % 8 == 0:
                LOGGER.info("  camada %d/%d | KL calib %.4f", layer, bank.num_layers, best[0])
        res = score_fidelity(loaded, corpus, ref=ref_full, batch_size=args.score_batch,
                             subset=subset_index(corpus.n_tokens, 2048))
        flip = (res["top1"] != ref_full["top1"]).double().mean().item()
        kl = res["kl"].double().mean().item()
        row = {"bits": bits, "flip": flip, "kl": kl, "kappa": flip / max(kl, 1e-12) ** 0.5,
               "seconds": time.time() - t0,
               "ratios": {k: v for k, v in list(chosen.items())[:8]},
               "ratio_hist": {str(r): sum(1 for v in chosen.values() if v == r) for r in args.grid}}
        results[f"klgreedy{bits}"] = row
        LOGGER.info("[KL-guloso %db] flip %.4f | KL %.4f | kappa %.3f | %.0fs | recortes %s", bits, flip, kl,
                    row["kappa"], row["seconds"], row["ratio_hist"])
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))
        # comparacao com GPTQ/AWQ/RTN do mesmo modelo no mesmo bitrate
        for name in (f"u{bits}", f"gptq{bits}", f"awq{bits}"):
            f = OUT / slug(args.model) / slug(args.model) / f"{name}.pt"
            if f.exists():
                c = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
                fl = (c["top1"] != ref_full["top1"]).double().mean().item()
                k = c["kl"].double().mean().item()
                LOGGER.info("    %-8s flip %.4f | KL %.4f | kappa %.3f", name, fl, k, fl / max(k, 1e-12) ** 0.5)
    bank.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
