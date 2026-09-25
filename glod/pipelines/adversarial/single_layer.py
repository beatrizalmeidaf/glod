#!/usr/bin/env python3
"""Experimento A: falsificador adversarial da lei flips ~ kappa*sqrt(KL).

A lei e uma afirmacao EMPIRICA sobre compressores honestos, nao uma identidade.
Para provar isso, construimos duas perturbacoes de peso otimizadas por gradiente
na ultima camada (down_proj, fatorada em rank r):

  maligna  : maximiza flips com KL minimo   -> deve ficar ACIMA da faixa
  benigna  : maximiza KL sem causar flips   -> deve ficar ABAIXO da faixa
             (o caso extremo conhecido e temperatura, que muda KL e nenhum flip)

Otimizacao em estados ocultos cacheados: para um conjunto de tokens guardamos a
entrada x de down_proj da ultima camada e o residual r antes da soma, de forma que
  logits(dW) = lm_head(norm(r + (W + dW) x))
e diferenciavel e barato. A perturbacao encontrada e depois aplicada ao modelo de
verdade e pontuada no corpus inteiro pelo pipeline normal.

    python -m glod adv-single --model Qwen/Qwen3-4B --device cuda:0 --mode malign
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch

from glod.core.fidelity import score_fidelity, subset_index
from glod.core.model_loader import load_model, resolve_decoder
from glod.corpora.token_oracle import GeneratedCorpus, teacher_forcing_batch
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, add_corpus_arg, ref_slug

LOGGER = logging.getLogger("glod.adv")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    add_corpus_arg(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--modes", nargs="+", default=["malign", "benign"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--kl-budget", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--score-batch", type=int, default=8)
    p.add_argument("--temperature-ref", action="store_true", help="ponto de referencia: escalar o lm_head")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    ms = ref_slug(args.model, corpus=args.corpus)
    out_dir = OUT / "adversarial" / ms
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{ms}.json").read_text()))
    ref_full = torch.load(OUT / ms / ms / "bf16.pt")
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    dec = resolve_decoder(loaded.model)
    layer = dec.layers[-1]
    inner = getattr(layer, "layer", layer)
    down = inner.mlp.down_proj
    W0 = down.weight.detach().clone()
    d_out, d_in = W0.shape

    # --- cache: entrada de down_proj e logits de referencia (top-2 indices)
    xs, tops = [], []
    box: dict[str, torch.Tensor] = {}
    hook = down.register_forward_pre_hook(lambda _m, i: box.__setitem__("x", i[0].detach()))
    got = 0
    with torch.no_grad():
        for s in range(0, len(corpus.gen_ids), args.batch_size):
            gens = corpus.gen_ids[s:s + args.batch_size]
            ids, mask, pos, keep = teacher_forcing_batch(corpus.prompt_ids[s:s + args.batch_size], gens,
                                                         loaded.tokenizer.pad_token_id)
            loaded.model(input_ids=ids.to(loaded.device), attention_mask=mask.to(loaded.device),
                         position_ids=pos.to(loaded.device), use_cache=False, logits_to_keep=keep)
            for row, g in enumerate(gens):
                n = len(g)
                xs.append(box["x"][row, -n:].to(torch.bfloat16).cpu())
                got += n
            if got >= args.tokens:
                break
    hook.remove()
    X = torch.cat(xs)[: args.tokens].to(args.device)                       # [T, d_in]
    T = X.shape[0]
    # logits de referencia nos MESMOS tokens (com dW = 0), por reprodutibilidade local:
    # a saida do modelo depende apenas de (res + W x) na ultima camada, entao capturamos
    # o efeito de dW aplicando a diferenca na saida do down_proj via o proprio grafo.
    dtype = torch.float32

    def logits_from(dW: torch.Tensor | None, chunk: slice) -> torch.Tensor:
        x = X[chunk].to(dtype)
        delta = None if dW is None else x @ dW.T                            # [c, d_out]
        base = base_out[chunk].to(dtype)
        h = base if delta is None else base + delta
        h = norm_fn(h.to(norm_dtype)).to(dtype)
        z = h @ Wu.T
        return cap(z if bias is None else z + bias)

    # captura base_out = residual + W0 x (i.e. a entrada do norm final), por token
    outs = []
    box2: dict[str, torch.Tensor] = {}
    hk = dec.norm.register_forward_pre_hook(lambda _m, i: box2.__setitem__("h", i[0].detach()))
    got = 0
    with torch.no_grad():
        for s in range(0, len(corpus.gen_ids), args.batch_size):
            gens = corpus.gen_ids[s:s + args.batch_size]
            ids, mask, pos, keep = teacher_forcing_batch(corpus.prompt_ids[s:s + args.batch_size], gens,
                                                         loaded.tokenizer.pad_token_id)
            loaded.model(input_ids=ids.to(loaded.device), attention_mask=mask.to(loaded.device),
                         position_ids=pos.to(loaded.device), use_cache=False, logits_to_keep=keep)
            for row, g in enumerate(gens):
                n = len(g)
                outs.append(box2["h"][row, -n:].to(torch.bfloat16).cpu())
                got += n
            if got >= args.tokens:
                break
    hk.remove()
    base_out = torch.cat(outs)[:T].to(args.device)
    norm_fn = dec.norm
    norm_dtype = next(dec.norm.parameters()).dtype
    cfg_text = loaded.model.config.get_text_config()
    softcap = getattr(cfg_text, "final_logit_softcapping", None)

    def cap(z: torch.Tensor) -> torch.Tensor:
        return z if not softcap else torch.tanh(z / softcap) * softcap
    head = loaded.model.get_output_embeddings()
    Wu = head.weight.detach().to(dtype)
    bias = head.bias.detach().to(dtype) if getattr(head, "bias", None) is not None else None

    with torch.no_grad():
        P = torch.cat([logits_from(None, slice(i, i + 256)).log_softmax(-1).to(torch.float16).cpu()
                       for i in range(0, T, 256)])
    ref_top = P.argmax(-1)
    LOGGER.info("cache pronto: %d tokens, d_in=%d, d_out=%d", T, d_in, d_out)

    from glod.pipelines.tasks.matched_kl import small_corpus
    probe_corpus, keep = small_corpus(corpus, 96)
    ref_probe = {k: ref_full[k][keep] for k in ("top1", "topv", "topi")}
    results = {}
    tied = head.weight.data_ptr() == loaded.model.get_input_embeddings().weight.data_ptr()
    if tied and args.temperature_ref:
        LOGGER.warning("lm_head amarrado as embeddings: escalar o lm_head nao e temperatura pura; referencia pulada")
    if args.temperature_ref and not tied:
        # referencia analitica: escalar o lm_head = temperatura -> KL > 0 e ZERO flips
        with torch.no_grad():
            W_head = head.weight.detach().clone()
            for tau in (1.1, 1.3):
                head.weight.copy_((W_head.float() / tau).to(W_head.dtype))
                r = score_fidelity(loaded, corpus, ref=ref_full, batch_size=args.score_batch,
                                   subset=subset_index(corpus.n_tokens, 2048))
                fl = (r["top1"] != ref_full["top1"]).double().mean().item()
                kl = r["kl"].double().mean().item()
                results[f"temperatura@tau={tau}"] = {"mode": "temperatura", "flip": fl, "kl": kl,
                                                     "kappa": fl / max(kl, 1e-12) ** 0.5}
                LOGGER.info("[temperatura tau=%.2f] flip %.4f | KL %.4f | kappa %.3f", tau, fl, kl,
                            results[f"temperatura@tau={tau}"]["kappa"])
            head.weight.copy_(W_head)
    for mode in args.modes:
        for budget in args.kl_budget:
            torch.manual_seed(0)
            full_rank = args.rank <= 0   # rank 0 = matriz completa (mais poder ao atacante)
            if full_rank:
                A = (1e-4 * torch.randn(d_out, d_in, device=args.device, dtype=dtype)).requires_grad_(True)
                B = None
                params = [A]
            else:
                A = (0.01 * torch.randn(d_out, args.rank, device=args.device, dtype=dtype)).requires_grad_(True)
                B = (0.01 * torch.randn(args.rank, d_in, device=args.device, dtype=dtype)).requires_grad_(True)
                params = [A, B]
            opt = torch.optim.Adam(params, lr=args.lr)
            lam = 10.0
            t0 = time.time()
            for step in range(args.steps):
                opt.zero_grad()
                idx = torch.randint(0, T, (512,), device=args.device)
                x = X[idx].to(dtype)
                base = base_out[idx].to(dtype)
                dWt = A if full_rank else (A @ B)
                h = norm_fn((base + x @ dWt.T).to(norm_dtype)).to(dtype)
                z = cap(h @ Wu.T if bias is None else h @ Wu.T + bias)
                lp = z.log_softmax(-1)
                p0 = P[idx.cpu()].to(args.device).to(dtype).log_softmax(-1)
                kl = (p0.exp() * (p0 - lp)).sum(-1).mean()
                gap = z.gather(-1, ref_top[idx.cpu()].to(args.device)[:, None]).squeeze(-1) - \
                    z.scatter(-1, ref_top[idx.cpu()].to(args.device)[:, None], -1e4).max(-1).values
                flip_surrogate = torch.nn.functional.softplus(gap)          # baixo = flip
                loss = (flip_surrogate.mean() if mode == "malign" else -kl) + lam * (kl - budget).clamp(min=0) ** 2
                if mode == "benign":
                    loss = -kl + lam * (kl - budget).clamp(min=0) ** 2 + 30.0 * torch.sigmoid(-gap).mean()
                loss.backward()
                opt.step()
                if step % 100 == 0:
                    LOGGER.info("  [%s KL<=%.3f] passo %d: KL %.4f | flips(sub) %.3f", mode, budget, step,
                                kl.item(), (gap < 0).float().mean().item())
            # a direcao encontrada e reescalada por bisseccao para bater o KL alvo
            # no corpus: so assim "flips a KL igual" e comparavel com os compressores
            with torch.no_grad():
                dW = (A if full_rank else (A @ B)).detach()

                def kl_at(scale: float) -> float:
                    down.weight.copy_((W0.float() + scale * dW).to(W0.dtype))
                    r = score_fidelity(loaded, probe_corpus, ref=ref_probe, batch_size=args.score_batch)
                    return r["kl"].double().mean().item()

                # bisseccao em escala LOG: a direcao otimizada pode ter norma enorme (visto no
                # Gemma), e 7 passos lineares em [0, 1] nao desciam o KL ate o alvo
                lo, hi = 1e-8, 1.0
                while kl_at(hi) < budget and hi < 1e3:
                    lo, hi = hi, hi * 4
                for _ in range(30):
                    mid = (lo * hi) ** 0.5
                    if kl_at(mid) < budget:
                        lo = mid
                    else:
                        hi = mid
                scale = (lo * hi) ** 0.5
                down.weight.copy_((W0.float() + scale * dW).to(W0.dtype))
                res = score_fidelity(loaded, corpus, ref=ref_full, batch_size=args.score_batch,
                                     subset=subset_index(corpus.n_tokens, 2048))
                down.weight.copy_(W0)
            flip = (res["top1"] != ref_full["top1"]).double().mean().item()
            klm = res["kl"].double().mean().item()
            row = {"mode": mode, "budget": budget, "scale": scale, "flip": flip, "kl": klm,
                   "kappa": flip / max(klm, 1e-12) ** 0.5, "rank": args.rank,
                   "seconds": time.time() - t0}
            results[f"{mode}@{budget}"] = row
            LOGGER.info("[%s KL<=%.3f] CORPUS: flip %.4f | KL %.4f | kappa %.3f (%.0fs)", mode, budget,
                        flip, klm, row["kappa"], row["seconds"])
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
