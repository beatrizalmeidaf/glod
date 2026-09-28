#!/usr/bin/env python3
"""Roteamento MoE sob compressao: o canal descontinuo que o argumento de 1a ordem nao cobre.

Num bloco MoE a perturbacao pode trocar QUAIS experts um token visita (top-k do
roteador), uma mudanca discreta como o proprio flip. Este estagio mede, por config:

  free    a config como ela e: flips, KL, TV e a taxa de troca do conjunto top-k
          do roteador (por camada e posicao) contra a referencia densa;
  frozen  a mesma config com os logits do roteador SUBSTITUIDOS pelos da referencia
          em toda camada: os experts sao os mesmos da referencia e so os pesos
          perturbados mudam a saida. flips(free) - flips(frozen) e o que o canal de
          roteamento acrescenta.

Tambem guarda a margem do roteador na referencia (logit do k-esimo expert menos o do
(k+1)-esimo), o analogo da margem de decisao top1-top2 do paper.

    python -m glod moe-routing --model allenai/OLMoE-1B-7B-0125-Instruct --corpus gsm8k \\
        --configs u8 u6 u4 g4 mag20 --logits-fp32

Precisa do corpus da referencia (`glod grid gen`). Grava OUT/analysis/moe_routing__<ref>.json,
incrementalmente (uma config por vez; reexecutar pula as prontas).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from glod.core import compressors as C
from glod.core.elastic_depth import make_elastic_depth
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import GeneratedCorpus
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.paths import OUT, add_corpus_arg, ref_slug
from glod.pipelines.fidelity.build_grid import apply_config, use_fp32_logits

LOGGER = logging.getLogger("glod.moe_routing")
K = 64  # top-K da referencia para KL/TV, como em glod.core.fidelity


def find_routers(decoder) -> list[tuple[str, torch.nn.Module, int]]:
    """(nome, Linear do roteador, top_k do bloco) para cada bloco MoE."""
    mods = dict(decoder.named_modules())
    out = []
    for name, m in mods.items():
        if isinstance(m, torch.nn.Linear) and C.ROUTER_RE.search(name):
            parent = mods[name.rsplit(".", 1)[0]]
            k = getattr(parent, "top_k", None) or getattr(parent, "num_experts_per_tok", None)
            if k is None:
                raise RuntimeError(f"bloco MoE sem top_k: {name}")
            out.append((name, m, int(k)))
    if not out:
        raise SystemExit("nenhum roteador MoE encontrado: este estagio e so para modelos MoE")
    return out


def kl_tv(logp: torch.Tensor, rv: torch.Tensor, ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(ref||cfg) e TV nos top-K da referencia + cauda agregada (mesma conta de fidelity.py)."""
    qv = logp.gather(-1, ri)
    P, Q = rv.exp(), qv.exp()
    tP, tQ = (1 - P.sum(-1)).clamp(min=1e-30), (1 - Q.sum(-1)).clamp(min=1e-30)
    kl = (P * (rv - qv)).sum(-1) + tP * (tP.log() - tQ.log())
    tv = 0.5 * ((P - Q).abs().sum(-1) + (tP - tQ).abs())
    return kl, tv


@torch.inference_mode()
def run_pass(loaded, routers, seqs, *, ref: dict | None, frozen: bool) -> dict:
    """Uma passada teacher-forcing sequencia a sequencia (lote 1, sem padding)."""
    dev = loaded.device
    captured: dict[int, torch.Tensor] = {}
    cur = {"seq": 0}
    hooks = []
    for li, (_n, mod, _k) in enumerate(routers):
        def hook(_m, _inp, out, li=li):
            if frozen:
                return ref["router"][cur["seq"]][li].to(out.device, out.dtype)
            captured[li] = out.detach()
            return None
        hooks.append(mod.register_forward_hook(hook))
    res = {"top1": [], "kl": [], "tv": [], "route_change": [], "overlap": []}
    if ref is None:
        res.update(router=[], topv=[], topi=[], rmargin=[])
    try:
        for si, (ids, npos) in enumerate(seqs):
            cur["seq"] = si
            captured.clear()
            x = torch.tensor([ids], device=dev)
            logits = loaded.model(input_ids=x, use_cache=False).logits[0]
            logp = logits[-npos - 1:-1].double().log_softmax(-1)      # prediz os tokens gerados
            res["top1"].append(logp.argmax(-1).cpu())
            if ref is None:
                v, i_ = logp.topk(K, dim=-1)
                res["topv"].append(v.float().cpu()); res["topi"].append(i_.cpu())
                res["router"].append([captured[li].half().cpu() for li in range(len(routers))])
                marg = []
                for li, (_n, _m, k) in enumerate(routers):
                    s = captured[li][-npos - 1:-1].float().sort(-1, descending=True).values
                    marg.append((s[:, k - 1] - s[:, k]).cpu())
                res["rmargin"].append(torch.stack(marg))              # [camadas, pos]
            else:
                kl, tv = kl_tv(logp, ref["topv"][si].to(dev).double(), ref["topi"][si].to(dev))
                res["kl"].append(kl.float().cpu()); res["tv"].append(tv.float().cpu())
                if not frozen:
                    ch, ov = [], []
                    for li, (_n, _m, k) in enumerate(routers):
                        a = captured[li][-npos - 1:-1].float().topk(k, -1).indices.sort(-1).values.cpu()
                        b = ref["router"][si][li][-npos - 1:-1].float().topk(k, -1).indices.sort(-1).values
                        ch.append((a != b).any(-1))
                        # fracao do conjunto preservada (|A inter B| / k)
                        ov.append((a[:, :, None] == b[:, None, :]).any(-1).float().mean(-1))
                    res["route_change"].append(torch.stack(ch))       # [camadas, pos]
                    res["overlap"].append(torch.stack(ov))
    finally:
        for h in hooks:
            h.remove()
    return res


def summarize(ref: dict, free: dict, frozen: dict) -> dict:
    t_ref = torch.cat(ref["top1"])
    fl_free = torch.cat(free["top1"]) != t_ref
    fl_frz = torch.cat(frozen["top1"]) != t_ref
    rc = torch.cat(free["route_change"], dim=1)                    # [camadas, T]
    any_ch = rc.any(0)
    kl_f, tv_f = torch.cat(free["kl"]), torch.cat(free["tv"])
    kl_z, tv_z = torch.cat(frozen["kl"]), torch.cat(frozen["tv"])
    p = lambda m: float(fl_free[m].float().mean()) if m.any() else None
    return {
        "n_tokens": int(t_ref.numel()),
        "free": {"flip": float(fl_free.float().mean()), "kl": float(kl_f.mean()), "tv": float(tv_f.mean()),
                 "flip_over_tv": float(fl_free.float().mean() / tv_f.mean())},
        "frozen": {"flip": float(fl_frz.float().mean()), "kl": float(kl_z.mean()), "tv": float(tv_z.mean()),
                   "flip_over_tv": float(fl_frz.float().mean() / tv_z.mean())},
        "routing_share_of_flips": float(1 - fl_frz.float().mean() / max(fl_free.float().mean(), 1e-12)),
        "route_change_rate": float(rc.float().mean()),                 # por (camada, posicao)
        "route_change_any_layer": float(any_ch.float().mean()),        # posicoes com >=1 camada trocada
        "expert_overlap": float(torch.cat(free["overlap"], dim=1).mean()),
        "flip_given_route_change": p(any_ch),
        "flip_given_no_route_change": p(~any_ch),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--configs", nargs="+", default=["u8", "u6", "u5", "u4", "g4", "mag20"])
    p.add_argument("--n-seqs", type=int, default=64)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--calib-source", choices=["wikitext", "c4", "code"], default="wikitext")
    p.add_argument("--calib-n", type=int, default=128)
    p.add_argument("--calib-len", type=int, default=512)
    p.add_argument("--calib-batch", type=int, default=8)
    p.add_argument("--logits-fp32", action="store_true")
    p.add_argument("--out", default=str(OUT))
    add_corpus_arg(p)
    args = p.parse_args(argv)
    out_root = Path(args.out)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)

    rs = ref_slug(args.model, corpus=args.corpus)
    corpus = GeneratedCorpus.from_dict(json.loads((out_root / "corpora" / f"{rs}.json").read_text()))
    target = out_root / "analysis" / f"moe_routing__{rs}{'__fp32' if args.logits_fp32 else ''}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(target.read_text()) if target.exists() else {"model": args.model, "corpus": args.corpus,
                                                                   "configs": {}}
    todo = [c for c in args.configs if c not in done["configs"]]
    if not todo:
        LOGGER.info("nada a fazer: %s", target)
        return 0

    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(len(corpus.gen_ids), generator=g)[:args.n_seqs].tolist()
    seqs = [(list(corpus.prompt_ids[i]) + list(corpus.gen_ids[i]), len(corpus.gen_ids[i])) for i in pick]

    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    if args.logits_fp32:
        use_fp32_logits(loaded.model)
    routers = find_routers(loaded.decoder)
    bank = C.WeightBank(loaded.decoder)
    LOGGER.info("%d roteadores (fora do banco: %d), top_k=%d, %d seqs", len(routers), len(bank.routers),
                routers[0][2], len(seqs))
    ctrl = make_elastic_depth(loaded.model)
    cal: dict = {}

    def calib_fn():
        if "x" not in cal:
            cal["x"] = C.CALIBRATION[args.calib_source](loaded.tokenizer, args.calib_n, args.calib_len,
                                                         cache_dir=f"{args.cache_dir}/datasets")
        return cal["x"]

    cfg_args = SimpleNamespace(calib_batch=args.calib_batch, cache_dir=args.cache_dir)
    bank.restore()
    ref = run_pass(loaded, routers, seqs, ref=None, frozen=False)
    rm = torch.cat(ref["rmargin"], dim=1)
    done["router_margin"] = {"median": float(rm.median()), "frac_below_0.1": float((rm < 0.1).float().mean()),
                             "frac_below_0.5": float((rm < 0.5).float().mean()),
                             "n_layers": len(routers), "top_k": routers[0][2]}
    for name in todo:
        t0 = time.time()
        with apply_config(name, loaded, bank, ctrl, calib_fn, cfg_args) as meta:
            free = run_pass(loaded, routers, seqs, ref=ref, frozen=False)
            frozen = run_pass(loaded, routers, seqs, ref=ref, frozen=True)
        s = summarize(ref, free, frozen)
        s["meta"] = {k: v for k, v in meta.items() if k != "alphas"}
        done["configs"][name] = s
        target.write_text(json.dumps(done, indent=1))
        LOGGER.info("[%s] flip livre %.4f | congelado %.4f | parte do roteamento %.0f%% | troca de rota "
                    "%.3f por (camada,pos), %.3f das posicoes | flips/TV %.2f -> %.2f | %.0fs",
                    name, s["free"]["flip"], s["frozen"]["flip"], 100 * s["routing_share_of_flips"],
                    s["route_change_rate"], s["route_change_any_layer"], s["free"]["flip_over_tv"],
                    s["frozen"]["flip_over_tv"], time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
