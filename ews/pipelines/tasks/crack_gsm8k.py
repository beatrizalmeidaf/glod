#!/usr/bin/env python3
"""Resolve a rachadura: no KL casado, poda preservou GSM8K +4pp melhor que quantizacao.

Hipoteses concorrentes:
  H1 alocacao por dominio: o KL foi casado no corpus MISTO (GSM8K + MMLU-PT). Se a poda
     coloca menos KL nos tokens de GSM8K, a comparacao no GSM8K nao estava casada.
  H2 artefato de alpha: W0 + alpha*dW com alpha < 1 nao e poda (encolhe os pesos em vez de
     zera-los); o braco "poda" do experimento com KL casado media encolhimento.
  H3 efeito real: poda e menos danosa ao raciocinio em varias etapas no mesmo KL.

Parte 1 (H1, sem geracao): para cada ponto de ews_matched_kl, reconstroi a perturbacao,
  aplica o mesmo alpha e mede KL e flips SO nos tokens de GSM8K; reanalisa a diferenca
  poda - quantizacao casando no KL de GSM8K.
Parte 2 (H2/H3): casa o KL NO DOMINIO GSM8K com CONTROLES NATIVOS e deployaveis:
     rtnmix / gptqmix  fracao f de projecoes em 3 bits, o resto em 4 (precisao mista real)
     wanda / sgpt / mag esparsidade real
  e mede acuracia GSM8K com n=400 prompts disjuntos. TOST poda x quantizacao.

    python ews_crack.py part1 --model Qwen/Qwen3-4B --device cuda:0
    python ews_crack.py part2 --model Qwen/Qwen3-4B --device cuda:0
    python ews_crack.py report
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys

import numpy as np
import torch

from ews.core import compressors as C
from ews.core.elastic_depth import make_elastic_depth
from ews.core.fidelity import score_fidelity
from ews.core.model_loader import load_model
from ews.corpora.token_oracle import GeneratedCorpus, generate_greedy
from ews.pipelines.tasks.closedloop import gsm8k_disjoint, parse_answer
from ews.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from ews.paths import OUT, slug
from ews.pipelines.fidelity.build_grid import apply_config
from ews.pipelines.tasks.matched_kl import apply_scaled, tost

LOGGER = logging.getLogger("ews.crack")
PRUNE = ("sgpt50", "wanda50", "mag40", "wanda", "sgpt", "mag")
QUANT = ("u4", "gptq4", "awq4", "g4", "rtnmix", "gptqmix")


def gsm_mask(corpus: GeneratedCorpus) -> torch.Tensor:
    return torch.tensor([r["source"] == "gsm8k" for r, g in zip(corpus.records, corpus.gen_ids) for _ in g])


def gsm_subcorpus(corpus: GeneratedCorpus, n_seq: int):
    idx = [i for i, r in enumerate(corpus.records) if r["source"] == "gsm8k"][:n_seq]
    offs = [0]
    for g in corpus.gen_ids:
        offs.append(offs[-1] + len(g))
    keep = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in idx])
    return GeneratedCorpus(prompt_ids=[corpus.prompt_ids[i] for i in idx], gen_ids=[corpus.gen_ids[i] for i in idx],
                           records=[corpus.records[i] for i in idx]), keep


def setup(args):
    corpus = GeneratedCorpus.from_dict(json.loads((OUT / "corpora" / f"{slug(args.model)}.json").read_text()))
    ref = torch.load(OUT / slug(args.model) / slug(args.model) / "bf16.pt")
    loaded = load_model(args.model, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    bank = C.WeightBank(loaded.decoder)
    ctrl = make_elastic_depth(loaded.model)
    ctrl.allow_kv_cache = True
    calib: dict = {}

    def calib_fn():
        if "x" not in calib:
            calib["x"] = C.wikitext_calibration(loaded.tokenizer, 128, 512, cache_dir=f"{args.cache_dir}/datasets")
        return calib["x"]
    return corpus, ref, loaded, bank, ctrl, calib_fn


# ------------------------------------------------------------------ parte 1
def part1(args) -> None:
    src = OUT / "matched_kl" / slug(args.model)
    out_f = OUT / "crack" / slug(args.model) / "part1.json"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_f.read_text()) if out_f.exists() else {}
    corpus, ref, loaded, bank, ctrl, calib_fn = setup(args)
    mask = gsm_mask(corpus)
    base = json.loads((src / "base.json").read_text())
    for f in sorted(src.glob("*.json")):
        if f.stem in ("base",) or f.stem.startswith("kv"):
            continue
        r = json.loads(f.read_text())
        method = r["method"]
        if all(f"{method}|{t}" in done for t in r["targets"]):
            continue
        with apply_config(method, loaded, bank, ctrl, calib_fn, args):
            deltas = {t.name: (t.module.weight.detach().float() - t.original.to(loaded.device).float()).to("cpu", torch.float16)
                      for t in bank.targets}
        for tgt, row in r["targets"].items():
            apply_scaled(bank, deltas, row["alpha"])
            s = score_fidelity(loaded, corpus, ref=ref, batch_size=args.score_batch)
            flip = (s["top1"] != ref["top1"])
            done[f"{method}|{tgt}"] = {
                "method": method, "target": float(tgt), "alpha": row["alpha"],
                "kl_mix": s["kl"].double().mean().item(),
                "kl_gsm": s["kl"][mask].double().mean().item(), "kl_mmlu": s["kl"][~mask].double().mean().item(),
                "flip_gsm": flip[mask].double().mean().item(), "flip_mmlu": flip[~mask].double().mean().item(),
                "dacc_gsm": row["gsm8k"]["acc"] - base["gsm8k"]["acc"],
                "dacc_mmlu": row["mmlu_pt"]["acc"] - base["mmlu_pt"]["acc"]}
            LOGGER.info("[%s alvo %s] KL misto %.4f | GSM8K %.4f | MMLU %.4f | dGSM8K %+.3f", method, tgt,
                        done[f"{method}|{tgt}"]["kl_mix"], done[f"{method}|{tgt}"]["kl_gsm"],
                        done[f"{method}|{tgt}"]["kl_mmlu"], done[f"{method}|{tgt}"]["dacc_gsm"])
            out_f.write_text(json.dumps(done, indent=1))
        bank.restore()


# ------------------------------------------------------------------ parte 2
def part2(args) -> None:
    out_f = OUT / "crack" / slug(args.model) / "part2.json"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_f.read_text()) if out_f.exists() else {}
    corpus, ref, loaded, bank, ctrl, calib_fn = setup(args)
    probe, keep = gsm_subcorpus(corpus, args.probe_seqs)
    ref_probe = {k: ref[k][keep] for k in ("top1", "topv", "topi")}
    mask = gsm_mask(corpus)
    used = {r["prompt"] for r in corpus.records}
    prompts = gsm8k_disjoint(loaded.tokenizer, args.n_task, used, args.cache_dir, seed=31337)

    def kl_probe() -> float:
        return score_fidelity(loaded, probe, ref=ref_probe, batch_size=args.score_batch)["kl"].double().mean().item()

    def task() -> dict:
        recs = [dict(r) for r in prompts]
        gen = generate_greedy(loaded, recs, max_new_tokens=args.max_new_tokens, batch_size=args.batch)
        ok = []
        for rec in gen.records:
            a = parse_answer(rec["completion"])
            ok.append(a is not None and abs(a - float(rec["gold"])) < 1e-6)
        return {"acc": sum(ok) / len(ok), "correct": [bool(x) for x in ok]}

    if "base" not in done:
        bank.restore()
        done["base"] = task()
        out_f.write_text(json.dumps(done, indent=1))
        LOGGER.info("bf16 GSM8K %.4f", done["base"]["acc"])

    # precisao mista NATIVA ao longo de uma escada de larguras: x em [0, 4] percorre
    # 8 -> 6 -> 5 -> 4 -> 3 bits; entre degraus vizinhos i e i+1, uma fracao frac(x) das
    # projecoes (ordem aleatoria fixa) desce para a largura i+1. Misturar so 3/4 bits nao
    # alcanca KL abaixo do RTN-4 (bug observado na primeira versao).
    ladder = [8, 6, 5, 4, 3]
    mixes = {}
    order = list(range(len(bank.targets)))
    random.Random(0).shuffle(order)
    rank_of = {j: k for k, j in enumerate(order)}

    def ladder_weights(kind: str):
        if kind in mixes:
            return mixes[kind]
        w = {}
        for bits in ladder:
            name = f"u{bits}" if kind == "rtnmix" else f"gptq{bits}"
            with apply_config(name, loaded, bank, ctrl, calib_fn, args):
                w[bits] = [t.module.weight.detach().to("cpu", torch.float16) for t in bank.targets]
        mixes[kind] = w
        return w

    @torch.no_grad()
    def apply_knob(kind: str, x: float) -> None:
        bank.restore()
        if kind in ("rtnmix", "gptqmix"):
            w = ladder_weights(kind)
            i = min(int(x), len(ladder) - 2)
            frac = x - i
            n_low = int(round(frac * len(order)))
            for j, t in enumerate(bank.targets):
                bits = ladder[i + 1] if rank_of[j] < n_low else ladder[i]
                t.module.weight.copy_(w[bits][j].to(t.module.weight.device, t.module.weight.dtype))
        elif kind == "mag":
            bank.map(lambda w_, t: C.magnitude_prune(w_, x))
        else:
            method = "sparsegpt" if kind == "sgpt" else "wanda"
            C.calibrated_compress(loaded.model, bank, loaded.decoder.layers, calib_fn(), method, sparsity=x,
                                  batch_size=args.calib_batch)

    ranges = {"rtnmix": (0.0, 4.0), "gptqmix": (0.0, 4.0), "mag": (0.0, 0.7), "wanda": (0.0, 0.7), "sgpt": (0.0, 0.75)}
    for kind in args.kinds:
        for target in args.targets:
            key = f"{kind}|{target}"
            if key in done:
                continue
            lo, hi = ranges[kind]
            apply_knob(kind, hi)
            if kl_probe() < target:
                LOGGER.warning("[%s] KL maximo do controle nativo abaixo do alvo %.3f; pulando", kind, target)
                continue
            for _ in range(args.bisect):
                mid = 0.5 * (lo + hi)
                apply_knob(kind, mid)
                if kl_probe() < target:
                    lo = mid
                else:
                    hi = mid
            knob = 0.5 * (lo + hi)
            apply_knob(kind, knob)
            s = score_fidelity(loaded, corpus, ref=ref, batch_size=args.score_batch)
            row = {"kind": kind, "target": target, "knob": knob, "kl_gsm": s["kl"][mask].double().mean().item(),
                   "flip_gsm": (s["top1"] != ref["top1"])[mask].double().mean().item(), **task()}
            row["dacc"] = row["acc"] - done["base"]["acc"]
            done[key] = row
            out_f.write_text(json.dumps(done, indent=1))
            LOGGER.info("[%s alvo %.3f] controle %.3f | KL GSM8K %.4f | flips %.4f | acc %.4f (d %+.4f)", kind, target,
                        knob, row["kl_gsm"], row["flip_gsm"], row["acc"], row["dacc"])
    bank.restore()


# ------------------------------------------------------------------ relatorio
def report(args) -> None:
    root = OUT / "crack"
    print("=== PARTE 1: o KL casado no corpus misto estava casado no GSM8K?")
    pairs_mix, pairs_gsm = [], []
    for mdir in sorted(root.iterdir()):
        f = mdir / "part1.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rows = list(d.values())
        ratio = {fam: np.mean([r["kl_gsm"] / max(r["kl_mix"], 1e-9) for r in rows if r["method"] in grp])
                 for fam, grp in (("poda", PRUNE), ("quant", QUANT))}
        print(f"{mdir.name}: KL_GSM8K / KL_misto  poda {ratio['poda']:.2f} | quant {ratio['quant']:.2f}")
        # casamento no KL de GSM8K: curva dacc ~ sqrt(KL_gsm) ajustada so na quantizacao
        q = [r for r in rows if r["method"] in QUANT]
        p = [r for r in rows if r["method"] in PRUNE]
        if len(q) >= 3 and p:
            c = np.polyfit(np.sqrt([r["kl_gsm"] for r in q]), [r["dacc_gsm"] for r in q], 1)
            for tgt in sorted({r["target"] for r in rows}):
                pm = [r for r in p if r["target"] == tgt]
                qm = [r for r in q if r["target"] == tgt]
                if pm and qm:
                    pairs_mix.append(np.mean([r["dacc_gsm"] for r in pm]) - np.mean([r["dacc_gsm"] for r in qm]))
                    resid = [r["dacc_gsm"] - np.polyval(c, math.sqrt(r["kl_gsm"])) for r in pm]
                    qres = [r["dacc_gsm"] - np.polyval(c, math.sqrt(r["kl_gsm"])) for r in qm]
                    pairs_gsm.append(np.mean(resid) - np.mean(qres))
    if len(pairs_mix) >= 2:
        for name, pr in (("casado no KL MISTO (original)", pairs_mix), ("casado no KL de GSM8K", pairs_gsm)):
            t = tost(pr, 0.03)
            print(f"   poda - quant, {name}: {100*t['mean']:+.2f}pp IC95 [{100*t['ci95'][0]:+.2f},{100*t['ci95'][1]:+.2f}] "
                  f"p_TOST={t['p_tost']:.4f} -> {'EQUIVALENTE' if t['equivalente'] else 'nao equivalente'} (n={t['n']})")
    print("\n=== PARTE 2: controles nativos casados no KL de GSM8K")
    pairs = []
    for mdir in sorted(root.iterdir()):
        f = mdir / "part2.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rows = [v for k, v in d.items() if k != "base"]
        for r in sorted(rows, key=lambda r: (r["target"], r["kind"])):
            print(f"   {mdir.name:26s} {r['kind']:8s} alvo {r['target']:.2f} controle {r['knob']:.3f} KL_gsm {r['kl_gsm']:.4f} "
                  f"flips {100*r['flip_gsm']:5.2f}% acc {100*r['acc']:5.1f} ({100*r['dacc']:+5.1f})")
        for tgt in sorted({r["target"] for r in rows}):
            pm = [r["dacc"] for r in rows if r["target"] == tgt and r["kind"] in PRUNE]
            qm = [r["dacc"] for r in rows if r["target"] == tgt and r["kind"] in QUANT]
            if pm and qm:
                pairs.append(np.mean(pm) - np.mean(qm))
    if len(pairs) >= 2:
        t = tost(pairs, 0.03)
        print(f"   poda - quant (nativo, KL de GSM8K): {100*t['mean']:+.2f}pp IC95 [{100*t['ci95'][0]:+.2f},"
              f"{100*t['ci95'][1]:+.2f}] p_TOST={t['p_tost']:.4f} -> {'EQUIVALENTE' if t['equivalente'] else 'nao equivalente'} (n={t['n']})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["part1", "part2", "report"])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--kinds", nargs="+", default=["rtnmix", "gptqmix", "mag", "wanda", "sgpt"])
    p.add_argument("--targets", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    p.add_argument("--probe-seqs", type=int, default=64)
    p.add_argument("--bisect", type=int, default=8)
    p.add_argument("--n-task", type=int, default=400)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--score-batch", type=int, default=8)
    p.add_argument("--calib-batch", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    if args.stage != "report":
        torch.cuda.set_device(torch.device(args.device))
    {"part1": part1, "part2": part2, "report": report}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
