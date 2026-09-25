#!/usr/bin/env python3
"""glod report: fidelidade de decisao de um modelo comprimido, pronta para reportar.

Dado o modelo denso e um comprimido, gera corpora pequenos de referencia (greedy do
modelo denso; texto natural sem geracao), pontua o comprimido em teacher forcing e
escreve report.json e report.md com, por corpus:

  flips (taxa de mudanca do arg-max), TV, KL (top-64 + cauda), flips/TV,
  kappa = flips/sqrt(KL) e o fator de Jensen J (por que o KL muda de corpus para corpus),
  aceitacao especulativa estimada: greedy ~ 1 - flips, amostragem ~ 1 - TV, e tokens
  por rodada para k rascunhos (rodadas independentes; superestima ~2 pontos),

mais o protocolo de reporte de 4 pontos do paper, marcando o que o relatorio cobre.

O comprimido pode ser:
  --compressed cfg:<nome>     qualquer configuracao da gramatica do grid (u4, gptq4, awq4,
                              wanda50, kv4, ofc@<repo>, ...), aplicada sobre o denso
  --compressed <repo|pasta>   outro checkpoint com o mesmo tokenizador (ex.: um draft)

    python -m glod report --dense Qwen/Qwen3-4B --compressed cfg:gptq4 --out relatorio/
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from glod.core import compressors as C
from glod.core.elastic_depth import make_elastic_depth
from glod.core.fidelity import score_fidelity
from glod.core.model_loader import load_model
from glod.corpora.registry import build_records
from glod.corpora.token_oracle import generate_greedy, natural_corpus
from glod.paths import CACHE_DIR as DEFAULT_CACHE_DIR
from glod.pipelines.fidelity.build_grid import apply_config, use_fp32_logits

LOGGER = logging.getLogger("glod.report")
PROTOCOL = [
    ("tv_per_corpus", "Report total variation next to KL, per corpus."),
    ("kl_within_corpus", "Compare KL values only within one corpus and one reference model."),
    ("flip_rate", "When the consumer needs exact agreement, report the flip rate itself; screen drafts by TV on task prompts."),
    ("matched_accuracy", "When comparing accuracy at matched divergence, report lost and repaired answers separately, "
                         "next to Gaussian noise of the same divergence over several seeds."),
]


def summarize(ref: dict, res: dict, k_draft: int) -> dict:
    flip = (res["top1"] != ref["top1"]).double().mean().item()
    kl_t = res["kl"].double().clamp_min(0)
    tv = res["tv"].double().mean().item()
    kl = kl_t.mean().item()
    J = kl_t.sqrt().mean().item() / math.sqrt(kl) if kl > 0 else float("nan")

    def tpr(a: float) -> float:          # tokens por rodada: aceitos + o do alvo
        return (1 - a ** (k_draft + 1)) / (1 - a) if a < 1 else k_draft + 1.0
    return {"n_tokens": int(ref["top1"].numel()), "flips": flip, "tv": tv, "kl": kl,
            "flips_over_tv": flip / tv if tv > 0 else float("nan"),
            "kappa": flip / math.sqrt(kl) if kl > 0 else float("nan"), "jensen_J": J,
            "spec_greedy_accept": 1 - flip, "spec_sample_accept": 1 - tv,
            "spec_greedy_tokens_per_round": tpr(1 - flip), "spec_sample_tokens_per_round": tpr(1 - tv)}


def markdown(rep: dict) -> str:
    L = [f"# Decision-fidelity report", "",
         f"- dense model: `{rep['dense']}`", f"- compressed: `{rep['compressed']}`",
         f"- {rep['n_prompts']} prompts per corpus, greedy continuations of up to {rep['max_new_tokens']} tokens "
         f"(natural text is scored as is), teacher forcing; KL and TV over the dense top-64 plus the tail.", "",
         "| corpus | tokens | flips | TV | KL | flips/TV | κ = flips/√KL | J | greedy accept | sampled accept |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, s in rep["corpora"].items():
        L.append(f"| {c} | {s['n_tokens']} | {100 * s['flips']:.2f}% | {s['tv']:.4f} | {s['kl']:.4f} | "
                 f"{s['flips_over_tv']:.2f} | {s['kappa']:.3f} | {s['jensen_J']:.2f} | "
                 f"{100 * s['spec_greedy_accept']:.1f}% | {100 * s['spec_sample_accept']:.1f}% |")
    L += ["", "**Reading it.** Flips/TV near one means total variation is a calibration-free proxy for changed "
          "decisions on that corpus. κ and J vary with the corpus: the same KL implies different flip rates "
          "on different text, because KL is averaged over tokens before its square root is taken (J measures how "
          "concentrated the divergence is). Speculative estimates are per proposed token; tokens per round for "
          f"k = {rep['k_draft']} drafts assume independent rounds, which overestimates acceptance by about 2 points.",
          ""]
    if rep.get("kl_rank_warning"):
        L += [f"> **KL and TV rank the corpora differently here** ({rep['kl_rank_warning']}). "
              "Compare KL values only within a corpus.", ""]
    L += ["## Reporting protocol", ""]
    for key, text in PROTOCOL:
        mark = "x" if rep["protocol"][key] else " "
        L.append(f"- [{mark}] {text}")
    L += ["", "Item 4 needs task accuracy, which this report does not compute."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dense", required=True)
    p.add_argument("--compressed", required=True)
    p.add_argument("--corpora", nargs="+", default=["gsm8k", "mmlu_en", "wikitext_nat", "code"])
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--k-draft", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dense = load_model(args.dense, role="ref", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
    use_fp32_logits(dense.model)
    corpora, refs = {}, {}
    for name in args.corpora:
        records = build_records(name, dense.tokenizer, args.n_prompts, args.seed, f"{args.cache_dir}/datasets")
        corpora[name] = (natural_corpus(dense.tokenizer, records) if records and "continuation" in records[0]
                         else generate_greedy(dense, records, max_new_tokens=args.max_new_tokens,
                                              batch_size=args.batch_size * 4))
        refs[name] = score_fidelity(dense, corpora[name], ref=None, batch_size=args.batch_size)
        LOGGER.info("referencia %s: %d tokens", name, corpora[name].n_tokens)

    results = {}
    if args.compressed.startswith("cfg:"):
        cfg = args.compressed[4:]
        bank = C.WeightBank(dense.decoder)
        ctrl = make_elastic_depth(dense.model)
        cache: dict = {}

        def calib_fn():
            if "x" not in cache:
                cache["x"] = C.wikitext_calibration(dense.tokenizer, 128, 512, cache_dir=f"{args.cache_dir}/datasets")
            return cache["x"]
        ns = SimpleNamespace(calib_batch=8, cache_dir=args.cache_dir)
        with apply_config(cfg, dense, bank, ctrl, calib_fn, ns):
            for name in args.corpora:
                results[name] = summarize(refs[name], score_fidelity(dense, corpora[name], ref=refs[name],
                                                                      batch_size=args.batch_size), args.k_draft)
    else:
        comp = load_model(args.compressed, role="cfg", device=args.device, dtype="bfloat16", cache_dir=args.cache_dir)
        use_fp32_logits(comp.model)
        for name in args.corpora:
            results[name] = summarize(refs[name], score_fidelity(comp, corpora[name], ref=refs[name],
                                                                  batch_size=args.batch_size), args.k_draft)
    for name, s in results.items():
        LOGGER.info("%-14s flips %.4f  TV %.4f  KL %.4f  flips/TV %.2f", name, s["flips"], s["tv"], s["kl"], s["flips_over_tv"])

    order_kl = sorted(results, key=lambda c: results[c]["kl"])
    order_tv = sorted(results, key=lambda c: results[c]["tv"])
    rep = {"dense": args.dense, "compressed": args.compressed, "n_prompts": args.n_prompts,
           "max_new_tokens": args.max_new_tokens, "k_draft": args.k_draft, "corpora": results,
           "kl_rank_warning": None if order_kl == order_tv else f"KL: {' < '.join(order_kl)}; TV: {' < '.join(order_tv)}",
           "protocol": {"tv_per_corpus": True, "kl_within_corpus": True, "flip_rate": True, "matched_accuracy": False}}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    (out / "report.md").write_text(markdown(rep))
    LOGGER.info("relatorio em %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
