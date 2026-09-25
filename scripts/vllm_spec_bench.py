#!/usr/bin/env python3
"""Tempo real da decodificacao especulativa no vLLM, com drafts quantizados de verdade.

Roda no venv do vLLM (nao no ambiente do glod):

    /local/user_beatrizalmeida/venv_vllm/bin/python scripts/vllm_spec_bench.py \\
        --target <pasta do modelo denso> --drafts gptq4=<pasta> amq=<pasta> \\
        --prompts prompts.json --out results.json

Para cada draft (e sem draft), gera as respostas dos mesmos prompts, uma requisicao por
vez (lote 1, o regime em que a especulativa compensa), greedy e por amostragem (T = 1),
e mede tokens gerados por segundo de parede, aceitacao por token proposto e tokens por
rodada, a partir das metricas do proprio vLLM. Cada configuracao roda num processo novo,
para que uma nao aqueca o cache da outra.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def run_one(args) -> dict:
    from vllm import LLM, SamplingParams
    spec = None
    if args.draft:
        spec = {"method": "draft_model", "model": args.draft, "num_speculative_tokens": args.k}
    llm = LLM(model=args.target, gpu_memory_utilization=args.mem, max_model_len=args.max_model_len, seed=0,
              attention_backend="FLASH_ATTN", speculative_config=spec, disable_log_stats=False)
    prompts = json.load(open(args.prompts))
    out = {}
    for mode in ("greedy", "sample"):
        sp = SamplingParams(temperature=0.0 if mode == "greedy" else 1.0, max_tokens=args.max_tokens, seed=0)
        llm.generate([{"prompt_token_ids": prompts[0]}], SamplingParams(temperature=0, max_tokens=16))   # aquecimento
        before = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
        t0 = time.perf_counter()
        n_tok = 0
        for p in prompts:
            r = llm.generate([{"prompt_token_ids": p}], sp, use_tqdm=False)
            n_tok += len(r[0].outputs[0].token_ids)
        dt = time.perf_counter() - t0
        after = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
        delta = lambda k: (after.get(k) or 0) - (before.get(k) or 0)  # noqa: E731
        drafts = delta("vllm:spec_decode_num_draft_tokens")
        acc = delta("vllm:spec_decode_num_accepted_tokens")
        drafts_n = delta("vllm:spec_decode_num_drafts")
        out[mode] = {"tokens": n_tok, "seconds": dt, "tokens_per_s": n_tok / dt,
                     "accept_rate": acc / drafts if drafts else None,
                     "mean_accepted_per_draft": acc / drafts_n if drafts_n else None}
        print(json.dumps({"draft": args.draft, "mode": mode, **out[mode]}), flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True)
    p.add_argument("--drafts", nargs="*", default=[], help="nome=pasta")
    p.add_argument("--prompts", required=True, help="json: lista de listas de ids")
    p.add_argument("--out")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--mem", type=float, default=0.5)
    p.add_argument("--draft", default=None, help=argparse.SUPPRESS)       # modo filho
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        print("RESULT " + json.dumps(run_one(args)), flush=True)
        return 0
    results = {}
    for name, path in [("none", None)] + [tuple(d.split("=", 1)) for d in args.drafts]:
        cmd = [sys.executable, __file__, "--child", "--target", args.target, "--prompts", args.prompts,
               "--k", str(args.k), "--max-tokens", str(args.max_tokens), "--max-model-len", str(args.max_model_len),
               "--mem", str(args.mem)] + (["--draft", path] if path else [])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(proc.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"falhou: {name}")
        results[name] = json.loads(line[7:])
        base = results["none"]
        for mode in ("greedy", "sample"):
            r = results[name][mode]
            r["speedup"] = r["tokens_per_s"] / base[mode]["tokens_per_s"]
        print(name, {m: {k: round(v, 4) if isinstance(v, float) else v for k, v in results[name][m].items()}
                     for m in ("greedy", "sample")}, flush=True)
        if args.out:
            json.dump(results, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
