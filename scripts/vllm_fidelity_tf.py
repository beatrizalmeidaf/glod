#!/usr/bin/env python3
"""Passo 2 (venv do vLLM): teacher forcing das sequencias do passo 1 no vLLM (kernel real;
AWQ-Marlin para o modelo exportado) e fracao de posicoes em que o top-1 difere do token
gerado pela simulacao no HF. Acrescenta o resultado em --json.

    python scripts/vllm_fidelity_tf.py <pasta do modelo> <prefixo> <nome> --json <arquivo>
"""
import argparse, json, os
from vllm import LLM, SamplingParams
ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("prefix"); ap.add_argument("name"); ap.add_argument("--json", required=True)
a = ap.parse_args()
prompts = json.load(open(a.prefix + ".prompts.json")); gens = json.load(open(a.prefix + ".hf.json"))
llm = LLM(model=a.path, gpu_memory_utilization=0.25, max_model_len=2048, enforce_eager=True, seed=0, attention_backend="FLASH_ATTN")
res = llm.generate([{"prompt_token_ids": p + g} for p, g in zip(prompts, gens)], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
dis = tot = 0
for r, p, g in zip(res, prompts, gens):
    for j, tok in enumerate(g):
        d = r.prompt_logprobs[len(p) + j]
        dis += max(d.items(), key=lambda kv: kv[1].logprob)[0] != tok; tot += 1
out = json.load(open(a.json)) if os.path.exists(a.json) else {}
out[a.name] = {"disagree": dis, "positions": tot, "rate": dis / tot, "quantization": str(llm.llm_engine.model_config.quantization)}
json.dump(out, open(a.json, "w"), indent=1)
print(a.name, out[a.name])
