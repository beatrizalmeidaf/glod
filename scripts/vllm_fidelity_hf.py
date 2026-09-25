#!/usr/bin/env python3
"""Passo 1 (ambiente do glod): prompts GSM8K e respostas greedy do Qwen3-4B no HF, densas
ou com a variante AMQ/GPTQ simulada. O passo 2 (scripts/vllm_fidelity_tf.py, no venv do
vLLM) pontua essas mesmas sequencias com o kernel real e conta os top-1 diferentes.

    python3 scripts/vllm_fidelity_hf.py <dense|gptq4> <prefixo de saida>
"""
import json, sys
import torch
from glod.core import compressors as C
from glod.core.model_loader import load_model
from glod.pipelines.compress.amq_eval import materialize
from glod.pipelines.tasks.closedloop import gsm8k_disjoint
torch.set_grad_enabled(False)
m, variant, out = "Qwen/Qwen3-4B", sys.argv[1], sys.argv[2]
L = load_model(m, role="cfg", device="cuda:0", dtype="bfloat16")
recs = gsm8k_disjoint(L.tokenizer, 16, set(), "/local/user_beatrizalmeida/hf_cache", seed=4242)
prompts = [L.tokenizer(r["prompt"], add_special_tokens=False).input_ids for r in recs]
json.dump(prompts, open(out + ".prompts.json", "w"))
bank = C.WeightBank(L.decoder)
if variant != "dense":
    materialize(L, bank, m, variant)
gens = []
for p in prompts:
    o = L.model.generate(torch.tensor([p], device="cuda:0"), max_new_tokens=64, do_sample=False, top_p=None, top_k=None)
    gens.append(o[0, len(p):].tolist())
json.dump(gens, open(out + ".hf.json", "w"))
