"""Sanidade da troca de precisao por token com KV misto (`glod adaptive-closedloop`).

Com promocao sempre ligada e alta = bf16, a geracao deve ser identica ao greedy do bf16
puro; com promocao desligada, identica ao greedy da base. Qualquer erro no alinhamento
passo/posicao do merge de KV quebra essas igualdades.

    python tests/test_adaptive_closedloop.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glod.core import compressors as C
from glod.core.closed_loop import cascade_generate
from glod.core.model_loader import load_model
from glod.corpora.token_oracle import _chat, _eos_ids
from glod.pipelines.adaptive.closedloop import generate_policy
from glod.paths import CACHE_DIR

dev = os.environ.get("DEV", "cuda:0")
torch.cuda.set_device(torch.device(dev))
cache = CACHE_DIR
base = load_model("Qwen/Qwen3-4B", role="base", device=dev, cache_dir=cache)
high = load_model("Qwen/Qwen3-4B", role="high", device=dev, cache_dir=cache)
C.WeightBank(base.decoder).map(lambda w, t: C.rtn(w, 3))
eos = _eos_ids(base)
prompts = [_chat(base.tokenizer, q) for q in ("What is 17*23? Answer: <number>",
                                               "Name three primary colors, briefly.")]
with torch.no_grad():
    ref_high = cascade_generate(high, None, prompts, threshold=float("inf"), eos_ids=eos, max_new_tokens=40).gen_ids
    ref_base = cascade_generate(base, None, prompts, threshold=float("inf"), eos_ids=eos, max_new_tokens=40).gen_ids
    always = generate_policy(base, high, prompts, policy="cascade", thr=-1.0, eos=eos, max_new=40)["gen"]
    never = generate_policy(base, high, prompts, policy="cascade", thr=float("inf"), eos=eos, max_new=40)["gen"]
    always_hp = generate_policy(base, high, prompts, policy="cascade", thr=-1.0, eos=eos, max_new=40,
                                prefill="high")["gen"]
# a primeira posicao do prompt vem da base (prefill na base), entao o "sempre alta" so pode
# divergir do bf16 puro por causa do KV do prompt; comparamos tambem com prefill em alta
print("nunca promover == greedy da base :", never == ref_base)
print("sempre promover == greedy do bf16 (com prompt em 3 bits):", always == ref_high,
      "| tokens iguais:", sum(a == b for x, y in zip(always, ref_high) for a, b in zip(x, y)),
      "de", sum(len(x) for x in ref_high))
print("sempre promover com prompt em alta == greedy do bf16:", always_hp == ref_high)
assert never == ref_base, "politica sem promocao divergiu da base: merge de KV incorreto"
assert always_hp == ref_high, "politica sempre-alta divergiu do bf16: merge de KV incorreto"
print("OK: merge de KV alinhado nas duas direcoes")
