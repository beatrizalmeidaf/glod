"""Regressao: layer-skipping em familias cujo DecoderLayer devolve tensor puro.

Em transformers 4.56, Qwen3/Mistral/OLMo2/Phi3 consomem a saida da camada
direto (`hidden_states = decoder_layer(...)`); o wrapper devolvia `(h,)` e o
RMSNorm seguinte quebrava com `'tuple' object has no attribute 'dtype'`.

    python tests/test_elastic_depth_families.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import (MistralConfig, MistralForCausalLM, Olmo2Config, Olmo2ForCausalLM,
                          Phi3Config, Phi3ForCausalLM, Qwen3Config, Qwen3ForCausalLM)

from ews.core.elastic_depth import make_elastic_depth
from ews.core.model_loader import resolve_decoder

common = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
families = {
    "qwen3": (Qwen3Config(head_dim=16, **common), Qwen3ForCausalLM),
    "mistral": (MistralConfig(head_dim=16, **common), MistralForCausalLM),
    "olmo2": (Olmo2Config(**common), Olmo2ForCausalLM),
    "phi3": (Phi3Config(pad_token_id=0, **common), Phi3ForCausalLM),
}
torch.manual_seed(0)
ids = torch.randint(1, 256, (2, 11))
for name, (cfg, cls) in families.items():
    m = cls(cfg).eval()
    with torch.no_grad():
        ref = m(input_ids=ids, use_cache=False).logits.clone()
    ctrl = make_elastic_depth(m)
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids, use_cache=False).logits, ref), name
        with ctrl.skipping([2]):
            out = m(input_ids=ids, use_cache=False).logits
        assert not torch.equal(out, ref), f"{name}: skip nao mudou a saida"
        with ctrl.skipping(range(6)):
            hs = resolve_decoder(m)(input_ids=ids, use_cache=False).last_hidden_state
            emb = resolve_decoder(m).embed_tokens(ids)
            assert torch.allclose(hs, resolve_decoder(m).norm(emb), atol=1e-5), f"{name}: nao e identidade"
    print(f"OK {name}: denso bit-exato, skip muda a saida, skip total == norm(emb)")
