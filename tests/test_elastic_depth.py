"""Testes do wrapper de layer-skipping (passo 3) num Gemma3 minusculo.

Roda em CPU, em segundos, sem precisar do checkpoint gated do Gemma:
    python tests/test_elastic_depth.py
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import Gemma3TextConfig, Gemma3ForCausalLM
from glod.core.elastic_depth import make_elastic_depth, uniform_skip_schedule, SkippableDecoderLayer
from glod.core.model_loader import resolve_decoder

torch.manual_seed(0)
cfg = Gemma3TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                       num_hidden_layers=12, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=16, sliding_window=8, max_position_embeddings=128)
m = Gemma3ForCausalLM(cfg).eval()
print("layer_types:", cfg.layer_types)
ids = torch.randint(0, 512, (2, 17))

with torch.no_grad():
    ref = m(input_ids=ids, use_cache=False).logits.clone()

ctrl = make_elastic_depth(m)
print("num_layers:", ctrl.num_layers, "| bytes/layer:", ctrl.layer_param_bytes[0])
print("wrapper type:", type(resolve_decoder(m).layers[0]).__name__)
print("attention_type preservado:", [resolve_decoder(m).layers[i].attention_type for i in range(6)])

with torch.no_grad():
    dense = m(input_ids=ids, use_cache=False).logits
assert torch.equal(dense, ref), "wrapper alterou a saida densa!"
print("OK: wrapper com todas as camadas ON == modelo original (bit-exato)")

# skip real muda a saida
sched = uniform_skip_schedule(12, 4)
print("schedule(12,4):", sched)
with ctrl.skipping(sched), torch.no_grad():
    print("  summary:", ctrl.summary())
    out = m(input_ids=ids, use_cache=False).logits
    assert not torch.equal(out, ref)
    print("  OK: saida mudou. max|d| =", (out-ref).abs().max().item())
print("restaurado:", ctrl.active_layers == list(range(12)))

# identidade verificada: pular TODAS as camadas do miolo
with ctrl.skipping(range(12)), torch.no_grad():
    hs = m.model(input_ids=ids, use_cache=False).last_hidden_state
    emb = m.model.embed_tokens(ids)
    assert torch.allclose(hs, m.model.norm(emb), atol=1e-5), "skip nao e identidade!"
    print("OK: skip total == norm(embeddings) -> identidade no residual confirmada")

# guarda de KV cache
try:
    with ctrl.skipping([5]), torch.no_grad():
        m(input_ids=ids, use_cache=True)
    print("FALHA: guarda de use_cache nao disparou")
except RuntimeError as e:
    print("OK: guarda de KV cache disparou ->", str(e)[:70], "...")

# output_attentions nao quebra
with ctrl.skipping([5]), torch.no_grad():
    o = m(input_ids=ids, use_cache=False, output_attentions=True)
    print("OK: output_attentions ->", len(o.attentions), "entradas, idx5 =", o.attentions[5])

# generate funciona com mascara densa
with torch.no_grad():
    g = m.generate(input_ids=ids[:1, :5], max_new_tokens=4, do_sample=False)
print("OK: generate denso ->", g.shape)

# idempotencia + detach
ctrl2 = make_elastic_depth(m)
assert not isinstance(ctrl2.wrapped[0].layer, SkippableDecoderLayer), "wrapper aninhou!"
print("OK: make_elastic_depth idempotente")
ctrl2.detach()
with torch.no_grad():
    assert torch.equal(m(input_ids=ids, use_cache=False).logits, ref)
print("OK: detach restaurou o modelo original")

for k in [0,1,3,6,9,11,12]:
    print(f"  uniform_skip_schedule(12,{k}) = {uniform_skip_schedule(12,k)}")
