"""Validacoes do simulador de quantizacao (Fase 1b).

Roda em CPU em segundos, sem checkpoint gated:
    python tests/test_quantize.py

A intencao e pegar erro AQUI, e nao depois de 20 min de H100 gerando uma
curva bits x acuracia silenciosamente errada.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from ews.core.model_loader import resolve_decoder
from ews.core.quantize import (
    QuantizationSimulator,
    effective_bits,
    quantize_dequantize,
)

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "OK  " if condition else "FALHA"
    print(f"  [{status}] {label}{(' | ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


# ---------------------------------------------------------------- QDQ puro
print("\n=== quantize_dequantize ===")
torch.manual_seed(0)
W = torch.randn(256, 512)

check(torch.equal(quantize_dequantize(W, 16), W), "bits>=16 e identidade exata")
check(quantize_dequantize(W, 4).shape == W.shape, "shape preservado")
check(quantize_dequantize(W, 4).dtype == W.dtype, "dtype preservado")

errs = {b: rel_err(quantize_dequantize(W, b, 128), W) for b in (2, 3, 4, 8)}
print(f"       erro relativo por bitrate: { {b: round(e, 5) for b, e in errs.items()} }")
check(
    errs[2] > errs[3] > errs[4] > errs[8],
    "erro decresce monotonicamente com os bits",
)
check(errs[8] < 0.01, "8-bit quase sem perda", f"rel_err={errs[8]:.5f}")

fine = rel_err(quantize_dequantize(W, 4, 32), W)
coarse = rel_err(quantize_dequantize(W, 4, 512), W)
check(fine < coarse, "grupo menor -> erro menor", f"g32={fine:.5f} < g512={coarse:.5f}")

# dimensao nao divisivel pelo grupo: a sobra vira um grupo proprio
odd = torch.randn(8, 300)
q_odd = quantize_dequantize(odd, 4, 128)
check(q_odd.shape == odd.shape, "dim nao divisivel pelo grupo nao quebra")
check(rel_err(q_odd, odd) < 0.2, "erro razoavel no caso nao divisivel",
      f"rel_err={rel_err(q_odd, odd):.5f}")

# grupo constante: sem divisao por zero, e reconstrucao exata
const = torch.full((4, 128), 0.7)
check(torch.allclose(quantize_dequantize(const, 2, 128), const),
      "grupo constante reconstruido exatamente (sem div/0)")

# valores extremos do grupo sao sempre representaveis
g = torch.randn(1, 128)
qg = quantize_dequantize(g, 2, 128)
check(torch.isclose(qg.min(), g.min()) and torch.isclose(qg.max(), g.max()),
      "min/max do grupo preservados pela quantizacao afim")

check(not torch.isnan(quantize_dequantize(W, 2)).any(), "sem NaN em 2-bit")
check(
    len(torch.unique(quantize_dequantize(g, 2, 128))) <= 4,
    "2-bit usa no maximo 4 niveis por grupo",
)

# --------------------------------------------------------- bits efetivos
print("\n=== effective_bits ===")
check(effective_bits(4, 128) == 4 + 32 / 128, "4-bit g128 = 4.25 bits/peso",
      f"{effective_bits(4, 128)}")
check(effective_bits(16, 128) == 16.0, "bf16 nao paga overhead de grupo")
check(effective_bits(2, 64) > effective_bits(2, 256), "grupo menor custa mais bytes")

# ------------------------------------------------- simulador end-to-end
print("\n=== QuantizationSimulator em Gemma3 ===")
torch.manual_seed(0)
cfg = Gemma3TextConfig(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=8,
    max_position_embeddings=128,
)
model = Gemma3ForCausalLM(cfg).eval()
decoder = resolve_decoder(model)
ids = torch.randint(0, 256, (2, 11))
with torch.no_grad():
    ref = model(input_ids=ids, use_cache=False).logits.clone()

sim = QuantizationSimulator(decoder, group_size=32)
check(sim.num_layers == 6, "detectou as 6 camadas", f"n={sim.num_layers}")
check(len(sim.targets) == 6 * 7, "7 projecoes por camada", f"n={len(sim.targets)}")
check(all("layers." in t.name for t in sim.targets), "so projecoes dentro de layers.")
check(
    not any(n.endswith(("lm_head", "embed_tokens")) for n in (t.name for t in sim.targets)),
    "embeddings e lm_head ficam fora",
)

emb_before = model.model.embed_tokens.weight.clone()
sim.apply_uniform(2)
with torch.no_grad():
    out2 = model(input_ids=ids, use_cache=False).logits.clone()
check(not torch.equal(out2, ref), "2-bit muda a saida")
check(torch.equal(model.model.embed_tokens.weight, emb_before),
      "embeddings intactos apos quantizar")

sim.restore()
with torch.no_grad():
    out_restored = model(input_ids=ids, use_cache=False).logits
check(torch.equal(out_restored, ref), "restore() e bit-exato")

# idempotencia: aplicar duas vezes nao acumula erro de requantizacao
sim.apply_uniform(4)
w_once = sim.targets[0].module.weight.clone()
sim.apply_uniform(4)
check(torch.equal(sim.targets[0].module.weight, w_once),
      "apply repetido e idempotente (parte sempre do original)")

# monotonicidade da degradacao no modelo inteiro
sim.restore()
dists = {}
for b in (2, 3, 4, 8):
    sim.apply_uniform(b)
    with torch.no_grad():
        dists[b] = rel_err(model(input_ids=ids, use_cache=False).logits, ref)
sim.restore()
print(f"       desvio dos logits por bitrate: { {b: round(d, 5) for b, d in dists.items()} }")
check(dists[2] > dists[4] > dists[8], "degradacao do modelo cresce ao baixar bits")

# precisao por camada: so as camadas pedidas mudam
sim.apply({0: 2, 3: 2})
changed = {
    t.layer_idx for t in sim.targets
    if not torch.equal(t.module.weight, t.original)
}
check(changed == {0, 3}, "apply por camada afeta exatamente as camadas pedidas",
      f"mudaram={sorted(changed)}")
check(sim.current_bits == {0: 2, 3: 2}, "current_bits reflete o estado")
sim.apply({0: 2})
changed = {t.layer_idx for t in sim.targets if not torch.equal(t.module.weight, t.original)}
check(changed == {0}, "camada omitida volta a bf16 automaticamente",
      f"mudaram={sorted(changed)}")
sim.restore()

# ---------------------------------------------------------- contabilidade
print("\n=== contabilidade de bytes ===")
total_numel = sum(t.numel for t in sim.targets)
full = sim.weight_bytes({})
check(abs(full - total_numel * 2) < 1e-6, "bf16 = 2 bytes/peso",
      f"{full} vs {total_numel * 2}")

b4 = sim.weight_bytes({i: 4 for i in range(6)})
expected = total_numel * effective_bits(4, 32) / 8
check(abs(b4 - expected) < 1e-6, "4-bit bate a formula analitica")
check(b4 < full, "4-bit le menos bytes que bf16", f"{b4:.0f} < {full:.0f}")

nd = 12345
check(abs(sim.weight_bytes({}, non_target_bytes=nd) - (full + nd)) < 1e-6,
      "non_target_bytes entra somando")

# delta_bytes: promover TODAS as camadas deve fechar a conta exatamente
base_all = sim.weight_bytes({i: 4 for i in range(6)})
delta_all = sim.delta_bytes(4, range(6))
check(abs(base_all + delta_all - full) < 1e-6,
      "base(4-bit) + delta(todas) == bf16 cheio",
      f"{base_all:.0f} + {delta_all:.0f} = {base_all + delta_all:.0f} vs {full:.0f}")
half = sim.delta_bytes(4, [0, 1, 2])
check(abs(2 * half - delta_all) < 1e-3, "delta e aditivo entre camadas iguais")
check(sim.delta_bytes(4, []) == 0.0, "delta de conjunto vazio e zero")
check(sim.delta_bytes(2, range(6)) > sim.delta_bytes(4, range(6)),
      "base mais agressiva -> delta maior")

# ------------------------------------------------ composicao com layer skip
print("\n=== quantizacao + elastic depth juntos ===")
from ews.core.elastic_depth import make_elastic_depth

ctrl = make_elastic_depth(model)
sim2 = QuantizationSimulator(resolve_decoder(model), group_size=32)
check(len(sim2.targets) == 6 * 7,
      "simulador ainda enxerga as projecoes atraves do SkippableDecoderLayer",
      f"n={len(sim2.targets)}")
sim2.apply_uniform(4)
with ctrl.skipping([1, 4]), torch.no_grad():
    out = model(input_ids=ids, use_cache=False).logits
check(out.shape == ref.shape and not torch.isnan(out).any(),
      "skip + quantizacao compoem sem quebrar")
sim2.restore()
ctrl.detach()
with torch.no_grad():
    check(torch.equal(model(input_ids=ids, use_cache=False).logits, ref),
          "estado original recuperado apos skip + quantizacao")

# ------------------------------------------------- aplicacao incremental
print("\n=== aplicacao incremental ===")
sim.restore()
cfg_a = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
cfg_b = {0: 3, 1: 3, 2: 16, 3: 3, 4: 8, 5: 3}

sim.apply(cfg_a)
sim.apply(cfg_b, incremental=True)
inc = [t.module.weight.clone() for t in sim.targets]
sim.restore()
sim.apply(cfg_b, incremental=False)
full_apply = [t.module.weight.clone() for t in sim.targets]
check(all(torch.equal(a, b) for a, b in zip(inc, full_apply)),
      "incremental == aplicacao completa (bit-exato)")
check(sim.bits_map == {0: 3, 1: 3, 2: 16, 3: 3, 4: 8, 5: 3},
      "bits_map cobre todas as camadas", str(sim.bits_map))
check(sim.current_bits == {0: 3, 1: 3, 3: 3, 4: 8, 5: 3},
      "current_bits omite as camadas em bf16", str(sim.current_bits))

# a cascata nao pode acumular erro: 3b depois de 2b == 3b a partir do original
sim.restore()
sim.apply_uniform(2)
sim.apply_uniform(3)
cascaded = sim.targets[0].module.weight.clone()
sim.restore()
sim.apply_uniform(3)
check(torch.equal(cascaded, sim.targets[0].module.weight),
      "requantizar apos outra precisao nao acumula erro")
sim.restore()
check(all(torch.equal(t.module.weight, t.original) for t in sim.targets),
      "restore() apos sequencia de applies volta ao original")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FALHA(S): {FAILURES}")
    sys.exit(1)
print("todos os testes passaram")
