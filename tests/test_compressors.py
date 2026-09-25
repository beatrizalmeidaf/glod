"""Testes dos compressores reais em CPU/GPU pequena.

    python tests/test_compressors.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import Qwen3Config, Qwen3ForCausalLM

from glod.core import compressors as C
from glod.core.model_loader import resolve_decoder

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# --- nivel de matriz: dados correlacionados, como ativacoes reais
d_in, d_out, n = 256, 96, 4096
A = torch.randn(d_in, d_in, device=dev) / d_in ** 0.5
X = torch.randn(n, d_in, device=dev) @ A.T + 0.3 * torch.randn(n, d_in, device=dev)
X[:, :4] *= 20  # canais outlier
W = torch.randn(d_out, d_in, device=dev)
H = 2 * X.T @ X / n
err = lambda Q: ((Q - W) @ X.T).pow(2).mean().item()

for bits in (3, 4):
    e_rtn = err(C.rtn(W, bits, 64).float())
    e_gptq = err(C.gptq(W, H, bits, group=64, blocksize=32))
    assert e_gptq < e_rtn, (bits, e_gptq, e_rtn)
    print(f"OK gptq {bits}b: erro de saida {e_gptq:.4f} < rtn {e_rtn:.4f}")

Wm = C.magnitude_prune(W, 0.5)
Ws = C.sparsegpt(W, H, 0.5, blocksize=32)
sp = (Ws == 0).float().mean().item()
assert abs(sp - 0.5) < 0.02, sp
assert err(Ws) < err(Wm), (err(Ws), err(Wm))
print(f"OK sparsegpt 50%: esparsidade {sp:.3f}, erro {err(Ws):.3f} < magnitude {err(Wm):.3f}")

W24 = C.sparsegpt(W, H, 0.5, prunen=2, prunem=4, blocksize=32)
assert ((W24.view(d_out, -1, 4) == 0).sum(-1) >= 2).all()
print("OK sparsegpt 2:4: todo bloco de 4 tem >= 2 zeros")

Ww = C.wanda(W, (X * X).sum(0), n, 0.5)
assert ((Ww == 0).sum(1) == d_in // 2).all()
assert err(Ww) < err(Wm)
print(f"OK wanda 50%: exatamente metade por linha, erro {err(Ww):.3f} < magnitude {err(Wm):.3f}")

new, alpha = C.awq_group([W], X.abs().mean(0), X[:1024], bits=3, group=64)
assert err(new[0]) <= err(C.rtn(W, 3, 64).float()) + 1e-6
print(f"OK awq 3b: alpha={alpha:.2f}, erro {err(new[0]):.4f} <= rtn {err(C.rtn(W, 3, 64).float()):.4f}")

gen = torch.Generator(device=dev).manual_seed(0)
Wg = C.gaussian_like_rtn(W, 4, gen, 64)
ratio = (Wg - W).pow(2).mean() / (C.rtn(W, 4, 64).float() - W).pow(2).mean()
assert 0.8 < ratio < 1.25, ratio.item()
print(f"OK ruido gaussiano: MSE / MSE_rtn = {ratio.item():.3f}")

# --- ponta a ponta num Qwen3 minusculo
cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=384, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=256)
m = Qwen3ForCausalLM(cfg).to(dev).eval()
dec = resolve_decoder(m)
bank = C.WeightBank(dec)
assert len(bank.targets) == 4 * 7
calib = torch.randint(0, 512, (16, 64))
ids = torch.randint(0, 512, (2, 32), device=dev)
with torch.no_grad():
    ref = m(input_ids=ids, use_cache=False).logits.clone()
# agrupamento AWQ: q/k/v, o, gate/up, down -> exatamente 4 grupos por camada
for idx in range(4):
    st = C.LayerStats(bank.layer(idx), need_hessian=False, n_samples=16)
    with st:
        C.run_until_layer(m, dec.layers, idx, calib, dev, 4)
    sizes = sorted(len(g) for g in st.groups())
    assert sizes == [1, 1, 2, 3], (idx, sizes)
print("OK grupos AWQ por camada: [down], [o], [gate, up], [q, k, v]")
for method, kw in (("gptq", dict(bits=3, group=32)), ("awq", dict(bits=3, group=32)),
                   ("sparsegpt", dict(sparsity=0.5)), ("wanda", dict(sparsity=0.5)),
                   ("wanda", dict(prunen=2, prunem=4))):
    info = C.calibrated_compress(m, bank, dec.layers, calib, method, batch_size=4, **kw)
    if method == "awq":
        assert len(info["alphas"]) == 4 * 4, len(info["alphas"])
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=False).logits
    assert not torch.equal(out, ref)
    bank.restore()
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids, use_cache=False).logits, ref)
    print(f"OK {method} {kw}: muda a saida e restaura bit-exato")

with torch.no_grad():
    diffs = {}
    for b in (8, 2):
        with C.kv_fake_quant(m, dec, b):
            diffs[b] = (m(input_ids=ids, use_cache=False).logits - ref).abs().mean().item()
    assert torch.equal(m(input_ids=ids, use_cache=False).logits, ref)
assert 0 < diffs[8] < diffs[2], diffs
print(f"OK kv fake-quant: |d| 8b {diffs[8]:.2e} < 2b {diffs[2]:.2e}, contexto restaura")
