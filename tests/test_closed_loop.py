"""Validacoes da decodificacao em malha fechada (Fase 1d).

    python tests/test_closed_loop.py

Invariantes que pegam erro de KV cache, position_ids ou mascara:
  * limiar +inf (nunca promove) == generate() da base, token a token;
  * limiar -inf (sempre promove) == generate() da referencia;
  * limiar intermediario: cada token emitido e o argmax do modelo escolhido
    NAQUELE passo, verificado por teacher forcing independente.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glod.paths import CACHE_DIR

from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig

from glod.core.closed_loop import cascade_generate
from glod.core.model_loader import LoadedModel
from glod.corpora.token_oracle import GeneratedCorpus, _eos_ids, generate_greedy, score_tokens

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FALHA'}] {label}{(' | ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


try:
    tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-it",
                                        cache_dir=CACHE_DIR)
except Exception as e:
    print(f"SKIP: test_closed_loop requires google/gemma-3-12b-it tokenizer: {e}")
    sys.exit(0)


def tiny(seed: int) -> LoadedModel:
    torch.manual_seed(seed)
    cfg = Gemma3TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=8,
        max_position_embeddings=512, pad_token_id=tok.pad_token_id,
    )
    m = Gemma3ForCausalLM(cfg).eval()
    m.generation_config.eos_token_id = [tok.eos_token_id]
    return LoadedModel(role=f"t{seed}", model_id="tiny", model=m, tokenizer=tok,
                       device=torch.device("cpu"), dtype=torch.float32)


base, ref = tiny(0), tiny(1)
texts = ["Oi.", "Uma pergunta bem mais longa para forcar padding a esquerda no batch todo.",
         "Media extensao aqui.", "x"]
prompts = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                   tokenize=False) for t in texts]
eos = _eos_ids(base)
N = 20  # > sliding_window, para exercitar o cache deslizante

print("\n=== limiar +inf reproduz a base ===")
g_base = generate_greedy(base, [{"prompt": p} for p in prompts], max_new_tokens=N, batch_size=4)
cl_base = cascade_generate(base, ref, prompts, threshold=float("inf"), eos_ids=eos, max_new_tokens=N)
check(cl_base.gen_ids == g_base.gen_ids, "cascata(+inf) == generate(base), batch com padding",
      f"iguais={[a == b for a, b in zip(cl_base.gen_ids, g_base.gen_ids)]}")
check(cl_base.promote_rate == 0.0, "nenhum token promovido")
cl_solo = cascade_generate(base, None, prompts, threshold=float("inf"), eos_ids=eos, max_new_tokens=N)
check(cl_solo.gen_ids == g_base.gen_ids, "modo so-base (ref=None) == generate(base)")

print("\n=== limiar -inf reproduz a referencia ===")
g_ref = generate_greedy(ref, [{"prompt": p} for p in prompts], max_new_tokens=N, batch_size=4)
cl_ref = cascade_generate(base, ref, prompts, threshold=float("-inf"), eos_ids=eos, max_new_tokens=N)
check(cl_ref.gen_ids == g_ref.gen_ids, "cascata(-inf) == generate(ref)",
      f"iguais={[a == b for a, b in zip(cl_ref.gen_ids, g_ref.gen_ids)]}")
check(cl_ref.promote_rate == 1.0, "todos os tokens promovidos")

print("\n=== limiar intermediario: cada token veio do modelo certo ===")
# entropia de modelo aleatorio ~ log(V); limiar na mediana aproximada promove parte dos passos
probe_ent = score_tokens(base, GeneratedCorpus(g_base.prompt_ids, g_base.gen_ids, []),
                         batch_size=4).entropy
thr = float(probe_ent.median())
cl_mix = cascade_generate(base, ref, prompts, threshold=thr, eos_ids=eos, max_new_tokens=N)
check(0.0 < cl_mix.promote_rate < 1.0, "taxa de promocao entre 0 e 1", f"{cl_mix.promote_rate:.2f}")
corpus_mix = GeneratedCorpus([tok(p, add_special_tokens=False)["input_ids"] for p in prompts],
                             cl_mix.gen_ids, [])
tb = score_tokens(base, corpus_mix, batch_size=4)
tr = score_tokens(ref, corpus_mix, batch_size=4)
flat = torch.tensor([t for g in cl_mix.gen_ids for t in g])
prom = torch.tensor([p for row in cl_mix.promoted for p in row])
from_ref = (tr.top1 == flat)[prom].float().mean().item()
from_base = (tb.top1 == flat)[~prom].float().mean().item()
check(from_ref == 1.0, "tokens promovidos == argmax da referencia naquele prefixo", f"{from_ref:.3f}")
check(from_base == 1.0, "tokens nao promovidos == argmax da base naquele prefixo", f"{from_base:.3f}")
# fora da faixa de empate numerico (KV incremental vs recomputo) a decisao tem de bater 100%
clear = (tb.entropy.double() - thr).abs() > 1e-5
gate_ok = ((tb.entropy > thr) == prom)[clear].float().mean().item()
check(gate_ok == 1.0, "decisao do gate == entropia recomputada (fora da faixa de empate 1e-5)",
      f"{gate_ok:.3f}, {int((~clear).sum())} tokens na faixa de empate")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FALHA(S): {FAILURES}")
    sys.exit(1)
print("todos os testes passaram")
