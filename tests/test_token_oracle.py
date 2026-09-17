"""Validacoes do oraculo por token (Fase 1c).

    python tests/test_token_oracle.py

O teste central: o proprio modelo, avaliado com teacher forcing sobre a sua
geracao greedy, precisa reproduzir 100% dos tokens. Qualquer desalinhamento de
padding, position_ids ou janela de logits aparece aqui como concordancia < 100%.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ews.paths import CACHE_DIR

from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig

from ews.core.model_loader import LoadedModel
from ews.corpora.token_oracle import (
    auroc,
    burstiness,
    burstiness_null,
    cached_streaming_cost,
    consistent_token_oracle,
    generate_greedy,
    run_lengths,
    score_tokens,
    speculative_expected_cost,
    teacher_forcing_batch,
    truncate_at_eos,
)

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FALHA'}] {label}{(' | ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


# ------------------------------------------------------ montagem do batch
print("\n=== teacher_forcing_batch ===")
prompts = [[5, 6, 7], [9], [1, 2, 3, 4, 5]]
gens = [[10, 11, 12], [20], [30, 31]]
ids, mask, pos, keep = teacher_forcing_batch(prompts, gens, pad_id=0)
check(keep == 3, "janela de logits = maior geracao", f"keep={keep}")
for i, (p, g) in enumerate(zip(prompts, gens)):
    row = ids[i][mask[i].bool()].tolist()
    check(row == p + g[:-1], f"linha {i} = prompt + gen[:-1]", str(row))
    # a posicao que preve g[j] e a (width - len(g) + j); seu input e o token anterior a g[j]
    width = ids.shape[1]
    inputs_before = [ids[i, width - len(g) + j].item() for j in range(len(g))]
    expected = [p[-1]] + g[:-1]
    check(inputs_before == expected, f"linha {i}: ultimos len(g) inputs precedem cada g[j]",
          f"{inputs_before} vs {expected}")
check(bool((pos[mask.bool()] >= 0).all()), "position_ids nao negativos")
check(pos[1, -1].item() == 0, "prompt de 1 token com gen de 1 token comeca na posicao 0")

check(truncate_at_eos([4, 5, 1, 7, 8], {1}) == [4, 5, 1], "truncate_at_eos inclui o EOS")
check(truncate_at_eos([4, 5, 6], {1}) == [4, 5, 6], "sem EOS devolve tudo")

# ------------------------------------------------ fim a fim em Gemma3 minusculo
print("\n=== auto-consistencia: teacher forcing reproduz a propria geracao ===")
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-it",
                                    cache_dir=CACHE_DIR)
cfg = Gemma3TextConfig(
    vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=16,
    max_position_embeddings=512, pad_token_id=tok.pad_token_id,
)
model = Gemma3ForCausalLM(cfg).eval()
model.generation_config.eos_token_id = [tok.eos_token_id]
loaded = LoadedModel(role="t", model_id="tiny", model=model, tokenizer=tok,
                     device=torch.device("cpu"), dtype=torch.float32)

texts = ["Oi.", "Uma pergunta bem mais longa para forcar padding a esquerda no batch.",
         "Media extensao aqui.", "x"]
records = [{"prompt": tok.apply_chat_template([{"role": "user", "content": t}],
                                              add_generation_prompt=True, tokenize=False)}
           for t in texts]

# geracao em batch (com padding) deve ser identica a geracao isolada (sem padding)
batched = generate_greedy(loaded, [dict(r) for r in records], max_new_tokens=12, batch_size=4)
alone = [generate_greedy(loaded, [dict(r)], max_new_tokens=12, batch_size=1).gen_ids[0]
         for r in records]
same = [b == a for b, a in zip(batched.gen_ids, alone)]
check(all(same), "generate em batch == generate isolado (padding nao altera a saida)",
      f"iguais={same}")

scores = score_tokens(loaded, batched, batch_size=4)
flat_gen = torch.tensor([t for g in batched.gen_ids for t in g], dtype=torch.int32)
agree = (scores.top1 == flat_gen).float().mean().item()
check(scores.top1.shape == flat_gen.shape, "um top-1 por token gerado",
      f"{tuple(scores.top1.shape)} vs {tuple(flat_gen.shape)}")
check(agree == 1.0, "teacher forcing reproduz 100% da propria geracao greedy",
      f"concordancia={agree:.4f}")

scores_bs1 = score_tokens(loaded, batched, batch_size=1)
check(torch.equal(scores.top1, scores_bs1.top1), "top-1 independe do batch_size")
check(bool((scores.ref_logprob <= 1e-5).all()), "log-prob <= 0")
check(bool((scores.margin >= 0).all()), "margem nao negativa")

hid = score_tokens(loaded, batched, batch_size=2, capture_layers=[1, 3])
check(set(hid.hidden) == {1, 3}, "camadas capturadas", str(sorted(hid.hidden)))
check(hid.hidden[1].shape == (batched.n_tokens, 64), "hidden [T, H] alinhado aos tokens",
      str(tuple(hid.hidden[1].shape)))
check(not any(len(m._forward_hooks) for m in model.model.layers), "hooks removidos apos uso")
check(hid.hidden[1].dtype == torch.bfloat16, "hidden capturado em bfloat16 (fp16 estoura no Gemma)",
      str(hid.hidden[1].dtype))
# ativacao acima do limite do fp16 precisa sobreviver a captura
big_hook = model.model.layers[1].register_forward_hook(
    lambda m, i, o: (o[0] * 0 + 1e5,) + tuple(o[1:]) if isinstance(o, tuple) else o * 0 + 1e5)
try:
    hbig = score_tokens(loaded, batched, batch_size=4, capture_layers=[1])
    check(bool(torch.isfinite(hbig.hidden[1]).all()) and float(hbig.hidden[1].float().max()) > 9e4,
          "ativacao de 1e5 capturada sem overflow", f"max={float(hbig.hidden[1].float().max()):.0f}")
finally:
    big_hook.remove()

seq_idx = batched.token_sequence_index()
check(len(seq_idx) == batched.n_tokens and int(seq_idx[-1]) == len(texts) - 1,
      "token_sequence_index cobre todos os tokens")

# ---------------------------------------------------------- oraculo
print("\n=== consistent_token_oracle ===")
# 2 configs (custos 1 e 5), referencia custa 10, 4 tokens
agree_m = torch.tensor([[1, 0, 0, 1],   # config barata
                        [1, 1, 0, 0]],  # config media
                       dtype=torch.bool)
res = consistent_token_oracle(agree_m, [1.0, 5.0], ref_cost=10.0)
# tokens: t0->1, t1->5, t2->10 (ninguem), t3->1  => media 4.25
check(abs(res["mean_cost"] - 4.25) < 1e-9, "custo medio do oraculo", f"{res['mean_cost']}")
res_rev = consistent_token_oracle(agree_m.flip(0), [5.0, 1.0], ref_cost=10.0)
check(abs(res_rev["mean_cost"] - 4.25) < 1e-9, "independe da ordem das linhas")

# ---------------------------------------------------------- rajadas
print("\n=== burstiness / run_lengths ===")
flags = torch.tensor([1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=torch.bool)
seqs = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
check(run_lengths(flags, seqs) == [2, 1, 3], "run_lengths nao cruza sequencias",
      str(run_lengths(flags, seqs)))
b = burstiness(flags, seqs)
# pares validos (mesma seq): (1,1)(1,0)(0,0)(0,1) | (0,1)(1,1)(1,1)
# prev=1 em 4 pares, cur=1 em 3 deles -> 0.75 ; taxa = 6/9
check(abs(b["p_given_prev"] - 0.75) < 1e-9, "P(h_t | h_{t-1})", f"{b['p_given_prev']}")
check(abs(b["rate"] - 6 / 9) < 1e-6, "taxa marginal (float32)")

torch.manual_seed(1)
bursty = torch.zeros(20000, dtype=torch.bool)
for s in range(0, 20000, 100):          # rajadas de 10 a cada 100 tokens
    bursty[s + 40 : s + 50] = True
seq_b = torch.arange(20000) // 1000
obs = burstiness(bursty, seq_b)
null = burstiness_null(bursty, seq_b, n_reps=5)
check(obs["lift"] > 5 and abs(null["lift"] - 1) < 0.1,
      "rajadas reais: lift observado >> nulo (~1)", f"obs={obs['lift']:.2f} nulo={null['lift']:.2f}")

# ---------------------------------------------------------- cache de ΔW
print("\n=== cached_streaming_cost ===")
hard = torch.tensor([0, 1, 0, 0, 0, 1, 1, 0], dtype=torch.bool)
seq_c = torch.zeros(8, dtype=torch.long)
c = cached_streaming_cost(hard, seq_c, window=2, base_cost=1, ref_cost=3, delta_io=10)
# t0 base(1) | t1 carrega(io10) ref(3) | t2 ref(3) streak1 | t3 ref(3) streak2 -> descarta
# t4 base(1) | t5 carrega(io10) ref(3) | t6 ref(3) | t7 ref(3) streak1
check(abs(c["read_per_token"] - 20 / 8) < 1e-9, "pesos lidos/token", f"{c['read_per_token']}")
check(abs(c["io_per_token"] - 20 / 8) < 1e-9, "I/O/token (2 recargas)", f"{c['io_per_token']}")
check(c["divergence"] == 0.0, "sem divergencia: todo token dificil caiu com ΔW carregado")
c0 = cached_streaming_cost(hard, seq_c, window=0, base_cost=1, ref_cost=3, delta_io=10)
check(c0["io_per_token"] == 30 / 8, "window=0 recarrega a cada token dificil",
      f"{c0['io_per_token']}")

# ------------------------------------------------ decodificacao especulativa
print("\n=== speculative_expected_cost ===")
all_ok = torch.ones(30, dtype=torch.bool)
s_all = speculative_expected_cost(all_ok, torch.zeros(30, dtype=torch.long),
                                  draft_cost=1, verify_cost=10, k=4)
# rodadas de 5 tokens a custo 4*1+10=14 -> 30 tokens em 6 rodadas
check(abs(s_all["cost_per_token"] - 14 / 5) < 1e-9, "draft sempre aceito: (k*d+v)/(k+1)",
      f"{s_all['cost_per_token']}")
none_ok = torch.zeros(12, dtype=torch.bool)
s_none = speculative_expected_cost(none_ok, torch.zeros(12, dtype=torch.long),
                                   draft_cost=1, verify_cost=10, k=4)
# perto do fim o draft propoe min(k, restantes): t=9,10,11 propoem 3,2,1 tokens.
# Limitar favorece levemente o baseline especulativo - escolha conservadora.
expected_none = (9 * 14 + 13 + 12 + 11) / 12
check(abs(s_none["cost_per_token"] - expected_none) < 1e-9,
      "draft nunca aceito: k*d+v por token, com proposta limitada no fim",
      f"{s_none['cost_per_token']} vs {expected_none}")
two = torch.ones(10, dtype=torch.bool)
s_two = speculative_expected_cost(two, torch.tensor([0] * 5 + [1] * 5),
                                  draft_cost=1, verify_cost=10, k=4)
# cada sequencia de 5: uma rodada propondo 4, emitindo 5 -> 2 rodadas, custo 28, 10 tokens
check(abs(s_two["cost_per_token"] - 2.8) < 1e-9, "respeita fronteira de sequencia",
      f"{s_two['cost_per_token']}")

# ------------------------------------------------------------------ auroc
print("\n=== auroc ===")
check(auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0, 0, 1, 1])) == 1.0, "separacao perfeita")
check(auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([0, 0, 1, 1])) == 0.0, "invertido")
check(abs(auroc(torch.ones(6), torch.tensor([0, 1, 0, 1, 0, 1])) - 0.5) < 1e-9, "empates -> 0.5")
torch.manual_seed(0)
x = torch.randn(1500); y = (x + torch.randn(1500)) > 0
pos_s, neg_s = x[y], x[~y]
brute = ((pos_s[:, None] > neg_s[None, :]).double().mean()
         + 0.5 * (pos_s[:, None] == neg_s[None, :]).double().mean()).item()
check(abs(auroc(x, y) - brute) < 1e-9, "AUROC por ranking == forca bruta O(n^2)",
      f"{auroc(x, y):.6f} vs {brute:.6f}")
xt = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 0.5]); yt = torch.tensor([0, 1, 1, 1, 0, 0])
pt, nt = xt[yt.bool()], xt[~yt.bool()]
brute_t = ((pt[:, None] > nt[None, :]).double().mean()
           + 0.5 * (pt[:, None] == nt[None, :]).double().mean()).item()
check(abs(auroc(xt, yt) - brute_t) < 1e-9, "AUROC com empates == forca bruta",
      f"{auroc(xt, yt):.6f} vs {brute_t:.6f}")

# ------------------------------------------------ granularidade e gates
print("\n=== chunk_oracle_cost ===")
from ews.corpora.token_oracle import chunk_oracle_cost, cost_at_divergence, gate_curve

# 1 config barata (custo 1), referencia 10; seq0 = [ok ok ok falha], seq1 = [ok ok]
ag = torch.tensor([[1, 1, 1, 0, 1, 1]], dtype=torch.bool)
sq = torch.tensor([0, 0, 0, 0, 1, 1]); ps = torch.tensor([0, 1, 2, 3, 0, 1])
c1 = chunk_oracle_cost(ag, [1.0], 10.0, sq, ps, chunk=1)
check(abs(c1 - (5 * 1 + 10) / 6) < 1e-9, "chunk=1 == oraculo por token", f"{c1}")
c2 = chunk_oracle_cost(ag, [1.0], 10.0, sq, ps, chunk=2)
# blocos: seq0 [0,1] ok ->1,1 ; seq0 [2,3] falha ->10,10 ; seq1 [0,1] ok ->1,1
check(abs(c2 - (1 + 1 + 10 + 10 + 1 + 1) / 6) < 1e-9, "chunk=2", f"{c2}")
cs = chunk_oracle_cost(ag, [1.0], 10.0, sq, ps, chunk=None)
check(abs(cs - (4 * 10 + 2 * 1) / 6) < 1e-9, "chunk=None == sequencia inteira", f"{cs}")
tok_level = consistent_token_oracle(ag, [1.0], 10.0)["mean_cost"]
check(abs(c1 - tok_level) < 1e-9, "chunk=1 concorda com consistent_token_oracle")
ag2 = torch.tensor([[1, 0, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1]], dtype=torch.bool)
check(abs(chunk_oracle_cost(ag2, [1.0, 5.0], 10.0, sq, ps, chunk=1)
          - consistent_token_oracle(ag2, [1.0, 5.0], 10.0)["mean_cost"]) < 1e-9,
      "chunk=1 com 2 configs concorda com o oraculo por token")
check(chunk_oracle_cost(ag2, [1.0, 5.0], 10.0, sq, ps, chunk=None)
      >= chunk_oracle_cost(ag2, [1.0, 5.0], 10.0, sq, ps, chunk=2)
      >= chunk_oracle_cost(ag2, [1.0, 5.0], 10.0, sq, ps, chunk=1) - 1e-12,
      "custo do oraculo nao diminui com blocos maiores")

print("\n=== gate_curve / cost_at_divergence ===")
hard_g = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
perfect = hard_g.float()
cur = gate_curve(perfect, hard_g, base_cost=1, ref_cost=10, mode="predictive", n_points=11)
best0 = cost_at_divergence(cur, 0.0)
check(abs(best0 - (0.5 * 1 + 0.5 * 10)) < 1e-9, "gate perfeito, preditivo: promove so os dificeis",
      f"{best0}")
cas = gate_curve(perfect, hard_g, base_cost=1, ref_cost=10, mode="cascade", n_points=11)
check(abs(cost_at_divergence(cas, 0.0) - (1 + 0.5 * 10)) < 1e-9, "cascata paga a base sempre")
ep = gate_curve(perfect, hard_g, base_cost=1, ref_cost=10, mode="early_probe",
                prefix_fraction=0.25, n_points=11)
check(abs(cost_at_divergence(ep, 0.0) - (0.25 * 1 + 0.75 * (0.5 * 1 + 0.5 * 10))) < 1e-9,
      "early_probe: prefixo na base + decisao no resto")
check(abs(cost_at_divergence(cur, 1.0) - 1.0) < 1e-9, "orcamento de divergencia total -> custo da base")
# prefixo mal-sucedido: promover ainda diverge em 1 dos 2 tokens promovidos
leaky = torch.tensor([0, 0, 1, 0], dtype=torch.bool)
ep_real = gate_curve(perfect, hard_g, base_cost=1, ref_cost=10, mode="early_probe",
                     prefix_fraction=0.25, n_points=11, hard_if_promoted=leaky)
check(cost_at_divergence(ep_real, 0.0) != cost_at_divergence(ep_real, 0.0),
      "com divergencia medida do prefixo, divergencia zero fica inatingivel (nan)")
check(min(pt["divergence"] for pt in ep_real) == 0.25,
      "divergencia minima = a do prefixo promovido", str(min(pt["divergence"] for pt in ep_real)))
check(cost_at_divergence(cur, -1.0) != cost_at_divergence(cur, -1.0), "alvo impossivel -> nan")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FALHA(S): {FAILURES}")
    sys.exit(1)
print("todos os testes passaram")
