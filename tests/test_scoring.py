"""Validacoes do scoring em batch (left padding + logits_to_keep).

    python tests/test_scoring.py

O ponto critico: `logits_to_keep=1` economiza GiB por batch, mas precisa
devolver EXATAMENTE os mesmos logits da ultima posicao. Um erro aqui
produziria uma curva inteira silenciosamente errada.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glod.paths import CACHE_DIR

from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from glod.core.model_loader import LoadedModel
from glod.core.scoring import (
    _left_pad_batch,
    gold_nll,
    predictions_and_correctness,
    score_choices,
)

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FALHA'}] {label}{(' | ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


torch.manual_seed(0)
cfg = Gemma3TextConfig(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=8,
    max_position_embeddings=256, pad_token_id=0,
)
model = Gemma3ForCausalLM(cfg).eval()

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "google/gemma-3-12b-it", cache_dir=CACHE_DIR
)
model.resize_token_embeddings(len(tok))
model.eval()

loaded = LoadedModel(
    role="test", model_id="tiny", model=model, tokenizer=tok,
    device=torch.device("cpu"), dtype=torch.float32,
)
# prompts de comprimentos deliberadamente diferentes, para exercitar o padding
prompts = [
    "Resposta:",
    "Pergunta: qual e a capital?\nA) x\nB) y\nC) z\nD) w\nResposta:",
    "Um prompt bem mais longo " * 12 + "\nResposta:",
    "Curto.\nResposta:",
]
choice_ids = [tok.encode(" " + c, add_special_tokens=False)[-1] for c in "ABCD"]

print("\n=== logits_to_keep=1 vs logits completos ===")
ids, mask, pos = _left_pad_batch(loaded, prompts)
with torch.inference_mode():
    full = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                 use_cache=False).logits[:, -1, :]
    kept = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                 use_cache=False, logits_to_keep=1).logits[:, -1, :]
check(kept.shape == full.shape, "shape da ultima posicao identico", str(tuple(kept.shape)))
check(torch.equal(kept, full), "logits_to_keep=1 e BIT-EXATO vs logits completos",
      f"max|d|={(kept - full).abs().max().item():.2e}")

print("\n=== left padding ===")
check(mask[:, -1].all(), "ultima coluna nunca e padding")
check(bool((pos.min() >= 0).all()), "position_ids nao negativos")
for i in range(len(prompts)):
    real = int(mask[i].sum())
    expected = torch.arange(real)
    check(torch.equal(pos[i, -real:], expected),
          f"position_ids do prompt {i} ignoram o padding")
# o tokenizer Gemma-3 ja vem com padding_side="left"; o invariante e que
# `_left_pad_batch` devolva o tokenizer ao estado em que o encontrou.
_original_side = tok.padding_side
tok.padding_side = "right"
_left_pad_batch(loaded, prompts[:2])
check(tok.padding_side == "right", "padding_side do tokenizer restaurado apos o uso",
      f"atual={tok.padding_side}")
tok.padding_side = _original_side

print("\n=== invariancia ao batch ===")
lg1, st1 = score_choices(loaded, prompts, choice_ids, batch_size=1, show_progress=False)
lg4, st4 = score_choices(loaded, prompts, choice_ids, batch_size=4, show_progress=False)
check(lg1.shape == (len(prompts), 4), "shape [N,4]", str(tuple(lg1.shape)))
check(torch.allclose(lg1, lg4, atol=1e-4),
      "mesmos logits com batch 1 e 4 (fp32)", f"max|d|={(lg1 - lg4).abs().max().item():.2e}")
p1, _ = predictions_and_correctness(lg1, [0] * len(prompts))
p4, _ = predictions_and_correctness(lg4, [0] * len(prompts))
check(torch.equal(p1, p4), "mesmas predicoes independentemente do batch")

print("\n=== diagnosticos ===")
check(set(st4) == {"choice_mass", "entropy"}, "chaves dos diagnosticos", str(sorted(st4)))
check(st4["choice_mass"].shape == (len(prompts),), "choice_mass tem um valor por exemplo")
check(bool(((st4["choice_mass"] >= 0) & (st4["choice_mass"] <= 1)).all()),
      "choice_mass dentro de [0,1]")
check(bool((st4["entropy"] >= 0).all()), "entropia nao negativa")
max_ent = torch.log(torch.tensor(float(model.config.vocab_size)))
check(bool((st4["entropy"] <= max_ent).all()),
      "entropia <= log(vocab) (exige acumulacao em fp64)",
      f"max={st4['entropy'].max():.5f} <= {max_ent:.5f}")

# massa calculada por batch deve bater com o calculo direto
with torch.inference_mode():
    probs = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                  use_cache=False, logits_to_keep=1).logits[:, -1, :].float().softmax(-1)
direct = probs[:, choice_ids].sum(-1)
check(torch.allclose(direct, st4["choice_mass"], atol=1e-5),
      "choice_mass por batch == calculo direto",
      f"max|d|={(direct - st4['choice_mass']).abs().max().item():.2e}")

print("\n=== gold_nll ===")
targets = [0, 1, 2, 3]
nll = gold_nll(lg4, targets)
check(nll.shape == (4,), "um NLL por exemplo")
check(bool((nll >= 0).all()), "NLL nao negativa")
manual = torch.nn.functional.cross_entropy(lg4, torch.tensor(targets), reduction="none")
check(torch.allclose(nll, manual), "gold_nll == cross_entropy renormalizada nas 4 opcoes")
confident = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
check(gold_nll(confident, [0]).item() < gold_nll(confident, [1]).item(),
      "NLL menor quando o gabarito e a opcao provavel")

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FALHA(S): {FAILURES}")
    sys.exit(1)
print("todos os testes passaram")
