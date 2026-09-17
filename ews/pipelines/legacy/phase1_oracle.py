#!/usr/bin/env python3
"""EWS | Fase 1 - Simulacao in-VRAM do Gate Oraculo.

Este script prova (ou mata) a tese matematica do Elastic Weight Streaming
ANTES de qualquer linha de codigo de I/O de disco, CUDA assincrono ou
Gumbel-Softmax. Tudo fica residente na VRAM do H100 e apenas *fingimos*
que camadas nao foram carregadas.

Pergunta que esta fase responde:
    "Se existisse um gate perfeito, quantos % da capacidade do 12B eu
     poderia nao carregar sem perder acuracia em relacao ao 12B denso?"

Estagios (`--stage`):
    env   - passo 1: valida ambiente (GPU, torch, transformers, cache).
    load  - passo 2: carrega Gemma 4B (base) e 12B (completo) em VRAM.
    skip  - passo 3: embrulha o 12B com elastic depth e varre fracoes de skip.
    oracle- passo 4: escada de capacidade + gate oraculo sobre MMLU.
    report- passo 5: logs crus da fronteira acuracia x bytes.

Uso:
    python ews_oracle.py --stage env
    python ews_oracle.py --stage skip --full-model google/gemma-3-12b-it
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from ews.core.elastic_depth import (
    ElasticDepthController,
    make_elastic_depth,
    uniform_skip_schedule,
)
from ews.corpora.mmlu import load_mmlu, resolve_choice_ids
from ews.core.oracle import (
    build_capacity_ladder,
    conditional_dependence,
    load_importance,
    permutation_null_test,
    compute_oracle,
    dump_raw,
    evaluate_ladder,
    layer_importance_profile,
    mcnemar_exact,
    wilson_interval,
)
from ews.core.quant_oracle import (
    build_quant_ladder,
    evaluate_quant_ladder,
    quant_sensitivity_profile,
)
from ews.core.quantize import QuantizationSimulator, effective_bits
from ews.core.scoring import predictions_and_correctness, score_choices
from ews.paths import MMLU_PT_CSV
from ews.core.model_loader import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CACHE_DIR,
    DEFAULT_FULL_MODEL,
    LoadedModel,
    cuda_memory_report,
    load_model,
)

LOGGER = logging.getLogger("ews.oracle")

# Prompt curto de fumaca para o passo 3. Nao e avaliacao: serve so para
# confirmar que o wrapper roda, que pular camada degrada monotonicamente e
# que a base 4B tem uma NLL de referencia no mesmo ballpark.
SMOKE_PROMPT = (
    "The capital of France is Paris. The capital of Japan is Tokyo. "
    "The capital of Brazil is Brasilia. The capital of Canada is"
)


# ---------------------------------------------------------------- passo 1
def stage_env(args: argparse.Namespace) -> None:
    """Passo 1 - sanidade do ambiente antes de gastar VRAM."""
    import transformers

    LOGGER.info("=" * 78)
    LOGGER.info("PASSO 1 | AMBIENTE")
    LOGGER.info("=" * 78)
    LOGGER.info("python               : %s", sys.version.split()[0])
    LOGGER.info("torch                : %s (cuda %s)", torch.__version__, torch.version.cuda)
    LOGGER.info("transformers         : %s", transformers.__version__)
    LOGGER.info("cuda disponivel      : %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            LOGGER.info(
                "  gpu %d              : %s | %.0f GiB | sm_%d%d",
                i, props.name, props.total_memory / 1024**3, props.major, props.minor,
            )
    LOGGER.info("bf16 suportado       : %s", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
    LOGGER.info("hf cache_dir         : %s", args.cache_dir)
    LOGGER.info("modelo base  (4B)    : %s", args.base_model)
    LOGGER.info("modelo full  (12B)   : %s", args.full_model)


# ---------------------------------------------------------------- passo 2
@dataclass
class ModelPair:
    base: LoadedModel | None
    full: LoadedModel


def stage_load(args: argparse.Namespace) -> ModelPair:
    """Passo 2 - carrega base (4B) e completo (12B) em VRAM."""
    LOGGER.info("=" * 78)
    LOGGER.info("PASSO 2 | CARREGAMENTO DOS MODELOS")
    LOGGER.info("=" * 78)

    full = load_model(
        args.full_model,
        role="full",
        device=args.device_full,
        dtype=args.dtype,
        attn_implementation=args.attn,
        cache_dir=args.cache_dir,
    )
    base: LoadedModel | None = None
    if not args.skip_base_model:
        base = load_model(
            args.base_model,
            role="base",
            device=args.device_base,
            dtype=args.dtype,
            attn_implementation=args.attn,
            cache_dir=args.cache_dir,
        )

    devices = [full.device] + ([base.device] if base else [])
    LOGGER.info("VRAM apos carregamento:\n%s", cuda_memory_report(devices))
    return ModelPair(base=base, full=full)


# ---------------------------------------------------------------- passo 3
@torch.inference_mode()
def mean_nll(loaded: LoadedModel, prompt: str) -> tuple[float, torch.Tensor]:
    """NLL media por token do prompt + argmax da distribuicao de cada posicao.

    `use_cache=False` e obrigatorio: uma camada pulada nao escreve no seu slot
    do KV cache (ver `SkippableDecoderLayer.forward`).
    """
    enc = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    out = loaded.model(**enc, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = enc["input_ids"][:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean"
    )
    return nll.item(), logits.argmax(dim=-1).squeeze(0).cpu()


def stage_skip(args: argparse.Namespace, pair: ModelPair) -> ElasticDepthController:
    """Passo 3 - habilita elastic depth no 12B e varre fracoes fixas de skip."""
    LOGGER.info("=" * 78)
    LOGGER.info("PASSO 3 | ELASTIC DEPTH (LAYER SKIPPING) NO MODELO COMPLETO")
    LOGGER.info("=" * 78)

    ctrl = make_elastic_depth(pair.full.model)
    LOGGER.info("camadas embrulhadas  : %d", ctrl.num_layers)
    LOGGER.info(
        "bytes/camada (media) : %.1f MiB | decoder total %.2f GiB",
        ctrl.total_param_bytes / ctrl.num_layers / 1024**2,
        ctrl.total_param_bytes / 1024**3,
    )

    # Referencia densa (g_l = 1 para todo l) - upper bound de qualidade.
    dense_nll, dense_pred = mean_nll(pair.full, SMOKE_PROMPT)
    LOGGER.info("-" * 78)
    LOGGER.info("[12B denso]  %s", ctrl.summary())
    LOGGER.info("[12B denso]  NLL media = %.4f  (referencia)", dense_nll)

    if pair.base is not None:
        base_nll, _ = mean_nll(pair.base, SMOKE_PROMPT)
        LOGGER.info("[4B  denso]  NLL media = %.4f  (baseline de capacidade)", base_nll)

    LOGGER.info("-" * 78)
    LOGGER.info("Varredura de skip uniforme (sem gate - ablacao 3 do documento):")
    LOGGER.info(
        "%6s | %6s | %6s | %9s | %9s | %s",
        "frac", "n_skip", "bytes%", "NLL", "dNLL", "concord. top-1 vs denso",
    )
    for frac in args.skip_fractions:
        n_skip = int(round(frac * ctrl.num_layers))
        schedule = uniform_skip_schedule(ctrl.num_layers, n_skip)
        with ctrl.skipping(schedule):
            nll, pred = mean_nll(pair.full, SMOKE_PROMPT)
            agree = (pred == dense_pred).float().mean().item()
            LOGGER.info(
                "%6.2f | %6d | %5.1f%% | %9.4f | %+9.4f | %6.1f%%   skip=%s",
                frac, n_skip, 100 * ctrl.skipped_fraction_bytes, nll,
                nll - dense_nll, 100 * agree, schedule,
            )

    LOGGER.info("-" * 78)
    LOGGER.info("Estado restaurado: %s", ctrl.summary())
    LOGGER.info(
        "NOTA: este e um smoke test de 1 prompt, NAO e evidencia. A varredura "
        "com significancia estatistica sobre MMLU entra no passo 4."
    )
    return ctrl


# ------------------------------------------------------------------- passo 4
def stage_oracle(
    args: argparse.Namespace, pair: ModelPair, ctrl: ElasticDepthController
) -> None:
    """Passo 4 - gate oraculo sobre a escada de capacidade do proprio 12B."""
    LOGGER.info("=" * 78)
    LOGGER.info("PASSO 4 | GATE ORACULO (ELASTIC DEPTH) SOBRE MMLU PT-BR")
    LOGGER.info("=" * 78)

    split = load_mmlu(
        args.mmlu_path, n_samples=args.n_samples, n_shot=args.n_shot, seed=args.seed
    )
    calib = load_mmlu(
        args.mmlu_path, n_samples=args.n_calib, n_shot=args.n_shot, seed=args.seed + 1
    )
    choice_ids = resolve_choice_ids(pair.full.tokenizer, leading_space=True)
    LOGGER.info("choice_ids (variante ' A'): %s", choice_ids)

    # --- baselines densos --------------------------------------------------
    with ctrl.dense():
        full_logits, full_stats = score_choices(
            pair.full, split.prompts, choice_ids, batch_size=args.batch_size,
            desc="12B denso",
        )
    _, full_correct = predictions_and_correctness(full_logits, split.targets)
    mass = full_stats["choice_mass"].mean().item()
    LOGGER.info("massa de prob. nos 4 ids de alternativa: %.4f", mass)
    if mass < 0.5:
        LOGGER.warning(
            "Massa baixa (%.4f): prompt e ids de alternativa mal casados. "
            "A acuracia pode sobreviver, mas qualquer leitura de confianca "
            "(Fase 2) sera ruido.", mass,
        )

    base_correct = None
    if pair.base is not None:
        base_ids = resolve_choice_ids(pair.base.tokenizer, leading_space=True)
        base_logits, _ = score_choices(
            pair.base, split.prompts, base_ids, batch_size=args.batch_size,
            desc="4B denso",
        )
        _, base_correct = predictions_and_correctness(base_logits, split.targets)

    # --- perfil de importancia por camada ----------------------------------
    LOGGER.info("-" * 78)
    LOGGER.info("Perfil de importancia leave-one-out (%d camadas, %d exemplos de calibracao)",
                ctrl.num_layers, len(calib))
    if args.importance_cache and Path(args.importance_cache).exists():
        importance = load_importance(args.importance_cache)
        LOGGER.info("  perfil reaproveitado de %s", args.importance_cache)
    else:
        importance = layer_importance_profile(
            pair.full, ctrl, calib, choice_ids, batch_size=args.batch_size
        )
    ranked = importance.order_ascending
    LOGGER.info("  NLL densa de referencia : %.4f", importance.dense_nll)
    LOGGER.info("  camadas MENOS criticas  : %s", ranked[:8])
    LOGGER.info("  camadas MAIS criticas   : %s", ranked[-8:][::-1])
    LOGGER.info("  dNLL min/mediana/max    : %.4f / %.4f / %.4f",
                min(importance.delta_nll),
                sorted(importance.delta_nll)[len(importance.delta_nll) // 2],
                max(importance.delta_nll))

    # --- escada de capacidade ----------------------------------------------
    skip_counts = [int(round(f * ctrl.num_layers)) for f in args.skip_fractions]
    ladder = build_capacity_ladder(
        importance, ctrl, pair.full, skip_counts, min_gap=args.min_layer_gap
    )
    LOGGER.info("-" * 78)
    LOGGER.info("Escada de capacidade (%d degraus, aninhados, min_gap=%d):",
                len(ladder), args.min_layer_gap)
    for rung in ladder:
        adj = sum(1 for x, y in zip(rung.skipped, rung.skipped[1:]) if y == x + 1)
        LOGGER.info("  degrau %d: %2d desligadas | %.2f GiB | %d pares adjacentes | %s",
                    rung.index, rung.n_skipped, rung.weight_bytes / 1024**3, adj,
                    rung.skipped)

    correct = evaluate_ladder(
        pair.full, ctrl, ladder, split, choice_ids, batch_size=args.batch_size
    )
    result = compute_oracle(correct, ladder)

    # --- relatorio ----------------------------------------------------------
    gib = 1024**3
    n = result.n
    LOGGER.info("-" * 78)
    LOGGER.info("SKIP FIXO, SEM GATE (ablacao 3 do documento) - n=%d", n)
    LOGGER.info("%7s | %8s | %9s | %8s | %s", "degrau", "n_skip", "GiB", "acc", "IC95% Wilson")
    for rung, acc in zip(ladder, result.rung_accuracy):
        lo, hi = wilson_interval(round(acc * n), n)
        LOGGER.info("%7d | %8d | %9.2f | %7.2f%% | [%.1f, %.1f]",
                    rung.index, rung.n_skipped, rung.weight_bytes / gib, 100 * acc, lo, hi)

    LOGGER.info("-" * 78)
    LOGGER.info("BASELINES E ORACULOS")
    if base_correct is not None:
        acc = base_correct.float().mean().item()
        lo, hi = wilson_interval(int(base_correct.sum()), n)
        base_gib = pair.base.param_bytes / gib
        LOGGER.info("  4B denso              : %6.2f%% [%.1f, %.1f] | %6.2f GiB",
                    100 * acc, lo, hi, base_gib)
        b = int((base_correct & ~full_correct).sum())
        c = int((~base_correct & full_correct).sum())
        LOGGER.info("      McNemar 12B vs 4B : b=%d c=%d p=%.4f%s",
                    b, c, mcnemar_exact(b, c),
                    "" if mcnemar_exact(b, c) < 0.05 else "  (NAO significativo a 5%)")
        casc_gib = base_gib + (1 - acc) * result.dense_bytes / gib
        LOGGER.info("      cascata 4B->12B   : %6.2f GiB  (paga o 4B em 100%% dos exemplos)",
                    casc_gib)

    lo, hi = wilson_interval(round(result.dense_accuracy * n), n)
    LOGGER.info("  12B denso             : %6.2f%% [%.1f, %.1f] | %6.2f GiB  <- teto de capacidade",
                100 * result.dense_accuracy, lo, hi, result.dense_bytes / gib)

    saving = 1 - result.consistent_bytes / result.dense_bytes
    LOGGER.info("  ORACULO consistente   : %6.2f%%            | %6.2f GiB  -> %.1f%% menos bytes",
                100 * result.consistent_accuracy, result.consistent_bytes / gib, 100 * saving)
    LOGGER.info("      (acuracia IGUAL a do denso por construcao; a metrica e byte economizado)")

    lo, hi = wilson_interval(round(result.unconstrained_accuracy * n), n)
    LOGGER.info("  ORACULO irrestrito    : %6.2f%% [%.1f, %.1f] | %6.2f GiB",
                100 * result.unconstrained_accuracy, lo, hi,
                result.unconstrained_bytes / gib)
    LOGGER.info("      (teto valido da classe de politica, mas inflado por acertos de sorte")
    LOGGER.info("       de degraus mutilados - NAO reportar como resultado principal)")

    LOGGER.info("-" * 78)
    LOGGER.info("HETEROGENEIDADE: a economia e sinal ou sorte? (nulo por permutacao)")
    null = permutation_null_test(correct, ladder, seed=args.seed)
    LOGGER.info("  economia observada    : %5.2f%%", 100 * null["observed_saving"])
    LOGGER.info("  economia do nulo      : %5.2f%% (dp %.2fpp)",
                100 * null["null_saving_mean"], 100 * null["null_saving_std"])
    LOGGER.info("  z / p empirico        : %+.2f / %.4f", null["z"], null["p_empirical"])
    LOGGER.info("  economia ATRIBUIVEL a heterogeneidade: %5.2f%%  -> %s",
                100 * null["attributable_saving"],
                "sinal real" if null["z"] > 2 else "indistinguivel de sorte")
    LOGGER.info("  P(degrau acerta | denso acerta) / P(degrau acerta):")
    for n_skip, p_marg, p_cond in conditional_dependence(correct, ladder):
        LOGGER.info("    skip %2d: %.4f -> %.4f  (razao %.2f)",
                    n_skip, p_marg, p_cond, p_cond / max(p_marg, 1e-9))

    LOGGER.info("-" * 78)
    LOGGER.info("CRITERIO DE VIDA OU MORTE DA FASE 1 (documento, secao 5):")
    LOGGER.info("  manter a acuracia do 12B lendo <= 70%% dos pesos?  %s  (lendo %.1f%%)",
                "SIM" if saving >= 0.30 else "NAO", 100 * (1 - saving))

    dump_raw(
        Path(args.results_dir) / args.results_name,
        importance=importance, ladder=ladder, result=result,
        null_test=null,
        baselines={
            "base_accuracy": float(base_correct.float().mean()) if base_correct is not None else None,
            "base_bytes": pair.base.param_bytes if pair.base is not None else None,
        },
        meta={
            "full_model": args.full_model, "base_model": args.base_model,
            "n_eval": n, "n_calib": len(calib), "n_shot": args.n_shot,
            "seed": args.seed, "dtype": args.dtype, "attn": args.attn,
            "batch_size": args.batch_size, "choice_mass": mass,
            "min_layer_gap": args.min_layer_gap,
        },
    )


# ------------------------------------------------------------- passo 4b
def stage_precision(args: argparse.Namespace, pair: ModelPair) -> None:
    """Fase 1b, passo 1 - varredura de precisao da base quantizada (QDQ).

    Decide o bitrate da base antes de construir qualquer gate. O criterio nao
    e "menor perda": e o gap util. Uma base que perde 2pp nao deixa nada para
    o gate recuperar; uma que perde 40pp nao e recuperavel por um dW pequeno.
    """
    LOGGER.info("=" * 78)
    LOGGER.info("FASE 1b | VARREDURA DE PRECISAO DA BASE (quantizacao simulada)")
    LOGGER.info("=" * 78)

    split = load_mmlu(
        args.mmlu_path, n_samples=args.n_samples, n_shot=args.n_shot, seed=args.seed
    )
    gib = 1024**3
    n = len(split)
    rows: list[tuple[str, float, float]] = []

    for loaded in [m for m in (pair.full, pair.base) if m is not None]:
        choice_ids = resolve_choice_ids(loaded.tokenizer, leading_space=True)
        sim = QuantizationSimulator(
            loaded.decoder, group_size=args.group_size,
            storage_device=args.quant_storage,
        )
        non_target = loaded.param_bytes - sum(t.numel * 2 for t in sim.targets)
        LOGGER.info("-" * 78)
        LOGGER.info("%s | %d projecoes | nao-quantizavel (emb/norm/lm_head): %.2f GiB",
                    loaded.model_id, len(sim.targets), non_target / gib)
        LOGGER.info("%6s | %11s | %9s | %8s | %s",
                    "bits", "bits/peso", "GiB", "acc", "IC95% Wilson")

        reference: float | None = None
        for bits in args.bitrates:
            sim.apply_uniform(bits)
            logits, _ = score_choices(
                loaded, split.prompts, choice_ids, batch_size=args.batch_size,
                desc=f"{loaded.role} {bits}b",
            )
            _, correct = predictions_and_correctness(logits, split.targets)
            acc = correct.float().mean().item()
            byt = sim.weight_bytes(
                {i: bits for i in range(sim.num_layers)}, non_target_bytes=non_target
            )
            lo, hi = wilson_interval(int(correct.sum()), n)
            if reference is None:
                reference = acc
            LOGGER.info("%6d | %11.2f | %9.2f | %7.2f%% | [%.1f, %.1f]  gap %+.2fpp",
                        bits, effective_bits(bits, args.group_size), byt / gib,
                        100 * acc, lo, hi, 100 * (acc - reference))
            rows.append((f"{loaded.role} @ {bits}b", byt / gib, acc))
        sim.restore()

    LOGGER.info("-" * 78)
    LOGGER.info("FRONTEIRA DE PARETO (nao-dominados)")
    for name, byt, acc in sorted(rows, key=lambda r: r[1]):
        dominators = [
            other for other, b2, a2 in rows
            if (b2 <= byt and a2 >= acc) and (b2 < byt or a2 > acc)
        ]
        LOGGER.info("  %-16s | %6.2f GiB | %6.2f%% | %s", name, byt, 100 * acc,
                    "PARETO" if not dominators else f"dominado por {dominators[0]}")

    out = Path(args.results_dir) / "precision_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": {"n": n, "group_size": args.group_size, "bitrates": args.bitrates,
                  "seed": args.seed, "n_shot": args.n_shot},
         "points": [{"name": nm, "gib": b, "accuracy": a} for nm, b, a in rows]},
        indent=2,
    ))
    LOGGER.info("Resultado cru gravado em %s", out)


# ------------------------------------------------------------- passo 4c
def stage_qoracle(args: argparse.Namespace, pair: ModelPair) -> None:
    """Fase 1b, passo 2 - oraculo sobre a escada de PRECISAO (Elastic Capacity)."""
    LOGGER.info("=" * 78)
    LOGGER.info("FASE 1b | ORACULO SOBRE ESCADA DE PRECISAO (base %db -> %db)",
                args.base_bits, args.target_bits)
    LOGGER.info("=" * 78)

    split = load_mmlu(args.mmlu_path, n_samples=args.n_samples,
                      n_shot=args.n_shot, seed=args.seed)
    calib = load_mmlu(args.mmlu_path, n_samples=args.n_calib,
                      n_shot=args.n_shot, seed=args.seed + 1)
    full = pair.full
    choice_ids = resolve_choice_ids(full.tokenizer, leading_space=True)
    sim = QuantizationSimulator(full.decoder, group_size=args.group_size,
                                storage_device=args.quant_storage)
    non_target = full.param_bytes - sum(t.numel * 2 for t in sim.targets)
    gib, n = 1024**3, len(split)

    LOGGER.info("-" * 78)
    LOGGER.info("Ganho de promocao por camada (%d camadas, %d exemplos de calibracao)",
                sim.num_layers, len(calib))
    sens = quant_sensitivity_profile(full, sim, calib, choice_ids,
                                     base_bits=args.base_bits,
                                     target_bits=args.target_bits,
                                     batch_size=args.batch_size)
    LOGGER.info("  NLL da base %db inteira  : %.4f", args.base_bits, sens.base_nll)
    LOGGER.info("  promover MAIS ajuda     : %s", sens.order_descending[:8])
    LOGGER.info("  promover MENOS ajuda    : %s", sens.order_descending[-8:][::-1])
    LOGGER.info("  dNLL min/mediana/max    : %+.4f / %+.4f / %+.4f",
                min(sens.delta_nll),
                sorted(sens.delta_nll)[len(sens.delta_nll) // 2],
                max(sens.delta_nll))
    LOGGER.info("  erro padrao pareado     : %.4f (mediana)",
                sorted(sens.delta_se)[len(sens.delta_se) // 2])
    LOGGER.info("  camadas com ganho significativo (|d| > 2 EP): %d/%d",
                sens.n_significant, sim.num_layers)
    LOGGER.info("  sinal/ruido do perfil   : %.2f  -> %s",
                sens.signal_to_noise(),
                "ordenacao confiavel" if sens.signal_to_noise() > 1.5
                else "ORDENACAO MAJORITARIAMENTE RUIDO - aumente --n-calib")

    counts = sorted({int(round(f * sim.num_layers)) for f in args.promote_fractions})
    ladder = build_quant_ladder(sens, sim, base_bits=args.base_bits,
                                target_bits=args.target_bits,
                                promote_counts=counts,
                                non_target_bytes=non_target)
    LOGGER.info("-" * 78)
    LOGGER.info("Escada de precisao (%d degraus, aninhados):", len(ladder))
    for rung in ladder:
        LOGGER.info("  degrau %d: %2d camadas promovidas a %db | %.2f GiB",
                    rung.index, rung.n_promoted, rung.target_bits,
                    rung.weight_bytes / gib)

    correct, base_stats = evaluate_quant_ladder(
        full, sim, ladder, split, choice_ids, batch_size=args.batch_size
    )
    result = compute_oracle(correct, ladder)

    LOGGER.info("-" * 78)
    LOGGER.info("ESCADA ESTATICA, SEM GATE - n=%d", n)
    LOGGER.info("%7s | %10s | %9s | %8s | %s",
                "degrau", "promovidas", "GiB", "acc", "IC95% Wilson")
    for rung, acc in zip(ladder, result.rung_accuracy):
        lo, hi = wilson_interval(round(acc * n), n)
        LOGGER.info("%7d | %10d | %9.2f | %7.2f%% | [%.1f, %.1f]",
                    rung.index, rung.n_promoted, rung.weight_bytes / gib,
                    100 * acc, lo, hi)

    LOGGER.info("-" * 78)
    LOGGER.info("ORACULO")
    top = ladder[-1]
    LOGGER.info("  topo da escada (%2d camadas a %db) : %6.2f%% | %6.2f GiB",
                top.n_promoted, top.target_bits,
                100 * result.dense_accuracy, top.weight_bytes / gib)
    saving = 1 - result.consistent_bytes / result.dense_bytes
    LOGGER.info("  ORACULO consistente               : %6.2f%% | %6.2f GiB -> %.1f%% menos bytes",
                100 * result.consistent_accuracy, result.consistent_bytes / gib,
                100 * saving)
    lo, hi = wilson_interval(round(result.unconstrained_accuracy * n), n)
    LOGGER.info("  ORACULO irrestrito                : %6.2f%% [%.1f, %.1f] | %6.2f GiB",
                100 * result.unconstrained_accuracy, lo, hi,
                result.unconstrained_bytes / gib)

    LOGGER.info("-" * 78)
    LOGGER.info("HETEROGENEIDADE (nulo por permutacao)")
    null = permutation_null_test(correct, ladder, seed=args.seed)
    LOGGER.info("  observada %.2f%% | nulo %.2f%% (dp %.2fpp) | z %+.2f | p %.4f",
                100 * null["observed_saving"], 100 * null["null_saving_mean"],
                100 * null["null_saving_std"], null["z"], null["p_empirical"])
    LOGGER.info("  economia ATRIBUIVEL a heterogeneidade: %.2f%%  -> %s",
                100 * null["attributable_saving"],
                "sinal real" if null["z"] > 2 else "indistinguivel de sorte")
    LOGGER.info("  P(degrau acerta | topo acerta) / P(degrau acerta):")
    for label, p_marg, p_cond in conditional_dependence(correct, ladder):
        LOGGER.info("    %8s: %.4f -> %.4f  (razao %.2f)",
                    label, p_marg, p_cond, p_cond / max(p_marg, 1e-9))

    out = Path(args.results_dir) / args.results_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"full_model": args.full_model, "n_eval": n, "n_calib": len(calib),
                 "base_bits": args.base_bits, "target_bits": args.target_bits,
                 "group_size": args.group_size, "seed": args.seed,
                 "n_shot": args.n_shot, "non_target_bytes": non_target},
        "sensitivity": {"base_nll": sens.base_nll, "delta_nll": sens.delta_nll,
                        "delta_se": sens.delta_se,
                        "n_significant": sens.n_significant,
                        "signal_to_noise": sens.signal_to_noise(),
                        "order_descending": sens.order_descending},
        "ladder": [{"index": r.index, "n_promoted": r.n_promoted,
                    "promoted": r.promoted, "weight_bytes": r.weight_bytes}
                   for r in ladder],
        "oracle": {"n": result.n, "top_accuracy": result.dense_accuracy,
                   "top_bytes": result.dense_bytes,
                   "consistent_bytes": result.consistent_bytes,
                   "unconstrained_accuracy": result.unconstrained_accuracy,
                   "unconstrained_bytes": result.unconstrained_bytes,
                   "rung_accuracy": result.rung_accuracy,
                   "rung_bytes": result.rung_bytes},
        "null_test": null,
        "base_entropy": base_stats["entropy"].tolist(),
        "base_choice_mass": base_stats["choice_mass"].tolist(),
        "correct_matrix": correct.to(torch.uint8).tolist(),
    }, indent=2))
    LOGGER.info("Resultado cru gravado em %s", out)


# ------------------------------------------------------------------- passo 5
def stage_report(args: argparse.Namespace) -> None:
    """Passo 5 - logs crus da fronteira de Pareto a partir do JSON do passo 4."""
    path = Path(args.results_dir) / args.results_name
    if not path.exists():
        raise FileNotFoundError(f"{path} nao existe - rode --stage oracle antes.")
    payload = json.loads(path.read_text())
    orc, gib = payload["oracle"], 1024**3
    LOGGER.info("=" * 78)
    LOGGER.info("PASSO 5 | FRONTEIRA DE PARETO (bruto)")
    LOGGER.info("=" * 78)
    LOGGER.info("%-24s | %9s | %8s | %s", "ponto", "GiB", "acc", "% dos bytes do 12B denso")
    dense = orc["dense_bytes"]
    rows = [(f"skip fixo ({r['n_skipped']} cam.)", b, a)
            for r, b, a in zip(payload["ladder"], orc["rung_bytes"], orc["rung_accuracy"])]
    if payload["baselines"]["base_bytes"]:
        rows.append(("4B denso", payload["baselines"]["base_bytes"],
                     payload["baselines"]["base_accuracy"]))
    rows.append(("ORACULO consistente", orc["consistent_bytes"], orc["consistent_accuracy"]))
    for name, byt, acc in rows:
        LOGGER.info("%-24s | %9.2f | %7.2f%% | %6.1f%%", name, byt / gib, 100 * acc,
                    100 * byt / dense)


# ---------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EWS Fase 1 - simulacao in-VRAM do gate oraculo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stage",
        default="all",
        choices=["env", "load", "skip", "precision", "qoracle", "oracle", "report", "all"],
        help="ate qual etapa executar ('all' = env+load+skip nesta entrega).",
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--full-model", default=DEFAULT_FULL_MODEL)
    p.add_argument("--device-base", default="cuda:1", help="GPU do modelo 4B.")
    p.add_argument("--device-full", default="cuda:0", help="GPU do modelo 12B.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--attn", default="sdpa", choices=["sdpa", "eager"],
        help="'eager' reproduz a numerica de referencia do Gemma; 'sdpa' e mais rapido.",
    )
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--skip-base-model", action="store_true",
        help="nao carrega o 4B (util para iterar so no 12B).",
    )
    p.add_argument(
        "--skip-fractions", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="fracoes de camadas a desligar na varredura do passo 3.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    p.add_argument("--mmlu-path", default=MMLU_PT_CSV, help="CSV do MMLU PT-BR.")
    p.add_argument("--n-samples", type=int, default=2000, help="Questoes de avaliacao (0 = todas).")
    p.add_argument("--n-calib", type=int, default=256,
                   help="Questoes do perfil de importancia (conjunto disjunto do de avaliacao).")
    p.add_argument("--n-shot", type=int, default=0, help="Exemplares few-shot no prompt.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch do scoring (left padding).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--min-layer-gap", type=int, default=1,
                   help="distancia minima entre camadas desligadas (1 = guloso puro, "
                        "2 = proibe adjacentes).")
    p.add_argument("--importance-cache", default=None,
                   help="JSON de uma rodada anterior para reaproveitar o perfil de importancia.")
    p.add_argument("--results-name", default="oracle_phase1.json")
    p.add_argument("--bitrates", type=int, nargs="+", default=[16, 8, 4, 3, 2],
                   help="bitrates da varredura de precisao (16 = bf16 intacto).")
    p.add_argument("--group-size", type=int, default=128,
                   help="tamanho do grupo da quantizacao afim (granularidade GPTQ/AWQ).")
    p.add_argument("--base-bits", type=int, default=3,
                   help="bitrate da base residente (janela util medida: 3 bits).")
    p.add_argument("--target-bits", type=int, default=16,
                   help="bitrate para onde o gate promove a camada (16 = bf16 cheio).")
    p.add_argument("--promote-fractions", type=float, nargs="+",
                   default=[0.0, 0.042, 0.083, 0.167, 0.25, 0.5, 0.75, 1.0],
                   help="fracoes de camadas promovidas em cada degrau.")
    p.add_argument("--quant-storage", default="cpu",
                   help="onde guardar a copia dos pesos originais ('cpu' poupa ~20 GiB "
                        "de VRAM no 12B; uma GPU com folga e mais rapida).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    torch.set_grad_enabled(False)

    stage_env(args)
    if args.stage == "env":
        return 0

    pair = stage_load(args)
    if args.stage == "load":
        return 0

    if args.stage == "precision":
        stage_precision(args, pair)
        return 0

    if args.stage == "qoracle":
        stage_qoracle(args, pair)
        return 0

    ctrl = stage_skip(args, pair) if args.stage == "skip" else make_elastic_depth(pair.full.model)
    if args.stage == "skip":
        return 0

    stage_oracle(args, pair, ctrl)
    stage_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
