#!/usr/bin/env bash
# Impacto (1/2): mixture-of-experts. Para cada modelo e corpus: referencia greedy, grade
# padrao de compressores com cabeca fp32, e o estagio moe-routing (roteamento livre x
# congelado na referencia: quanto dos flips vem da troca de experts). Termina no ext-report.
#
#   GPU_MOE=cuda:2 nohup scripts/bg/run_moe.sh &          # saida em logs.txt, tag [moe]
#   MOE_MODELS="allenai/OLMoE-1B-7B-0125-Instruct" MOE_CORPORA="gsm8k" scripts/bg/run_moe.sh
#
# O roteador fica em bf16 (fora do WeightBank), como nos quantizadores usados na pratica.
# Qwen3-30B-A3B ocupa ~61 GB: precisa de uma H100 livre.
TAG=moe
source "$(dirname "$0")/_common.sh"
DEVICE=${GPU_MOE:-cuda:2}
MODELS=${MOE_MODELS:-"allenai/OLMoE-1B-7B-0125-Instruct Qwen/Qwen3-30B-A3B-Instruct-2507"}
CORPORA=${MOE_CORPORA:-"gsm8k mmlu_en wikitext wikitext_nat code"}
# nao calibrados primeiro: se a calibracao por expert falhar, a curva ja existe
CONF=${CONF:-"u8 u6 u5 u4 g4 mag20 kv4 kv3 gptq4 awq4 sgpt50 wanda50"}
ROUTE_CONF=${ROUTE_CONF:-"u8 u6 u5 u4 g4 mag20 kv4 gptq4 wanda50"}
log "=== inicio | $DEVICE | modelos: $MODELS | corpora: $CORPORA | saida: $GLOD_RESULTS"
gpu_check "$DEVICE"
for m in $MODELS; do
  prefetch "$m" || continue
  for c in $CORPORA; do
    step "$PY -m glod grid gen --model $m --corpus $c --device $DEVICE" || continue
    for cfg in bf16 $CONF; do      # uma config por etapa: uma falha nao derruba as outras
      step "$PY -m glod grid score --model $m --corpus $c --device $DEVICE --logits-fp32 --configs bf16 $cfg"
    done
    step "$PY -m glod moe-routing --model $m --corpus $c --device $DEVICE --logits-fp32 --configs $ROUTE_CONF"
  done
done
step "$PY -m glod ext-report"
finish
