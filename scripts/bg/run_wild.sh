#!/usr/bin/env bash
# Teste "in the wild" PRE-REGISTRADO (results/inthewild_predictions.json): modelos fora do
# estudo, checkpoints publicados e kernels reais (bitsandbytes NF4/int8, FP8). Termina no
# wild-check, que aplica os criterios W1-W5 como registrados.
#
#   GPU_WILD=cuda:2 nohup scripts/bg/run_wild.sh &           # saida em logs.txt, tag [wild]
TAG=wild
export GLOD_RESULTS_EXT=${GLOD_RESULTS_EXT:-/local/$USER/ews_results/fid_wild}
source "$(dirname "$0")/_common.sh"
export PYTHONPATH=/local/$USER/pylib_extra${PYTHONPATH:+:$PYTHONPATH}   # bitsandbytes isolado
DEVICE=${GPU_WILD:-cuda:2}
MODELS=${WILD_MODELS:-"Qwen/Qwen2.5-14B-Instruct mistralai/Mistral-Nemo-Instruct-2407 ibm-granite/granite-3.3-8b-instruct"}
CORPORA=${WILD_CORPORA:-"gsm8k mmlu_en wikitext_nat code"}
CONF="u8 u6 u5 u4 g4 mag20 kv4 kv3 gptq4 awq4 sgpt50 wanda50"
REAL="nf4 int8 fp8"
declare -A RELEASED=( ["Qwen/Qwen2.5-14B-Instruct"]="Qwen/Qwen2.5-14B-Instruct-AWQ Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8" )
log "=== inicio | $DEVICE | registro $(grep -o '"sha256_of_body": "[0-9a-f]*' results/inthewild_predictions.json | cut -c20-35) | saida $GLOD_RESULTS"
gpu_check "$DEVICE"
for m in $MODELS; do
  prefetch "$m" || continue
  for r in ${RELEASED[$m]:-}; do prefetch "$r"; done
  for c in $CORPORA; do
    step "$PY -m glod grid gen --model $m --corpus $c --device $DEVICE" || continue
    for cfg in bf16 $CONF; do
      step "$PY -m glod grid score --model $m --corpus $c --device $DEVICE --logits-fp32 --configs bf16 $cfg"
    done
    for q in $REAL; do
      step "$PY -m glod grid score --model $m --corpus $c --device $DEVICE --logits-fp32 --load-in $q --configs $q"
    done
    for r in ${RELEASED[$m]:-}; do
      step "$PY -m glod grid score --model $m --corpus $c --device $DEVICE --logits-fp32 --configs ofc@$r"
    done
  done
done
step "$PY -m glod wild-check"
finish
