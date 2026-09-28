#!/usr/bin/env bash
# Impacto (2/2): quantization-aware training. Os checkpoints QAT do Gemma 3 (pesos bf16
# treinados para o formato Q4_0) contra o Gemma denso, nos mesmos corpora do paper:
#   (a) referencia densa: grade padrao + q40 (PTQ no formato Q4_0)
#   (b) referencia QAT propria: grade padrao + q40 (a lei vale num modelo QAT?)
#   (c) cruzado: o QAT inteiro (raw) e o QAT em Q4_0 (o que se implanta) contra o denso:
#       caem na curva de KL e no flips/TV ~ 1 da referencia densa?
#
#   GPU_QAT=cuda:2 nohup scripts/bg/run_qat.sh &          # saida em logs.txt, tag [qat]
TAG=qat
source "$(dirname "$0")/_common.sh"
DEVICE=${GPU_QAT:-cuda:2}
SIZES=${QAT_SIZES:-"1b 4b 12b"}
CORPORA=${QAT_CORPORA:-"gsm8k mmlu_en wikitext wikitext_nat"}
CONF=${CONF:-"q40 u8 u6 u5 u4 g4 mag20 kv4 kv3 gptq4 awq4 sgpt50 wanda50"}
log "=== inicio | $DEVICE | tamanhos: $SIZES | corpora: $CORPORA | saida: $GLOD_RESULTS"
gpu_check "$DEVICE"
for s in $SIZES; do
  dense=google/gemma-3-${s}-it
  qat=google/gemma-3-${s}-it-qat-q4_0-unquantized
  prefetch "$dense" || continue
  prefetch "$qat" || continue
  for c in $CORPORA; do
    rs=gemma-3-${s}-it__${c}
    # (a) denso sobre o MESMO texto do paper, quando ele existe
    if [ -f "$FID/corpora/$rs.json" ] && [ ! -e "$GLOD_RESULTS/corpora/$rs.json" ]; then
      ln -s "$FID/corpora/$rs.json" "$GLOD_RESULTS/corpora/$rs.json"
      log "corpus denso reaproveitado do paper: $rs"
    fi
    step "$PY -m glod grid gen --model $dense --corpus $c --device $DEVICE" || continue
    for cfg in bf16 $CONF; do
      step "$PY -m glod grid score --model $dense --corpus $c --device $DEVICE --logits-fp32 --configs bf16 $cfg"
    done
    # (b) QAT com a propria referencia
    step "$PY -m glod grid gen --model $qat --corpus $c --device $DEVICE" || continue
    for cfg in bf16 $CONF; do
      step "$PY -m glod grid score --model $qat --corpus $c --device $DEVICE --logits-fp32 --configs bf16 $cfg"
    done
    # (c) QAT (e QAT em Q4_0) contra a referencia densa
    step "$PY -m glod grid score --model $qat --ref-model $dense --corpus $c --device $DEVICE --logits-fp32 --configs raw q40"
  done
done
step "$PY -m glod ext-report"
finish
