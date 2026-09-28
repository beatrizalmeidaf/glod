#!/usr/bin/env bash
# Especulativa ponta a ponta: escolher o draft por TV ou por KL (teacher forcing, GSM8K) e
# medir o speedup realizado no vLLM (kernels reais, prompts GSM8K disjuntos da calibracao).
#
#   GPU_DS=cuda:2 nohup scripts/bg/run_draftselect.sh &      # saida em logs.txt, tag [draft]
TAG=draft
export GLOD_RESULTS_EXT=${GLOD_RESULTS_EXT:-/local/$USER/ews_results/fid_ds}
source "$(dirname "$0")/_common.sh"
DEVICE=${GPU_DS:-cuda:2}
VLLM_PY=${VLLM_PY:-/local/$USER/venv_vllm/bin/python}
# o vLLM compila kernels na primeira execucao: precisa do nvcc (pacote nvidia/cu13 do venv;
# nao ha /usr/local/cuda nesta maquina) e do ninja (bin do venv, fora do PATH quando o python
# e chamado por caminho absoluto)
VENV_BIN=$(dirname "$VLLM_PY")
export CUDA_HOME=${CUDA_HOME:-$VENV_BIN/../lib/python3.12/site-packages/nvidia/cu13}
export PATH=$VENV_BIN:$CUDA_HOME/bin:$PATH
# o sampler do FlashInfer compila por JIT e seus headers nao batem com o nvcc do cu13;
# o sampler nativo do vLLM faz a mesma amostragem sem compilar nada
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_CACHE=$GLOD_HF_CACHE
N_PROMPTS=${N_PROMPTS:-64}
C=${DS_CORPUS:-gsm8k}                       # tarefa: gsm8k ou code (MBPP)
SFX=""; [ "$C" != gsm8k ] && SFX="__$C"
# alvo -> drafts como nome=modelo:config (config raw = o modelo como publicado; ofc@ = checkpoint quantizado)
declare -A DRAFTS=(
  ["Qwen/Qwen3-14B"]="q0.6b=Qwen/Qwen3-0.6B:raw q1.7b=Qwen/Qwen3-1.7B:raw q4b=Qwen/Qwen3-4B:raw q8b=Qwen/Qwen3-8B:raw q1.7b-gptq8=Qwen/Qwen3-1.7B:ofc@Qwen/Qwen3-1.7B-GPTQ-Int8 q4b-awq=Qwen/Qwen3-4B:ofc@Qwen/Qwen3-4B-AWQ q8b-awq=Qwen/Qwen3-8B:ofc@Qwen/Qwen3-8B-AWQ"
  ["Qwen/Qwen3-8B"]="q0.6b=Qwen/Qwen3-0.6B:raw q1.7b=Qwen/Qwen3-1.7B:raw q4b=Qwen/Qwen3-4B:raw q1.7b-gptq8=Qwen/Qwen3-1.7B:ofc@Qwen/Qwen3-1.7B-GPTQ-Int8 q4b-awq=Qwen/Qwen3-4B:ofc@Qwen/Qwen3-4B-AWQ"
)
TARGETS=${DS_TARGETS:-"Qwen/Qwen3-14B Qwen/Qwen3-8B"}
log "=== inicio | $DEVICE | alvos: $TARGETS | saida $GLOD_RESULTS"
gpu_check "$DEVICE"
for t in $TARGETS; do
  prefetch "$t" || continue
  step "$PY -m glod grid gen --model $t --corpus $C --device $DEVICE" || continue
  step "$PY -m glod grid score --model $t --corpus $C --device $DEVICE --logits-fp32 --configs bf16" || continue
  vdrafts=""
  for d in ${DRAFTS[$t]}; do
    name=${d%%=*}; rest=${d#*=}; model=${rest%%:*}; cfg=${rest#*:}
    repo=$model; [[ $cfg == ofc@* ]] && repo=${cfg#ofc@}
    prefetch "$model"; [ "$repo" != "$model" ] && prefetch "$repo"
    step "$PY -m glod grid score --model $model --ref-model $t --corpus $C --device $DEVICE --logits-fp32 --configs $cfg"
    vdrafts="$vdrafts $name=$repo"
  done
  s=$(basename "$t")
  step "$PY -m glod draft-select prompts --target $t --corpus $C --n $N_PROMPTS --out $GLOD_RESULTS/ds_prompts__$s$SFX.json" || continue
  dev=${DEVICE#cuda:}
  step "CUDA_VISIBLE_DEVICES=$dev $VLLM_PY scripts/vllm_spec_bench.py --target $t --drafts$vdrafts --prompts $GLOD_RESULTS/ds_prompts__$s$SFX.json --out $GLOD_RESULTS/ds_vllm__$s$SFX.json --mem ${VLLM_MEM:-0.45}" || continue
  step "$PY -m glod draft-select report --target $t --corpus $C --vllm $GLOD_RESULTS/ds_vllm__$s$SFX.json --drafts ${DRAFTS[$t]}"
done
finish
