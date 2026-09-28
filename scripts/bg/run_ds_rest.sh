#!/usr/bin/env bash
# Completa a selecao de draft: drafts restantes do Qwen3-14B em GSM8K (retomada) e a segunda
# tarefa (codigo) nos dois alvos. VLLM_MEM alto: 14B + draft de 8B precisam de ~44 GB so de pesos.
cd "$(dirname "$0")/../.."
export VLLM_MEM=${VLLM_MEM:-0.85} GPU_DS=${GPU_DS:-cuda:2}
DS_TARGETS=Qwen/Qwen3-14B scripts/bg/run_draftselect.sh
DS_CORPUS=code scripts/bg/run_draftselect.sh
