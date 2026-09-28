#!/usr/bin/env bash
# Espera a cadeia da rodada 2 terminar e refaz a selecao de draft com o nvcc do venv do vLLM
# (a 1a tentativa falhou: "Could not find nvcc and default cuda_home='/usr/local/cuda'").
# As etapas HF ja feitas sao puladas pelos marcos; so as do vLLM rodam de novo.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat var/bg_pids2.txt)" 2>/dev/null; do sleep 120; done
export CUDA_HOME=/local/$USER/venv_vllm/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
GPU_DS=${GPU_DS:-cuda:2} scripts/bg/run_draftselect.sh
