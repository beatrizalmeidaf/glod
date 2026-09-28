#!/usr/bin/env bash
# Dispara tudo em fundo (sobrevive ao fechar o terminal). Saida unica em logs.txt.
#
#   scripts/bg/run_all.sh                        # QAT e depois MoE na cuda:2; substancia em paralelo (CPU)
#   GPU_QAT=cuda:1 GPU_MOE=cuda:2 scripts/bg/run_all.sh   # QAT e MoE em paralelo, em GPUs diferentes
#   tail -f logs.txt                             # acompanhar;  grep FALHOU logs.txt  # problemas
#   kill $(cat var/bg_pids.txt)                  # parar
set -u
cd "$(dirname "$0")/../.."
mkdir -p var
GPU_QAT=${GPU_QAT:-cuda:2}; GPU_MOE=${GPU_MOE:-cuda:2}
export GPU_QAT GPU_MOE
: > var/bg_pids.txt
setsid nohup scripts/bg/run_substancia.sh >/dev/null 2>&1 & echo $! >> var/bg_pids.txt
if [ "$GPU_QAT" = "$GPU_MOE" ]; then
  # mesma GPU: em serie (o Qwen3-30B-A3B sozinho ocupa ~61 GB); QAT primeiro, e mais barato
  setsid nohup bash -c 'scripts/bg/run_qat.sh; scripts/bg/run_moe.sh' >/dev/null 2>&1 &
  echo $! >> var/bg_pids.txt
else
  setsid nohup scripts/bg/run_qat.sh >/dev/null 2>&1 & echo $! >> var/bg_pids.txt
  setsid nohup scripts/bg/run_moe.sh >/dev/null 2>&1 & echo $! >> var/bg_pids.txt
fi
echo "disparado (pids em var/bg_pids.txt): QAT em $GPU_QAT, MoE em $GPU_MOE; log: $(pwd)/logs.txt"
