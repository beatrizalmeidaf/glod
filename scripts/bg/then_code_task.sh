#!/usr/bin/env bash
# Espera a selecao de draft em GSM8K terminar, instala a versao do script com DS_CORPUS e roda
# a segunda tarefa (codigo, MBPP) nos dois alvos. Saida em logs.txt, tag [draft].
cd "$(dirname "$0")/../.."
while pgrep -f "scripts/bg/run_draftselect.sh" >/dev/null; do sleep 60; done
[ -f scripts/bg/run_draftselect.sh.new ] && mv scripts/bg/run_draftselect.sh.new scripts/bg/run_draftselect.sh
DS_CORPUS=code VLLM_MEM=0.6 GPU_DS=${GPU_DS:-cuda:2} scripts/bg/run_draftselect.sh
