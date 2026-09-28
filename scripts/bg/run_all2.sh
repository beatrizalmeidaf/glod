#!/usr/bin/env bash
# Rodada 2 em fundo: selecao de draft no vLLM e depois o teste in-the-wild pre-registrado,
# em serie na mesma GPU (o vLLM reserva ~45% da placa). Saida em logs.txt.
#
#   GPU=cuda:2 scripts/bg/run_all2.sh ;  tail -f logs.txt ;  kill $(cat var/bg_pids2.txt)
set -u
cd "$(dirname "$0")/../.."
mkdir -p var
export GPU_DS=${GPU:-cuda:2} GPU_WILD=${GPU:-cuda:2}
setsid nohup bash -c 'scripts/bg/run_draftselect.sh; scripts/bg/run_wild.sh' >/dev/null 2>&1 &
echo $! > var/bg_pids2.txt
echo "disparado em $GPU_DS (pid $(cat var/bg_pids2.txt)); log: $(pwd)/logs.txt"
