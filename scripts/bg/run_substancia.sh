#!/usr/bin/env bash
# Substancia: de onde vem o desvio do expoente de 1/2 (lacuna do App. A.2 do paper).
# So CPU, le os JSON publicados de $FID/analysis e grava em $GLOD_RESULTS/analysis.
#
#   nohup scripts/bg/run_substancia.sh &     # saida em logs.txt, tag [substancia]
TAG=substancia
source "$(dirname "$0")/_common.sh"
log "=== inicio (identidade alpha = 1/2 + beta_A + beta_top2 + beta_fora_do_par, refs fp32)"
step "$PY -m glod exponent-budget --src $FID/analysis/prop.json --out $GLOD_RESULTS/analysis/exponent_budget.json"
finish
