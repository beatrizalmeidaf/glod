#!/usr/bin/env bash
# Varredura modelos x corpora, distribuida entre GPUs (uma fila em serie por GPU).
#
#   scripts/run_sweep.sh --dry-run                       # mostra o plano e nao roda nada
#   scripts/run_sweep.sh --corpora "mmlu_en wikitext gsm8k" --devices "cuda:0 cuda:2"
#   scripts/run_sweep.sh --models configs/models.txt --corpora mmlu_en --devices cuda:0
#   scripts/run_sweep.sh --models "Qwen/Qwen3-4B google/gemma-3-4b-it" --profile core
#
# Os pares (modelo, corpus) sao distribuidos round-robin entre as GPUs; cada GPU
# processa a sua fila em serie, e as filas correm em paralelo. Um par que falha nao
# interrompe os outros; o resumo no fim diz quais falharam.
#
# ATENCAO ao custo: o perfil `paper` roda 20 pontos de ataque por par (4 do principal,
# 4 em fp32, 2 sem norma, 8 das sementes) mais a grade de compressores. No Qwen3-4B
# isso da ~2 h por par; no Gemma-12B, ~5 h. Use --dry-run primeiro.
set -euo pipefail

MODELS_ARG="configs/models.txt"
CORPORA="mmlu_en wikitext gsm8k"
DEVICES="cuda:0"
PROFILE="paper"
DRY=0
PY=${PY:-python3}

while [ $# -gt 0 ]; do
  case "$1" in
    --models)  MODELS_ARG=$2; shift 2 ;;
    --corpora) CORPORA=$2; shift 2 ;;
    --devices) DEVICES=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --print-pairs) PRINT_PAIRS=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
  esac
done

# --models aceita um arquivo (uma linha por modelo, # comenta) ou uma lista solta
if [ -f "$MODELS_ARG" ]; then
  MODELS=$(grep -vE '^\s*(#|$)' "$MODELS_ARG" | awk '{print $1}')
else
  MODELS=$MODELS_ARG
fi

read -r -a DEV_ARR <<< "$DEVICES"
n_dev=${#DEV_ARR[@]}
[ "$n_dev" -gt 0 ] || { echo "falta --devices" >&2; exit 2; }

# --------------------------------------------------- monta as filas por GPU
declare -a QUEUE
i=0
for model in $MODELS; do
  for corpus in $CORPORA; do
    d=$((i % n_dev))
    QUEUE[$d]="${QUEUE[$d]:-}${model} ${corpus}"$'\n'
    i=$((i + 1))
  done
done
total=$i

# --print-pairs: uma linha "modelo corpus" por par, na ordem. E o que o job array do
# Slurm consome (indice da tarefa -> linha do arquivo).
if [ "${PRINT_PAIRS:-0}" = 1 ]; then
  for model in $MODELS; do
    for corpus in $CORPORA; do
      echo "$model $corpus"
    done
  done
  exit 0
fi

echo "== varredura: $total pares | perfil=$PROFILE | GPUs: ${DEVICES}"
for d in $(seq 0 $((n_dev - 1))); do
  echo "-- ${DEV_ARR[$d]}:"
  printf '%s' "${QUEUE[$d]:-}" | sed 's/^/     /'
done

if [ "$DRY" = 1 ]; then
  echo
  echo "== comandos de um par (exemplo: primeiro da fila da primeira GPU)"
  first=$(printf '%s' "${QUEUE[0]:-}" | head -n 1)
  if [ -n "$first" ]; then
    set -- $first
    scripts/run_corpus.sh --model "$1" --corpus "$2" --device "${DEV_ARR[0]}" \
        --profile "$PROFILE" --dry-run | sed 's/^/     /'
  fi
  exit 0
fi

LOGDIR=${LOGDIR:-var/logs}
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/sweep_$(date +%Y%m%d_%H%M%S).status"
: > "$STATUS"

for d in $(seq 0 $((n_dev - 1))); do
  dev=${DEV_ARR[$d]}
  queue=${QUEUE[$d]:-}
  [ -n "$queue" ] || continue
  (
    printf '%s' "$queue" | while read -r model corpus; do
      [ -n "${model:-}" ] || continue
      if scripts/run_corpus.sh --model "$model" --corpus "$corpus" --device "$dev" \
             --profile "$PROFILE" --no-global; then
        echo "OK     $model $corpus $dev" >> "$STATUS"
      else
        echo "FALHOU $model $corpus $dev" >> "$STATUS"
      fi
    done
  ) &
done
wait

# As analises globais escrevem os mesmos analysis/*.json para todos os modelos, entao
# rodam UMA vez, depois que todas as filas terminaram (rodar em paralelo corromperia).
echo "== analises globais"
{
  for model in $MODELS; do
    $PY -m glod flip-dirs --model "$model" || echo "flip-dirs falhou: $model" >&2
  done
  $PY -m glod analyze law
  if [ "$PROFILE" = paper ]; then
    $PY -m glod analyze theory
    $PY -m glod analyze prop
    $PY -m glod slope
    $PY -m glod domain
  fi
  $PY -m glod adv-report
  $PY -m glod figures
} 2>&1 | tee -a "$LOGDIR/sweep_global.log"

echo "== resumo ($STATUS)"
sort "$STATUS"
! grep -q '^FALHOU' "$STATUS"
