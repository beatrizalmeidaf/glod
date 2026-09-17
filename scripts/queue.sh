#!/usr/bin/env bash
# Fila em serie numa GPU: recebe linhas "modelo corpus" na entrada padrao.
#
#   printf 'Qwen/Qwen3-4B mmlu_en\nQwen/Qwen3-4B wikitext\n' | scripts/queue.sh cuda:0
#
# Serve para nao disputar a mesma GPU: um job por vez, na ordem dada. Nada de
# esperar por PID de outro processo (o que travou as filas antigas do projeto).
set -euo pipefail
DEVICE=${1:?uso: queue.sh <device>}
while read -r model corpus; do
  [ -z "${model:-}" ] && continue
  scripts/run_corpus.sh "$model" "${corpus:-mix}" "$DEVICE" || echo "FALHOU: $model $corpus" >&2
done
