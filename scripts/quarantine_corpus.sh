#!/usr/bin/env bash
# Tira do caminho todos os resultados de um corpus, sem apagar: move para
# $EWS_RESULTS/_quarantine/<corpus>_<data>/ junto com os marcos das etapas dele, para
# que uma nova rodada refaca o corpus do zero em vez de pular etapas "ja feitas".
#
#   scripts/quarantine_corpus.sh wikitext --dry-run
#   scripts/quarantine_corpus.sh wikitext
#
# Para desfazer, mova o conteudo da quarentena de volta para $EWS_RESULTS.
set -euo pipefail

CORPUS=${1:?uso: quarantine_corpus.sh <corpus> [--dry-run]}
DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1
[ "$CORPUS" != mix ] || { echo "recuso mover o corpus do paper (mix)" >&2; exit 2; }
R=${EWS_RESULTS:?defina EWS_RESULTS}
Q="$R/_quarantine/${CORPUS}_$(date +%Y%m%d_%H%M%S)"

mapfile -t items < <(
  cd "$R"
  ls -d corpora/*__"$CORPUS".json 2>/dev/null
  ls -d *__"$CORPUS" *__"$CORPUS"__fp32 2>/dev/null
  ls -d adversarial/*__"$CORPUS" adversarial/*__"$CORPUS"__fp32 2>/dev/null
  ls -d fungibility/*__"$CORPUS" 2>/dev/null
  # marcos: a primeira linha e o comando da etapa
  grep -l -- "--corpus $CORPUS\b" _stamps/* 2>/dev/null
)

echo "== $CORPUS: ${#items[@]} itens -> $Q"
printf '   %s\n' "${items[@]}" | awk '{print}' | sed -n '1,12p'
[ "${#items[@]}" -gt 12 ] && echo "   ... (+$(( ${#items[@]} - 12 )))"
[ "$DRY" = 1 ] && exit 0

for it in "${items[@]}"; do
  mkdir -p "$Q/$(dirname "$it")"
  mv "$R/$it" "$Q/$it"
done
echo "movido. para desfazer: cp -r $Q/* $R/ (e remova a quarentena)"
