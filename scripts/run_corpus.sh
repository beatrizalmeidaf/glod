#!/usr/bin/env bash
# Cadeia de um (modelo, corpus) numa GPU, em serie.
#
#   scripts/run_corpus.sh --model Qwen/Qwen3-4B --corpus mmlu_en --device cuda:0
#   scripts/run_corpus.sh --model Qwen/Qwen3-4B --corpus wikitext --profile paper
#   scripts/run_corpus.sh --model Qwen/Qwen3-4B --corpus gsm8k --dry-run
#
# Perfis (o que cada um roda, na ordem):
#   core   corpus greedy -> grade de compressores -> analise da lei -> ataque -> relatorio
#   paper  core + grade em fp32 + kappa x geometria + dominio + teto + fungibilidade
#          + ataque em fp32 (P5) + ataque sem a norma final (P7) + sementes 1 e 2 (P8)
#
# Cada etapa e idempotente (pula o que ja existe em $EWS_RESULTS), entao reexecutar
# depois de uma falha nao repete trabalho. Os estagios de tarefa (matched-kl, crack),
# de especulativa e de adaptatividade NAO entram aqui: eles nao recebem --corpus.
set -euo pipefail

MODEL="" CORPUS="mix" DEVICE="cuda:0" PROFILE="paper" DRY=0 GLOBAL=1 FORCE=0
CONFIGS=${CONFIGS:-"g4 gptq4 awq4 sgpt50 wanda50 kv4 kv3"}
KL=${KL:-"0.02 0.05"}
STEPS=${STEPS:-400}
RANK=${RANK:-16}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)   MODEL=$2; shift 2 ;;
    --corpus)  CORPUS=$2; shift 2 ;;
    --device)  DEVICE=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --configs) CONFIGS=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --force)   FORCE=1; shift ;;      # ignora os marcos e refaz tudo
    --no-global) GLOBAL=0; shift ;;   # pula as analises que escrevem JSONs compartilhados
                                      # (analyze/slope/domain/flip-dirs/adv-report): em varredura
                                      # paralela elas correm uma vez no fim, sem corrida
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] || { echo "falta --model" >&2; exit 2; }

PY=${PY:-python3}
EWS=("$PY" -m ews)

cmds=()
add() { cmds+=("$*"); }

# ---------------------------------------------------------------- fidelidade
add "${EWS[*]} grid gen   --model $MODEL --device $DEVICE --corpus $CORPUS"
add "${EWS[*]} grid score --model $MODEL --device $DEVICE --corpus $CORPUS --configs bf16 $CONFIGS"
if [ "$PROFILE" = paper ]; then
  add "${EWS[*]} grid score --model $MODEL --device $DEVICE --corpus $CORPUS --logits-fp32 --configs bf16 $CONFIGS"
fi
if [ "$GLOBAL" = 1 ]; then
  add "${EWS[*]} analyze law"
fi
if [ "$PROFILE" = paper ]; then
  if [ "$GLOBAL" = 1 ]; then
    add "${EWS[*]} analyze theory"
    add "${EWS[*]} analyze prop"
    add "${EWS[*]} slope"
    add "${EWS[*]} domain"
    add "${EWS[*]} flip-dirs --model $MODEL"
  fi
  add "${EWS[*]} fungibility --model $MODEL --device $DEVICE --corpus $CORPUS"
fi

# ------------------------------------------------------------------- ataque
adv() { # <tag> <extra flags> <kl>
  add "${EWS[*]} adv-multi --model $MODEL --device $DEVICE --corpus $CORPUS --steps $STEPS --rank $RANK --kl-budget $3 --modes malign benign --tag $1 $2"
}
adv multi "" "$KL"
if [ "$PROFILE" = paper ]; then
  adv fp32 "--logits-fp32" "$KL"                 # P5: sem os empates do bf16
  adv semnorma "--no-output-scale" "0.05"        # P7: sem a escala de saida
  adv seed1 "--seed 1" "$KL"                     # P8: variancia entre sementes
  adv seed2 "--seed 2" "$KL"
fi
if [ "$GLOBAL" = 1 ]; then
  add "${EWS[*]} adv-report"
fi

# --------------------------------------------------------------------- roda
if [ "$DRY" = 1 ]; then
  printf '%s\n' "${cmds[@]}"
  exit 0
fi

LOGDIR=${LOGDIR:-var/logs}
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(basename "$MODEL")__${CORPUS}__${PROFILE}.log"

# Marcos de progresso: uma etapa que terminou com sucesso grava um arquivo com o hash
# do seu comando. Reexecutar (depois de queda, timeout ou requeue do Slurm) pula essas
# etapas na hora, sem nem carregar o modelo. As etapas ja eram idempotentes por dentro
# (o ataque guarda cada ponto em results_<tag>.json assim que ele termina, e a grade
# guarda cada config em .pt), entao o pior caso e repetir a etapa em andamento.
STAMPS=${EWS_STAMPS:-${EWS_RESULTS:-.}/_stamps}
mkdir -p "$STAMPS"
# O marco identifica a ETAPA, nao a GPU: o --device sai do hash, senao a mesma etapa
# rodada em cuda:2 local e em cuda:0 no Slurm contaria como duas e repetiria o trabalho.
stamp_of() { printf '%s' "$1" | sed 's/--device [^ ]*//' | sha1sum | cut -c1-16; }

echo "== $MODEL | corpus=$CORPUS | $DEVICE | perfil=$PROFILE | log=$LOG"
echo "   marcos em $STAMPS (--force refaz tudo)"
rc_total=0
{
  echo "### inicio: $(date -Is) | host=$(hostname) | job=${SLURM_JOB_ID:-nenhum}"
  for c in "${cmds[@]}"; do
    st="$STAMPS/$(stamp_of "$c")"
    if [ "$FORCE" = 0 ] && [ -f "$st" ]; then
      echo "### JA FEITO (marco $(basename "$st")): $c"
      continue
    fi
    echo "### $c"
    # shellcheck disable=SC2086
    if eval $c; then
      { echo "$c"; echo "ok: $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-nenhum}"; } > "$st"
    else
      echo "### FALHOU (segue para a proxima etapa): $c"
      rc_total=1
    fi
  done
  echo "### fim: $(date -Is) | rc=$rc_total"
} 2>&1 | tee -a "$LOG"
exit $rc_total
