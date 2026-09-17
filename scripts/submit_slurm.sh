#!/usr/bin/env bash
# Submete a varredura ao Slurm: um job array (um par por tarefa) + um job dependente
# com as analises globais.
#
#   scripts/submit_slurm.sh --dry-run
#   scripts/submit_slurm.sh --corpora "mmlu_en wikitext gsm8k"
#   scripts/submit_slurm.sh --models "Qwen/Qwen3-4B" --corpora mmlu_en --concurrent 1
#
# Robustez: as tarefas tem --requeue e o run_corpus.sh grava um marco por etapa
# concluida, entao uma tarefa que cai (timeout, preempcao, no que reinicia) recomeca
# do ponto em que estava, nao do zero. Resubmeter o mesmo comando tambem e seguro: o
# que ja terminou e pulado na hora.
set -euo pipefail

MODELS=${MODELS:-configs/models.txt}
CORPORA=${CORPORA:-"mmlu_en wikitext gsm8k"}
PROFILE=${PROFILE:-paper}
CONCURRENT=${CONCURRENT:-2}      # QOS onejob = 2 jobs rodando por usuario
PARTITION=${PARTITION:-h100n2,h100n3}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --models)     MODELS=$2; shift 2 ;;
    --corpora)    CORPORA=$2; shift 2 ;;
    --profile)    PROFILE=$2; shift 2 ;;
    --concurrent) CONCURRENT=$2; shift 2 ;;
    --partition)  PARTITION=$2; shift 2 ;;
    --time)       TIME_LIMIT=$2; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
  esac
done

mkdir -p var/logs/slurm var/slurm
STAMP=$(date +%Y%m%d_%H%M%S)
PAIRS=var/slurm/pairs_${STAMP}.txt
scripts/run_sweep.sh --print-pairs --models "$MODELS" --corpora "$CORPORA" > "$PAIRS"
N=$(wc -l < "$PAIRS")
[ "$N" -gt 0 ] || { echo "nenhum par gerado" >&2; exit 1; }

echo "== $N pares -> $PAIRS"
cat -n "$PAIRS"
echo "== array 0-$((N - 1))%$CONCURRENT | particao $PARTITION | perfil $PROFILE | limite $TIME_LIMIT"

if [ "$DRY" = 1 ]; then
  echo
  echo "sbatch --parsable --array=0-$((N - 1))%$CONCURRENT --partition=$PARTITION --time=$TIME_LIMIT \\"
  echo "    --export=ALL,PAIRS=$PAIRS,PROFILE=$PROFILE slurm/sweep.sbatch"
  echo "sbatch --dependency=afterany:<jobid> --partition=$PARTITION \\"
  echo "    --export=ALL,PAIRS=$PAIRS,PROFILE=$PROFILE slurm/global.sbatch"
  exit 0
fi

ARRAY_ID=$(sbatch --parsable --array=0-$((N - 1))%"$CONCURRENT" --partition="$PARTITION" \
    --time="$TIME_LIMIT" --export=ALL,PAIRS="$PAIRS",PROFILE="$PROFILE" slurm/sweep.sbatch)
echo "array submetido: $ARRAY_ID"

# afterany (nao afterok): se um par falhar, as analises globais ainda consolidam o que
# terminou - e assim que se descobre o que faltou.
GLOBAL_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" --partition="$PARTITION" \
    --export=ALL,PAIRS="$PAIRS",PROFILE="$PROFILE" slurm/global.sbatch)
echo "analises globais: $GLOBAL_ID (depende de $ARRAY_ID)"
echo
echo "acompanhe:   squeue -u $USER"
echo "logs:        var/logs/slurm/ews-sweep_${ARRAY_ID}_*.out"
echo "progresso:   ls \${EWS_RESULTS:-.}/_stamps | wc -l"
echo "resubmeter o que falhou: scripts/submit_slurm.sh --models '$MODELS' --corpora '$CORPORA'"
