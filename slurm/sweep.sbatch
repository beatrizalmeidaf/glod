#!/usr/bin/env bash
# Um par (modelo, corpus) por tarefa do array. Submeta com scripts/submit_slurm.sh,
# que monta o arquivo de pares e acerta o tamanho do array.
#
# A QOS `onejob` deste cluster permite 2 jobs rodando por usuario, entao o array vai
# com %2: as tarefas restantes ficam na fila e entram sozinhas quando uma termina.
#SBATCH --job-name=glod-sweep
#SBATCH --partition=h100n2,h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=var/logs/slurm/%x_%A_%a.out
#SBATCH --error=var/logs/slurm/%x_%A_%a.out
set -euo pipefail

: "${PAIRS:?defina PAIRS=<arquivo de pares> (scripts/submit_slurm.sh faz isso)}"
PROFILE=${PROFILE:-paper}
cd "${SLURM_SUBMIT_DIR:-.}"

line=$((SLURM_ARRAY_TASK_ID + 1))
read -r MODEL CORPUS < <(sed -n "${line}p" "$PAIRS")
[ -n "${MODEL:-}" ] || { echo "linha $line vazia em $PAIRS"; exit 1; }

echo "== tarefa $SLURM_ARRAY_TASK_ID: $MODEL | $CORPUS | perfil=$PROFILE"
echo "   no: $(hostname) | GPUs visiveis: ${CUDA_VISIBLE_DEVICES:-nenhuma} | tentativa: ${SLURM_RESTART_COUNT:-0}"

# Com --gres=gpu:1 o Slurm expoe uma GPU so, que para o processo e sempre a cuda:0.
# As analises globais NAO rodam aqui (--no-global): elas escrevem os mesmos
# analysis/*.json e sao o job dependente, uma vez, no fim.
srun --unbuffered scripts/run_corpus.sh \
    --model "$MODEL" --corpus "$CORPUS" --device cuda:0 \
    --profile "$PROFILE" --no-global
