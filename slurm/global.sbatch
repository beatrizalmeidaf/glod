#!/usr/bin/env bash
# Analises globais, uma vez, depois que o array terminou. Submetido com
# --dependency=afterany:<jobid do array> por scripts/submit_slurm.sh.
#
# Sao globais porque escrevem os mesmos analysis/*.json para todos os modelos: rodar
# em paralelo corromperia os arquivos. Por isso ficam fora das tarefas do array.
#SBATCH --job-name=ews-global
#SBATCH --partition=h100n2,h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=var/logs/slurm/%x_%j.out
#SBATCH --error=var/logs/slurm/%x_%j.out
set -euo pipefail

: "${PAIRS:?defina PAIRS=<arquivo de pares>}"
PROFILE=${PROFILE:-paper}
cd "${SLURM_SUBMIT_DIR:-.}"
PY=${PY:-python3}

echo "== analises globais | no: $(hostname) | perfil=$PROFILE"

# flip-dirs le os pesos de cada modelo e grava no analysis/flipdirs.json compartilhado:
# em serie, um modelo por vez.
for model in $(awk '{print $1}' "$PAIRS" | sort -u); do
  srun --unbuffered $PY -m ews flip-dirs --model "$model" || echo "flip-dirs falhou: $model" >&2
done

srun --unbuffered $PY -m ews analyze law
if [ "$PROFILE" = paper ]; then
  srun --unbuffered $PY -m ews analyze theory
  srun --unbuffered $PY -m ews analyze prop
  srun --unbuffered $PY -m ews slope
  srun --unbuffered $PY -m ews domain
fi
srun --unbuffered $PY -m ews adv-report
srun --unbuffered $PY -m ews figures
