#!/usr/bin/env bash
# Confere o basico antes de gastar GPU: diretorios montados e GPU visivel.
set -euo pipefail

for d in "$EWS_RESULTS" "$EWS_HF_CACHE"; do
  if [ ! -d "$d" ]; then
    echo "aviso: $d nao existe no container (monte com -v); criando vazio" >&2
    mkdir -p "$d"
  fi
done

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "aviso: nenhuma GPU visivel (falta --gpus all?). Estagios de analise (CPU) ainda rodam." >&2
fi

exec "$@"
