# Funcoes compartilhadas pelos scripts de fundo. Uso: `source scripts/bg/_common.sh` com TAG definido.
#
# Tudo escreve em logs.txt (raiz do repo), uma linha por evento: "<data hora> [TAG] mensagem".
# Os resultados vao para uma arvore SEPARADA da do paper (GLOD_RESULTS=.../fid_ext): nenhuma
# analise publicada ve estas referencias. Cada etapa grava um marco ao terminar; reexecutar
# pula as etapas prontas (e as etapas do glod ja sao idempotentes por dentro).
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
LOG=${LOG:-$ROOT/logs.txt}
PY=${PY:-/usr/bin/python3}                  # o venv nao tem torch
FID=${FID:-/local/$USER/ews_results/fid}     # arvore do paper (so leitura aqui)
export GLOD_RESULTS=${GLOD_RESULTS_EXT:-/local/$USER/ews_results/fid_ext}
export GLOD_HF_CACHE=${GLOD_HF_CACHE:-/local/$USER/hf_cache}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# processos em fundo nao leem o ~/.bashrc interativo: pega so o HF_TOKEN de la (sem imprimir)
if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.bashrc ]; then
  HF_TOKEN=$(grep -oE 'HF_TOKEN=[^ ;]+' ~/.bashrc | tail -1 | cut -d= -f2- | tr -d "\"'")
  export HF_TOKEN
fi
STAMPS=$GLOD_RESULTS/_stamps
mkdir -p "$STAMPS" "$GLOD_RESULTS/corpora" "$GLOD_RESULTS/analysis"
FAILS=0

log() { echo "$(date '+%F %T') [$TAG] $*" >> "$LOG"; }

# step <comando...>: roda, prefixa cada linha da saida no log, grava marco se der certo
step() {
  local cmd="$*" st
  st="$STAMPS/$(printf '%s' "$cmd" | sed -e 's/--device [^ ]*//' | sha1sum | cut -c1-16)"
  if [ -f "$st" ]; then log "JA FEITO: $cmd"; return 0; fi
  log "INICIO: $cmd"
  local t0=$SECONDS
  if eval "$cmd" 2>&1 | while IFS= read -r l; do echo "$(date '+%F %T') [$TAG]   $l"; done >> "$LOG"; then
    printf '%s\nok %s\n' "$cmd" "$(date -Is)" > "$st"
    log "OK ($(( (SECONDS - t0) / 60 )) min): $cmd"
  else
    FAILS=$((FAILS + 1))
    log "FALHOU (segue para a proxima etapa): $cmd"
    return 1
  fi
}

# baixa so os arquivos que o from_pretrained usa, no cache do estudo; falha cedo se gated
prefetch() {
  step "$PY -c \"from huggingface_hub import snapshot_download as s; import os; \
print(s('$1', cache_dir=os.environ['GLOD_HF_CACHE'], token=os.environ.get('HF_TOKEN'), \
allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','tokenizer*']))\""
}

# avisa se a GPU ja esta ocupada por outro processo (nao bloqueia)
gpu_check() {
  local idx=${1#cuda:} used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
  log "GPU $idx: ${used:-?} MiB em uso antes de comecar"
  if [ -n "$used" ] && [ "$used" -gt 10000 ]; then
    log "AVISO: GPU $idx ja tem ${used} MiB ocupados; modelos grandes podem dar OOM"
  fi
}

finish() { log "FIM: $FAILS etapa(s) falharam"; exit $(( FAILS > 0 )); }
