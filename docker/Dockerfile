# Imagem para rodar o pipeline em outra maquina com GPU NVIDIA.
#
#   docker build -f docker/Dockerfile -t ews:latest .
#
# O torch vem do indice cu128 (a DGX de origem roda CUDA 12.8, driver 580.x).
# Nada de pesos nem de resultados dentro da imagem: modelos e saidas entram por
# volume (/hf_cache e /results), porque um checkpoint de 12B nao cabe em camada.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    EWS_RESULTS=/results \
    EWS_HF_CACHE=/hf_cache \
    EWS_RAW=/workspace/results/raw \
    EWS_DATA=/workspace/data \
    EWS_FIGS=/workspace/results/figs

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git make ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# dependencias primeiro: a camada so refaz quando requirements.txt muda
COPY requirements.txt pyproject.toml ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.6,<3.0" && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install pytest ruff

COPY ews ./ews
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile README.md ./
RUN python3 -m pip install -e . --no-deps

RUN mkdir -p /results /hf_cache
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["ews", "--help"]
