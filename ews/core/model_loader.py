"""Passo 2 - Carregamento dos modelos base (4B) e completo (12B) em VRAM.

Fase 1 roda tudo residente na GPU: o objetivo e isolar a *matematica* do
EWS do gargalo de I/O. Um H100 de 80GB comporta Gemma-3-12B (~24 GiB em
bf16) e Gemma-3-4B (~9 GiB) simultaneamente na mesma placa, o que permite
comparar base x completo sem trocar pesos de lugar.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)

# Checkpoints de referencia do projeto. Ambos sao *gated* no Hugging Face:
# e preciso aceitar a licenca Gemma na pagina do modelo e exportar HF_TOKEN.
DEFAULT_BASE_MODEL = "google/gemma-3-4b-it"
DEFAULT_FULL_MODEL = "google/gemma-3-12b-it"

# /home tem pouco espaco livre nesta DGX; o cache vai para o array RAID.
from ews.paths import CACHE_DIR as DEFAULT_CACHE_DIR

DTYPE_ALIASES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    """Converte um alias textual (`"bf16"`) em `torch.dtype`."""
    if isinstance(name, torch.dtype):
        return name
    try:
        return DTYPE_ALIASES[name.lower()]
    except KeyError as exc:  # pragma: no cover - erro de configuracao
        raise ValueError(
            f"dtype desconhecido: {name!r}. Opcoes: {sorted(DTYPE_ALIASES)}"
        ) from exc


@dataclass
class LoadedModel:
    """Um checkpoint residente em VRAM + metadados usados nos logs da Fase 1."""

    role: str  # "base" (4B) ou "full" (12B)
    model_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype

    @property
    def decoder(self) -> nn.Module:
        """Modulo que hospeda a `ModuleList` de camadas do decoder."""
        return resolve_decoder(self.model)

    @property
    def layers(self) -> nn.ModuleList:
        return self.decoder.layers  # type: ignore[return-value]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def param_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    @property
    def decoder_param_bytes(self) -> int:
        """Bytes apenas das camadas do decoder (o que o EWS pode 'streamar')."""
        return sum(p.numel() * p.element_size() for p in self.layers.parameters())

    def describe(self) -> str:
        gib = 1024**3
        return (
            f"[{self.role:>4}] {self.model_id}\n"
            f"         classe        : {type(self.model).__name__}\n"
            f"         device/dtype  : {self.device} / {self.dtype}\n"
            f"         camadas       : {self.num_layers}\n"
            f"         hidden_size   : {getattr(self.decoder.config, 'hidden_size', '?')}\n"
            f"         parametros    : {self.param_count / 1e9:.2f} B\n"
            f"         peso total    : {self.param_bytes / gib:.2f} GiB\n"
            f"         peso decoder  : {self.decoder_param_bytes / gib:.2f} GiB "
            f"({100 * self.decoder_param_bytes / max(self.param_bytes, 1):.1f}% do total)"
        )


def resolve_decoder(model: nn.Module) -> nn.Module:
    """Encontra o stack de decoder de forma robusta a checkpoints multimodais.

    Gemma-3 4B/12B sao carregados como `Gemma3ForConditionalGeneration`, cujo
    backbone textual vive em `model.model.language_model`. `get_decoder()`
    resolve isso, mas mantemos um fallback por busca em largura para que o
    mesmo codigo sirva a Llama/Qwen na ablacao de segunda familia (Reviewer 6).
    """
    getter = getattr(model, "get_decoder", None)
    if callable(getter):
        try:
            candidate = getter()
        except (AttributeError, NotImplementedError):
            candidate = None
        if candidate is not None and isinstance(
            getattr(candidate, "layers", None), nn.ModuleList
        ):
            return candidate

    queue: list[nn.Module] = [model]
    while queue:
        current = queue.pop(0)
        if isinstance(getattr(current, "layers", None), nn.ModuleList) and len(
            current.layers  # type: ignore[attr-defined]
        ):
            return current
        queue.extend(current.children())

    raise RuntimeError(
        f"Nao foi possivel localizar o decoder (ModuleList `layers`) em "
        f"{type(model).__name__}."
    )


def load_model(
    model_id: str,
    *,
    role: str,
    device: str | torch.device = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "sdpa",
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> LoadedModel:
    """Carrega um checkpoint inteiro numa unica GPU, em modo inferencia.

    Args:
        model_id: repo do Hugging Face (ex.: ``google/gemma-3-12b-it``).
        role: rotulo usado nos logs (``"base"`` ou ``"full"``).
        device: GPU alvo. Fase 1 usa uma placa por modelo, sem `device_map`
            automatico, para que a medicao de memoria seja atribuivel.
        dtype: bf16 e o padrao no H100 (Gemma-3 foi treinado em bf16).
        attn_implementation: ``"sdpa"`` (rapido) ou ``"eager"`` (referencia
            numerica exata recomendada pelo Google para a familia Gemma).
        cache_dir: diretorio de cache do HF. O padrao aponta para o RAID.
        token: token do HF. Gemma e gated; se ``None``, cai no token salvo
            em ``~/.cache/huggingface/token`` ou na env ``HF_TOKEN``.

    Returns:
        `LoadedModel` com o modelo em `eval()` e grad desabilitado.
    """
    torch_dtype = resolve_dtype(dtype)
    # device="auto": reparte o modelo entre as GPUs visiveis (accelerate). So para
    # checkpoints que nao cabem numa placa (o ponto de escala em 70B); nos demais casos
    # uma placa por modelo mantem a medicao de memoria atribuivel.
    shard = str(device) == "auto"
    torch_device = torch.device("cuda:0") if shard else torch.device(device)
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    LOGGER.info("Carregando [%s] %s em %s (%s)...", role, model_id, torch_device, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, token=hf_token, revision=revision
    )
    if tokenizer.pad_token_id is None:  # Mistral/OLMo base nao definem pad
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
        token=hf_token,
        low_cpu_mem_usage=True,
        revision=revision,
        **({"device_map": "auto"} if shard else {}),
    )
    if shard:
        # com o modelo repartido, a entrada vai para a placa da primeira camada e o
        # accelerate move os estados entre as placas
        torch_device = next(model.parameters()).device
    else:
        model.to(torch_device)
    model.eval()
    model.requires_grad_(False)

    loaded = LoadedModel(
        role=role,
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        dtype=torch_dtype,
    )
    LOGGER.info("Carregado:\n%s", loaded.describe())
    return loaded


def cuda_memory_report(devices: list[torch.device]) -> str:
    """Snapshot de VRAM alocada por GPU - entra nos logs crus da Fase 1."""
    if not torch.cuda.is_available():
        return "CUDA indisponivel."
    gib = 1024**3
    lines = []
    for dev in sorted({d for d in devices if d.type == "cuda"}, key=lambda d: d.index or 0):
        allocated = torch.cuda.memory_allocated(dev) / gib
        reserved = torch.cuda.memory_reserved(dev) / gib
        total = torch.cuda.get_device_properties(dev).total_memory / gib
        lines.append(
            f"  {dev}: alocado {allocated:6.2f} GiB | reservado {reserved:6.2f} GiB "
            f"| total {total:6.2f} GiB"
        )
    return "\n".join(lines) or "  (nenhuma GPU em uso)"
