"""EWS - Elastic Weight Streaming.

Fase 1: simulacao in-VRAM do Gate Oraculo (upper bound teorico).
Nesta fase nao existe I/O de disco, quantizacao nem Gumbel-Softmax: o
modelo completo fica residente na VRAM e apenas *fingimos* que camadas
foram desligadas, para medir quanto de capacidade e de fato necessario.
"""

from ews.core.elastic_depth import (
    ElasticDepthController,
    SkippableDecoderLayer,
    make_elastic_depth,
    uniform_skip_schedule,
)
from ews.core.model_loader import (
    DEFAULT_BASE_MODEL,
    DEFAULT_FULL_MODEL,
    LoadedModel,
    load_model,
)

__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_FULL_MODEL",
    "ElasticDepthController",
    "LoadedModel",
    "SkippableDecoderLayer",
    "load_model",
    "make_elastic_depth",
    "uniform_skip_schedule",
]

__version__ = "0.1.0"
