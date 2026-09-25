"""Passo 3 - Elastic Depth: desligar camadas do decoder em tempo de inferencia.

Corresponde a "Versao 2 - Elastic Depth" do documento raiz: o gate binario
`g_l(x) = 0` faz a camada `l` virar identidade no fluxo residual,

    h_{l+1} = h_l                          (g_l = 0, camada desligada)
    h_{l+1} = h_l + f_l(h_l, W_l)          (g_l = 1, camada ativa)

Implementacao: em vez de monkey-patch no `forward` do modelo (fragil a cada
release do transformers), embrulhamos cada `DecoderLayer` da `ModuleList` num
`SkippableDecoderLayer` que preserva a assinatura original e delega atributos.
O loop do `Gemma3TextModel` le `decoder_layer.attention_type` para escolher a
mascara (full vs sliding) - por isso o wrapper precisa ser transparente.

Nada aqui move bytes: a camada continua residente em VRAM. Estamos medindo o
*upper bound* de quanta capacidade poderia nao ter sido carregada.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


class SkippableDecoderLayer(nn.Module):
    """Envolve um `DecoderLayer` e o transforma em identidade quando desligado.

    O wrapper e deliberadamente burro: toda a decisao mora no
    `ElasticDepthController` compartilhado, para que uma mudanca de mascara
    valha instantaneamente para as N camadas sem re-embrulhar nada.
    """

    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        controller: "ElasticDepthController",
    ) -> None:
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        # Mantido fora de `self._modules`/`_parameters` para nao virar filho
        # do modulo (evita recursao em `.to()`, `state_dict()` etc.).
        object.__setattr__(self, "_controller", controller)
        # `Gemma3TextModel.forward` consulta este atributo antes de chamar a
        # camada, para escolher entre mascara full_attention e sliding.
        attention_type = getattr(layer, "attention_type", None)
        if attention_type is not None:
            self.attention_type = attention_type

    def __getattr__(self, name: str) -> Any:
        """Delega atributos desconhecidos para a camada original."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("layer")
            if inner is None:
                raise
            return getattr(inner, name)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        controller: ElasticDepthController = object.__getattribute__(self, "_controller")

        if controller.is_active(self.layer_idx):
            controller.note_call(self.layer_idx, skipped=False)
            out = self.layer(hidden_states, **kwargs)
            if controller.tuple_output is None:
                controller.tuple_output = isinstance(out, tuple)
            return out

        controller.note_call(self.layer_idx, skipped=True)

        # KV cache: uma camada pulada nao escreve no seu slot do cache. Isso e
        # inofensivo num prefill unico (o caso da Fase 1), mas corrompe
        # geracao multi-step se a mascara mudar entre os passos - a camada
        # teria cache de comprimento 0 enquanto `cache_position` ja avancou.
        if kwargs.get("use_cache") and not controller.allow_kv_cache:
            raise RuntimeError(
                f"Camada {self.layer_idx} foi pulada com use_cache=True. Na Fase 1 "
                "avalie com use_cache=False (prefill unico). Se voce sabe o que "
                "esta fazendo, set controller.allow_kv_cache = True."
            )

        # Identidade no fluxo residual, no MESMO formato de saida da camada
        # envolvida: `Gemma3DecoderLayer` devolve (hidden_states,) + (attn,)
        # opcional, mas Qwen3/Mistral/OLMo2/Phi3 (transformers 4.56) devolvem o
        # tensor puro e o loop do modelo o consome direto - uma tupla ali quebra.
        # A anotacao de retorno nao serve (Phi3 anota tupla e devolve tensor):
        # o formato e aprendido da primeira chamada ativa ou, se a primeira
        # chamada ja for um skip, de uma unica execucao descartada da camada.
        if controller.tuple_output is None:
            probe = self.layer(hidden_states, **kwargs)
            controller.tuple_output = isinstance(probe, tuple)
        if not controller.tuple_output:
            return hidden_states
        if kwargs.get("output_attentions"):
            return (hidden_states, None)
        return (hidden_states,)

    def extra_repr(self) -> str:
        controller: ElasticDepthController = object.__getattribute__(self, "_controller")
        state = "ON " if controller.is_active(self.layer_idx) else "OFF"
        return f"layer_idx={self.layer_idx}, gate={state}"


class ElasticDepthController:
    """Estado compartilhado do gate `g_l` + contabilidade de bytes/camadas.

    A contabilidade de *bytes por camada* existe desde ja porque a metrica
    central do paper e acuracia x bytes (e depois x latencia), nao acuracia x
    numero de camadas - camadas de um mesmo modelo tem custos parecidos, mas
    isso deixa de valer assim que entrarem residuais de baixo rank na Fase 2.
    """

    def __init__(self, decoder: nn.Module, wrapped: Sequence[SkippableDecoderLayer]) -> None:
        self.decoder = decoder
        self.wrapped = list(wrapped)
        self._active: list[bool] = [True] * len(self.wrapped)
        self._calls: list[int] = [0] * len(self.wrapped)
        self._skips: list[int] = [0] * len(self.wrapped)
        self.allow_kv_cache: bool = False
        # formato de saida das camadas envolvidas (tupla no Gemma3, tensor no
        # Qwen3/Mistral/OLMo2/Phi3); None ate a primeira chamada
        self.tuple_output: Optional[bool] = None
        self.layer_param_bytes: list[int] = [
            sum(p.numel() * p.element_size() for p in w.layer.parameters())
            for w in self.wrapped
        ]

    # ------------------------------------------------------------------ estado
    @property
    def num_layers(self) -> int:
        return len(self.wrapped)

    @property
    def active_mask(self) -> list[bool]:
        return list(self._active)

    @property
    def active_layers(self) -> list[int]:
        return [i for i, on in enumerate(self._active) if on]

    @property
    def skipped_layers(self) -> list[int]:
        return [i for i, on in enumerate(self._active) if not on]

    def is_active(self, layer_idx: int) -> bool:
        return self._active[layer_idx]

    # ------------------------------------------------------------------ setters
    def reset(self) -> None:
        """Religa todas as camadas (modelo denso de referencia)."""
        self._active = [True] * self.num_layers

    def set_mask(self, mask: Iterable[bool | int]) -> None:
        values = [bool(v) for v in mask]
        if len(values) != self.num_layers:
            raise ValueError(
                f"mascara de tamanho {len(values)} != {self.num_layers} camadas"
            )
        self._active = values

    def set_skipped(self, layer_ids: Iterable[int]) -> None:
        ids = self._validate(layer_ids)
        self._active = [i not in ids for i in range(self.num_layers)]

    def set_active(self, layer_ids: Iterable[int]) -> None:
        ids = self._validate(layer_ids)
        self._active = [i in ids for i in range(self.num_layers)]

    def _validate(self, layer_ids: Iterable[int]) -> set[int]:
        ids = {int(i) for i in layer_ids}
        bad = {i for i in ids if not 0 <= i < self.num_layers}
        if bad:
            raise IndexError(f"indices fora de [0,{self.num_layers}): {sorted(bad)}")
        return ids

    # ------------------------------------------------------- context managers
    @contextlib.contextmanager
    def skipping(self, layer_ids: Iterable[int]) -> Iterator["ElasticDepthController"]:
        """`with ctrl.skipping([3,7]):` - desliga camadas e restaura ao sair."""
        previous = self.active_mask
        try:
            self.set_skipped(layer_ids)
            yield self
        finally:
            self._active = previous

    @contextlib.contextmanager
    def using_mask(self, mask: Iterable[bool | int]) -> Iterator["ElasticDepthController"]:
        previous = self.active_mask
        try:
            self.set_mask(mask)
            yield self
        finally:
            self._active = previous

    @contextlib.contextmanager
    def dense(self) -> Iterator["ElasticDepthController"]:
        """Forca o 12B denso completo (baseline / upper bound de qualidade)."""
        previous = self.active_mask
        try:
            self.reset()
            yield self
        finally:
            self._active = previous

    # ------------------------------------------------------------- metricas
    @property
    def total_param_bytes(self) -> int:
        return sum(self.layer_param_bytes)

    @property
    def active_param_bytes(self) -> int:
        return sum(b for b, on in zip(self.layer_param_bytes, self._active) if on)

    @property
    def skipped_fraction_layers(self) -> float:
        return len(self.skipped_layers) / max(self.num_layers, 1)

    @property
    def skipped_fraction_bytes(self) -> float:
        total = self.total_param_bytes
        if total == 0:
            return 0.0
        return 1.0 - self.active_param_bytes / total

    # ------------------------------------------------------- instrumentacao
    def note_call(self, layer_idx: int, *, skipped: bool) -> None:
        self._calls[layer_idx] += 1
        if skipped:
            self._skips[layer_idx] += 1

    def reset_counters(self) -> None:
        self._calls = [0] * self.num_layers
        self._skips = [0] * self.num_layers

    @property
    def counters(self) -> dict[str, list[int]]:
        return {"calls": list(self._calls), "skips": list(self._skips)}

    def summary(self) -> str:
        mask = "".join("1" if on else "0" for on in self._active)
        mib = 1024**2
        return (
            f"gate g_l = {mask}  "
            f"({len(self.active_layers)}/{self.num_layers} camadas ativas | "
            f"skip {100 * self.skipped_fraction_layers:5.1f}% camadas, "
            f"{100 * self.skipped_fraction_bytes:5.1f}% bytes | "
            f"residente {self.active_param_bytes / mib:.0f} MiB de "
            f"{self.total_param_bytes / mib:.0f} MiB)"
        )

    # ------------------------------------------------------------- teardown
    def detach(self) -> None:
        """Desfaz o wrapping, devolvendo o modelo ao estado original."""
        layers: nn.ModuleList = self.decoder.layers  # type: ignore[assignment]
        for i, wrapper in enumerate(self.wrapped):
            layers[i] = wrapper.layer
        self.wrapped = []
        self._active = []


def make_elastic_depth(model: nn.Module, decoder: Optional[nn.Module] = None) -> ElasticDepthController:
    """Embrulha in-place todas as camadas do decoder e devolve o controlador.

    Idempotente: chamar duas vezes no mesmo modelo nao aninha wrappers.
    """
    from glod.core.model_loader import resolve_decoder  # import tardio: evita ciclo

    target = decoder if decoder is not None else resolve_decoder(model)
    layers: nn.ModuleList = target.layers  # type: ignore[assignment]

    controller = ElasticDepthController.__new__(ElasticDepthController)
    wrapped: list[SkippableDecoderLayer] = []
    for idx, layer in enumerate(layers):
        inner = layer.layer if isinstance(layer, SkippableDecoderLayer) else layer
        wrapper = SkippableDecoderLayer(inner, idx, controller)
        layers[idx] = wrapper
        wrapped.append(wrapper)

    ElasticDepthController.__init__(controller, target, wrapped)
    LOGGER.info("Elastic depth habilitado em %d camadas de %s.", len(wrapped), type(model).__name__)
    return controller


def uniform_skip_schedule(
    num_layers: int,
    num_to_skip: int,
    *,
    protect_first: int = 1,
    protect_last: int = 1,
) -> list[int]:
    """Escolhe `num_to_skip` camadas espacadas uniformemente no miolo do stack.

    Baseline "layer-skipping fixo, sem gate" (ablacao 3 do documento). As
    primeiras e ultimas camadas sao protegidas por padrao: a literatura de
    early-exit/layer-pruning mostra consistentemente que elas carregam
    embedding e formatacao de saida, e puni-las colapsa o modelo por um
    motivo que nao tem nada a ver com a hipotese do GLOD.
    """
    if num_to_skip <= 0:
        return []
    lo, hi = protect_first, num_layers - protect_last
    candidates = list(range(lo, hi))
    if num_to_skip >= len(candidates):
        return candidates
    step = len(candidates) / num_to_skip
    picked = {candidates[min(int(i * step + step / 2), len(candidates) - 1)] for i in range(num_to_skip)}
    # desempate determinista caso o arredondamento tenha colidido indices
    for cand in candidates:
        if len(picked) >= num_to_skip:
            break
        picked.add(cand)
    return sorted(picked)[:num_to_skip]
