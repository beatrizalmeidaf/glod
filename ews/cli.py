"""Ponto de entrada unico: `ews <estagio> [args...]` (ou `python -m ews ...`).

Cada estagio e um modulo de `ews.pipelines` com sua propria `main(argv)`; este
dispatcher nao interpreta nenhum argumento alem do nome do estagio, entao tudo o
que vem depois vai inalterado para o pipeline.

    ews grid      --model Qwen/Qwen3-4B --configs bf16 g4 --corpus mmlu_en
    ews adv-multi --model Qwen/Qwen3-4B --kl-budget 0.05
    ews adv-report
"""
from __future__ import annotations

import importlib
import sys

#: nome no CLI -> modulo do pipeline
STAGES: dict[str, str] = {
    # fidelidade: corpus, grade de compressores e as analises que saem dela
    "grid": "ews.pipelines.fidelity.build_grid",
    "analyze": "ews.pipelines.fidelity.analyze",
    "slope": "ews.pipelines.fidelity.slope",
    "domain": "ews.pipelines.fidelity.domain",
    "flip-dirs": "ews.pipelines.fidelity.flip_dirs",
    "fungibility": "ews.pipelines.fidelity.fungibility",
    "kl-greedy": "ews.pipelines.fidelity.kl_greedy",
    "mcq-permutation": "ews.pipelines.fidelity.mcq_permutation",
    # ataque adversarial (Secao 4)
    "adv-single": "ews.pipelines.adversarial.single_layer",
    "adv-multi": "ews.pipelines.adversarial.multi_layer",
    "adv-report": "ews.pipelines.adversarial.report",
    # equivalencia condicional em tarefa (Secao 5)
    "matched-kl": "ews.pipelines.tasks.matched_kl",
    "crack": "ews.pipelines.tasks.crack_gsm8k",
    "closedloop": "ews.pipelines.tasks.closedloop",
    "compound": "ews.pipelines.tasks.compound",   # a lei de um passo compoe em geracao livre?
    # decodificacao especulativa (Secao 6)
    "spec-bench": "ews.pipelines.speculative.bench",
    "spec-law": "ews.pipelines.speculative.law",
    # adaptatividade por token (Secao 7)
    "adaptive-alpha": "ews.pipelines.adaptive.alpha",
    "adaptive-bits": "ews.pipelines.adaptive.bits",
    "adaptive-closedloop": "ews.pipelines.adaptive.closedloop",
    "adaptive-report": "ews.pipelines.adaptive.report",
    # saidas
    "figures": "ews.pipelines.report.figures",
    "html-report": "ews.pipelines.report.html_report",
    # Fase 1 (elastic depth, descartada; mantida para reproduzir o historico)
    "legacy-oracle": "ews.pipelines.legacy.phase1_oracle",
    "legacy-tokens": "ews.pipelines.legacy.phase1c_tokens",
    "legacy-tokens-analyze": "ews.pipelines.legacy.phase1c_analyze",
    "legacy-closedloop": "ews.pipelines.legacy.phase1d_closedloop",
    "legacy-phase-diagram": "ews.pipelines.legacy.phase_diagram",
}


def _usage() -> str:
    largest = max(len(k) for k in STAGES)
    lines = [__doc__.strip(), "", "estagios:"]
    lines += [f"  {k.ljust(largest)}  {v}" for k, v in STAGES.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    stage, rest = argv[0], argv[1:]
    if stage not in STAGES:
        print(f"estagio desconhecido: {stage}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    mod = importlib.import_module(STAGES[stage])
    fn = getattr(mod, "main", None)
    if fn is None:                       # pragma: no cover - pipeline sem main()
        raise SystemExit(f"{STAGES[stage]} nao tem main()")
    try:                                 # alguns relatorios nao recebem argv
        return int(fn(rest) or 0)
    except TypeError:
        if rest:
            raise
        return int(fn() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
