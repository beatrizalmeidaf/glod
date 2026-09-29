"""Ponto de entrada unico: `glod <estagio> [args...]` (ou `python -m glod ...`).

Cada estagio e um modulo de `glod.pipelines` com sua propria `main(argv)`; este
dispatcher nao interpreta nenhum argumento alem do nome do estagio, entao tudo o
que vem depois vai inalterado para o pipeline.

    glod grid      --model Qwen/Qwen3-4B --configs bf16 g4 --corpus mmlu_en
    glod adv-multi --model Qwen/Qwen3-4B --kl-budget 0.05
    glod adv-report
"""
from __future__ import annotations

import importlib
import sys

#: nome no CLI -> modulo do pipeline
STAGES: dict[str, str] = {
    # fidelidade: corpus, grade de compressores e as analises que saem dela
    "grid": "glod.pipelines.fidelity.build_grid",
    "analyze": "glod.pipelines.fidelity.analyze",
    "slope": "glod.pipelines.fidelity.slope",
    "domain": "glod.pipelines.fidelity.domain",
    "flip-dirs": "glod.pipelines.fidelity.flip_dirs",
    "flip-kinds": "glod.pipelines.fidelity.flip_kinds",
    "flip-mechanism": "glod.pipelines.fidelity.flip_mechanism",
    "translation-control": "glod.pipelines.fidelity.translation_control",
    "margin-shape": "glod.pipelines.fidelity.margin_shape",
    "delta-g": "glod.pipelines.fidelity.delta_g",
    "tv-mechanism": "glod.pipelines.fidelity.tv_mechanism",
    "tv-theory": "glod.pipelines.fidelity.tv_theory",
    "tv-exact": "glod.pipelines.fidelity.tv_exact",
    "calib-control": "glod.pipelines.fidelity.calib_control",
    "divergences": "glod.pipelines.fidelity.divergences",
    "holdout": "glod.pipelines.fidelity.holdout",
    "holdout-mech": "glod.pipelines.fidelity.holdout_mech",
    "extreme": "glod.pipelines.fidelity.extreme",
    "fungibility": "glod.pipelines.fidelity.fungibility",
    "kl-greedy": "glod.pipelines.fidelity.kl_greedy",
    "mcq-permutation": "glod.pipelines.fidelity.mcq_permutation",
    # extensoes de escopo (substancia e impacto): nada entra no numbers.json do paper
    "exponent-budget": "glod.pipelines.fidelity.exponent_budget",
    "tv-bound": "glod.pipelines.fidelity.tv_bound",
    "kl-reversals": "glod.pipelines.fidelity.kl_reversals",
    "wild-check": "glod.pipelines.fidelity.wild_check",
    "moe-routing": "glod.pipelines.fidelity.moe_routing",
    "ext-report": "glod.pipelines.fidelity.ext_report",
    # ataque adversarial
    "adv-single": "glod.pipelines.adversarial.single_layer",
    "adv-multi": "glod.pipelines.adversarial.multi_layer",
    "adv-report": "glod.pipelines.adversarial.report",
    "rank-ablation": "glod.pipelines.adversarial.rank_ablation",
    # equivalencia condicional em tarefa
    "matched-kl": "glod.pipelines.tasks.matched_kl",
    "crack": "glod.pipelines.tasks.crack_gsm8k",
    "semantic-judge": "glod.pipelines.tasks.semantic_judge",
    "closedloop": "glod.pipelines.tasks.closedloop",
    "compound": "glod.pipelines.tasks.compound",   # a lei de um passo compoe em geracao livre?
    # decodificacao especulativa
    "amq": "glod.pipelines.compress.amq",           # quantizacao ciente do arg-max
    "amq-eval": "glod.pipelines.compress.amq_eval",
    "spec-bench": "glod.pipelines.speculative.bench",
    "spec-law": "glod.pipelines.speculative.law",
    "draft-select": "glod.pipelines.speculative.draft_select",
    # adaptatividade por token
    "adaptive-alpha": "glod.pipelines.adaptive.alpha",
    "adaptive-bits": "glod.pipelines.adaptive.bits",
    "adaptive-closedloop": "glod.pipelines.adaptive.closedloop",
    "adaptive-report": "glod.pipelines.adaptive.report",
    # saidas
    "report": "glod.pipelines.report.glod_report",   # relatorio de fidelidade para usuarios
    "figures": "glod.pipelines.report.figures",
    "html-report": "glod.pipelines.report.html_report",
    # Fase 1 (elastic depth, descartada; mantida para reproduzir o historico)
    "legacy-oracle": "glod.pipelines.legacy.phase1_oracle",
    "legacy-tokens": "glod.pipelines.legacy.phase1c_tokens",
    "legacy-tokens-analyze": "glod.pipelines.legacy.phase1c_analyze",
    "legacy-closedloop": "glod.pipelines.legacy.phase1d_closedloop",
    "legacy-phase-diagram": "glod.pipelines.legacy.phase_diagram",
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
