<div align="center">
  
<img src="results/graficos/logo.png" alt="GLOD Logo" width="200" />

# GLOD (Geometric Law of Damage)

> **Note on "Damage"**: In the context of GLOD, "Damage" refers strictly to **distributional deviation** from the dense oracle (teacher-forcing decision flips), not a loss in semantic capability or downstream task accuracy.

**GLOD** is the analytical framework and evaluation suite introduced in the paper: *How Divergence Becomes Decision Flips in Compressed Language Models*.

While the paper describes the theoretical discovery, the **GLOD** package provides the empirical infrastructure to measure how the margin geometry of an LLM converts statistical perturbations (such as compression) into decision changes ("damage").

[![license](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![paper](https://img.shields.io/badge/arxiv-Paper-red.svg)]()
[![simulator](https://img.shields.io/badge/try-Simulator-blue.svg)](https://github.com/beatrizalmeidaf/elastic_weight_streaming/tree/gh-pages)

</div>

---

## What is the Geometric Law of Damage?

Today we have dozens of techniques to compress LLM weights and accelerate inference (pruning, quantization, layer skipping, etc.). When we choose one of these methods, the big question is: **do they cause different types of "damage" to the model's decisions, or do they merely differ in the amount of damage?**

We measured the effect of 10+ compressors across **19 models** from six families (Gemma-3, Qwen3/2.5, Mistral, Phi, OLMo-2 and Llama-3), from 1B to 72B parameters, on five corpora — **802 configurations** in total — under teacher forcing, counting how often the compressed model's arg-max token differs from the dense model's (the *flip rate*). The central findings:

> **Total Variation (TV)** — a first-order divergence — tracks the flip rate at a ratio near one (median 1.05; model-level mean 1.075, 95% CI [1.04, 1.11]) with **no fitted constant**. KL tracks flips too, but only through an conversion factor κ = flips/√KL that varies 4.3× across models and corpora.

<div align="center">
  <h3><strong>flips ≈ TV,&nbsp;&nbsp;&nbsp; flips ≈ κ · √KL</strong></h3>
  <img src="results/graficos/tv_chart.png" alt="Flips against total variation, and kappa against the Jensen factor" width="820"/>
</div>

**Why KL needs a coefficient.** The square-root form is not an empirical finding: divergence is **second** order in the perturbation (for weights, the Fisher form below), while the decision margin moves at **first** order, so flips grow as √KL.

$$ \mathrm{KL} = \frac{1}{2} \delta^\top F \delta + O(\lVert\delta\rVert^3) $$

**Why κ varies (the Jensen factor).** κ decomposes exactly as κ = (flips/TV) · c · J, where J = E[√KL_t] / √E[KL_t]. Because KL is averaged over tokens *before* the square root, κ is small when the divergence is concentrated on a few tokens (GSM8K, where most tokens are near-certain; J ≈ 0.38) and larger when it is spread out (natural text; J ≈ 0.75). J carries **70%** of the variance of log κ across references; TV is first order at every token and has no such factor. A first-order model that uses only the dense model, flips/TV = √2·ρ(0)/E[h(p)] with h(p) = Σᵢ pᵢ√(1 − 2pᵢ + Σⱼpⱼ²) (isotropic logit displacement), predicts the per-reference ratio with a mean error of 10% and orders the references as measured (`python -m glod tv-theory`). As temperature goes to zero, flips/TV tends to exactly one, and its median stays between 1.01 and 1.11 for temperatures 0.25–2. The split is by order, not by statistic: Hellinger distance, also first order at every token, is nearly as stable across references (per-reference spread 1.54× vs 1.33× for TV), while √JS and √KL, both second order before the root, spread 3.6× and 4.0×.

**Where it holds.** The quadratic approximation is within 11% where κ is measured (KL < 0.05). Flips/√KL is flat up to ~0.25 nats, rises above it, and breaks at 2-bit weights, where KL reaches 8–18 nats and nearly every decision flips; flips/TV stays within about 25% of its window value throughout.

**Out of sample (pre-registered).** Five predictions were registered before generating a held-out code corpus (MBPP) for 3 models, then extended unchanged to 5 more before their completions existed. In all 8 models the exponent (0.48–0.57) and the near-unit flips/TV ratio (0.98–1.05) held. κ predicted from the Jensen factor overestimated κ in every model (by 5–25%; within the 15% tolerance in 3 of 8), and per-family deviations were about three times larger than in sample (median 7.5% vs 2.4%) — but the failure is KL's: measured against TV, the same deviations are 2.7% (vs 7.5% against √KL). 50% sparsity spreads its divergence 1.3–1.5× more evenly across code tokens (the Jensen factor J) at an unchanged flips/TV, so KL understates its flips; recalibrating on Python code lowers KL by 28% but leaves the deviations (7.7% → 7.3%). Across domains, families are interchangeable in TV, not in KL (`python -m glod holdout-mech`). Predictions: [`results/holdout_predictions.json`](results/holdout_predictions.json), [`results/holdout_predictions_ext.json`](results/holdout_predictions_ext.json).

<div align="center">
  <img src="results/graficos/regime_chart.png" alt="Where the square-root relation bends and breaks" width="700"/>
</div>

### How much the method family still matters, at matched KL

For each reference, we fit the reference's own power law **without** the family being measured and report how far that family's flip rate sits from it (multiplicative; intervals clustered by model; 802 configurations). The last column restricts the comparison to the KL window where all families coexist.

| Compressor | Family | Flips vs. curve (held out) | 95% CI | Common KL window |
|:---|:---|:---:|:---:|:---:|
| **Layer removal** | Structural | **0.945** ▼ | [0.920, 0.970] | 0.936 |
| KV-cache (quant.) | Cache | 0.986 | [0.974, 0.997] | 0.984 |
| Magnitude | Pruning | 0.990 | [0.955, 1.026] | 0.989 |
| Gaussian noise | Control | 0.995 | [0.980, 1.011] | 0.997 |
| GPTQ | Quantization | 0.996 | [0.988, 1.005] | 1.004 |
| AWQ | Quantization | 1.003 | [0.991, 1.016] | 1.006 |
| Wanda | Pruning | 1.007 | [0.970, 1.044] | 0.986 |
| RTN | Rounding | 1.022 | [1.005, 1.039] | 1.014 |
| SparseGPT | Pruning | 1.024 | [1.001, 1.048] | 1.009 |

*All families stay within 5.5% of the curve; in the common window ($0.03 \leq \mathrm{KL} \leq 0.20$) eight of the nine stay within 1.6%. After a Holm correction only layer removal deviates significantly. Recalibrating GPTQ/AWQ/SparseGPT/Wanda on C4 instead of WikiText-2 moves them along the curve, not off it. Closeness to the curve says how many decisions change, not how good a compressor is.*

<div align="center">
  <img src="results/graficos/methods_chart.png" alt="Deviation of each family from the reference curve" width="700"/>
</div>

- **Standard compressors look like generic perturbations:** they align with the flip directions no better than isotropic Gaussian noise of the same Fisher norm.
- **Optimized perturbations:** a gradient search at a fixed KL budget raised flips by 13–17% on two corpora and not detectably on two others — a lower bound on the worst case, not a ceiling — while the opposite objective **halved** them on every corpus. To first order, standard compressors produce 17.4% of the flip rate of a token-by-token oracle; a fixed perturbation can at most double that, and the rest requires choosing token by token where the divergence goes. The practical room is in the other direction: **arg-max-aware compressors** that keep their error out of the flip directions could change half as many decisions at the same divergence (per-token precision allocation, which we also tested, does *not* pay off in free-running generation).

<div align="center">
  <img src="results/graficos/attacks_chart.png" alt="Optimized perturbations vs standard compressors" width="700"/>
</div>

### Empirical Validation by Domain

Two distinct quantities: the **exponent** (the slope in log-log, which theory predicts to be ≈ ½) and **κ** (the conversion factor, which depends on the model+corpus pair).

| Corpus | Scope | Pooled Exponent | Pooled R² | κ (range across models) |
|:---|:---|:---:|:---:|:---:|
| **GSM8K** | Exact logical answers | 0.511 | 0.954 | 0.13 – 0.24 |
| **Mix (GSM8K+MMLU-PT)** | Mixed | 0.511 | 0.961 | 0.19 – 0.35 |
| **MMLU-en** | General knowledge | 0.482 | 0.941 | 0.21 – 0.45 |
| **WikiText** | Free text generation | 0.464 | 0.937 | 0.27 – 0.56 |
| **WikiText (natural)** | Real text, no generation | 0.471 | 0.980 | 0.39 – 0.56 |

*The pooled R² is lower than that of any isolated reference (all ≥ 0.990) because the references differ in intercept; per-reference exponents range 0.46–0.58. κ varies 1.6–2.6× between corpora **within the same model**: reporting "the κ of model X" without naming the distribution means nothing.*

### What Flip Counts Do Not Measure (Matched Divergence, Unmatched Accuracy)

GLOD measures **decision fidelity** to the dense model, not downstream competence (accuracy):
1. **Fidelity ≠ accuracy.** On Gemma-3-4B at the same GSM8K divergence, Wanda gains 4.8 points while magnitude pruning loses 33.3 — at the same flip rate. That contrast is specific to that model (elsewhere magnitude pruning costs at most 9.5 points), and three random draws of the same Gaussian noise at the same divergence span 18 points.
2. **Repair is largely re-sampling.** Over 139 (model, method, divergence) cells, every perturbation — Gaussian noise included — fixes about a quarter of the dense model's wrong answers (median 25%), because changing the trajectory of a failed problem re-samples its answer. Wanda's 40% on Gemma-3-4B is the top of that range, a few points above the best Gaussian seed.
3. **Across models, divergence predicts changed answers poorly** (leave-one-model-out R² = 0.31 for √KL).
4. **A count cannot tell a paraphrase from a changed quantity.** Two LLM judges from different families (Qwen2.5-72B, Mistral-Small-24B) disagree on how many flips change mathematical content (25.5% vs 43.2% for magnitude pruning) but agree that magnitude pruning's flips are only modestly more consequential than Wanda's (+5.7 and +6.5 points); inter-judge agreement 82% (Cohen's kappa 0.60). On a blind sample of 99 flips, a human annotator agrees with each judge on 85% of items (kappa 0.53 / 0.61), marks 24% as consequential (between the judges' 15% and 29% on the same items), and finds no difference between the two methods. Items, key and labels are in [`results/human_eval/`](results/human_eval/).

Divergences and flip counts therefore screen decision fidelity; they **do not** replace benchmarks of task capability.

### What to report when evaluating a compressed model

`python -m glod report --dense <model> --compressed cfg:gptq4 --out report/` writes the three items below (TV, flips and KL per corpus, flips/TV, κ, J, and speculative-acceptance estimates) as JSON and Markdown, for any grid configuration or any checkpoint that shares the tokenizer, and flags a flips/TV outside the central 95% of the published configurations (0.89–1.29).

1. Report **total variation next to KL, per corpus** — TV converts into flips at a ratio near one; KL's conversion moves with the corpus.
2. Compare KL values **only within one corpus and one reference model**.
3. When the consumer needs exact agreement (speculative drafts, cached outputs, regression tests), report the **flip rate** itself, and screen drafts by TV on task prompts.
Flips and divergences describe fidelity to the dense model, not task accuracy: at matched divergence, accuracy can differ by tens of points.

### Can a compressor exploit the flip directions? (negative result)

At fixed divergence, an optimized perturbation halves the flip rate — which suggests a quantizer that keeps its error out of the flip directions. We tested it (`python -m glod amq`): re-fitting the group scales of 4-bit GPTQ on Qwen3-4B end to end (same codes, same bits, same kernel), on the model's own responses to 6000 prompts disjoint from every evaluation. **Plain KL self-distillation lowered flips by 11–35%** on the task corpora and on held-out code; three decision-aimed objectives (smooth flip count, margin displacement at fragile tokens, TV at fragile tokens) **at best matched it**, also with a rank-16 correction. With these parameterizations, lowering divergence is what lowers flips. The variants export to AWQ format and run in vLLM with Marlin kernels (`python -m glod amq-eval export`; `scripts/vllm_spec_bench.py` measures wall-clock speculative decoding).

### Speculative Decoding Predictor

Greedy speculative decoding accepts a draft token exactly when it matches the target's decision, so the relevant quantity is the draft's flip rate against the target. Over 43 target–draft pairs (23 compressed self-drafts, 20 smaller same-family drafts), **TV predicts that per-token flip rate with a mean error of 4.3% (self-drafts) and 3.6% (cross-model) with no calibration**, whereas κ·√KL must be calibrated on the task corpus and still errs by 9–12%. Converted to acceptance, 1 − TV gives **R² = 0.955 (MAPE 2.5%)** on self-drafts and **R² = 0.85** cross-model. If both models are already run under teacher forcing, the measured agreement sequence itself is the best predictor (R² ≈ 0.98–0.99). Under **sampling**, acceptance is exactly 1 − TV per position; TV measured under teacher forcing transfers to the positions sampled speculative decoding actually visits with R² = 0.95 / 0.89 and mean error 1.3% / 1.0% (self / cross-model).

<div align="center">
  <img src="results/graficos/speculative_chart.png" alt="Speculative decoding acceptance predicted from total variation" width="520"/>
</div>

---

## Impact Metrics

| 72B | 802 | 70% | ~25% |
| :---: | :---: | :---: | :---: |
| **Model Scale** | **Configurations** | **Jensen share of κ** | **Repair is largely re-sampling** |
| Up to Qwen2.5-72B | 19 models, 9 perturbation families, 5 corpora | of the variance of log κ comes from KL being averaged before the square root | of the dense model's wrong answers are fixed by any perturbation, noise included |

---

## Quick Start

We created two practical tools so the community can validate geometric predictions without running heavy GPU simulations:

### 1. Web Simulator (Landing Page)
The interactive simulator lives on the [`gh-pages` branch](https://github.com/beatrizalmeidaf/elastic_weight_streaming/tree/gh-pages) (`index.html`, `index-pt.html`): pick a model, corpus and compressor, and it compares the paper's two predictions, κ·√KL and total variation, with the closest real measurement.

### 2. Test the law against real measurements, without a GPU

The file [`data/measurements.csv`](data/measurements.csv) contains the **802 measured configurations** from the paper — model, corpus, family, configuration, KL, flips and TV. The script below uses only the Python standard library and **simulates nothing**: it compares the prediction against what was actually observed.

```bash
# predicted vs measured, configuration by configuration, on one reference
$ python3 scripts/test_formula.py --model gemma-3-4b-it --corpus gsm8k

measured kappa (Eq. 4, window 0.001 < KL < 0.05) : 0.1299
fitted exponent in log-log                       : 0.5179  (R2 0.9960)

config       family               KL  measured flips predicted      error
----------------------------------------------------------------------
u8           rtn             0.00067       0.00361    0.00336   -6.8%
kv4          kv              0.01043       0.01327    0.01327   +0.0%
u5           rtn             0.01835       0.01740    0.01760   +1.1%
mag20        magnitude       0.04413       0.02839    0.02729   -3.9%
...
median absolute error inside the kappa window     : 1.1%
```

```bash
$ python3 scripts/test_formula.py --families   # the central question, at matched KL
$ python3 scripts/test_formula.py --list       # the 54 available references
$ python3 scripts/test_formula.py --kappa 0.35 --kl 0.10   # just the prediction
```

The `--families` mode reproduces the in-sample family deviations behind the paper's family table (clustered by reference; the paper additionally refits each curve without the family and clusters by model) using solely the public CSV, and [`tests/test_measurements_csv.py`](tests/test_measurements_csv.py) locks this agreement in place.

---

## Repository Map

The complete thesis structure, historical results, and rebuttals are detailed in [docs/thesis_structure.md](docs/thesis_structure.md). The rigorous mathematical tests for idempotency of manipulations are in [DOC_TERMOS_E_TESTES.md](DOC_TERMOS_E_TESTES.md).

```text
data/
└── measurements.csv      # the 802 measurements from the paper (KL, flips, TV) — testable without GPU
glod/
├── paths.py              # paths and references (environment variables)
├── cli.py                # `glod <stage>` — central dispatcher
├── core/                 # main library: compressors, scoring, inference
├── corpora/              # datasets and oracles (MMLU, GSM8K, etc)
└── pipelines/
    ├── fidelity/         # compression grid and law measurement
    ├── adversarial/      # gradient attacks
    ├── speculative/      # theoretical predictors
    └── ...
```

---

## Installation & Pipeline

### Local Environment
```bash
make install          # pip install -e .
make install-dev      # + ruff
make test             # Validation (some require a local GPU)
```

**Environment Variables (`glod/paths.py`):**
| Variable | Default | Description |
|---|---|---|
| `GLOD_RESULTS` | `/local/$USER/ews_results/fid` | Output directory for calculations |
| `GLOD_HF_CACHE` | `/local/$USER/hf_cache` | Downloaded HuggingFace weights |
| `GLOD_DATA` | `data` | Smaller static artifacts |

### Running the Grid
A main end-to-end sweep (Qwen3-4B):
```bash
make corpus  MODEL=Qwen/Qwen3-4B DEVICE=cuda:0
make grid    MODEL=Qwen/Qwen3-4B DEVICE=cuda:0
make analyze
make adv     MODEL=Qwen/Qwen3-4B DEVICE=cuda:0
```

### Slurm Integration
Idempotent parallel distributed execution (tasks resume from where they left off):
```bash
make sweep-dry     # validates array 
make slurm         # queues the tasks
make slurm-status  # logs
```

### Docker
```bash
make docker-build
make docker-run TARGET="grid MODEL=Qwen/Qwen3-4B DEVICE=cuda:0"
```
