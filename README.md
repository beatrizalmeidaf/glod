<div align="center">
  
<img src="results/graficos/logo.png" alt="GLOD Logo" width="200" />

# GLOD (Geometric Law of Damage)

> **Note on "Damage"**: In the context of GLOD, "Damage" refers strictly to **distributional deviation** from the dense oracle (teacher-forcing decision flips), not a loss in semantic capability or downstream task accuracy.

**GLOD** is the analytical framework and evaluation suite introduced in the paper: *How Divergence Becomes Decision Flips in Compressed Language Models*.

While the paper describes the theoretical discovery, the **GLOD** package provides the empirical infrastructure to measure how the margin geometry of an LLM converts statistical perturbations (such as compression) into decision changes ("damage").

[![license](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![paper](https://img.shields.io/badge/arxiv-Paper-red.svg)]()
[![simulator](https://img.shields.io/badge/try-Simulator-blue.svg)](https://beatrizalmeidaf.github.io/glod/index-en.html)

</div>

---

## What is the Geometric Law of Damage?

Today we have dozens of techniques to compress LLM weights and accelerate inference (Pruning, Quantization, Layer Skipping, etc). When we choose one of these methods, the big question is: **do they cause different types of "damage" to the model's reasoning, or do they merely differ in the amount of damage?**

**GLOD answers this question mathematically.** We exhaustively measured the effect of 10+ compressors across **19 models** from six families (Gemma-3, Qwen3/2.5, Mistral, Phi, OLMo-2, and Llama-3), ranging from 1B to 72B parameters, on five corpora — **802 configurations** in total. The central discovery is that:

> For a given reference (model + evaluation distribution pair), **Total Variation (TV)** — a first-order divergence — tracks the decision flip rate nearly one-for-one (**flips ≈ TV**) with no free constant. Meanwhile, KL Divergence works only because it approximates TV via the **model's margin geometry**.

<div align="center">
  <h3><strong>flips ≈ TV ≈ κ · √KL</strong></h3>
</div>

**Why KL needs a coefficient.** The square root form is not an empirical finding: divergence is **second** order in the weight perturbation (Taylor expansion with the Fisher Information metric):

$$ \mathrm{KL} = \frac{1}{2} \delta^\top F \delta + O(\lVert\delta\rVert^3) $$

while the decision margin displacement is **first** order. The proportionality with $\sqrt{\mathrm{KL}}$ follows directly. We verified that the quadratic regime holds up to ~11% deviation within the window where κ is defined, breaking down above ~2 nats.

**What is empirical** is that a **single** κ describes rounding, pruning, KV-cache quantization, layer removal, and isotropic noise on the same model — the proportionality factor varies by only 2.7% (median CV) across families within a reference.

- **The algorithm matters little, but it is not irrelevant:** conditional on KL, 4-bit AWQ and Wanda pruning are within 1.2% of each other in *flips*. The maximum deviation among all nine tested families is 5%, and four of them have a statistically detectable deviation.

### How much the method family still matters, at matched KL

For each reference, we fit the reference's own power law and measure how much each family deviates from it. The deviation is **multiplicative over the flip rate**; intervals are clustered by reference (the independent unit), over all 802 configurations.

| Compressor | Family | Flips vs. reference law | 95% CI | refs |
|:---|:---|:---:|:---:|:---:|
| **Layer Removal** | Structural | **0.950** ▼ | [0.927, 0.973] | 14 |
| **KV-cache (quant.)** | Cache | **0.989** ▼ | [0.982, 0.997] | 51 |
| Gaussian Noise | Control | 0.993 | [0.985, 1.001] | 52 |
| Magnitude | Pruning | 0.995 | [0.982, 1.009] | 51 |
| **GPTQ** | Quantization | 0.996 | [0.988, 1.003] | 52 |
| **AWQ** | Quantization | 0.997 | [0.989, 1.005] | 52 |
| Wanda | Pruning | 1.005 | [0.991, 1.020] | 51 |
| **RTN** | Rounding | **1.008** ▲ | [1.004, 1.012] | 52 |
| **SparseGPT** | Pruning | **1.023** ▲ | [1.009, 1.038] | 50 |

*In the common KL window where all families coexist ($0.03 \leq \KL \leq 0.20$), eight of the nine methods have a maximum deviation of just 1.6%. After Holm correction, only Layer Removal significantly alters the geometry (5.5% fewer flips). Strict fungibility holds closely in the common regime.*

<div align="center">
  <img src="results/graficos/methods_chart.png" alt="Methods Comparison vs Theory" width="700"/>
</div>

- **The Static Perturbation Ceiling:** an optimizer with full gradient access and a fixed KL budget achieves at most **1.17×** the rate of a standard compressor (and **nothing** on two of the four corpora), while the inverse objective cuts the rate in half. The asymmetry — easy to lose decision damage at fixed divergence, hard to gain it — is the robust result. We present the ceilings as first-order estimates under the stated assumptions, not as proven bounds.
- **The Future:** Because static compressors use only ~17% of the perfect oracle's informational budget, GLOD demonstrates that next-generation optimization will require **argmax-aware compressors** (per-token bit allocation).

### Empirical Validation by Domain

Two distinct quantities, which earlier versions of this table conflated: the **exponent** (the slope in log-log, which theory predicts to be ≈ ½) and **κ** (the exchange rate, which depends on the model+corpus pair).

| Corpus | Scope | Pooled Exponent | Pooled R² | κ (range across models) |
|:---|:---|:---:|:---:|:---:|
| **GSM8K** | Exact logical answers | 0.511 | 0.954 | 0.13 – 0.24 |
| **Mix (GSM8K+MMLU-PT)** | Mixed | 0.511 | 0.961 | 0.19 – 0.35 |
| **MMLU-en** | General knowledge | 0.482 | 0.941 | 0.21 – 0.45 |
| **WikiText** | Free text generation | 0.464 | 0.937 | 0.27 – 0.56 |
| **WikiText (natural)** | Real text, no generation | 0.471 | 0.980 | 0.39 – 0.56 |

*The pooled R² is lower than that of any isolated reference (all ≥ 0.990) because the references differ in intercept. κ varies 1.6–2.6× between corpora **within the same model**: reporting "the κ of model X" without naming the distribution means nothing.*

**Why does κ vary? (The Jensen Factor):** The paper shows that 70% of this variance across corpora is caused by the **Jensen inequality**. Because KL divergence averages the per-token divergence *before* the square root is applied, it structurally penalizes high-entropy distributions (like MMLU or natural text) compared to low-entropy ones (like GSM8K). Total Variation (TV) is first-order at every token and avoids this Jensen flattening entirely.

### What Flip Counts Do Not Measure (Matched Divergence, Unmatched Accuracy)

GLOD measures **decision stability (fidelity)**, not downstream competence (accuracy). The paper conclusively demonstrates that:
1. **Fidelity $\neq$ Accuracy:** At the exact same KL divergence, one method (Wanda) can gain +4.8 points in GSM8K, while another (Magnitude Pruning) loses -33.3 points. Random draws of pure Gaussian noise span 18 points of accuracy.
2. **Repair is Re-sampling:** It is commonly claimed that some compression methods "repair" the dense model's wrong answers. The paper proves this is merely a **trajectory re-sampling effect** shared equally by random Gaussian noise. All perturbations correct ~25% of errors simply by shaking the model out of local minima.

Therefore, while TV replaces the need for statistical fidelity evaluation, it **does not** replace empirical zero-shot benchmarking for task capability.

### Insurmountable Ceiling

We tested the resilience of the law by trying to force the network to make mistakes (Adversarial Attacks focused on maximizing flips, conditioned on a KL ceiling). The result shows that even an omnipotent attacker barely manages to extract 17% more errors than a simple honest compressor, proving that the exchange rate is, in fact, a fundamental geometric barrier.

<div align="center">
  <img src="results/graficos/attacks_chart.png" alt="Adversarial Attacks vs Honest Baseline" width="700"/>
</div>

### Speculative Decoding Predictor

Speculative decoding accepts a draft token exactly when it matches the target's decision. Because Total Variation (and its geometric KL approximation) tracks decision flips, **GLOD can predict speculative decoding acceptance without instantiating the speculative system.**

On compressed self-drafts, the theory predicts the actual measured speculative acceptance with **$R^2 = 0.95$ (MAPE 2.5%)**. Furthermore, the paper demonstrates that while the KL-based approximation breaks down for cross-model drafts, **Total Variation (TV)** continues to predict acceptance reliably without any fitted constants.

<div align="center">
  <img src="results/graficos/speculative_chart.png" alt="Speculative Decoding Prediction" width="700"/>
</div>

---

## Impact Metrics

| 72B+ | 10+ | 5.7x | ~25% |
| :---: | :---: | :---: | :---: |
| **Model Scale** | **Compressors Tested** | **Oracle Gap** | **Repair is Re-sampling** |
| Validity confirmed on Qwen2.5-72B | Quantization, Pruning, and Attacks | Compressors use 17% of the oracle budget | Apparent repair of wrong answers is a trajectory re-sampling effect shared by Gaussian noise |

---

## Quick Start

We created two practical tools so the community can validate geometric predictions without running heavy GPU simulations:

### 1. Web Simulator (Landing Page)
Open the [GLOD Web Simulator](https://beatrizalmeidaf.github.io/glod/index-en.html) in your browser to access an interactive graphical visualization that compares traditional techniques with the geometric limit.

### 2. Test the law against real measurements, without a GPU

The file [`data/measurements.csv`](data/measurements.csv) contains the **802 measured configurations** from the paper — model, corpus, family, configuration, KL, and flips. The script below uses only the Python standard library and **simulates nothing**: it compares the prediction against what was actually observed.

```bash
# predicted vs measured, configuration by configuration, on one reference
$ python3 scripts/test_formula.py --model gemma-3-4b-it --corpus gsm8k

measured kappa (Eq. 3, window 0.001 < KL < 0.05) : 0.1299
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

The `--families` mode reproduces Table 5 of the paper using solely the public CSV, and [`tests/test_measurements_csv.py`](tests/test_measurements_csv.py) locks this agreement in place.

---

## Repository Map

The complete thesis structure, historical results, and rebuttals are detailed in [docs/thesis_structure.md](docs/thesis_structure.md). The rigorous mathematical tests for idempotency of manipulations are in [DOC_TERMOS_E_TESTES.md](DOC_TERMOS_E_TESTES.md).

```text
data/
└── measurements.csv      # the 802 measurements from the paper (KL, flips) — testable without GPU
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
