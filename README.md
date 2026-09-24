<div align="center">
  
<img src="results/graficos/logo.png" alt="GLOD Logo" width="200" />

# Geometric Law of Damage (GLOD)

**A Lei Geométrica do Dano de Compressão em LLMs.**

[![license](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![paper](https://img.shields.io/badge/arxiv-Paper-red.svg)]()
[![simulator](https://img.shields.io/badge/try-Simulator-blue.svg)](https://beatrizalmeidaf.github.io/glod/index-pt.html)

</div>

---

## What is Geometric Law of Damage?

Hoje temos dezenas de técnicas para comprimir pesos de LLMs e acelerar a inferência (Poda, Quantização, Layer Skipping, etc). Quando escolhemos um desses métodos, a grande dúvida é: **eles causam tipos diferentes de "dano" ao raciocínio do modelo, ou apenas diferem na quantidade de dano?**

**GLOD responde a essa pergunta matematicamente.** Foi medido exaustivamente o efeito de 10+ compressores através de múltiplos modelos (de 1B até 72B parâmetros) e múltiplos domínios. A descoberta central é que:

> Para uma dada referência (par modelo + distribuição de avaliação), perturbações estáticas de pesos mecanicamente distintas convertem divergência KL em mudanças de decisão (*flips*) **a uma taxa que difere entre si em no máximo 5%**, fixada pela **geometria das margens do modelo**.

<div align="center">
  <h3><strong>flips ≈ κ · √KL</strong></h3>
</div>

**De onde vem o expoente.** A forma de raiz quadrada não é um achado empírico: a divergência é de **segunda** ordem na perturbação de pesos (expansão de Taylor com a métrica de Informação de Fisher):

$$ \mathrm{KL} = \frac{1}{2} \delta^\top F \delta + O(\lVert\delta\rVert^3) $$

enquanto o deslocamento da margem de decisão é de **primeira** ordem. A proporcionalidade com $\sqrt{\mathrm{KL}}$ segue daí. Verificamos que o regime quadrático vale até ~11% de desvio dentro da janela onde κ é definido, e quebra acima de ~2 nats.

**O que é empírico** é que um **único** κ descreve arredondamento, poda, quantização de KV-cache, remoção de camadas e ruído isotrópico do mesmo modelo — o fator de proporcionalidade varia apenas 2,7% (CV mediano) entre famílias dentro de uma referência.

- **O algoritmo importa pouco, mas não é irrelevante:** condicionado ao KL, AWQ de 4 bits e poda Wanda ficam a menos de 1,2% um do outro em *flips*. O desvio máximo entre todas as nove famílias testadas é de 5%, e quatro delas têm desvio estatisticamente detectável.

### Quanto a família do método ainda importa, a KL casado

Para cada referência ajustamos a lei de potência da própria referência e medimos o quanto cada família se desvia dela. O desvio é **multiplicativo sobre a taxa de flips**; os intervalos são agrupados por referência (a unidade independente), sobre as 742 configurações.

| Compressor | Família | Flips vs. a lei da referência | IC 95% | refs |
|:---|:---|:---:|:---:|:---:|
| **Remoção de camadas** | Estrutural | **0.950** ▼ | [0.927, 0.973] | 14 |
| **KV-cache (quant.)** | Cache | **0.987** ▼ | [0.979, 0.995] | 46 |
| Magnitude | Poda | 0.990 | [0.976, 1.004] | 46 |
| Ruído gaussiano | Controle | 0.994 | [0.985, 1.003] | 47 |
| **GPTQ** | Quantização | 0.997 | [0.988, 1.005] | 47 |
| **AWQ** | Quantização | 0.997 | [0.988, 1.006] | 47 |
| **RTN** | Arredondamento | **1.009** ▲ | [1.004, 1.013] | 47 |
| Wanda | Poda | 1.009 | [0.993, 1.024] | 46 |
| **SparseGPT** | Poda | **1.028** ▲ | [1.012, 1.043] | 45 |

*Fungibilidade estrita é falsa: quatro das nove famílias têm intervalo excluindo 1, e a ordem é interpretável — remover camadas inteiras produz 5,0% **menos** flips por unidade de divergência, poda calibrada 2,8% **mais**. O enunciado correto é fungibilidade a menos de 5%, não igualdade.*

<div align="center">
  <img src="results/graficos/methods_chart.png" alt="Comparação de Métodos vs Teoria" width="700"/>
</div>

- **Teto da Perturbação Estática:** um otimizador com acesso total ao gradiente e orçamento de KL fixo atinge no máximo **1.17×** a taxa de um compressor padrão (e **nada** em dois dos quatro corpora), enquanto o objetivo inverso corta a taxa pela metade. A assimetria — fácil perder dano decisório a divergência fixa, difícil ganhá-lo — é o resultado robusto. Apresentamos os tetos como estimativas de primeira ordem sob as hipóteses declaradas, não como limites provados.
- **O Futuro:** Como compressores estáticos usam apenas ~17% do orçamento informacional do oráculo perfeito, GLOD demonstra que a otimização de próxima geração exigirá **compressores argmax-aware** (alocação de bits per-token).

### Validação Empírica por Domínio

Duas grandezas distintas, que a versão anterior desta tabela confundia: o **expoente** (a inclinação em log-log, que a teoria prevê ser ≈ ½) e **κ** (a taxa de câmbio, que depende do par modelo+corpus).

| Corpus | Escopo | Expoente conjunto | R² conjunto | κ (faixa entre modelos) |
|:---|:---|:---:|:---:|:---:|
| **GSM8K** | Respostas lógicas exatas | 0.513 | 0.954 | 0.13 – 0.24 |
| **Mix (GSM8K+MMLU-PT)** | Misto | 0.512 | 0.960 | 0.19 – 0.35 |
| **MMLU-en** | Conhecimento geral | 0.483 | 0.937 | 0.21 – 0.45 |
| **WikiText** | Geração de texto livre | 0.462 | 0.931 | 0.27 – 0.56 |
| **WikiText (natural)** | Texto real, sem geração | 0.468 | 0.978 | 0.39 – 0.56 |

*O R² conjunto é menor que o de qualquer referência isolada (todos ≥ 0.990) porque as referências diferem no intercepto. κ varia 1,6–2,6× entre corpora **dentro de um mesmo modelo**: reportar "o κ do modelo X" sem nomear a distribuição não significa nada.*

### Teto Intransponível

Testamos a resiliência da lei tentando forçar a rede a errar (Ataques Adversariais focados em maximizar flips, condicionados a um teto de KL). O resultado mostra que mesmo um atacante onipotente mal consegue arrancar 17% a mais de erros do que um compressor honesto simples, comprovando que a taxa de câmbio é, de fato, uma barreira geométrica fundamental.

<div align="center">
  <img src="results/graficos/attacks_chart.png" alt="Ataques Adversariais vs Baseline Honesto" width="700"/>
</div>

---

## Impact Metrics

| 72B+ | 10+ | 5.7x | 0.99 |
| :---: | :---: | :---: | :---: |
| **Model Scale** | **Compressors Tested** | **Oracle Gap** | **R² Accuracy** |
| Validade confirmada no Qwen2.5-72B | Quantização, Poda e Ataques | Compressores usam 17% do orçamento do oráculo | R² por referência (o conjunto é 0.93–0.98) |

---

## Quick Start

Criamos duas ferramentas práticas para que a comunidade possa validar as previsões geométricas sem precisar rodar simulações pesadas em GPU:

### 1. Simulador Web (Landing Page)
Abra o [Simulador GLOD Web](https://beatrizalmeidaf.github.io/glod/index-pt.html) no seu navegador para acessar uma visualização gráfica interativa que compara técnicas tradicionais com o limite geométrico.

### 2. Teste a lei contra as medições reais, sem GPU

O arquivo [`data/measurements.csv`](data/measurements.csv) (39 KB) traz as **742 configurações
medidas** do artigo — modelo, corpus, família, configuração, KL e flips. O script abaixo usa
apenas a biblioteca padrão do Python e **não simula nada**: confronta a previsão com o que foi
de fato observado.

```bash
# previsto x medido, configuração por configuração, numa referência
$ python3 scripts/test_formula.py --model gemma-3-4b-it --corpus gsm8k

kappa medido (Eq. 3, janela 0.001 < KL < 0.05) : 0.1299
expoente ajustado em log-log                   : 0.5179  (R2 0.9960)

config       familia              KL  flips medido   previsto     erro
----------------------------------------------------------------------
u8           rtn             0.00067       0.00361    0.00336   -6.8%
kv4          kv              0.01043       0.01327    0.01327   +0.0%
u5           rtn             0.01835       0.01740    0.01760   +1.1%
mag20        magnitude       0.04413       0.02839    0.02729   -3.9%
...
erro absoluto mediano dentro da janela de kappa   : 1.1%
```

```bash
$ python3 scripts/test_formula.py --families   # a pergunta central, a KL casado
$ python3 scripts/test_formula.py --list       # as 49 referências disponíveis
$ python3 scripts/test_formula.py --kappa 0.35 --kl 0.10   # só a previsão
```

O modo `--families` reproduz a Tabela 5 do artigo a partir do CSV público sozinho, e
[`tests/test_measurements_csv.py`](tests/test_measurements_csv.py) trava essa concordância.

---

## Repository Map

A tese completa, o histórico de resultados e as refutações estão detalhados em [docs/thesis_structure.md](docs/thesis_structure.md). O rigor dos testes matemáticos de idempotência das manipulações está em [DOC_TERMOS_E_TESTES.md](DOC_TERMOS_E_TESTES.md).

```text
data/
└── measurements.csv      # as 742 medições do artigo (KL, flips) — testáveis sem GPU
ews/
├── paths.py              # caminhos e referências (variáveis de ambiente)
├── cli.py                # `ews <estágio>` — dispatcher central
├── core/                 # biblioteca principal: compressores, pontuação, inferência
├── corpora/              # datasets e oráculos (MMLU, GSM8K, etc)
└── pipelines/
    ├── fidelity/         # grade de compressão e medição da lei
    ├── adversarial/      # ataques por gradiente
    ├── speculative/      # previsores teóricos
    └── ...
```

---

## Installation & Pipeline

### Local Environment
```bash
make install          # pip install -e .
make install-dev      # + ruff
make test             # Validação (alguns necessitam de GPU local)
```

**Variáveis de Ambiente (`ews/paths.py`):**
| Variável | Default | Descrição |
|---|---|---|
| `EWS_RESULTS` | `/local/$USER/ews_results/fid` | Diretório destino dos cálculos |
| `EWS_HF_CACHE` | `/local/$USER/hf_cache` | Pesos baixados do HuggingFace |
| `EWS_DATA` | `data` | Artefatos estáticos menores |

### Running the Grid
Uma varredura principal ponta a ponta (Qwen3-4B):
```bash
make corpus  MODEL=Qwen/Qwen3-4B DEVICE=cuda:0
make grid    MODEL=Qwen/Qwen3-4B DEVICE=cuda:0
make analyze
make adv     MODEL=Qwen/Qwen3-4B DEVICE=cuda:0
```

### Slurm Integration
Execução distribuída paralela idempotente (as tasks retomam de onde pararam):
```bash
make sweep-dry     # valida array 
make slurm         # enfileira as tarefas
make slurm-status  # logs
```

### Docker
```bash
make docker-build
make docker-run TARGET="grid MODEL=Qwen/Qwen3-4B DEVICE=cuda:0"
```
