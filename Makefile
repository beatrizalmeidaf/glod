# EWS - pipeline do paper. Cada alvo e um estagio; nada aqui esconde um experimento
# longo atras de um nome curto: os alvos que usam GPU dizem quanto custam no README.
#
#   make help                      lista os alvos
#   make corpus grid MODEL=Qwen/Qwen3-4B DEVICE=cuda:0
#   make all-corpus CORPUS=mmlu_en MODEL=Qwen/Qwen3-4B
#
# Variaveis (sobrescreva na linha de comando):
MODEL   ?= Qwen/Qwen3-4B
DEVICE  ?= cuda:0
CORPUS  ?= mix
CONFIGS ?= bf16 g4 gptq4 awq4 sgpt50 wanda50 kv4 kv3
KL      ?= 0.02 0.05
STEPS   ?= 400
RANK    ?= 16
SEED    ?= 0
N_PROMPTS ?= 256
PY      ?= python3
EWS     ?= $(PY) -m ews

# caminhos (tambem lidos pelo codigo via ews/paths.py)
export EWS_RESULTS  ?= /local/$(USER)/ews_results/fid
export EWS_RAW      ?= results/raw
export EWS_HF_CACHE ?= /local/$(USER)/hf_cache
export EWS_DATA     ?= data
export EWS_FIGS     ?= results/figs

.DEFAULT_GOAL := help
.PHONY: help install install-dev test lint datasets corpus grid analyze law slope domain flip-dirs \
        fungibility adv adv-fp32 adv-report matched-kl crack closedloop spec-bench spec-law \
        adaptive-bits adaptive-closedloop adaptive-report figures fidelity-all adversarial-all \
        all all-corpus sweep sweep-dry sweep-one slurm slurm-dry slurm-status docker-build docker-shell docker-run clean-pyc

help:
	@printf "alvos:\n"
	@grep -E '^[a-z][a-z0-9-]*:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 24
	@printf "\nvariaveis: MODEL=$(MODEL) DEVICE=$(DEVICE) CORPUS=$(CORPUS) SEED=$(SEED)\n"
	@printf "resultados: EWS_RESULTS=$(EWS_RESULTS)\n"

# ---------------------------------------------------------------- ambiente
install:            ## instala o pacote (editavel) e as dependencias
	$(PY) -m pip install -e .

install-dev: install  ## idem, mais ruff (lint)
	$(PY) -m pip install ruff

test:               ## testes (scripts com asserts; os que carregam modelo exigem GPU)
	@fail=0; for t in tests/*.py; do \
	  printf '%-46s' "$$t"; \
	  if $(PY) "$$t" >/dev/null 2>var/logs/$$(basename $$t).log; then echo PASS; \
	  else echo FAIL; tail -3 var/logs/$$(basename $$t).log; fail=1; fi; \
	done; exit $$fail

lint:               ## ruff no pacote
	$(PY) -m ruff check ews tests scripts

datasets:           ## baixa GSM8K, MMLU (en), wikitext-2 para EWS_DATA/datasets
	$(PY) scripts/download_datasets.py

# ------------------------------------------------- fidelidade (Secoes 2 e 3)
corpus:             ## corpus greedy da referencia (MODEL, CORPUS)
	$(EWS) grid gen --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS) --n-prompts $(N_PROMPTS)

grid:               ## pontua a grade de compressores no corpus (CONFIGS)
	$(EWS) grid score --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS) --configs bf16 $(CONFIGS)

grid-fp32:          ## idem com lm_head em fp32 (sem empates do bf16)
	$(EWS) grid score --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS) --logits-fp32 \
	    --configs bf16 $(CONFIGS)

analyze:            ## law/theory/prop/d3/d4 -> analysis/*.json
	$(EWS) analyze law
	$(EWS) analyze theory
	$(EWS) analyze prop

law: analyze        ## alias de analyze

slope:              ## kappa x geometria de margens
	$(EWS) slope

domain:             ## kappa por dominio
	$(EWS) domain

flip-dirs:          ## rank efetivo das direcoes de flip e teto do oraculo
	$(EWS) flip-dirs --model $(MODEL)

fungibility:        ## kappa por subconjunto de modulos, a KL casado
	$(EWS) fungibility --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS)

fidelity-all: corpus grid analyze slope domain flip-dirs  ## a cadeia de fidelidade inteira

# ------------------------------------------------------ adversarial (Secao 4)
adv:                ## ataque multicamada, maligno e benigno (KL, STEPS, RANK, SEED)
	$(EWS) adv-multi --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS) --seed $(SEED) \
	    --steps $(STEPS) --rank $(RANK) --kl-budget $(KL) --modes malign benign \
	    --tag $(if $(filter 0,$(SEED)),multi,seed$(SEED))

adv-fp32:           ## idem com logits em fp32 (P5)
	$(EWS) adv-multi --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS) --logits-fp32 \
	    --steps $(STEPS) --rank $(RANK) --kl-budget $(KL) --modes malign benign --tag fp32

adv-no-norm:        ## ablacao sem a escala de saida (P7)
	$(EWS) adv-multi --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS) --no-output-scale \
	    --steps $(STEPS) --rank $(RANK) --kl-budget 0.05 --modes malign benign --tag semnorma

adv-single:         ## ataque de 1 camada, rank completo
	$(EWS) adv-single --model $(MODEL) --device $(DEVICE) --corpus $(CORPUS)

adv-report:         ## consolida a Secao 4 -> analysis/adversarial.json
	$(EWS) adv-report

adversarial-all: adv adv-fp32 adv-no-norm adv-report  ## ataque + ablacoes + relatorio

# ------------------------------------------------- tarefa e especulativa (5 e 6)
matched-kl:         ## equivalencia a KL casado + TOST
	$(EWS) matched-kl --model $(MODEL) --device $(DEVICE)

crack:              ## rachadura do GSM8K (H1/H2/H3)
	$(EWS) crack --model $(MODEL) --device $(DEVICE)

closedloop:         ## geracao real e especulativa real
	$(EWS) closedloop gsm8k --model $(MODEL) --device $(DEVICE)

spec-bench:         ## especulativa com relogio
	$(EWS) spec-bench --target $(MODEL) --device $(DEVICE)

spec-law:           ## os tres previsores -> analysis/spec_law.json
	$(EWS) spec-law

# ------------------------------------------------------- adaptativo (Secao 7)
adaptive-bits:      ## A1-A2 (teacher forcing)
	$(EWS) adaptive-bits

adaptive-closedloop:  ## A3 (geracao real, KV misto)
	$(EWS) adaptive-closedloop --model $(MODEL) --device $(DEVICE)

adaptive-report:    ## A3 contra a fronteira estatica
	$(EWS) adaptive-report

# --------------------------------------------------------------------- saidas
figures:            ## as 6 figuras do paper em EWS_FIGS
	$(EWS) figures

all: fidelity-all adversarial-all spec-law adaptive-report figures  ## tudo o que entra no paper

all-corpus:         ## cadeia minima num corpus novo: make all-corpus CORPUS=mmlu_en
	$(MAKE) corpus grid analyze CORPUS=$(CORPUS) MODEL=$(MODEL) DEVICE=$(DEVICE)
	$(MAKE) adv adv-report CORPUS=$(CORPUS) MODEL=$(MODEL) DEVICE=$(DEVICE)

# ------------------------------------------------------------------ varredura
# Tudo o que foi medido no corpus do paper (grade + fp32 + ataque + P5 + P7 + P8),
# repetido em cada corpus e em cada modelo. Veja o custo com sweep-dry antes.
MODELS  ?= configs/models.txt
CORPORA ?= mmlu_en wikitext gsm8k
DEVICES ?= cuda:0
PROFILE ?= paper

sweep-dry:          ## mostra o plano da varredura (nao roda nada)
	scripts/run_sweep.sh --models "$(MODELS)" --corpora "$(CORPORA)" \
	    --devices "$(DEVICES)" --profile $(PROFILE) --dry-run

sweep:              ## roda a varredura: make sweep CORPORA="mmlu_en wikitext" DEVICES="cuda:0 cuda:2"
	scripts/run_sweep.sh --models "$(MODELS)" --corpora "$(CORPORA)" \
	    --devices "$(DEVICES)" --profile $(PROFILE)

sweep-one:          ## a varredura completa de um modelo so, num corpus: make sweep-one MODEL=... CORPUS=...
	scripts/run_corpus.sh --model $(MODEL) --corpus $(CORPUS) --device $(DEVICE) --profile $(PROFILE)

# ---------------------------------------------------------------------- slurm
# QOS deste cluster (onejob): 2 jobs rodando por usuario, entao o array vai com %2.
CONCURRENT ?= 2
PARTITION  ?= h100n2,h100n3

slurm-dry:          ## mostra os pares e os sbatch que seriam submetidos
	scripts/submit_slurm.sh --models "$(MODELS)" --corpora "$(CORPORA)" \
	    --profile $(PROFILE) --concurrent $(CONCURRENT) --partition $(PARTITION) --dry-run

slurm:              ## submete a varredura: job array (um par por tarefa) + analises globais dependentes
	scripts/submit_slurm.sh --models "$(MODELS)" --corpora "$(CORPORA)" \
	    --profile $(PROFILE) --concurrent $(CONCURRENT) --partition $(PARTITION)

slurm-status:       ## fila, tarefas do array e marcos concluidos
	@squeue -u $(USER) -o "%.10i %.10P %.14j %.2t %.11M %.6D %R" || true
	@echo "marcos concluidos: $$(ls $(EWS_RESULTS)/_stamps 2>/dev/null | wc -l)"

# --------------------------------------------------------------------- docker
docker-build:       ## imagem com CUDA + dependencias
	docker build -f docker/Dockerfile -t ews:latest .

docker-shell:       ## shell no container, com GPUs e volumes
	docker run --rm -it --gpus all $(DOCKER_MOUNTS) ews:latest bash

docker-run:         ## roda um alvo do Makefile dentro do container: make docker-run TARGET="grid MODEL=..."
	docker run --rm -it --gpus all $(DOCKER_MOUNTS) ews:latest make $(TARGET)

DOCKER_MOUNTS = -v $(CURDIR):/workspace \
                -v $(EWS_RESULTS):/results \
                -v $(EWS_HF_CACHE):/hf_cache \
                -e EWS_RESULTS=/results -e EWS_HF_CACHE=/hf_cache -e HF_TOKEN

clean-pyc:          ## remove bytecode
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
