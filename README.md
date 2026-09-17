# EWS — o dano de compressão é uma taxa de câmbio fixa

Código do paper: **flips ≈ κ·√KL**. Qualquer perturbação estática de pesos (arredondamento, poda,
quantização de KV, remoção de camadas, checkpoints de terceiros, ataques por gradiente) converte
divergência KL em mudanças de decisão a uma taxa fixada pela geometria de margens do modelo denso.

A tese, o estado de cada resultado e o que foi descartado estão em
[docs/thesis_structure.md](docs/thesis_structure.md). Esse README é só o mapa do código.

## Estrutura

```
ews/
├── paths.py              caminhos e nomes de referência (tudo por variável de ambiente)
├── cli.py                `ews <estágio>` — um dispatcher, nenhuma lógica
├── core/                 biblioteca: compressores, quantização, carga de modelo, scoring, fidelidade
├── corpora/              prompts e corpora: token_oracle, mmlu, registry (o registro de datasets)
└── pipelines/
    ├── fidelity/         corpus + grade de compressores + análises (law, κ×geometria, domínio, teto)
    ├── adversarial/      ataque de 1 camada, ataque multicamada e o relatório da §4
    ├── tasks/            equivalência a KL casado, GSM8K, geração real
    ├── speculative/      decodificação especulativa: medição e os três previsores
    ├── adaptive/         adaptatividade por token (A1–A3, o resultado negativo)
    ├── report/           figuras do paper e relatório HTML
    └── legacy/           Fase 1 (elastic depth, descartada; mantida para reproduzir o histórico)
docker/                   Dockerfile, compose e entrypoint
scripts/                  download de datasets, cadeia por (modelo, corpus), fila por GPU
tests/                    scripts com asserts; alguns exigem GPU e checkpoint local
docs/                     tese, histórico de descobertas, TODO
```

Os resultados **não** ficam no repositório: cada estágio escreve em `$EWS_RESULTS` e é idempotente
(pula o que já existe), então reexecutar um alvo é barato.

## Instalação

```bash
make install          # pip install -e .  (torch já instalado no host da DGX)
make install-dev      # + ruff
make test             # roda tests/*.py; os que carregam modelo exigem GPU
```

## Caminhos

Nada de caminho absoluto no código: tudo passa por `ews/paths.py`.

| variável | default | o que é |
|---|---|---|
| `EWS_RESULTS` | `/local/$USER/ews_results/fid` | corpora, grades, `analysis/*.json`, ataques |
| `EWS_RAW` | `results/raw` | saídas da Fase 1 (pipelines `legacy-*`) |
| `EWS_HF_CACHE` | `/local/$USER/hf_cache` | checkpoints e datasets do Hugging Face |
| `EWS_DATA` | `data` | CSV do MMLU PT-BR, versionado à mão |
| `EWS_FIGS` | `results/figs` | figuras do paper |

Na DGX os dois primeiros ficam no disco local de propósito: o `/raid` tem cota de 500 G por usuário.

## Rodando

Cada estágio é um módulo com seu próprio `--help`:

```bash
python3 -m ews                       # lista os estágios
python3 -m ews grid --help
make help                            # os mesmos estágios como alvos, com as variáveis
```

A cadeia principal de um modelo:

```bash
make corpus  MODEL=Qwen/Qwen3-4B DEVICE=cuda:0     # corpus greedy da referência
make grid    MODEL=Qwen/Qwen3-4B DEVICE=cuda:0     # grade de compressores no corpus
make analyze                                       # law/theory/prop -> analysis/*.json
make adv     MODEL=Qwen/Qwen3-4B DEVICE=cuda:0     # ataque multicamada (maligno + benigno)
make adv-report figures
```

## Outros datasets

Um corpus é um id em `ews/corpora/registry.py`. Trocar de dataset é trocar `--corpus`:

| id | conteúdo | gabarito |
|---|---|---|
| `mix` | GSM8K + MMLU PT-BR (o dos resultados do paper) | sim |
| `gsm8k` | só GSM8K | sim |
| `mmlu_pt` | só MMLU PT-BR (CSV em `$EWS_DATA`) | sim |
| `mmlu_en` | MMLU em inglês (`cais/mmlu`, split de teste) | sim |
| `wikitext` | continuação de texto livre (`wikitext-2-raw-v1`) | não |

O corpus entra no nome da referência no disco: `mix` não tem sufixo (os resultados antigos seguem
valendo) e os outros ganham um, como `Qwen3-4B__mmlu_en`. Assim as análises, que varrem
`$EWS_RESULTS`, tratam cada corpus como uma referência separada, e nada se mistura.

```bash
make datasets                                    # baixa gsm8k, mmlu_en, wikitext
make all-corpus CORPUS=mmlu_en MODEL=Qwen/Qwen3-4B DEVICE=cuda:0   # cadeia minima
make sweep-one  CORPUS=mmlu_en MODEL=Qwen/Qwen3-4B DEVICE=cuda:0   # tudo, um modelo
```

### Varredura completa

```bash
make sweep-dry                        # o plano: pares, filas e comandos (não roda nada)
make sweep DEVICES="cuda:0 cuda:2"    # local, fora do Slurm
```

Isso cobre `configs/models.txt` × (`mmlu_en`, `wikitext`, `gsm8k`) no perfil paper: grade em bf16 e fp32, fungibilidade, ataque multicamada, o mesmo ataque em fp32 (P5), sem a escala de saída (P7) e com as sementes 1 e 2 (P8), mais uma passada final das análises globais e das figuras.

### No Slurm

```bash
make slurm-dry     # mostra os pares e os sbatch exatos
make slurm         # submete o array + o job dependente
make slurm-status  # fila e quantas etapas já concluíram
```

Um detalhe que mudou o desenho: estamos logados direto no dgx-H100-03, fora do Slurm — é por isso que disputamos GPU com os jobs de outras pessoas a noite toda. E a QOS desta conta (`onejob`) permite 2 jobs rodando por usuário. Então:

- `slurm/sweep.sbatch` é um job array com um par (modelo, corpus) por tarefa, `--array=0-N%2` e `--gres=gpu:1`. Cada tarefa lê a sua linha do arquivo de pares em `var/slurm/` e roda a cadeia com `--no-global`. Com uma GPU alocada, o dispositivo é sempre `cuda:0`.
- `slurm/global.sbatch` entra com `--dependency=afterany:<array>` e roda as análises globais uma vez. Usei afterany e não afterok de propósito: se um par falhar, as análises ainda consolidam o que terminou — é assim que se descobre o que faltou.
- Partição `h100n2,h100n3`. Não usei a `b200n1`: está em drng e com uma fila grande de outras pessoas.
- Logs em `var/logs/slurm/ews-sweep_<jobid>_<tarefa>.out`.

O array é a granularidade que dá robustez: uma tarefa que morre não afeta as outras 23.

### Retomada sem recomeçar

Três camadas, e testei a primeira sem gastar GPU (substituí o executor por true: 5 etapas gravaram marco, a segunda passada pulou todas as 5, e `--force` refez):

1. **Marcos por etapa.** Cada etapa concluída grava `$EWS_RESULTS/_stamps/<hash>`, e na reexecução é pulada na hora, sem carregar modelo. O hash ignora o `--device` — senão a mesma etapa rodada em cuda:2 local e em cuda:0 no Slurm contaria como duas e repetiria o trabalho.
2. **Idempotência interna**, que já existia. O ataque grava cada ponto em `results_<tag>.json` assim que ele termina e pula os já presentes; a grade grava um `.pt` por configuração; o corpus não é regerado. Uma tarefa morta no meio perde no máximo a etapa em andamento, que é um ponto de ataque: 4 min no 4B, 11 min no 12B.
3. **`--requeue` nas tarefas.** Timeout, preempção ou nó reiniciado devolvem a tarefa à fila, e ela retoma pelos marcos.

Resubmeter o mesmo comando é a forma de fechar o que faltou:

```bash
make slurm CORPORA="mmlu_en wikitext gsm8k"    # o que terminou é pulado
grep -l FALHOU var/logs/*.log                  # pares com etapa em erro
ls $EWS_RESULTS/_stamps | wc -l                # progresso
```

Sobre o custo, que continua sendo a decisão: 24 pares no perfil paper, com 2 rodando por vez, dão algo como 100 h de GPU, ou seja uns 2 dias de fila. Minha sugestão é submeter primeiro um recorte que já responde a pergunta em aberto (κ e a razão do ataque mudam com o corpus?):

```bash
make slurm CORPORA="mmlu_en" MODELS="Qwen/Qwen3-4B mistralai/Mistral-7B-Instruct-v0.3 google/gemma-3-12b-it"
```

Aceitam `--corpus`: `grid`, `adv-multi`, `adv-single`, `fungibility`. Os estágios de análise
(`analyze`, `slope`, `domain`, `adv-report`, `figures`) descobrem as referências pelo disco e não
precisam do flag.

## Docker

```bash
make docker-build
make docker-run TARGET="grid MODEL=Qwen/Qwen3-4B DEVICE=cuda:0"
docker compose -f docker/docker-compose.yml run --rm ews make analyze
```

Os pesos e os resultados entram por volume (`/hf_cache` e `/results`), nunca em camada da imagem.
Modelos gated (Gemma) precisam de `HF_TOKEN` no ambiente.
