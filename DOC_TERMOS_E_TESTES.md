# Guia Completo: Termos e Testes do Elastic Weight Streaming (EWS)

Projeto focado em otimizar e simular a inferência de Large Language Models (LLMs), especificamente usando a arquitetura Gemma 3. O objetivo principal é descobrir como rodar esses modelos gigantes de forma mais barata e rápida (usando menos precisão ou pulando camadas), sem perder a qualidade final na resposta.

---

## 1. Conceitos Fundamentais 


### A. Quantização (`quantize.py`)
**O que é:** Quantizar significa reduzir a precisão dos "pesos" matemáticos do LLM. O modelo normal usa números de 16-bits (bfloat16). O simulador consegue reduzi-los para 8-bits, 4-bits ou até 2-bits.
**Por que é feito:** Ocupa menos memória (VRAM) na GPU e permite contas mais rápidas.
**Termos relacionados:**
- **Group Size (Agrupamento):** A quantização é feita em blocos (ex: blocos de 32 pesos).
- **Effective Bits (Bits Efetivos):** Como agrupar exige guardar pequenos fatores de escala extra, o tamanho real na memória não é exatamente "4 bits", e sim "4.25 bits". O código faz a contabilidade exata disso.
- **QuantizationSimulator:** Em vez de de fato compilar os pesos na placa de vídeo (o que exige kernels de baixo nível), o código simula como o modelo degradado se comportaria, injetando o ruído correspondente.

### B. Elastic Depth / Layer Skipping (`elastic_depth.py`)
**O que é:** Profundidade Elástica. As LLMs têm dezenas de "camadas" (layers). Essa técnica basicamente decide "pular" algumas dessas camadas durante a geração.
**Termos relacionados:**
- **Skip Schedule:** A regra de quais camadas pular (ex: se o modelo tem 12 camadas, o `uniform_skip_schedule` diz exatamente quais jogar fora para ficar apenas com as 8 mais importantes).
- **SkippableDecoderLayer:** É o "wrapper" (um embrulho) que o seu código coloca em volta das camadas originais do HuggingFace para permitir que elas sejam ligadas ou desligadas dinamicamente.

### C. Malha Fechada / Cascata (`closed_loop.py`)
**O que é:** Ao invés de usar apenas o modelo "Base" (quantizado e/ou com camadas puladas, que é rápido mas burro), usa ele em dupla com um modelo "Referência" (original de 16-bits, perfeito porém lento).
**Como funciona:**
- O modelo base vai gerando os tokens. Se ele estiver confiante (Entropia baixa), ele gera sozinho.
- Se a incerteza aumentar e ultrapassar um **Limiar (Threshold)**, o sistema aciona um "Gate" (Portão), promovendo esse token para ser gerado pelo modelo referência. 
- Assim, ganha velocidade nos tokens fáceis (a maioria) e gasta processamento pesado apenas nos tokens difíceis.

### D. Token Oracle (`token_oracle.py`)
**O que é:** O Oráculo. Trata-se de uma análise feita no passado (ou em modo de "avaliação") onde o sistema usa **Teacher Forcing**. 
- **Teacher Forcing:** É a técnica de alimentar o modelo não com as respostas que ele acabou de gerar, mas forçando o "gabarito" original. 
- O oráculo serve para responder: *"O modelo barato teria acertado essa palavra que o modelo caro gerou?"*. A partir disso, calcula-se o **Custo Especulativo** ou **Custo de Streaming**.

### E. Cost Model (`cost_model.py`)
**O que é:** Modelo Matemático de Custo. Em vez de rodar milhares de simulações iterativas de tokens, essa parte aplica estatística (Distribuições Binormais e Teoria de Detecção de Sinal).
- **AUROC:** Área sob a curva. Uma métrica de "separabilidade". Mede o quão bem a incerteza da Base consegue separar os tokens que ela acerta dos tokens que ela erra. 
- Usando equações exatas fechadas (closed-form), ele prevê o ganho computacional da decodificação antes de você gastar dinheiro rodando as GPUs reais.

---

## 2. A Arquitetura de Testes (Pasta `tests/`)

### Eles usam "Modelos Tiny" (Minúsculos) Locais
Note que nenhum teste usa um modelo real gigante (tipo um Gemma 3 de 14 ou 27 Bilhões de parâmetros). Os testes instanciam um "Gemma3 minúsculo" local em memória (com `hidden_size=64`, apenas 4 camadas e vocabulário pequeno).
**Motivo:** Evita baixar pesos enormes da internet, roda instantaneamente na CPU (em poucos segundos) e dispensa o uso de GPUs pesadas como a H100 para apenas debugar lógica.

### Foco em "Idempotência" e "Bit-Exato"
Muitos testes (ex: `test_quantize.py` e `test_elastic_depth.py`) testam o seguinte:
- Se eu ligo o modificador elástico, e depois "desligo", o modelo tem que retornar exatamente a mesma matriz bit a bit (bit-exato).
- Se eu mando um limiar (threshold) que diz "nunca promova nada", a saída precisa bater `100%` com o modelo base sozinho. E vice-versa.
Isso garante que as manipulações dinâmicas de tensor não estejam quebrando a rede neural sem você perceber.

### Principais Invariantes dos Arquivos de Teste:
1. **`test_quantize.py`**: Garante que o erro do LLM cresce gradualmente quando você baixa o número de bits (ex: erro de 2 bits > erro de 4 bits > erro de 8 bits). Garante que a contabilidade de bytes bate com as fórmulas.
2. **`test_elastic_depth.py`**: Checa que pular 100% das camadas não quebra a rede (apenas produz uma matriz idêntica aos Embeddings de entrada). Garante que a KV Cache é tratada corretamente.
3. **`test_closed_loop.py`**: Verifica a "porta" (gate) da malha fechada. Limiar de entropia muito alto gera igual a Base. Limiar muito baixo gera igual a Referência.
4. **`test_token_oracle.py`**: Avalia o oráculo. Testa se o modelo, avaliado consigo mesmo, gera 100% de precisão de acerto (auto-consistência). Mede Rajadas (Burstiness) e Custos Especulativos.
5. **`test_scoring.py`**: Assegura que cálculos em batches (grupos) grandes não estragam a posição dos tokens no LLM (Left Padding), e garante que calcular tudo versus manter apenas o último Logit (`logits_to_keep=1`) gera exatamente as mesmas probabilidades, só que poupando muito mais RAM.
6. **`test_cost_model.py`**: Garante que as contas difíceis de estatística (modelos binormais) cruzam perfeitamente com a simulação iterativa (força-bruta).

---

## Resumo do contexto
Tratei do **montante de testes matematicamente rigorosos**. Eles não estão testando se o LLM "fala bem", estão testando se as operações matemáticas de compressão de pesos e pular camadas (Elastic Weight Streaming) estão sendo injetadas corretamente sem vazar memória, sem estragar os estados internos (KV Caches), e sem distorcer as estimativas de custo e uso de RAM. 

Qualquer alteração na pasta `ews/core/` que quebre o fluxo dos dados será capturada em segundos rodando `pytest tests/` (ou chamando os arquivos um a um).
