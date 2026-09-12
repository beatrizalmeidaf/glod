# Projeto EWS: Elastic Weight Streaming

## 1. PESQUISA (Visão Geral, Teoria e Estado da Arte)

Por que "12B mais rápido E mais preciso que 4B" faz sentido nessa linha:
Um modelo dense de 4B usa todos os seus parâmetros em todo input, sempre. Um modelo de 12B com carregamento condicional (layer-skipping / streaming de resíduo) só "gasta" a capacidade extra quando o input realmente precisa — pra maioria dos tokens/exemplos fáceis, ele pode efetivamente rodar equivalente a um modelo bem menor que o 4B (poucas camadas, sem correção), e só "vira" um 12B de verdade nos casos difíceis. Ou seja: o 12B tem uma capacidade de pico maior, mas um custo médio menor, porque a maior parte da massa de parâmetros dele é redundante pra inputs fáceis — redundância essa que o 4B, sendo denso e "no limite" do seu tamanho, não tem pra explorar. Isso é literalmente o argumento central da idea 4 (layer-skipping condicionado ao input).

A versão que eu proporia como paper, fundindo 4 + 1:
Chama isso de algo como Elastic Weight Streaming — um framework único onde:

Um gate barato (treinado, ou até baseado em incerteza/confiança da camada anterior) decide, camada por camada, três opções por input: (a) pular a camada inteira, (b) usar só a versão base comprimida/residente em RAM, ou (c) streamar do disco a correção de baixo rank daquela camada pra "ativar" a capacidade completa do modelo grande.
Isso generaliza tanto early-exit quanto LoRA-em-inferência num único mecanismo contínuo de "quanto do modelo de 12B eu realmente preciso carregar agora".
A métrica que fecha o argumento pro NeurIPS: uma curva de Pareto entre bytes streamados por token/exemplo × acurácia, mostrando que o modelo grande com streaming condicional domina tanto o modelo pequeno denso (mesma latência, mais preciso) quanto o modelo grande denso tradicional (mesma acurácia, muito mais rápido em hardware fraco) — os dois pontos de comparação que sua observação empírica já sugere que existem.

O que dá o "wow" e a originalidade pro comitê:
Não é só "quantização adaptativa" (isso já existe) — é mostrar formalmente que capacidade nominal (nº de parâmetros) e custo efetivo de I/O podem ser desacoplados por input, com uma política aprendida que explora isso, e provar (ou pelo menos caracterizar empiricamente com bound) que essa política domina qualquer modelo denso de tamanho fixo na fronteira latência-acurácia em hardware com memória limitada. Esse é um resultado "surpreendente mas explicável" — exatamente o tipo de história que engana a intuição ingênua ("modelo maior = sempre mais lento") e por isso rende bem em venue como NeurIPS.

### 1. Formulação da pergunta de pesquisa

A ideia final, batizando de **Elastic Weight Streaming (EWS)**: um modelo grande congelado tem sua maior parte comprimida/quantizada e sempre residente em RAM/VRAM (um "esqueleto" barato), e por camada, um gate leve decide — pra cada input — se precisa streamar do disco uma **correção de baixo rank** (estilo LoRA) daquela camada pra recuperar a capacidade completa do modelo grande. A hipótese central: o custo médio de I/O fica baixo (porque a maioria dos inputs é "fácil" e não precisa da correção), mas o teto de qualidade continua sendo o do modelo grande — dominando um modelo pequeno denso do mesmo tamanho médio de memória, tanto em acurácia quanto em latência.

Decidir o que usar dinamicamente (Roteamento)

Carregar pesos do disco pra RAM (Offloading): Já existe mas de forma estática (carregam sempre a mesma coisa).


Ideia: (Elastic Capacity Inference): Usar um gate aprendido para fazer streaming sob demanda de resíduos de baixo rank, baseado na dificuldade do input, otimizando I/O no disco e não apenas FLOPs na GPU.
Essa interseção exata (streaming condicional de capacidade via disco) é um buraco na literatura
hipótese: é possível desacoplar a capacidade do modelo do seu custo de I/O em inferência. Apresenta-se o EWS, que usa controle de risco para streamar tensores apenas quando necessário, fazendo um modelo de 12B rodar em laptops comuns com a latência de um 4B, batendo baselines estáticos em X%

### 2. O que já existe (fiz a checagem — isso é importante antes de você investir tempo)

Já tem bastante prior art em partes disso, especificamente no nível de sistemas:

- **PowerInfer** já faz exatamente o "gate decide o que carregar por input", mas no nível de *neurônio*, não de correção de baixo rank: neurônios ativados com frequência ficam pré-carregados na GPU, enquanto neurônios "frios" são computados na CPU, reduzindo demanda de memória de GPU e transferências CPU-GPU.
- **MCAP** também faz profiling de importância por camada em tempo de deployment pra decidir alocação de precisão mista, e se posiciona explicitamente contra PowerInfer como o "vizinho de pesquisa" mais próximo, notando que o roteamento neurônio-a-neurônio do PowerInfer é mais fino que o roteamento por camada.
- **FlexGen** já resolve o offloading de peso/ativação/KV-cache entre GPU/CPU/disco pra throughput alto numa única GPU, mas com uma política fixa, não condicionada ao input.
- **Mixture-of-Depths (DeepMind)** já treina um roteador que determina quais tokens passam pelo bloco computacional completo vs. uma conexão residual barata, com orçamento de compute fixo por camada — mas isso é sobre *computação*, assumindo o modelo já carregado, não sobre *streaming de peso do disco*.
- Existem também ferramentas mais "hobbyist"/produto (StreamLLM, digvijay_llm, a issue aberta no MLX) que já fazem streaming de camada do disco pra RAM sob demanda — mas de forma sequencial fixa, sem um gate condicionado ao conteúdo do input.

**Conclusão honesta**: a peça "carregar peso condicionalmente por input" já existe (PowerInfer é o mais próximo). O que **não existe**, até onde encontrei, é a combinação específica de: (a) correção contínua de baixo rank (não binário liga/desliga de neurônio ou camada inteira) streamada sob demanda, (b) treinada como um trade-off explícito acurácia-vs-bytes (não FLOPs-vs-acurácia como o MoD), e (c) validada com a narrativa "modelo grande streamado domina modelo pequeno denso em latência real de hardware fraco" como resultado central e comparação cabeça-a-cabeça. Isso ainda é um espaço aberto, mas você vai precisar posicionar o paper com muito cuidado contra PowerInfer e MCAP — reviewers de sistemas vão bater nisso primeiro.

### Crítica como revisor (a parte que você pediu pra lapidar de verdade)

**Reviewer 1 (sistemas)**: "Isso é PowerInfer com uma correção de baixo rank em vez de roteamento de neurônio. A diferença é incremental." — *Mitigação*: você precisa de uma comparação empírica direta, não só conceitual, mostrando que a correção contínua de baixo rank recupera qualidade em menos bytes streamados do que ligar/desligar neurônios inteiros, numa curva de Pareto lado a lado.

**Reviewer 2 (cético em geral)**: "12B mais rápido que 4B é uma afirmação extraordinária — suspeito de comparação injusta (precisões diferentes, quantização não controlada)." — *Mitigação*: toda comparação tem que ser em precisão efetiva equivalente controlada (ex: ambos em 4-bit, ou reportar curva pra várias precisões), senão o resultado é descartado como artefato.

**Reviewer 3**: "O overhead do próprio gate não foi contabilizado — ele consome tempo/memória e isso pode anular o ganho." — *Mitigação*: reportar o custo do gate separadamente em toda tabela, não escondido dentro do número "total".

**Reviewer 4**: "Resultado depende demais de banda de I/O do disco específico — não generaliza." — *Mitigação*: rodar em pelo menos 3 perfis de hardware com banda de disco diferentes e/ou um modelo de custo teórico (não só medição empírica) que caracterize o ponto de cruzamento como função da banda.

**Reviewer 5**: "Treinar o gate exige rótulos de 'dificuldade' que são caros de obter — como isso escala pra outros domínios/idiomas de poucos recursos?" — *Mitigação*: comparar explicitamente gate heurístico (barato, sem supervisão) vs. gate treinado (caro, melhor) como uma ablação central, não uma nota de rodapé — isso também é honesto sobre a limitação e transforma a crítica em um resultado do paper.

**Reviewer 6**: "Só uma família de modelo (Gemma) — generaliza?" — *Mitigação*: rodar pelo menos uma segunda família antes de submeter.

### Principais problemas conceituais que eu corrigiria

Li o texto. A ideia é viável como linha de pesquisa, mas eu faria uma mudança importante: **não venderia “12B é mais rápido que 4B” como premissa**. Isso é um resultado que o experimento precisa demonstrar sob condições bem definidas. O núcleo cientificamente mais forte é **desacoplar capacidade nominal de custo de inferência por entrada**, usando armazenamento hierárquico e ativação condicional de pesos.

Também há algumas inconsistências conceituais no texto atual que vale corrigir antes de transformar isso em proposta de paper.

1. **“O 12B só gasta a capacidade extra quando precisa”**
   Isso não é automaticamente verdade. Se o modelo 12B precisa executar as mesmas camadas do 4B, mesmo sem carregar a correção LoRA, ele ainda tem custo computacional associado à base residente. O ganho precisa vir de uma combinação de:
   * pesos residentes menores;
   * skipping de computação;
   * pouca transferência de dados;
   * e/ou uma decomposição realmente eficiente do peso.

2. **“Correção LoRA recupera a capacidade completa do 12B”**
   Essa afirmação é forte demais. Uma aproximação do tipo
   $W = W_{\text{base}} + AB^\top$
   não garante que uma correção de baixo posto reproduza exatamente o modelo original. O correto seria dizer **“aproxima a transformação do modelo original”** ou **“recupera parte da capacidade perdida pela compressão”**.

3. **“Pular camada + base + correção” são três coisas diferentes**
   Você está misturando:
   * **compute sparsity** → pular camada;
   * **weight compression** → base comprimida;
   * **weight streaming** → carregar informação adicional.
   Isso pode ser uma contribuição interessante, mas também pode deixar o paper excessivamente amplo. Eu começaria com **uma contribuição principal**.

4. **O maior problema é I/O**
   Se você realmente precisa acessar o SSD para cada camada/token, o método pode ficar muito mais lento que um modelo menor. Portanto, a unidade correta provavelmente não é “streamar uma camada inteira por token”. Você precisa explorar **granularidade, cache e reutilização**.

5. **Bytes streamados não são suficientes**
   Dois métodos podem transferir os mesmos 500 MB e ter latências completamente diferentes dependendo de:
   * tamanho dos blocos;
   * acesso sequencial vs. aleatório;
   * cache;
   * concorrência;
   * compressão/descompressão;
   * bandwidth;
   * latência do dispositivo.
   Então **latência real deve ser a métrica primária**, e bytes transferidos uma métrica explicativa.

6. **“Domina qualquer modelo denso” é uma reivindicação perigosa**
   Você não consegue provar empiricamente que domina *qualquer* modelo denso. Melhor:
   > “domina modelos densos de referência sob um orçamento comparável de memória/latência”
   Isso é defensável.

7. **PowerInfer é uma ameaça séria à novidade**
   A contribuição não pode ser simplesmente “decidir dinamicamente o que carregar”. O diferencial precisa estar na **representação da capacidade adicional** e na política de ativação.

8. **LoRA talvez nem seja a melhor representação**
   Esse é um ponto que eu investigaria seriamente. Talvez a ideia mais interessante não seja “LoRA + streaming”, mas:
   > **um modelo grande decomposto em uma representação barata residente + componentes de capacidade recuperáveis sob demanda.**
   A correção pode ser low-rank, residual quantizada, delta weights, adapters ou outra decomposição. Assim você não prende a contribuição a LoRA antes de saber se é realmente a melhor solução.

### O verdadeiro problema científico

Eu escreveria a pergunta de pesquisa mais ou menos assim:

> **É possível reduzir o custo médio de inferência de um modelo de linguagem grande, mantendo sua capacidade de pico, por meio da recuperação condicional de pesos adicionais em função da dificuldade da entrada?**

E a hipótese:

> **A recuperação condicional de capacidade permite que um modelo de maior capacidade alcance uma fronteira qualidade–latência superior à de modelos densos menores sob um orçamento limitado de memória rápida.**

Isso é muito mais acadêmico que:
> “12B mais rápido e mais preciso que 4B”.
O segundo é **o resultado que você quer obter**, não a pergunta científica.

### Uma correção importante na sua narrativa

Eu **não escreveria**:
> “a maior parte da massa de parâmetros do 12B é redundante para inputs fáceis”.

Você ainda não sabe disso. Eu escreveria:
> “A hipótese é que a contribuição marginal de determinados componentes de capacidade varia entre entradas. Se essa hipótese for válida, uma representação compacta residente pode preservar o desempenho em entradas fáceis, enquanto componentes adicionais são recuperados seletivamente para entradas que exigem maior capacidade.”

Isso é muito mais forte cientificamente porque **transforma a afirmação em hipótese testável**.

### Por que o "Paradoxo" faz sentido teórico?

O texto que você consolidou já traz uma maturidade excelente para uma submissão de alto nível. A transição de uma afirmação comercial ("12B mais rápido que 4B") para uma formulação científica rigorosa ("Inferência de Capacidade Elástica") é exatamente o que um comitê do NeurIPS ou ACL espera ver.

A viabilidade dessa ideia repousa em uma característica fundamental da modelagem de linguagem que os modelos densos ignoram: **a assimetria da entropia da informação**.

Na linguagem natural, a grande maioria dos tokens é altamente previsível (conjunções, estrutura sintática, conhecimento de senso comum). Uma minoria de tokens carrega o peso semântico real (fatos específicos, saltos lógicos de raciocínio, desambiguação complexa).

1. **O desperdício do modelo denso (4B):** Um modelo denso aloca um custo computacional estático $C$ para todo token. Para tokens fáceis, o 4B está superparametrizado, gastando FLOPs à toa. Para tokens difíceis, ele está subparametrizado, limitando o teto de acurácia.
2. **A eficiência do modelo elástico (12B EWS):** O EWS transforma o custo computacional em uma variável aleatória $C(x)$ condicionada à dificuldade da entrada $x$. Se a representação base residente em RAM tem a capacidade equivalente a um modelo de 1.5B (suficiente para resolver a gramática e a estrutura básica), o gate manterá $g_l(x) = 0$ para a maioria dos tokens.

O paradoxo funciona porque você está operando com **esperança matemática**. O custo médio esperado do 12B elástico, $E[C(x)]$, torna-se menor que a constante $C$ do 4B denso, enquanto a capacidade máxima (quando $\sum g_l(x) = L$) iguala a do 12B completo. Você ganha na latência explorando a redundância da linguagem, e ganha na acurácia ativando a representação completa apenas quando a incerteza estatística exige.

### Minha avaliação da ideia

| Aspecto                     | Avaliação                       |
| --------------------------- | ------------------------------- |
| Viabilidade técnica         | **Alta, com ressalvas**         |
| Novidade potencial          | **Boa**                         |
| Risco de já existir         | **Alto nas partes isoladas**    |
| Diferencial potencial       | **Bom**                         |
| Risco experimental          | **Alto**                        |
| Potencial de paper forte    | **Sim**                         |
| “12B > 4B”                  | **Resultado, não premissa**     |
| LoRA como solução           | **Ainda precisa ser validado**  |
| Streaming SSD por token     | **Principal risco técnico**     |
| Gate condicionado à entrada | **Central**                     |
| Pareto qualidade–latência   | **Excelente métrica principal** |

**Minha conclusão:** eu não descartaria a ideia. Pelo contrário, acho que existe uma linha de paper aqui, mas eu **tiraria o foco de “12B mais rápido que 4B” e colocaria em “elasticidade de capacidade”**. O resultado 12B > 4B seria então a demonstração mais chamativa da hipótese.
E eu mudaria o nome também. **Elastic Weight Streaming** é bom como nome provisório, mas talvez “weight” seja estreito demais se você posteriormente incluir skipping de camadas. Algo como **Elastic Capacity Inference (ECI)** ou **Elastic Model Capacity** pode deixar a contribuição mais abrangente.

---

## 2. TESTES E DESENHO EXPERIMENTAL

### Desenho experimental

**Modelos**: família Gemma (você já tem dado empírico com 4B e 12B) — mas pra reivindicar generalidade, o paper precisa de pelo menos mais uma família (ex: Qwen ou Llama numa faixa de tamanho equivalente).

**Hardware**: reproduza o cenário do Samsung Galaxy Book 2 formalmente (CPU, RAM, tipo de armazenamento — SATA SSD vs NVMe importa MUITO pro resultado) e adicione 2-3 perfis controlados adicionais (ex: NVMe rápido vs SSD lento vs simulação de rede), porque o ponto de cruzamento onde streaming compensa depende diretamente da banda de I/O.

**Tarefas**: aproveite seus 14 corpora em português (heterogeneidade de dificuldade é o que faz o método brilhar — tarefa homogênea em dificuldade não mostra ganho nenhum) + 1-2 benchmarks padrão em inglês (ex: subset do MMLU, GSM8K) pra dar credibilidade internacional ao resultado.

**Gate**: comece com uma versão simples (heurística de confiança/entropia da camada anterior) como baseline, depois uma versão treinada (probe leve, com Gumbel-softmax/straight-through, penalizando bytes streamados na loss).

**Baselines que você PRECISA vencer** (não é opcional, é o que vai determinar aceite ou rejeição):
1. Modelo pequeno denso (Gemma 4B) na mesma latência média
2. Modelo grande denso sem streaming condicional (upper bound de referência)
3. Layer-skipping fixo, sem gate (ablação — mostra que o gate importa)
4. Reimplementação simplificada do PowerInfer (hot/cold split) — não pode só citar números do paper deles, tem que rodar no seu próprio hardware pra comparação justa
5. Quantização uniforme no mesmo bitrate médio

**Métricas**: acurácia/tarefa, bytes streamados por exemplo, latência real (tokens/s), pico de RAM/VRAM, e a curva de Pareto acurácia × bytes-streamados / acurácia × latência como resultado central (gráfico principal do paper).

### Resultado esperado (hipótese a testar, não garantia)

Que numa tarefa de dificuldade heterogênea, o Gemma-12B-com-EWS domine o Gemma-4B-denso: mesma latência média, acurácia maior — e domine o Gemma-12B-denso-offloaded (FlexGen-style): mesma acurácia, bytes/latência muito menores. Se isso não aparecer, o resultado mais provável é que o ganho só existe em tarefas com cauda de dificuldade bem definida (muitos exemplos fáceis, poucos difíceis) — o que já seria um achado publicável (caracterizar *quando* o método funciona), só que menos "wow".

### Ablações - Versões

Eu testaria primeiro:
### Versão 1 — Elastic Capacity
$W_l = W_l^{base}+\Delta W_l$
Gate:
$g_l \in \{0,1\}$
Só duas decisões:
**base** ou **base + residual**.
Isso te dá uma hipótese muito limpa.

Depois você pode adicionar:
### Versão 2 — Elastic Depth
$g_l=0$
significa pular a camada.

### Versão 3 — Elastic Capacity + Depth
$g_l\in\{skip,base,full\}$
Essa última seria a versão completa.

Assim o paper consegue mostrar incrementalmente:

| Método           | Skip | Residual streaming | Gate |
| ---------------- | ---: | -----------------: | ---: |
| Dense            |    ✗ |                  ✗ |    ✗ |
| Layer skipping   |    ✓ |                  ✗ |    ✓ |
| Weight streaming |    ✗ |                  ✓ |    ✓ |
| **EWS**          |    ✓ |                  ✓ |    ✓ |

Isso seria uma ablação muito mais convincente.

### O experimento decisivo

Eu faria um gráfico central assim:

**x:** latência por token
**y:** qualidade

e colocaria:
* 4B dense;
* 4B quantizado;
* 12B dense;
* 12B offloaded;
* 12B + fixed residual;
* 12B + random gate;
* 12B + confidence gate;
* **12B + EWS**.

Se aparecer:
$$ \boxed{ \text{EWS-12B} > \text{Dense-4B} } $$
em qualidade para o mesmo orçamento de latência, aí você tem uma história muito interessante.

E ainda melhor se aparecer:
$$ \boxed{ \text{EWS-12B} \approx \text{Dense-12B} } $$
em qualidade com muito menos I/O.

### Com que máquina eu testo isso?

Você vai precisar de **duas** abordagens de hardware, uma para desenvolver e outra para provar seu ponto.

**Para Desenvolvimento e Simulação (A Oficina):**
Para treinar o gate e simular a ideia (carregando o 12B inteiro na memória para testar as ablações), você precisa de VRAM alta. O ideal é usar plataformas em nuvem como RunPod, Lambda Labs ou Colab Pro, alugando uma GPU como a RTX 3090 ou RTX 4090 (24GB de VRAM). O custo é baixo (alguns centavos de dólar por hora) e permite que você não perca tempo com gargalos de memória enquanto coda.

**Para o Experimento Real (O Palco):**
O argumento central dessa pesquisa é: "fazer modelos grandes rodarem rápido em hardware fraco". Portanto, a sua máquina de teste final para coletar os números de latência que vão pro paper é um hardware de consumo padrão.
O seu próprio Samsung Galaxy Book 2 é o laboratório perfeito. A estrela do seu experimento vai ser mostrar que o seu método consegue fazer um modelo de 12B rodar nele via CPU/RAM com uma latência (tokens/segundo) melhor do que a versão densa tradicional, contornando o limite do hardware local ao aproveitar o SSD/NVMe apenas quando estritamente necessário.

---

## 3. IMPLEMENTAÇÃO E ARQUITETURA TÉCNICA

### Eu reformularia a ideia matematicamente assim

Um modelo grande é decomposto em:
$$ W_l = W_l^{base} + \Delta W_l $$
onde:
* $W_l^{base}$ é uma representação compacta, quantizada e residente;
* $\Delta W_l$ contém informação adicional armazenada fora da memória rápida;
* um gate condicionado ao estado do modelo decide se essa informação adicional deve ser recuperada.

Para cada camada, temos:
$$ g_l(x) \in \{0,1\} $$
e:
$$ h_{l+1} = f_l(h_l,W_l^{base}) + g_l(x)\,f_l(h_l,\Delta W_l) $$

A interpretação é muito mais limpa: **o modelo completo existe, mas sua capacidade é elástica.**
Para entradas fáceis: $g_l(x)=0$ e o modelo utiliza apenas a representação barata.
Para entradas difíceis: $g_l(x)=1$ e recupera capacidade adicional.
Isso é conceitualmente mais forte que dizer simplesmente “layer skipping”.

### O ponto que eu investigaria antes de implementar

Existe uma pergunta fundamental: **o residual de baixo rank realmente entrega uma boa relação qualidade/bytes?**
Porque pode acontecer isto:
$$ \text{12B base quantizado} + \text{LoRA} $$
precisar carregar tanto residual que o benefício desaparece.

Nesse caso, sua ideia não morre. Você simplesmente descobre que **low-rank não é a representação adequada**.
Por isso eu trataria a representação como variável experimental:
$$ \Delta W \in \{ \text{LoRA}, \text{quantized residual}, \text{adapter}, \text{delta compression} \} $$
e escolheria pela curva:
$$ \text{quality} \quad vs. \quad \text{bytes transferred} $$
Isso transforma uma possível fragilidade em uma questão científica.

### Desenhando a Arquitetura Técnica do Gate

Como o texto aponta, o gargalo não é computacional, é de **I/O**. O disco (mesmo um NVMe) é ordens de grandeza mais lento que a VRAM. Se o gate parar a inferência, for ao disco, buscar a matriz residual $\Delta W$, e só então computar a camada, a latência real destruirá qualquer benefício teórico.

Para que isso funcione na prática, o gate precisa ser desenhado integrando **classificação seletiva com controle de risco** e otimização de baixo nível.

**1. O Mecanismo do Gate (Otimização de Risco)**
O gate não pode ser apenas uma rede neural ingênua. Ele é fundamentalmente um **classificador seletivo**. O modelo base faz uma previsão latente, e o gate deve decidir se "abstém" de usar apenas a base e pede ajuda (stream de $\Delta W$).

Você pode definir o gate $g_l$ de uma camada $l$ como uma função baseada na entropia das ativações intermediárias ou um classificador linear leve treinado para prever o erro residual:
$$g_l(h_l) = \sigma(W_{gate} \cdot h_l + b_{gate})$$

Para tornar isso diferenciável durante o treinamento e forçar a esparsidade do streaming, você introduz a **Gumbel-Softmax**. A função de perda precisa penalizar explicitamente a quantidade de bytes trafegados:
$$\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \sum_{l=1}^{L} \mathbb{E}[g_l(x)] \cdot \text{Bytes}(\Delta W_l)$$
Onde $\lambda$ é o hiperparâmetro que controla diretamente a fronteira de Pareto (Acurácia vs. I/O). Ao variar $\lambda$, você gera a curva exata que precisa para o paper.

**2. A Realidade do Hardware (Streaming Assíncrono)**
Se o gate da camada $L$ só decidir carregar o peso no exato momento em que a camada precisa dele, o modelo ficará travado esperando o PCIe. A solução exige antecipação.

O design mais robusto envolve um **Early-Probe Gate**:
Em vez de decidir camada por camada de forma isolada, um probe acoplado nas primeiras camadas (ex: na camada 3 de um modelo de 32 camadas) avalia a incerteza geral do token e gera um vetor de roteamento antecipado para as próximas camadas.

Enquanto a GPU calcula as camadas $L_4$ a $L_7$ usando apenas a base residente, o sistema já iniciou o pre-fetch assíncrono (streaming) dos tensores $\Delta W_8 \dots \Delta W_{10}$ do disco para a RAM/VRAM. Para que esse paralelismo de I/O e computação funcione sem travar o GIL do Python ou gerar overhead no PyTorch, será necessário escrever **kernels CUDA customizados em C++** que gerenciem a movimentação de memória de forma assíncrona (`cudaMemcpyAsync`) enquanto a matriz base é multiplicada.

**3. A Natureza de $\Delta W$**
Como discutido nos seus apontamentos, LoRA puro pode ser ineficiente em I/O se o rank precisar ser muito alto para recuperar a qualidade. Uma alternativa é que $\Delta W$ seja um tensor altamente esparso (SparseGPT) ou quantizado agressivamente (ex: 2-bit), tornando o custo de carregar a capacidade extra tão baixo que a banda do SSD não sofra saturação.

---

## 4. SKILLS.MD IMPORTANTES (VOU USAR AGENTES)

*(Estrutura criada com base nas demandas técnicas do texto para direcionar seus agentes autônomos de IA nas próximas etapas do desenvolvimento)*

Para orquestrar o desenvolvimento com agentes, você precisará configurar os seguintes perfis em seu `skills.md` ou prompt de sistema:

1. **Agente de Pesquisa e Estado da Arte:**
   - **Habilidades:** Analisar implementações open-source de sistemas MLSys (PowerInfer, Mixture-of-Depths, FlexGen, llama.cpp).
   - **Missão:** Vasculhar repositórios, extrair as métricas exatas de throughput dos baselines e configurar os ambientes de reprodutibilidade para as comparações.

2. **Agente Engenheiro de Machine Learning / NLP (PyTorch):**
   - **Habilidades:** Hackear a função `forward` do PyTorch (HuggingFace Transformers). Domínio em classificação seletiva, controle de risco, e funções de perda customizadas (Gumbel-Softmax e trade-off $\lambda$).
   - **Missão (Fase 1 e 2):** Implementar o Gate Oráculo, calcular o upper bound teórico carregando o 12B na VRAM. Em seguida, treinar o classificador linear preditor de entropia acoplado nas camadas intermediárias.

3. **Agente Engenheiro de Sistemas e Hardware (C++ / CUDA):**
   - **Habilidades:** Sistemas de baixo nível. Domínio em concorrência I/O, `mmap` (memory-mapped files), PCIe latency, e `cudaMemcpyAsync`.
   - **Missão (Fase 3):** Construir o mecanismo de Early-Probe Gate, escrevendo kernels CUDA em C++ integrados ao PyTorch para garantir que a GPU e o SSD funcionem de forma paralela e assíncrona sem travar o GIL do Python.

4. **Agente Cientista de Dados (Análise de Benchmarks):**
   - **Habilidades:** Visualização de dados (Pareto curves), análise estatística rigorosa.
   - **Missão (Fase 4):** Gerar os gráficos centrais do paper comparando Acurácia vs Latência por Token.

---

## 5. WORKFLOW (CRONOGRAMA E MESTRADO)

O deadline do NeurIPS 2027 ainda não é oficial, mas projeções baseadas no padrão histórico apontam pra **maio de 2027** (registro de abstract por volta de 14 de maio, paper completo por volta de 20-21 de maio, AoE) — confirme no call for papers oficial quando sair. Isso te dá cerca de 8 meses a partir de agora, o que é apertado mas factível se você escopar pra **uma família de modelo + uma variante do método (correção contínua de baixo rank, não misturar com skip binário) + 3 baselines fortes**, deixando a segunda família de modelo e a versão com garantia de risco certificado (ligando com seu trabalho de SGR) como extensão pra uma versão de workshop ou trabalho futuro citado no paper.

### É viável para um Mestrado? (E como conectar com o agora)

A resposta curta é: **A combinação é original.**
A pesquisa em IA funciona como blocos de Lego. As peças separadas já existem, mas ninguém montou esse castelo específico:
* **Decidir o que usar dinamicamente (Roteamento):** Já existe. O *Mixture-of-Depths* (DeepMind) e o *PowerInfer* fazem isso.
* **Carregar pesos do disco pra RAM (Offloading):** Já existe. O *FlexGen* e o *llama.cpp* fazem isso, mas de forma estática (carregam sempre a mesma coisa).
* **A sua originalidade (Elastic Capacity Inference):** Usar um gate aprendido para fazer *streaming sob demanda* de resíduos de baixo rank, baseado na dificuldade do input, otimizando I/O no disco e não apenas FLOPs na GPU.
Essa interseção exata (streaming condicional de capacidade via disco) é um buraco na literatura. É publicável.

É o escopo perfeito. O mestrado no Brasil dura 24 meses. Dá tempo de estudar o estado da arte, errar a implementação, corrigir e publicar.
Como você está exatamente no momento de transição de orientação e precisa definir o escopo do seu trabalho de conclusão de curso, você pode quebrar isso em duas fases, usando a primeira como ponte:

* **TCC (O Oráculo / A Prova de Conceito):** Você prova que a *matemática* funciona. Você não programa nada de disco ou streaming real. Você carrega tudo na VRAM e apenas "finge" que pulou camadas ou ignorou o $\Delta W$. O objetivo do TCC é responder: *"Se existisse um gate perfeito que sabe a dificuldade do token, eu conseguiria jogar fora 70% dos cálculos sem perder acurácia?"*
* **Mestrado (O Sistema Real / O Gate):** Aqui você constrói o sistema de verdade. É onde entra a sua base em **classificação seletiva com controle de risco**. O gate que decide "pular" ou "carregar" o peso extra é, na essência, um classificador seletivo que diz: "tenho confiança alta usando só a base" ou "o risco de errar é alto, acione a capacidade extra". Você usa o que já domina de NLP para resolver o gargalo do hardware.

### Plano de Batalha de 8 Meses

Além disso, usar a submissão de um paper forte como seu projeto de qualificação ou proposta de ingresso no mestrado é a **melhor estratégia possível**. No Brasil, processos seletivos de mestrado (como na UFG ou outras federais) valorizam imensamente alunos que já entram com um problema de pesquisa formulado, cronograma claro e um protótipo inicial.
Como você já tem bagagem com **classificação seletiva e controle de risco**, você tem uma vantagem injusta aqui.

Aqui está o **Plano de Batalha de 8 Meses** para sair do zero hoje e chegar no NeurIPS em maio de 2027, estruturado como um projeto de mestrado.

**Fase 1: A "Mentira" Necessária (Simulação in-VRAM)**
*Objetivo:* Provar que a matemática funciona antes de brigar com o hardware.
*Prazo:* Setembro e Outubro de 2026 (Ideal para fechar seu TCC atual)
Você não vai escrever uma linha de código de I/O de disco agora. Você vai alugar uma GPU na nuvem (ex: RunPod, RTX 4090) e carregar o **Gemma 12B inteiro na VRAM**.
1. **O Gate Oráculo:** Em vez de treinar um gate, você calcula a resposta com o modelo base (esqueleto) e com o modelo completo. Se a resposta da base for igual ao gabarito, você anota: `gate = 0`. Se errar, `gate = 1`.
2. **O Limite Teórico (Upper Bound):** Com esse "oráculo", você calcula: "Se meu gate fosse perfeito, quantos % das camadas eu poderia pular sem perder 1% de acurácia?"
3. **Métrica:** Gráfico de Pareto teórico. Eixo X: % de tensores extras necessários. Eixo Y: Acurácia. Se você não conseguir manter a acurácia do 12B lendo apenas 30% dos pesos extras para a maioria dos inputs, a ideia morre aqui (e você troca de tema sem ter perdido meses programando C++).

**Fase 2: O Treinamento do Gate (A sua zona de conforto)**
*Objetivo:* Fazer o modelo adivinhar o "oráculo" sem olhar a resposta.
*Prazo:* Novembro e Dezembro de 2026
Aqui entra a sua pesquisa atual de Risco Controlado e NLP.
1. Desenhar uma função de perda (loss) que penaliza duas coisas: errar a predição (Cross-Entropy) e acionar a leitura do disco (Gumbel-Softmax para penalizar os "bytes").
2. Treinar um classificador linear minúsculo (o Gate) acoplado nas camadas intermediárias.
3. **Validação:** Mostrar empíricamente que o seu gate treinado consegue se aproximar da curva de Pareto do gate Oráculo.
*(Nota: Esse material (Fase 1 + Fase 2) é o documento perfeito para você submeter no processo seletivo do Mestrado (entre outubro e o início do ano que vem).*

**Fase 3: A Trincheira da Engenharia (Sistemas)**
*Objetivo:* Fazer o streaming condicional real acontecer no disco.
*Prazo:* Janeiro a Março de 2027
É aqui que o paper ganha o nível "NeurIPS". Você vai implementar o *Elastic Weight Streaming* (EWS) de verdade.
1. **Implementação PyTorch/C++:** Usar técnicas como `mmap` (memory-mapped files) para carregar os tensores quantizados de baixo rank (ex: GGUF) direto do seu SSD para a RAM.
2. **Streaming Assíncrono:** Fazer com que o gate da camada 2 já dê a ordem para o disco começar a ler os pesos da camada 5, escondendo a latência do hardware.

**Fase 4: O "Palco" (Avaliação e Benchmarks)**
*Objetivo:* Executar o desenho experimental para calar os revisores.
*Prazo:* Abril de 2027
Os revisores de Machine Learning Systems (MLSys/NeurIPS) são obcecados por comparações justas. Seu palco de testes deve ser o seu **Samsung Galaxy Book 2** (hardware de borda real, limitado por memória e CPU/GPU integrada).
Os baselines que você vai rodar e comparar contra o seu método:
1. Gemma 4B Denso (O modelo rápido, mas "burro").
2. Gemma 12B Denso offloaded via *llama.cpp* ou *FlexGen* (O modelo inteligente, mas super lento porque lê tudo do disco sempre).
3. Gemma 12B com PowerInfer (O estado da arte em esparsidade).
As Tarefas: MMLU, GSM8K, e uns 3 a 5 datasets daquele seu corpus em português para mostrar que funciona em línguas multilingues/menos representadas.

**Fase 5: A Escrita da História**
*Objetivo:* Transformar dados em um paper de 9 páginas.
*Prazo:* Maio de 2027
A narrativa do paper não será "criei uma arquitetura nova". Será: *"Nós provamos que é possível desacoplar a capacidade do modelo do seu custo de I/O em inferência. Apresentamos o EWS, que usa controle de risco para streamar tensores apenas quando necessário, fazendo um modelo de 12B rodar em laptops comuns com a latência de um 4B, batendo baselines estáticos em X%."*

### O Plano para o Mestrado
Um mestrado exige uma contribuição inédita. Submeter para o NeurIPS já valida isso.
* **Se o paper for aceito no NeurIPS (Novembro de 2027):** Seu mestrado está praticamente pronto no primeiro ano. Você gasta o segundo ano só aprofundando a teoria e defendendo.
* **Se o paper for rejeitado:** O NeurIPS tem uma taxa de aceite de ~26%. Rejeição é normal. Mas os revisores vão te dar um feedback brutal e valioso. Você corrige os erros, expande o trabalho (ex: implementa para a família LLaMA também) e submete para a ACL, EMNLP, ICLR ou MLSys no início de 2028. De qualquer forma, você já terá a sua dissertação pronta.

---

## 6. COISAS IMPORTANTES PRA COMEÇAR O DESENVOLVIMENTO (AGORA)

Como começar do zero HOJE?
Não tente escrever kernels de GPU no primeiro dia. Siga o princípio da "complexidade progressiva".
Não tente resolver o PyTorch agora. Seu primeiro passo nesta semana é conceitual. Crie um repositório no GitHub ou um documento no Notion chamado `EWS_NeurIPS2027`. Sua única tarefa é baixar os papers do **PowerInfer** e do **Mixture-of-Depths**, ler as seções de "Methodology" e desenhar num papel, com setinhas, onde exatamente a sua ideia difere da deles.

E em seguida, comece desenhando o código mais básico dessa Fase 1 (a simulação do "oráculo") para você rodar no Colab/RunPod amanhã.
Pegue o Gemma 4B e o Gemma 12B. Rode inferência normal usando o Hugging Face em alguma tarefa como o MMLU ou um dos seus datasets de português. Hackeie o código fonte do modelo (no PyTorch é só alterar o `forward`). Diga para o 12B pular as camadas pares. A acurácia despencou? E se pular só as últimas 5? Você precisa "sentir" como o modelo quebra quando partes dele são desligadas.