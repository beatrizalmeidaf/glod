# O dano de compressão é uma taxa de câmbio fixa: KL, margens e os limites das perturbações estáticas de pesos

Documento de estrutura do paper. Contém só o que está sustentado por experimento, o que ainda precisa
ser provado e o que foi descartado. Cada afirmação aponta para o script e o dado que a sustenta.


---

## 1. Tese central

> Para um modelo de referência, **qualquer perturbação estática de pesos converte divergência KL em
> mudanças de decisão a uma taxa única**, flips ≈ κ·√KL, onde κ é fixado pela geometria de margens do
> modelo denso. Arredondamento (RTN, GPTQ, AWQ), poda (magnitude, Wanda, SparseGPT), quantização de KV
> cache, remoção de camadas e checkpoints de terceiros caem na mesma taxa — e **ataques por gradiente
> também**: otimizar um delta em todas as camadas para maximizar flips a KL casado dá 1,13× a taxa dos
> compressores honestos em tokens não vistos, enquanto o mesmo otimizador ao contrário a corta pela
> metade. Essa taxa fica em ~0,17 do teto informacional de um perturbador livre por token, e a folga é
> decomposta: no máximo 2× é acessível a qualquer perturbação fixa (o fator 1/2 do sinal na previsão de
> 1ª ordem), e os 2,9× restantes exigem escolher o KL token a token.
>
> Consequências: (i) condicionado ao KL, a identidade do método não prevê flips; (ii) a fidelidade por
> token é previsível só a partir do modelo denso, e isso tem um consumidor exato — a aceitação da
> decodificação especulativa é prevista com ~2% de erro a partir de um único escalar, sem instanciar o
> sistema; (iii) a métrica por token **não** basta para decidir política de compressão: o ganho de um
> gate adaptativo por token medido em flips desaparece em geração real, onde a alocação estática de bits
> entre projeções empata ou vence.

O que a tese **não** afirma:
- que GPTQ e RTN "são a mesma coisa" — GPTQ produz **menos KL** nos mesmos bits; a equivalência é
  condicional ao KL;
- que a lei prevê acurácia de tarefa — ela prevê decisões por token, não trajetórias longas;
- que o teto de ~0,17 é um limite para compressão — ele mede quão **menos** dano os compressores
  causam em relação ao pior caso possível;
- que *nenhuma* perturbação escapa da taxa — a afirmação é sobre perturbações fixas de pesos, e o
  limite de 2× da 1ª ordem é explicitamente falsificável;
- que adaptatividade por token supera precisão mista estática — medido em geração real, não supera.

---

## 2. Contribuições e evidência

| # | contribuição | número-chave | status | script |
|---|---|---|---|---|
| C1 | Lei flips ≈ κ√KL em compressores reais | 15 referências, 377 configs; inclinação 0,478–0,544; R² ≥ 0,990 por modelo | ✅ | `ews_fidelity.py`, `ews_fid_analysis.py law` |
| C2 | Previsão de 1ª ordem sem parâmetros | flips = ρ_c(0)·E\|Δgap\|/2; obs/prev 0,96 (Mistral), 0,99 (Qwen3‑4B), 0,88 (Gemma‑12B), logits fp32 | ✅ | `ews_fid_analysis.py prop` |
| C3 | κ é geometria do modelo denso | κ ~ fração de gaps < 1 nat: r = 0,98 (18 refs); transfere para domínios sem reajuste, R² = 0,979 | ✅ | `ews_slope.py`, `ews_domain.py` |
| C4 | KL é fungível na arquitetura | κ varia ≤ 1,11–1,20× entre atenção/MLP × terços de profundidade (3 modelos) | ✅ | `ews_fungibility.py` |
| C5 | Taxa em ~0,17 do teto informacional, e a folga é decomposta | 0,174 do teto (62 pontos, 6 modelos); limite de 1ª ordem de qualquer perturbação fixa = 0,348 (fator 1/2 do sinal); melhor ataque medido 0,205; os 2,9× restantes exigem alocação por token | ✅ | `ews_flipdirs.py`, `ews_adv_report.py` |
| C6 | Equivalência condicional ao KL | TOST flips +0,10pp e MMLU‑PT +0,65pp **equivalentes**; GSM8K: os +4pp eram artefato do α e com controles nativos dão **−0,06pp [−2,30, +2,18], p_TOST = 0,016 → equivalente** (n=9) | ✅ | `ews_matched_kl.py`, `ews_crack.py` |
| C7 | Aceitação especulativa prevista **só pelo KL** | 24 pares em 5 alvos, sem instanciar o sistema: R² 0,952 (aceitação) e 0,941 (tokens/rodada), erro ~2% — melhor que usar o flip medido (0,940); com a sequência de TF: 0,986 / 0,980; amostragem (1−TV, identidade): 0,949 | ✅ | `ews_spec_law.py`, `ews_spec_bench.py` |
| C8 | ❌ O ganho adaptativo por token **não** transfere para geração | 1,5–4,4× menos flips em teacher forcing; em geração real, contra a fronteira estática no mesmo orçamento: −0,88pp [−1,97, +0,20] (contabilidade incremental) e −3,53pp [−5,50, −1,55] (cascata), n=24, 3 modelos | ❌ | `ews_adaptive_closedloop.py`, `ews_adaptive_report.py` |
| C9 | Higiene de avaliação | empates exatos em ~0,8% dos passos greedy (grade do bf16); viés de letra em MCQ explica erro da métrica bruta (Pearson −0,82) | ✅ | `ews_fid_analysis.py theory`, `ews_d2_search.py` |

---

## 3. Estrutura do paper

### Seção 1 — Introdução
Pergunta: métodos de compressão diferem na *forma* do dano ou só na *quantidade*? Resposta: na quantidade,
medida em KL; a forma é fixada pela geometria do modelo denso.

### Seção 2 — Medição
- Métrica primária: **flip** = mudança do top‑1 sob teacher forcing no corpus greedy da referência
  (GSM8K + MMLU‑PT, ~80–90k tokens por modelo). KL(p_ref ‖ p_cfg) por token, top‑64 + cauda.
- Validação: KL truncado = KL em vocabulário completo (razão mediana 1,003, p95 1,06).
- **Artefato bf16:** logits em bfloat16 caem numa grade de 0,0625–0,25 nat; ~0,8% dos passos têm
  empate exato top‑1/top‑2. É isso — não não determinismo — que gera o piso de ~0,5% de discordância
  e domina os flips em perturbações muito pequenas. Solução: lm_head em fp32 para a análise teórica.
- Compressores: implementações próprias (GPTQ, AWQ, SparseGPT, Wanda sequenciais; KV estilo KIVI) e
  checkpoints oficiais (Qwen2.5‑GPTQ‑Int4/Int8, Qwen3‑8B‑AWQ) como controle de terceiros.

### Seção 3 — A taxa de câmbio (C1–C4)
- **Figura 1:** log flips × log KL, 377 pontos, cor por família de compressor, uma reta por modelo.
- Mecanismo: proposição de 1ª ordem (C2). O expoente ½ vem de o fator de anisotropia
  A = E\|Δgap\|/√(2·KL) ser quase o mesmo para todas as perturbações.
- **Figura 2:** κ × fração de gaps < 1 nat, pontos por modelo e por domínio sobre a mesma reta (C3).
- Fungibilidade (C4): 1 nat de KL em atenção ou MLP, início ou fim, produz os mesmos flips (±20%).
- Exceção diagnosticável: modelos menores da mesma família ficam abaixo da reta por **descasamento de
  temperatura** (τ* 1,7–2,1); corrigido, voltam à reta (z ≈ −0,3). Checkpoints de treino não se
  distinguem da compressão.

### Seção 4 — Por que a taxa é essa (C5)
- Teto analítico: o KL mínimo para virar o token t é c_t = p₁ln(2p₁/(p₁+p₂)) + p₂ln(2p₂/(p₁+p₂)).
  Um perturbador livre por token gasta o orçamento nos tokens mais baratos.
- **Figura 3:** fração do teto × d_model, 13 modelos, faixa 0,16–0,22 sem tendência com escala.
- Direções de flip (u_t = W_U[i₁] − W_U[i₂]) têm rank efetivo 50–226, convergido com o número de
  tokens. O rank **não** é constante arquitetural (plano no Gemma, cresce no OLMo); o invariante é a
  fração do teto.
#### O que um atacante por gradiente consegue (P1)
A lei é empírica: vale para os compressores que a literatura usa. A pergunta que decide a seção é se
a taxa é propriedade dos **algoritmos** ou da **restrição** ("a perturbação é fixa nos pesos, a mesma
para todos os tokens"). Quatro perturbações, em ordem de poder (`ews_adversarial.py`,
`ews_adversarial_multi.py`, agregadas por `ews_adv_report.py`), todas medidas a KL casado e no mesmo
corpus, como κ em razão do κ dos compressores honestos do mesmo modelo:

| perturbação | liberdade dada | κ / κ_honesto |
|---|---|---|
| delta aleatório de baixo rank em todas as camadas | nenhuma otimização (n=16) | 1,10× [0,86, 1,28] |
| 1 camada, rank completo (`down_proj` final) | gradiente em cache de estados, 5 modelos | 0,95× [0,84, 1,07] |
| **multicamada** (todas as projeções de todas as camadas + peso da norma final) | gradiente pelo modelo inteiro, objetivo = maximizar log κ; 5 modelos × 2–3 orçamentos (n=13) | **1,13× IC95 [1,07, 1,18]** (faixa 0,94–1,30; cresce com o orçamento, ver abaixo) |
| mesma máquina, sinal invertido (**controle positivo**) | minimizar flips a KL fixo (n=12) | **0,51× IC95 [0,47, 0,54]** |
| multicamada com **rank completo** (698M parâmetros, 100% do decoder livre) | 400 e 1200 passos, dois lr | 0,96× e **0,70×** — não consegue superar o rank 16 |
| temperatura (referência analítica) | — | 0,06–0,42× |

Todos os números acima são **em tokens de teste** (sequências que o ataque nunca viu, split por passo
fixo), **pareados por KL** (cada ponto do ataque contra os compressores honestos medidos no mesmo KL,
±35%) e tomando o **melhor esforço do atacante** por célula modelo × orçamento.

A conclusão **não** é "nada quebra a taxa" — o ataque a supera em 13% [7%, 18%]. É mais forte e
quantitativa:
**a taxa é invariante a ±25% sobre dez famílias de compressores, deltas aleatórios e ataques por
gradiente, enquanto o mesmo otimizador com o sinal trocado a derruba 2×.** A assimetria é o resultado.

A assimetria tem uma explicação geométrica única, e ela usa a medida que já estava na seção: as
direções de flip u_t = W_U[i₁] − W_U[i₂] ocupam **rank efetivo 95–110** de 2560–4096 dimensões.
- **Derrubar a taxa é fácil**: um delta fixo pode viver quase todo no complemento ortogonal desse
  subespaço — gera KL em todos os logits e quase nenhum flip. Medido: 0,51× [0,47, 0,54], faixa
  0,39–0,58, e **não** precisa da escala de saída (temperatura é só o caso extremo conhecido desse
  mecanismo, não o mecanismo).
- **Subir a taxa é difícil**: exige alinhar com as u_t, que variam **por token** e são quase
  ortogonais entre si. Um único mapa linear fixo não se alinha com ~100 direções ao mesmo tempo.

E há uma decomposição de primeira ordem de *onde* cada lado ganha. A previsão é
flips = ρ_c(0)·E|Δgap|/2, o que dá exatamente **dois** parafusos para uma perturbação fixa:
a magnitude do deslocamento de gap por unidade de KL, A = E|Δgap|/√(2KL), e o **sinal** (o fator 1/2
supõe sinal aleatório, então alinhá-lo perfeitamente valeria 2×). Medidos nos cinco modelos:

| | A (magnitude) | fração de Δgap que fecha o gap | κ/κ_honesto |
|---|---|---|---|
| compressores honestos | 2,29–3,20 | 48,1–50,1% (aleatório, como a teoria supõe) | 1,00 |
| ataque **maligno** | 2,10–2,91 (**abaixo** dos honestos) | 54–58% | 1,13 |
| ataque **benigno** | 1,04–1,93 (muito abaixo) | 44–63% | 0,51 |

Duas leituras, e a segunda corrige uma interpretação intuitiva que não sobrevive aos dados:
- **O lado benigno é explicado pela magnitude.** O ataque benigno esmaga A para ~0,4 do valor honesto:
  ele põe o deslocamento no complemento ortogonal das direções de flip. κ cai junto.
- **O sinal não distingue os dois lados.** O braço benigno também sobe a fração de sinal (até 63%),
  logo "alinhar o sinal" **não** é o mecanismo do maligno — hipótese descartada. O maligno ganha
  apesar de ter A *menor* que os honestos, o que só é possível concentrando o deslocamento nos tokens
  de gap mais próximo de zero (dentro do conjunto de gap < 1 nat). É uma redistribuição, não uma
  amplificação, e é por isso que o ganho é de ~12% e não de 2×: para concentrar mais seria preciso
  um mapa que varie **por token**.
- Checagem que fecha uma objeção: o ataque **não** vive dos empates exatos do bf16. Repetido com
  lm_head em fp32 (sem empates) em Mistral‑7B, Qwen3‑4B e Gemma‑12B, a razão do maligno muda
  +0,02 [−0,03, +0,07] (n=6) e κ_honesto muda ≤ 0,006 (P5).

O ganho do ataque **cresce com o orçamento de KL**, e isso é medido, não ruído. Pareado por KL nos
cinco modelos: no KL 0,02 a razão é 0,94–1,10× (média 1,04), no KL 0,05 é 1,09–1,30× e no KL 0,10 é
1,14–1,24×. Três sementes em Qwen3‑4B e Gemma‑12B dão desvio de 0,007–0,033 e amplitude ≤ 0,07 no
maligno, uma ordem de grandeza menor que a diferença entre orçamentos — logo a tendência é real. A
leitura que entra no paper: **no KL baixo, onde a compressão útil vive, o ataque praticamente empata
com os compressores honestos**; ele só encosta nos 1,2–1,3× em KL alto, onde o dano já é grande. O
controle benigno é bem mais instável entre sementes (desvio 0,04–0,09), o que é esperado: há muitas
direções quase ortogonais às u_t e várias servem.

Isso decompõe a folga de 5,7× até o teto do oráculo (média de 62 pontos, 6 modelos):

| | fração do teto | interpretação |
|---|---|---|
| compressores honestos | 0,174 | o que se usa hoje |
| melhor ataque medido | 0,217 | +13% na média, +30% no melhor ponto |
| limite de 1ª ordem com sinal perfeitamente alinhado | 0,348 | **máximo de qualquer perturbação fixa** (remover o fator 1/2) |
| teto do oráculo por token | 1,000 | exige escolher o KL token a token |

Ou seja: dos 5,7× de folga, **no máximo 2× é acessível** a uma perturbação fixa de pesos (e medimos
1,19×); os 2,9× restantes exigem alocar KL token a token, o que uma perturbação fixa não pode fazer
por construção. Essa é a afirmação que o paper faz — e é falsificável: basta exibir uma perturbação
fixa que passe de 2×.

Checagens que precisam estar no apêndice, porque um revisor vai perguntar:
- o ataque **não** explora os empates exatos do bf16: em fp32 a razão κ/κ_honesto do maligno muda
  +0,02 [−0,03, +0,07], e a do benigno +0,05 [+0,01, +0,09] (3 modelos × 2 orçamentos,
  `ews_adv_report.py`, seção P5). A fração bruta de flips vindos de empates não serve para isso: o
  ataque fica em 4,5–13,5% contra 5,3–12,0% dos honestos, acima deles em Mistral e Gemma‑12B no KL 0,02;
- treino e teste por passo fixo dão o mesmo κ (0,247 vs 0,235): o ataque não decorou tokens;
- repetir a mesma semente não dá bit a bit igual (Gemma‑1B, KL 0,05: κ de teste 1,21× e 1,24×): o
  forward em GPU não é determinístico, e a dispersão entre sementes (≤ 0,07) já cobre isso;
- o ataque **não** depende de mexer no peso da norma final, liberdade que os compressores honestos não
  têm: sem ela o maligno fica igual ou acima (+0,01 a +0,04 em 3 modelos, P7);
- ‖ΔW‖/‖W‖ do ataque é ~1e‑4 a 4e‑3, **abaixo do passo de arredondamento relativo do bf16**, então o
  delta é aplicado em fp32 e o efeito de materializá-lo em bf16 é reportado à parte.

### Seção 5 — Equivalência condicional e o limite da previsão (C6)
- Desenho: cada método gera uma perturbação ΔW; α é ajustado até o mesmo KL alvo (0,02–0,20);
  teste de equivalência TOST pareado por modelo × alvo (4 modelos, 16 pares).
- Resultado: flips equivalentes; MMLU‑PT equivalente; GSM8K não (ver abaixo).
- **Limite honesto:** acurácia de tarefa em geração longa não segue a lei (mudança de resposta prevista
  com R² 0,36–0,39). Mesmo 8 bits muda 9–17% das respostas no GSM8K: um flip reescreve a trajetória.
  A lei governa um passo; trajetórias exigem avaliação ponta a ponta.

#### A rachadura resolvida: os +4pp da poda eram artefato do α ✅
Resultado original: poda − quantização = +4,03pp [IC +2,61, +5,46] no GSM8K com KL casado, enquanto
flips e MMLU‑PT eram equivalentes.

| hipótese | teste (`ews_crack.py`) | resultado |
|---|---|---|
| **H1** o KL casado no corpus misto não estava casado no GSM8K | recalcular KL só nos tokens de GSM8K e recasar (n=16, 4 modelos) | ❌ refutada: a alocação difere (poda põe 0,39–0,84 do KL misto no GSM8K, quantização 0,29–0,87), mas a diferença não muda: +4,03 → **+4,37pp** |
| **H2** escalar a perturbação de poda por α < 1 encolhe pesos em vez de zerá-los | controles **nativos** casados no KL de GSM8K — esparsidade real (magnitude, Wanda, SparseGPT) e precisão mista real na escada 8→6→5→4→3 bits — GSM8K n=400 (n=9 pares, 3 modelos, 45 execuções) | ✅ **confirmada**: poda − quantização = **−0,06pp** [IC −2,30, +2,18], p_TOST = 0,016 → **equivalente** |
| **H3** poda genuinamente menos danosa a raciocínio | não é necessária | descartada |

Consequência de método, que entra no paper: **o eixo de intensidade tem de ser o controle nativo do
método** (esparsidade, largura de bits), nunca uma interpolação linear da perturbação. Interpolar uma
poda produz encolhimento de pesos, que é outra perturbação.

Com os controles nativos o resultado deixa de ser "sem diferença detectável" e passa a ser
**equivalência positiva** (p_TOST = 0,016 contra margem de 3pp): no GSM8K, a KL casado, a família do
método (poda vs quantização) não determina a acurácia.

Ressalva que precisa ficar no paper: a equivalência vale para a **média da família**, e a variância
*dentro* dela é enorme. No Gemma‑3‑4B a KL = 0,10 a poda por magnitude derruba o GSM8K em 33,2pp
enquanto a Wanda sobe 4,7pp — no mesmo KL e no mesmo modelo. Isso é a mesma limitação de C6/§5 por
outro ângulo: o KL casa fidelidade por token, não trajetória. Métodos que concentram o erro em poucos
pesos muito grandes (magnitude) quebram a geração longa de um jeito que o KL médio não vê.

### Seção 6 — A lei em uso: escolher um draft de decodificação especulativa sem construir o sistema
Esta **não** é uma seção de "aplicações". É a prova de que a grandeza que a lei governa — fidelidade de
decisão por token — é exatamente a grandeza que um sistema real consome. A decodificação especulativa
aceita um token do draft se e só se ele concorda com o alvo (greedy) ou passa a regra de rejeição
(amostragem); ou seja, a taxa de aceitação **é** uma medida de flips, e tokens por rodada é a sua
tradução operacional.

A pergunta que um engenheiro faz antes de escrever qualquer código é: *quantos tokens esse draft vai
emplacar?* Três previsores respondem, do que usa menos informação ao que usa mais (`ews_spec_law.py`);
nenhum deles roda decodificação especulativa:

  | previsor | informação usada | aceitação/proposta | tokens/rodada |
  |---|---|---|---|
  | **lei** | um escalar (KL) + κ do alvo | R² 0,952, erro 2,2% | R² 0,941, erro 1,9% |
  | iid | flips medidos em TF | R² 0,940, erro 3,0% | — |
  | sequência | sequência de concordância do TF | R² 0,986, erro 1,3% | R² 0,980, erro 1,1% |
  | amostragem | 1 − TV (identidade de Leviathan) | R² 0,949, erro 1,3% | R² 0,887, erro 3,6% |

  (n = 24 pares alvo×draft em 5 alvos: Mistral‑7B, Phi‑3.5, Qwen3‑4B, Qwen3‑14B, Gemma‑3‑4B; drafts
  próprios quantizados/podados e modelos menores da mesma família; KL de 0,0005 a 0,41.)

  Detalhe que vale uma frase no paper: **a lei prevê melhor que o flip medido** (0,952 vs 0,940). O
  flip de um draft é uma média de Bernoulli ruidosa; passar pelo KL (uma média de uma quantidade
  contínua) e converter pela lei suaviza esse ruído.

  Nenhum deles roda decodificação especulativa. O previsor mais fraco — **um único escalar medido em
  teacher forcing** — acerta tokens/rodada com ~3% de erro.
- Duas armadilhas de medição que tinham de ser corrigidas: (i) comparar `aceitos/propostos` com
  `1 − flips` é inválido, porque as propostas depois da primeira rejeição na rodada são contadas e
  nunca aceitas (essa diluição é o que o previsor "sequência" reproduz); (ii) a previsão tem de usar o
  mesmo domínio dos prompts (usar o corpus misto errou ~0,7 token/rodada).
- **Onde a previsão para: relógio.** A lei prevê a grandeza estatística (aceitação, tokens/rodada) com
  ~2% de erro, mas converter isso em speedup exige um modelo de latência do stack. Com o modelo de
  custo ingênuo (bytes de peso do draft / bytes do alvo), a correlação com o relógio medido é fraca —
  Spearman +0,30 (Qwen3‑14B, 5 drafts) e +0,54 (Gemma‑12B, 6 drafts) — porque no lote 1 o forward do
  draft não é proporcional aos seus bytes (o overhead de lançamento domina no draft de 1B). Escolher o
  draft pela previsão acerta o melhor no Qwen3‑14B e perde 4,5% do melhor no Gemma‑12B
  (`ews_spec_law.py decision_table`). Relógio real medido: 1,18–1,64× no Gemma‑12B e 0,90–1,07× no
  Qwen3‑14B.
- Isso é uma **separação limpa de escopo**, e o paper deve dizê-la: a contribuição é prever *quantos
  tokens o draft emplaca*, não prever tempo de parede. Tempo de parede depende de kernels, de o alvo
  já ser rápido e do custo real do draft, nenhum dos quais é objeto desta teoria.
- **Por que isso é a prova de valor e não um brinde:** a alternativa a prever é medir, e medir exige
  instanciar o sistema (dois modelos na memória, cache duplo, regra de rejeição) e rodá-lo por draft
  candidato. Nós medimos **24 pares alvo×draft em 5 alvos** para *verificar* a previsão; quem só quer
  escolher o draft precisa de um forward de teacher forcing por candidato e nada mais.

#### Apêndice: higiene de avaliação (C9)
A acurácia bruta em MCQ mede deslocamento do prior de letra junto com dano; erro da métrica bruta × TV
do prior: Pearson −0,82 (33 configs, MMLU inglês). Selecionar camadas para poda pela NLL do gabarito
escolhe camadas prejudiciais. Entra como apêndice metodológico, não como contribuição.

### Seção 7 — Adaptatividade por token: um resultado negativo informativo
A taxa estática diz quais tokens são frágeis (margem baixa), e medir flips sugeria ganho grande em
gastar bits neles (1,5–4,4× menos flips no mesmo orçamento, em teacher forcing). **Em geração real
esse ganho não existe.**

Desenho (`ews_adaptive_closedloop.py`): duas cópias do modelo (base RTN 3 bits, alta 8 bits), KV cache
**misto e consistente** (o token promovido escreve K/V da cópia alta na mesma posição), limiares vindos
de quantis da entropia da cópia base, GSM8K n=400, 3 modelos. Duas políticas por token — cascata
(decide depois de ver a saída base) e preditiva (decide antes, sem refazer o forward) — contra a
fronteira **estática** de precisão mista entre projeções, interpolada no mesmo orçamento de bits
(`ews_adaptive_report.py`, 24 pares política × orçamento):

| contabilidade de bits | adaptativo − estático | leitura |
|---|---|---|
| **incremental** (só o residual de bits dos tokens promovidos: memória/IO) | **−0,88pp** [−1,97, +0,20] | sem diferença detectável |
| **cascata** (o token promovido paga base + alta: compute refeito) | **−3,53pp** [−5,50, −1,55] | adaptativo **perde** |

O ponto que torna o resultado defensável: mesmo na contabilidade **mais generosa possível** para a
adaptatividade — a que supõe que só o tráfego de pesos custa — não há ganho. Na contabilidade realista
de compute, perde. A política preditiva é igual ou pior que a cascata, o que descarta a hipótese de que
o custo do reforward fosse o problema.

O instrumento foi validado antes de acreditar no negativo (`tests/test_adaptive_closedloop.py`): nunca
promover reproduz a base **bit a bit**, e promover sempre com prefill em alta reproduz o bf16 **bit a
bit**. O resultado não é bug de cache.

Por que a métrica por token enganou: flips medem divergência contra a trajetória da *referência*, um
passo à frente. Numa geração real o modelo base processa todos os outros tokens em 3 bits e degrada os
estados ocultos de forma cumulativa; promover pontualmente o token de maior entropia não recupera isso.
Precisão mista estática melhora todos os tokens, inclusive o prefill.

Como isso entra no paper: como **limite de validade da métrica**, não como derrota. A afirmação é que
fidelidade por token e utilidade em geração são objetivos diferentes, e que otimizar a primeira não
implica ganho na segunda — o que também explica por que a previsão de acurácia pela lei falha
(R² 0,36–0,39) e por que a decodificação especulativa, que é exatamente um mecanismo de fidelidade por
token, é o consumidor natural desta teoria.

---

## 4. Programa de experimentos adaptativos: estado

Moeda: **bits médios lidos por token** (não KL — com liberdade por token dá para esconder KL onde não
vira decisão).

| id | pergunta | resultado | status |
|---|---|---|---|
| **A1** | ganho no mesmo orçamento de bits, medido em flips (teacher forcing) | gate por entropia da base: 1,5–4,4× menos flips que precisão mista estática interpolada, em 6 modelos | ✅ mas ver A3 |
| **A2** | o ganho depende do recurso escasso? | incremental (resíduo de planos de bits, memória/E‑S) 1,5–4,4×; cascata (computação) 0,86–2,3×, pior que estático em 2 de 6 modelos a 4 bits | ✅ |
| **A3** | **sobrevive à geração real?** | **não.** Com KV misto e consistente, precisão mista estática empata ou vence em Mistral‑7B, Qwen3‑4B e Gemma‑4B; cascata e preditiva indistinguíveis entre si | ❌ |
| **A4** | quanto do ganho é recuperável sem oráculo? | oráculo por sequência: só 1,1–1,4×; oráculo por token é trivial (promove o que vai errar) | ⚠️ perdeu relevância com A3 |
| **A5** | generaliza além de RTN (base GPTQ‑3, resíduo GPTQ‑4)? | pendente; só vale a pena se A3 for revertido com outra granularidade | ⏸ |
| **A6** | latência real de E/S | não executar: A3 removeu a premissa | ❌ cancelado |

Validação do instrumento de A3 (importante para o paper): sem promoção, a geração reproduz a base
**exatamente**; promovendo sempre com prefill em alta precisão, reproduz o bf16 **exatamente**
(`tests/test_adaptive_closedloop.py`). O resultado negativo não vem de bug de cache.

O que ainda poderia salvar a linha adaptativa (trabalho futuro honesto, não afirmação):
granularidade maior que o token (trechos ou sequências inteiras), decisão sobre *quais projeções*
promover em vez de quando, ou alvo de fidelidade por token — que é o caso da decodificação especulativa,
onde a teoria já entrega previsão útil (R² 0,952 só pelo KL).

Apêndice (não é resultado de compressão): adaptatividade **maligna** em α por token chega a 0,41 do teto
no Qwen3‑4B, contra 0,19 do estático — evidência de que a taxa fixa vem de a perturbação ser estática.

## 5. Pendências obrigatórias antes da submissão

Nenhuma aberta: P1–P5, P7 e P8 fechadas, P6 cancelada com justificativa.

| id | pendência | estado | por quê |
|---|---|---|---|
| P1 | Ataque adversarial multicamada | ✅ feito (`ews_adversarial_multi.py`) | virou o resultado da §4: taxa invariante a ±20%, assimetria com controle positivo, decomposição da folga |
| P2 | Aumentar n do TOST em tarefa com controles nativos | ✅ feito | com controles nativos deu equivalência positiva (p_TOST = 0,016, n=9) |
| P3 | Mais alvos na especulativa | ✅ feito (5 alvos, 24 pares) | previsão só pelo KL: R² 0,952 / 0,941 (aceitação / tokens por rodada) |
| P4 | Figuras | ✅ 6 figuras em `results/figs/` | fig1 lei + ataques + teto; fig2 κ×geometria; fig3 teto; fig4 especulativa; fig5 adaptativo; fig6 trajetória do ataque |
| P5 | Repetir o ataque com logits em fp32 | ✅ feito (Mistral‑7B, Qwen3‑4B, Gemma‑12B; bf16 e fp32 no KL 0,02 e 0,05) | hipótese refutada: κ_honesto de teste muda ≤ 0,006 em fp32 e a razão do maligno muda +0,02 [−0,03, +0,07]. O 0,93× no KL 0,02 **não** é artefato de precisão nem variância de semente (ver P8): é a ponta baixa de uma tendência com o orçamento |
| P6 | Relógio real com self-drafts de 14B | ⏸ cancelado | o u4 foi medido e u3/gptq3/wanda50 deram falta de memória; drafts simulados em bf16 têm razão de bytes 1,0 e não podem dar speedup, e a §6 já declara o relógio fora do escopo |
| P7 | Ablação do ataque sem acesso à norma final | ✅ feito (Mistral‑7B, Qwen3‑4B, Gemma‑1B, KL 0,05) | o 1,13× **não** depende dessa liberdade: sem a norma o maligno fica igual ou acima (+0,01 a +0,04). O benigno é que usa a norma, e só em um modelo (Gemma‑1B −0,13) |
| P8 | Variância entre sementes do ataque | ✅ feito (3 sementes em Qwen3‑4B e Gemma‑12B, 2 no Gemma‑1B) | o maligno é estável (desvio 0,007–0,033, amplitude ≤ 0,07) e o benigno não (desvio 0,04–0,09). Isso é o que descarta "ruído" como explicação da tendência com o orçamento |

---

## 6. Descartado (não usar no paper)

| afirmação | por que caiu |
|---|---|
| "√ΣKL por camada prevê flips melhor que o KL real" | interação entre camadas mudaca de sinal por modelo (soma/conjunto: OLMo‑1B ×0,76, Qwen3‑4B ×1,18) |
| "O rank baixo é uma constante arquitetural do Transformer" | rank vai de 50 a 226; o invariante é a fração do teto |
| "A compressão estática tem um teto de vidro de 19%" | 0,17 é fração do **pior caso**; significa menos dano, não limite de compressão. O teto real de uma perturbação fixa é 2× a taxa honesta (fator do sinal), não 1/0,17 |
| "Adaptatividade rompe os 19% e é a única fundação viável" | romper para cima = mais flips; em custo de computação o ganho some e a especulativa sem perda domina |
| "κ é uma constante universal" | κ varia 0,15–0,36 entre modelos e domínios; o que é universal é a relação κ ↔ geometria de margens. Dentro de um modelo, é invariante a ±20% sobre famílias de compressores, deltas aleatórios e ataques otimizados |
| "Nem um ataque adversarial escapa da taxa" | o ataque multicamada supera em 1,13× [1,07, 1,18] em teste e até ~1,3× por ponto; a afirmação certa é a **invariância a ±20%** mais o limite de 2× da 1ª ordem |
| ~~"Prever aceitação especulativa com R² > 0,95 só pelo KL"~~ | **reabilitada**: a comparação antiga era inválida (media `aceitos/propostos`, que inclui as propostas pós‑rejeição, contra `1−flips`). Com grandezas iguais e domínio casado: R² 0,952 só pelo KL, 0,986 com a sequência de TF (n=24, 5 alvos) |
| "Checkpoints de treino têm assinatura sub-isotrópica" | indistinguíveis da compressão no mesmo KL |
| "Comprimido supera bf16 no MMLU" | deslocamento do prior de letra (Gemma PT‑BR); não replica em inglês |
| "Poda preserva raciocínio melhor que quantização no mesmo KL" | artefato de escalar a perturbação por α; com controles nativos o efeito inverte e perde significância |
| "Gate adaptativo por token reduz o dano de compressão" | vale em flips (teacher forcing) e desaparece em geração real (A3) |
| Elastic depth como mecanismo | Fase 1: 2% de economia atribuível à heterogeneidade |

---

## 7. Vocabulário e defesa

- Sempre "equivalentes **condicionados ao KL**". GPTQ/AWQ são superiores porque produzem menos KL por bit.
- Não escrever números esperados antes de rodar.
- Declarar a aproximação de teacher forcing em todo resultado por token e em A1–A2.
- Declarar que quantização e poda são simuladas em bf16; relógio real só para drafts menores.
- "Pinsker + Taylor" é a objeção esperada: responder com a previsão sem parâmetros (C2), o teto analítico
  (C5) e — só depois de P1 — o ataque.

---

## 8. Mapa de scripts e dados

O codigo esta reestruturado como pacote (`ews/`), com um ponto de entrada unico: `python -m ews
<estagio>` (ou `make <alvo>`). Ver [README.md](../README.md) para a estrutura e as variaveis de
ambiente. Resultados em `$EWS_RESULTS` (default `/local/$USER/ews_results/fid`; a cota do /raid esta
na memoria do projeto).

| estagio (`ews ...`) | modulo | produz | pasta |
|---|---|---|---|
| `grid gen` / `grid score` | `ews.pipelines.fidelity.build_grid` | corpora, grade por modelo (flips, KL, TV, logprobs nos top-64) | `corpora/`, `<ref>/<modelo>/*.pt` |
| `analyze` | `ews.pipelines.fidelity.analyze` | `law`, `theory`, `prop`, `d3`, `d4`, `d2`, `closedloop` | `analysis/*.json` |
| `slope`, `domain` | `ews.pipelines.fidelity.{slope,domain}` | κ × geometria, dominios | `analysis/slope.json`, `analysis/domain.json` |
| `flip-dirs` | `ews.pipelines.fidelity.flip_dirs` | rank das direcoes de flip, teto analitico | `analysis/flipdirs.json` |
| `fungibility` | `ews.pipelines.fidelity.fungibility` | κ por subconjunto com KL casado | `fungibility/` |
| `mcq-permutation` | `ews.pipelines.fidelity.mcq_permutation` | MCQ invariante a permutacao (D2) | `d2/` |
| `kl-greedy` | `ews.pipelines.fidelity.kl_greedy` | quantizador KL-guloso | `klgreedy/` |
| `adv-single` | `ews.pipelines.adversarial.single_layer` | ataque de 1 camada, rank completo, KL casado | `adversarial/` |
| `adv-multi` | `ews.pipelines.adversarial.multi_layer` | **ataque multicamada** (P1): maligno/benigno, rank 16 ou completo, com/sem escala de saida, fp32 e `--seed` | `adversarial/<ref>/results_<tag>.json`, `fid_<tag>_*.pt` |
| `adv-report` | `ews.pipelines.adversarial.report` | consolida a §4: κ pareado por KL em teste, anisotropia, sinal, decomposicao do teto, `fp32_check()` (P5) e `seed_check()` (P7/P8) | `analysis/adversarial.json` |
| `matched-kl` | `ews.pipelines.tasks.matched_kl` | equivalencia com KL casado + TOST | `matched_kl/`, `analysis/matched_kl.json` |
| `crack` | `ews.pipelines.tasks.crack_gsm8k` | rachadura do GSM8K (H1/H2/H3) | `crack/` |
| `closedloop` | `ews.pipelines.tasks.closedloop` | GSM8K real e especulativa real | `closedloop/` |
| `spec-bench` | `ews.pipelines.speculative.bench` | especulativa real com relogio, previsao × medicao | `spec_bench/` |
| `spec-law` | `ews.pipelines.speculative.law` | **C7**: os tres previsores (lei / iid / sequencia); `decision_table()` | `analysis/spec_law.json` |
| `adaptive-alpha` | `ews.pipelines.adaptive.alpha` | adaptativo em α (apendice maligno) | `adaptive/` |
| `adaptive-bits` | `ews.pipelines.adaptive.bits` | A1–A2 | `analysis/adaptive_bits.json` |
| `adaptive-closedloop` | `ews.pipelines.adaptive.closedloop` | A3 (KV misto, politicas por token) | `adaptive_closedloop/` |
| `adaptive-report` | `ews.pipelines.adaptive.report` | A3 contra a fronteira estatica interpolada | `analysis/adaptive_closedloop.json` |
| `figures` | `ews.pipelines.report.figures` | as 6 figuras do paper | `$EWS_FIGS/*.pdf` |
| `legacy-*` | `ews.pipelines.legacy.*` | Fase 1 (elastic depth, descartada): oraculo, tokens, malha fechada, diagrama de fases | `results/raw/` |

**Corpora.** O corpus de referencia e um id (`ews/corpora/registry.py`): `mix` (GSM8K + MMLU PT-BR,
o dos resultados deste documento), `gsm8k`, `mmlu_pt`, `mmlu_en`, `wikitext`. Ele entra no nome da
referencia no disco — `mix` sem sufixo, os outros como `Qwen3-4B__mmlu_en` — e vale como um eixo de
generalizacao ainda **nao medido**: os numeros do paper sao todos em `mix`.
