# Resultados

Consulta única ao teste final em **18/10/2026** (protocolo congelado, SHA-256 `4a2ef2442edd9f3ce943ad454e7c67d74a29c5d407d523c1b378eb484156e94f`): 452 vídeos, 16 canais, três sub-blocos temporais. As tabelas abaixo são **geradas** dos JSON da avaliação final por `VERIFICACOES/distribuicao/gerar_tabelas_docs.py` (o comentário no início de cada bloco traz o SHA-256 dos JSON de origem); nenhum número foi digitado à mão. O teste foi lido uma única vez e nada foi ajustado depois.

**Como ler.** O erro é o MAE em escala de log (`log(1 + curtidas)`): 1,126 significa errar, em média, por um fator de cerca de 3. O acerto par a par vale dentro do mesmo canal, em pares distinguíveis (a maior contagem é pelo menos 1,2 vezes a menor); 50% é o acaso. O IC de 95% vem de 2.000 reamostragens de canais. "Favorável" quer dizer que o IC inteiro está do lado do modelo; "inconclusivo", que o IC inclui zero, o que **não** prova ausência de diferença.

## Resumo honesto

- O modelo completo de vídeo é **melhor que a média aritmética histórica** (a referência mínima do TAPI) em erro e em acerto par a par.
- Contra a **mediana** a vantagem em erro é pequena (−0,094) e o IC não inclui zero; contra a **média móvel exponencial** o erro é **inconclusivo**. O acerto par a par é favorável contra as três referências. O Spearman intra-canal (diferença na mediana entre canais) é favorável só contra a média aritmética (+0,366; IC +0,134 a +0,573); contra a mediana (+0,277; IC −0,003 a +0,505) e a média móvel (+0,279; IC −0,071 a +0,508) é inconclusivo.
- A **faixa de 80% cobriu 64,2%**, abaixo do pretendido e sem garantia. Pior em beleza (50,0%, só 4 canais, IC largo) e na dobra mais recente (49,7%).
- Em **canais não vistos** o erro sobe (1,642) e a vantagem em erro sobre a mediana e a média móvel desaparece.
- Modelos de **imagem** e de **texto** melhoram o erro frente à média aritmética, mas **não demonstram ganho de ordenação**.
- **Ablação:** legenda e modelo completo reduzem o erro; a capa sozinha é inconclusiva; **nenhum contraste de ordenação é conclusivo** e a ablação não detecta efeitos pequenos (menor efeito detectável de cerca de 0,21 no Spearman intra-canal, contra δ = 0,02 pré-definido).
- Baseline 4 (regressão sobre inscritos atuais, só no relatório): MAE 1,749; o modelo é favorável em erro e em acerto par a par; ele atribui um valor por canal e portanto não ordena vídeos do mesmo canal.
- As métricas descrevem o **procedimento** de avaliação. O pacote servido (`final=v1`) foi retreinado com todos os vídeos anotados, inclusive os de teste; seu desempenho em vídeos novos **não foi medido**.

## Tabelas

<!-- gerado por VERIFICACOES/distribuicao/gerar_tabelas_docs.py; principal.json 4216e90c5d375f62ac5c1f11d3cef54a7b8a494f5b518440ac91234012b5915f; canais_nao_vistos.json 050557f249bc06956abc04dc2a94e4994d8f7dd6ec02148609b8c4623b094665; ablacao.json 6f696114439be9d08e1ff7c26a9f408878959d76237e31ccea3c40b0282f2c4f -->

Avaliação temporal principal: 452 vídeos de teste, 16 canais.

| | MAE em log | Acerto par a par (pares distinguíveis) | Spearman intra-canal (mediana) |
|---|---|---|---|
| **modelo completo de vídeo** | **1,126** | **58,6%** | **0,314** |
| média aritmética histórica | 1,637 | 51,2% | -0,030 |
| mediana histórica | 1,220 | 51,3% | 0,058 |
| média móvel exponencial | 1,207 | 51,9% | 0,056 |

Contrastes pareados por canal (bootstrap de 2.000 reamostragens, semente 20261015):

| Contra | MAE em log: modelo − referência (IC 95% por canal) | Leitura | Par a par: diferença (IC 95%) | Leitura |
|---|---|---|---|---|
| média aritmética histórica | -0,511 [-0,822; -0,305] | favorável ao modelo | 0,074 [0,030; 0,159] | favorável ao modelo |
| mediana histórica | -0,094 [-0,168; -0,025] | favorável ao modelo | 0,073 [0,025; 0,156] | favorável ao modelo |
| média móvel exponencial | -0,080 [-0,209; 0,001] | inconclusivo | 0,066 [0,011; 0,175] | favorável ao modelo |

Os três modelos e a cobertura da faixa:

| Modelo | MAE em log | Acerto par a par | Spearman intra-canal (mediana) | Cobertura da faixa (intervalo pretendido de 80%) |
|---|---|---|---|---|
| modelo completo de vídeo | 1,126 | 58,6% | 0,314 | 64,2% (290 de 452; IC 95% por canal 50,3% a 79,8%) |
| modelo de imagem | 1,181 | 50,2% | 0,150 | 63,1% (285 de 452; IC 95% por canal 50,5% a 77,6%) |
| modelo de texto | 1,166 | 47,5% | -0,012 | 63,1% (285 de 452; IC 95% por canal 49,7% a 78,1%) |

| Setor (modelo completo de vídeo) | Cobertura | Canais | Observação |
|---|---|---|---|
| automotivo | 68,5% (237 de 346) IC 55,8% a 81,2% | 12 |  |
| beleza | 50,0% (53 de 106) IC 27,0% a 84,5% | 4 | IC aproximado: poucos canais no setor; o percentil de poucos grupos tende a ser estreito demais |

| Dobra do teste | Cobertura da faixa (modelo completo de vídeo) | MAE em log |
|---|---|---|
| teste_1 | 66,7% | 1,046 |
| teste_2 | 75,7% | 0,876 |
| teste_3 | 49,7% | 1,469 |

Canais não vistos:

Três sorteios de 6 canais retidos; 384 vídeos únicos em 12 canais.

| | MAE em log | Acerto par a par | Spearman intra-canal (mediana) |
|---|---|---|---|
| **modelo completo de vídeo** | **1,642** | **56,8%** | **0,147** |
| média aritmética histórica | 2,082 | 51,0% | -0,071 |
| mediana histórica | 1,706 | 49,7% | -0,015 |
| média móvel exponencial | 1,618 | 49,7% | -0,007 |

Ablação multimodal:

Corte único, 929 linhas de treino e 452 de avaliação, as mesmas em todos os níveis.

| Nível | MAE em log | Acerto par a par | Spearman intra-canal (mediana) |
|---|---|---|---|
| E1: histórico | 1,309 | 50,2% | 0,061 |
| E2: histórico e legenda | 1,201 | 51,7% | 0,053 |
| E3: histórico, legenda e capa | 1,185 | 52,7% | 0,057 |
| E4: modelo completo | 1,161 | 54,3% | 0,225 |

| Contraste | MAE em log (IC 95%) | Leitura do MAE | Spearman intra-canal (IC 95%) | Leitura da ordenação |
|---|---|---|---|---|
| E2 − E1 | -0,108 [-0,186; -0,034] | favorável ao nível mais completo | [-0,236; 0,140] | inconclusivo |
| E3 − E2 | -0,017 [-0,057; 0,030] | inconclusivo | [-0,238; 0,231] | inconclusivo |
| E4 − E3 | -0,024 [-0,040; -0,010] | favorável ao nível mais completo | [-0,095; 0,329] | inconclusivo |


## Notas

- **Dobras:** o procedimento usa três treinos expansivos (929, 1.066 e 1.240 linhas) e avalia 153, 152 e 147 vídeos. A ablação usa corte único (929 linhas de treino, 452 de avaliação), por isso o MAE do nível E4 (1,161) difere do MAE do modelo servido nas três dobras (1,126); os dois conjuntos são reportados separados.
- **Spearman intra-canal** usa só as células canal × sub-bloco com pelo menos 5 vídeos; as indefinidas não viram zero.
- **MAPE adaptado** é alto (razão 8,2) por vídeos com poucas curtidas e não é usado como critério.
- **F1 macro em 3 faixas:** 0,421 contra 0,333 do acaso; R² do alvo relativo 0,107.
- **Sensibilidade (exploratória):** histórico com defasagem de 90 dias (MAE 1,132) e intervalo treino–teste de 90 dias (MAE 1,151) ficaram inconclusivos contra o principal; a estabilidade é compatível com um efeito pequeno da aproximação, não é prova.
- **Limiar de baixa confiança** do vídeo (2,9733 em log, n = 100): não foi avaliado no teste; o diagnóstico do pacote de desenvolvimento foi inconclusivo (erro médio 2,283 nos sinalizados contra 0,894; IC 95% da diferença de −0,148 a 2,215).
- Documentos de decisão e protocolo completos ficam com o autor; ver [registro_de_experimentos.md](registro_de_experimentos.md).
