# CaseGalaxy: agente preditivo de curtidas em YouTube Shorts

Projeto individual do Inteli Academy x Galaxies. Dado um criativo (vídeo, imagem de capa ou texto de legenda) de um canal de **beleza** ou **automotivo**, o agente devolve a quantidade de curtidas que o Short tende a acumular, uma faixa de 80% pretendidos, a comparação com o histórico do canal, os fatores que mais pesaram, avisos de baixa confiança e a versão e a cobertura do modelo que respondeu. Também ordena de 2 a 5 criativos do **mesmo canal**.

> **Leia antes de usar os números.** O modelo vence a média histórica aritmética, mas contra a mediana e a média móvel exponencial a vantagem em erro é pequena ou inconclusiva. A faixa de 80% cobriu **64,2%** dos vídeos de teste (não 80%) e não tem cobertura garantida. A base é pequena e concentrada (16 canais, 82% automotivo). A Camada 1 local do TAPI §6.3 **não é atendida** (desvio declarado e não resolvido). Não há promessa de percentual de acerto nem de conformidade integral ao TAPI. Detalhes em [Limitações](#8-limitações).

Índice: [1 Problema](#1-problema-e-recorte) · [2 Base](#2-base) · [3 Arquitetura](#3-arquitetura) · [4 Instalação e execução](#4-instalação-e-execução) · [5 Modelos](#5-modelos-e-configurações) · [6 Metodologia](#6-metodologia) · [7 Resultados](#7-resultados) · [8 Limitações](#8-limitações) · [9 Evolução](#9-evolução)

## 1. Problema e recorte

- **Pergunta:** quantas curtidas um Short de uma marca ou criador tende a acumular, e entre dois criativos do mesmo canal, qual tende a ir melhor?
- **Alvo:** `log(1 + curtidas)` de Shorts com pelo menos 30 dias, **relativo ao canal**: o que o vídeo faz acima ou abaixo da mediana dos últimos 20 posts do canal publicados até 30 dias antes. A saída principal para decisão é essa razão e o ranking dentro do canal. O valor absoluto é uma estimativa e não uma previsão exata aos 30 dias.
- **Recorte:** Shorts (até 180 s, vertical ou quadrado), canais do Brasil, setores beleza e automotivo (outros setores, como moda, são recusados com erro explícito). Sem tela, sem mídia paga, sem público específico de cliente, sem previsão de visualizações.
- **Entradas aceitas:** URL pública de Short do YouTube, arquivo de vídeo local, imagem (interpretada como a capa planejada) e texto (interpretado como a legenda planejada). Cada uma é atendida pelo seu próprio modelo (seção 5).
- **O que não afirma:** causalidade. Os fatores mostrados descrevem o modelo, não o efeito de mudar o criativo.

## 2. Base

| | |
|---|---|
| Origem | YouTube Data API v3, coleta própria e única (snapshot `20261014T021437Z`, coletado em 2026-10-14T02:14:37Z). Nenhum vídeo ou áudio é baixado do YouTube. |
| Canais | 16 elegíveis (12 automotivo, 4 beleza), de uma lista de 30 canais aprovados pela identidade (site oficial liga ao canal) e filtrados por país Brasil (três exceções documentadas para canais de atuação brasileira). |
| Período | Alvos: Shorts publicados nos 365 dias até a coleta, menos os 30 últimos dias. Histórico: mais 183 dias anteriores. Amostra LIM (menos de 30 dias): só para um limiar de largura da faixa, sem rótulos. |
| Anotados | 1.500 vídeos-alvo (automotivo 1.227, beleza 273) anotados pelo Gemini, mais 100 da LIM. |
| Rótulo | Curtidas lidas **na data da coleta** (avaliação retrospectiva aproximada; ver Limitações). |
| Licença e termos | Dados da API do YouTube e saídas do Gemini: sem licença de redistribuição presumida. Retenção de **30 dias** (menor prazo desta entrega: **2026-11-03T23:46:26Z**). Ver [docs/dados_e_retencao.md](docs/dados_e_retencao.md). |
| Limites | 16 canais (muito abaixo da meta de 60+ do briefing, declarado); 82% automotivo; beleza com 4 canais; uma coleta, sem recoleta. |

O que a distribuição leva e por quanto tempo está em [docs/dados_e_retencao.md](docs/dados_e_retencao.md). **Pedro, autor, pode estender os prazos se a regra de origem permitir; nenhuma permissão indefinida é presumida.**

## 3. Arquitetura

```
criativo ─► MCP (stdio) ─► prever / ordenar ─► grafo fixo (LangGraph)
                                                 validar_entrada
                                                 1 extrair_atributos   (Gemini observa; capa e legenda locais)
                                                 2 historico_e_comparaveis (YouTube/base; 3 do canal + 2 do setor)
                                                 3 prever_modelo       (LightGBM q10/q50/q90, um modelo por modalidade)
                                                 avaliar_confianca
                                                 4 explicar            (TreeSHAP + redator + validador)
                                                 montar_saida          (5 ordenar: só mesmo canal)
```

- **O Gemini observa, o LightGBM calcula, o LangGraph organiza, o SHAP e o redator explicam, o MCP expõe.** Todo número vem de modelo salvo; o único nó com LLM só redige a explicação e o validador reprova frases com números inventados.
- Uma função (`from src.agente.api import prever, ordenar`) e o servidor MCP chamam a mesma lógica; os testes conferem que dão os mesmos números.
- Falhas viram erro ou aviso tipado, nunca previsão calada: se o Gemini falha, o vídeo cai no **modelo de imagem** (a saída diz isso, traz a versão e a cobertura do modelo de imagem e baixa confiança). Detalhes em [docs/arquitetura.md](docs/arquitetura.md).

**Tecnologias e por quê** (o TAPI §8 diz que nada é obrigatório e que a escolha deve ser justificada):

| Escolha | Motivo |
|---|---|
| Python 3.12 com `uv` e `uv.lock` | ambiente reproduzível e versões fixadas; torch só na versão CPU |
| LightGBM (quantis 0,1/0,5/0,9) | dados tabulares e pequenos, ausências tratadas nativamente, treino determinístico em CPU, intervalo por quantis sem calibração extra, TreeSHAP embutido (`pred_contrib`, o pacote `shap` não é usado) |
| scikit-learn | pré-processamento e PCA dos embeddings, ajustados só no treino |
| Gemini (`gemini-3.5-flash-lite`, URL pública) | lê o vídeo sem baixá-lo do YouTube e devolve rubrica e transcrição; substitui o Whisper (desvio declarado) |
| e5-small multilíngue, CLIP ViT-B/32, EasyOCR, MediaPipe, OpenCV | atributos locais de legenda e capa, em CPU |
| LangGraph | grafo fixo e testável (não é agente livre) |
| MCP SDK (stdio) | a superfície que o agente da Galaxies consome |
| Pydantic | contratos de entrada e saída |
| Parquet/JSON com SHA-256 | rastreabilidade (ver "Registro de experimentos" na seção 6) |

Não usados: MLflow, faster-whisper, SigLIP 2, YOLO, PySceneDetect, conformal/CQR, RAG e banco vetorial, tela.

## 4. Instalação e execução

Pré-requisitos: Linux (testado em Ubuntu sob WSL2), [`uv`](https://docs.astral.sh/uv/), Python 3.12 (o `uv` instala), cerca de 3 GB de disco para o ambiente e 1,2 GB para os pesos dos codificadores, e as bibliotecas do sistema `libgomp1` (LightGBM) e `libgles2`/`libegl1` (MediaPipe) (nomes Debian/Ubuntu, **não verificados em máquina limpa**). Se faltarem, `uv run casegalaxy-demo --verificar-ambiente` diz; uma pasta com as `.so` pode ser indicada por `LD_LIBRARY_PATH`.

Os comandos abaixo valem a partir da pasta `Entrega Galaxy/` e não dependem de caminho absoluto do autor.

```bash
# 1. ambiente travado
uv sync --locked

# 2. chaves (opcionais offline; só o modo online do MCP e a coleta as usam)
cp .env.example .env        # preencha YOUTUBE_API_KEY e GEMINI_API_KEY apenas se for usar o modo online

# 3. distribuição de modelos e dados de serviço (arquivo fornecido pelo autor, fora do Git)
tar -xzf casegalaxy_distribuicao_v1.tar.gz casegalaxy_distribuicao_v1/verificar_distribuicao.py
uv run python casegalaxy_distribuicao_v1/verificar_distribuicao.py instalar casegalaxy_distribuicao_v1.tar.gz --destino .

# 4. pesos dos codificadores (não vão na distribuição)
uv run casegalaxy-demo --verificar-ambiente         # somente leitura: lista o que falta, com tamanho e SHA-256
uv run casegalaxy-demo --baixar-codificadores       # opt-in, só o que falta; pede rede

# 5. demonstração offline (6 criativos do de-para, sem rede)
uv run casegalaxy-demo
uv run casegalaxy-demo --mostrar-salvas             # plano B: relê as respostas salvas, sem carregar modelos

# 6. servidor MCP (stdio; offline por padrão)
uv run casegalaxy-mcp
```

### Chaves de API

O `.env` (ignorado pelo Git) aceita `YOUTUBE_API_KEY` (Google Cloud, YouTube Data API v3, cota de 10.000 unidades por dia) e `GEMINI_API_KEY` (Google AI Studio, nível gratuito: 15 requisições por minuto, 500 por dia e 28.800 s de vídeo por dia, premissa operacional contada pelo dia do Pacífico). O servidor MCP só liga o modo online com `CASEGALAXY_MCP_ONLINE=1` e `CASEGALAXY_MCP_LIVRO=<arquivo do livro de consumo>`; tetos por execução em [docs/uso.md](docs/uso.md). **A demonstração nunca faz chamada online.**

### Offline e online

Offline, o agente responde com modelos, base curada e caches locais; vídeo por URL só tem os atributos do Gemini se a anotação estiver no cache (a distribuição leva as anotações dos 6 vídeos do de-para, não as dos 1.500 alvos). Sem anotação, o vídeo cai no modelo de imagem se houver capa; sem anotação e sem capa dá erro, para não responder só com o histórico. Online, o servidor anota vídeos novos por URL (consome cota), envia vídeo local pela Files API e consulta o histórico do canal.
- **Reproduzir o treino** (precisa de chaves, de vários dias de cota do Gemini e gera outra base): [docs/reproduzir_o_treino.md](docs/reproduzir_o_treino.md). Os testes e as verificações de desenvolvimento ficam fora deste repositório e não são entregues.
- Guia de uso, formatos de pedido e resposta, MCP e exemplos: [docs/uso.md](docs/uso.md). Demonstração: [docs/demonstracao.md](docs/demonstracao.md).

## 5. Modelos e configurações

Versão ativa: **`final=v1`** (manifesto `data/models/pacote_servido=v1.json`), refit final com os 1.500 vídeos-alvo anotados (LIM fora). LightGBM, configuração A fixa: 300 árvores, taxa de aprendizado 0,05, 7 folhas, `min_child_samples` 10, semente 20261015. Sem busca nem calibração no refit.

| Modalidade | Modelo (nome comum) | Versão servida | Atributos |
|---|---|---|---|
| vídeo | modelo completo de vídeo (N4) | `final=v1/video-6bb801f43992-c62b81db` | histórico, legenda, capa, rubrica e transcrição do Gemini, duração |
| imagem | modelo de imagem (N3) | `final=v1/imagem-361551e0e0f7-c62b81db` | histórico, legenda, capa (também é o modelo de contingência do vídeo) |
| texto | modelo de texto (N2) | `final=v1/texto-05fb7d7a5f8f-c62b81db` | histórico e legenda |

- Cada resposta traz **a versão e a cobertura do próprio modelo**. A cobertura exibida é a do **procedimento de avaliação** (não foi medida no artefato servido, porque ele foi retreinado com todos os dados) e **não é garantida**: modelo completo de vídeo 64,2%, imagem 63,1%, texto 63,1%, para um nível pretendido de 80%.
- **Faixa:** o q50 é sempre o valor central; os limites são o mínimo e o máximo das três saídas (q10, q50, q90), não negativas.
- **Baixa confiança:** imagem, texto e contingência sempre avisam "evidência limitada". No vídeo há também uma regra pela largura da faixa: limiar **2,9733** em log (percentil 80 das larguras na LIM, n = 100), vinculado ao pacote servido. A validade dessa regra no pacote servido é **presumida**; no pacote de desenvolvimento o diagnóstico foi **inconclusivo** (erro médio 2,283 nos sinalizados contra 0,894; IC 95% da diferença de −0,148 a 2,215). A regra não garante cobertura nem desempenho.
- **Rollback** do pacote ativo: `uv run casegalaxy-pacote-servido reverter --motivo "..."` (a distribuição não leva os pacotes de desenvolvimento, então só funciona na máquina do autor).

## 6. Metodologia

- **Partição temporal por contagem, em ordem de publicação:** 55% treino, 15% validação, 30% teste em três sub-blocos, com intervalo de 30 dias antes de cada bloco avaliado. Canais não vistos: 30% dos canais retidos, em três sorteios. Divisão aleatória nunca é resultado.
- **Contra o vazamento:** estatísticas do próprio vídeo nunca entram; inscritos e totais do canal ficam fora; histórico só com posts até 30 dias antes; PCA, tercis e referência do setor ajustados só no treino de cada dobra. Aproximações declaradas: as curtidas do histórico e dos rótulos foram lidas na data da coleta, não na data simulada.
- **Referências:** (1) média histórica aritmética (o mínimo do TAPI), (2) mediana, (3) média móvel exponencial, (4) regressão sobre inscritos atuais (só no relatório). Todas com os últimos 20 posts elegíveis.
- **Métricas:** MAE e RMSE em log, MAPE adaptado, Spearman global e intra-canal, acerto par a par dentro do canal, cobertura da faixa por modelo e por setor, com IC 95% por bootstrap de canais (2.000 reamostragens, semente 20261015).
- **Teste final:** protocolo congelado com SHA-256 antes da consulta, **consulta única** em 18/10/2026 aos 452 vídeos de teste, depois nenhum ajuste. A seleção do modelo (LightGBM configuração A, sem CatBoost) ocorreu antes, na validação.
- **Ablação multimodal** (histórico, histórico e legenda, mais capa, modelo completo) em corte único, com as mesmas linhas.
- **Registro de experimentos (equivalente ao MLflow).** O projeto **não instala MLflow**. O registro é feito por manifestos com SHA-256, protocolo congelado com hash, partição e amostra congeladas e registros de decisão. O que a pessoa avaliadora **pode abrir na distribuição**: `data/models/pacote_servido=v1.json`; em cada pacote, `pacote.json` (população, atributos, parâmetros, sementes, hashes das fontes e do código, versões das bibliotecas, retenção, limitações, cobertura do procedimento), `manifesto_arquivos.json` e `ids.json`; o `manifesto.json` do dataset de referência e o `REGISTRO_REFIT.json`; o `MANIFEST.json` da distribuição. O protocolo, os JSON de resultado e os registros de decisão ficam com o autor; os números centrais estão em [docs/resultados.md](docs/resultados.md). Ver [docs/registro_de_experimentos.md](docs/registro_de_experimentos.md).

## 7. Resultados

Teste final retrospectivo: 452 vídeos, 16 canais, consulta única. Números lidos dos JSON da avaliação (tabelas completas em [docs/resultados.md](docs/resultados.md)).

| | MAE em log | Acerto par a par (pares distinguíveis) | Spearman intra-canal (mediana) |
|---|---|---|---|
| **modelo completo de vídeo** | **1,126** | **58,6%** | **0,314** |
| média aritmética histórica | 1,637 | 51,2% | -0,030 |
| mediana histórica | 1,220 | 51,3% | 0,058 |
| média móvel exponencial | 1,207 | 51,9% | 0,056 |

- Contra a **média aritmética** o modelo é favorável em erro e em acerto par a par (ICs de 95% por canal do lado favorável). Contra a **mediana** é favorável em erro (−0,094; IC −0,168 a −0,025) e em acerto par a par. Contra a **média móvel exponencial** o erro é **inconclusivo** (−0,080; IC −0,209 a 0,001); só o acerto par a par é favorável.
- **A faixa subcobre:** 64,2% de 452 vídeos ficaram dentro dela, para 80% pretendidos (IC 50,3% a 79,8%); beleza 50,0% (4 canais); a dobra mais recente 49,7%. Não há cobertura garantida.
- **Canais não vistos:** MAE 1,642 (média móvel 1,618): sem vantagem em erro e pior que nos canais vistos.
- **Modelos de imagem e de texto:** reduzem o erro frente à média aritmética, mas **sem ganho de ordenação demonstrado** (acerto par a par 50,2% e 47,5%).
- **Ganho multimodal:** o erro cai com a legenda e com o modelo completo; a **ordenação não mostra ganho** (todos os contrastes inconclusivos; a ablação não detecta efeitos pequenos).
- Nenhuma meta de acerto percentual é prometida.

## 8. Limitações

1. **Camada 1 local do TAPI §6.3 não atendida (desvio declarado, não resolvido).** O projeto usa capa e legenda locais mais o Gemini por URL pública; não há processamento local obrigatório de frames, transcrição local, detecção de objetos nem de cortes sobre o vídeo, porque não baixa vídeo do YouTube e não obteve os arquivos originais. **Não se afirma conformidade integral ao TAPI.**
2. **Avaliação retrospectiva aproximada:** curtidas atuais no histórico e nos rótulos; sensibilidade 30 × 90 dias estável, o que é compatível com efeito pequeno e não prova ausência.
3. **Base pequena e concentrada:** 16 canais, 82% automotivo, beleza com 4 canais, poucos vídeos por canal no teste.
4. **Erro e ordenação:** vantagem clara só sobre a média aritmética; vantagem em erro sobre a mediana é pequena; contra a média móvel é inconclusiva; pior em canais não vistos.
5. **Faixa subcalibrada** (64,2% para 80%), sem conformal; cobertura não medida no artefato servido; limiar de baixa confiança do vídeo com validade presumida e diagnóstico inconclusivo.
6. **Transcrição do Gemini validada só qualitativamente** (exemplos inspecionados pelo autor), sem taxa de erro; o Gemini substitui o Whisper; a temperatura do Gemini é 1,0 e uma chamada por previsão não entra no intervalo.
7. **Upload de vídeo local não foi verificado ao vivo.** Está implementado e verificado offline e com Files API simulada; a verificação real depende de um vídeo local autorizado, ainda não fornecido.
8. **Gemini e YouTube ao vivo no pacote final:** não verificados após o refit (a última verificação ao vivo foi em 16/10 com o pacote de desenvolvimento).
9. **Retenção:** a base, os datasets, os embeddings, os pacotes derivados das anotações e os caches têm prazo (menor: 2026-11-03T23:46:26Z). Depois dele, a demonstração, o servidor e a reprodução dependem de nova autorização ou nova coleta.
10. **Demonstração:** 4 dos 6 canais sem confirmação de país ou idioma do Brasil (1 "US", 3 sem país), conteúdo em inglês e espanhol em parte, tipo de canal ausente e um canal prolífico sem histórico utilizável (cai na referência do setor com aviso de histórico curto). É só ilustrativa da mecânica e rotulada "de-para retrospectivo".
11. **Licenças dos pesos de e5-small, CLIP, EasyOCR e MediaPipe: NÃO CONFIRMADAS.** Os pesos não vão na distribuição; onde conferir: [docs/dados_e_retencao.md](docs/dados_e_retencao.md).
12. **Termos das plataformas:** métricas derivadas e agregação entre canais (YouTube) e uso do Gemini gratuito sobre URLs do YouTube são questões pendentes, declaradas; nenhuma permissão indefinida é presumida.

O que foi verificado ao vivo, offline com artefatos reais e com dados sintéticos está em [docs/verificado_ao_vivo_offline_sintetico.md](docs/verificado_ao_vivo_offline_sintetico.md).

## 9. Evolução

- Verificar o upload de vídeo local ao vivo (um arquivo vertical ou quadrado de até 180 s) e repetir a verificação do Gemini e do YouTube com o pacote final.
- Cumprir a Camada 1 local se houver fonte autorizada de arquivos de vídeo, e validar a transcrição com taxa de erro.
- Validação prospectiva (curtidas lidas depois da data planejada) e base maior, com mais canais e setores.
- Calibrar a faixa (por exemplo conformal) e validar de fato a regra de baixa confiança no pacote servido.
- Agrupar descrições audiovisuais e testar ordenação por modalidade com mais canais, para dar poder estatístico à ablação.
