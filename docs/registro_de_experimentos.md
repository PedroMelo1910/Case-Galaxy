# Registro de experimentos (equivalente ao MLflow)

O TAPI pede "registro de experimentos (MLflow ou equivalente)". O projeto **não instala MLflow**. O equivalente são **manifestos com SHA-256, protocolos congelados com hash e registros de decisão**: cada artefato diz de quais dados, código e parâmetros veio, e cada hash pode ser conferido.

## O que a pessoa avaliadora pode abrir (vai na distribuição)

| Arquivo | O que registra |
|---|---|
| `data/models/pacote_servido=v1.json` | versão ativa (`final=v1`), SHA-256 e identidade de cada pacote, limiar vinculado e seu diagnóstico, dataset de referência, base histórica, histórico de ativações |
| `data/models/final=v1/<modalidade>/pacote.json` | população (n, canais, setores, blocos), definição e ordem dos atributos, parâmetros e sementes, hashes das fontes e dos módulos de código, versões das bibliotecas, retenção, limitações e **cobertura do procedimento** (com `garantia_de_cobertura: false`) |
| `.../manifesto_arquivos.json`, `.../ids.json` | SHA-256 de cada arquivo do pacote e lista dos IDs do ajuste |
| `.../video-.../limiar/limiar_largura_servido.json` | limiar 2,9733 em log, n = 100, vinculado à identidade do pacote |
| `data/features/.../desenvolvimento_final=v1/6bb801f43992/manifesto.json` e `.../REGISTRO_REFIT.json` | linhagem do dataset, contagens, exclusões e hashes do refit |
| `data/MANIFEST_DISTRIBUICAO.json` | grupos da distribuição, SHA-256 de cada arquivo e prazos de retenção |
| `data/demonstracao/congelado/` | regra, lista e manifesto do de-para, cada um com o SHA-256 |

`uv run casegalaxy-pacote-servido mostrar` resume a versão ativa.

## O que fica com o autor (não vai na distribuição)

O protocolo final congelado (SHA-256 `4a2ef2442edd9f3ce943ad454e7c67d74a29c5d407d523c1b378eb484156e94f`), a revisão de ML registrada com o aceite do autor, o marcador e o registro da consulta única ao teste, os JSON de resultado e os registros de decisão. Os números centrais estão em [resultados.md](resultados.md), geradas desses JSON. Sob pedido, o autor mostra os originais.

## O que garante a rastreabilidade

- Partição, amostra, LIM e entradas da avaliação congeladas com hash antes de qualquer consulta ao teste; consulta única registrada; módulos de código do método hasheados e arquivados em cópia somente leitura.
- Refit final com especificação fixa (mesmos atributos, mesma ordem e mesma configuração dos pacotes de desenvolvimento aprovados), recarregado em processo novo com previsões idênticas bit a bit.
- Sementes registradas (20261015 para modelos e bootstrap).
- Limites: os manifestos de treino gravam `revisao: null` para os codificadores e5/CLIP; a reprodutibilidade deles depende das revisões fixadas em [instalacao_e_modelos.md](instalacao_e_modelos.md).
