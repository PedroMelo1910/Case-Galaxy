# Arquitetura

O produto é um pacote Python (`src/`) executado localmente, em CPU. Não há banco de dados, banco vetorial, fila externa nem servidor HTTP. Tem três partes: o **pipeline de dados** (coleta, curadoria, extração), o **pipeline de treino e avaliação** e o **serviço de predição** (função Python e servidor MCP por stdio, que chamam a mesma lógica).

## Fluxo de predição

O grafo é fixo (LangGraph), não é um agente livre. Cada pedido passa por:

| Passo | O que faz | LLM? | Se falhar |
|---|---|---|---|
| `validar_entrada` | contrato Pydantic; setor `beleza` ou `automotivo`; modalidade; vídeo até 180 s, vertical ou quadrado | não | erro `ENTRADA_INVALIDA` |
| 1 `extrair_atributos` | vídeo: capa (enviada, miniatura da API ou primeiro quadro), legenda local, rubrica e transcrição do Gemini (URL ou arquivo, com cache); imagem: capa e legenda; texto: só a legenda. Nunca inventa o que a modalidade não tem | Gemini como extrator, só no vídeo | sem anotação, o vídeo cai no modelo de imagem (com capa); sem capa, erro `MIDIA_INVALIDA` ou `COTA_ESGOTADA`; transcrição `incerta` deixa os atributos de transcrição ausentes |
| 2 `historico_e_comparaveis` | referência do canal (mediana dos últimos 20 posts até 30 dias antes da data planejada); 3 comparáveis do canal e 2 do setor, só com vídeos de pelo menos 30 dias | não | sem fonte de histórico: `HISTORICO_INDISPONIVEL`; menos de 5 posts: referência do setor e aviso de histórico curto |
| 3 `prever_modelo` | escolhe o modelo pela modalidade e calcula q10, q50 e q90; curtidas previstas; razão com a referência do canal | não | `MODELO_INDISPONIVEL` |
| `avaliar_confianca` | motivos tipados de baixa confiança | não | |
| 4 `explicar` | TreeSHAP do q50; os 5 atributos interpretáveis que mais pesam e as somas por família vão a um redator; um validador confere a frase contra o JSON | sim, só o redator | erro ou reprovação: texto de modelo fixo (sempre assim no modo offline) |
| `montar_saida` | contrato de saída | não | |
| 5 `ordenar` | de 2 a 5 criativos do **mesmo canal** e da mesma modalidade; roda o grafo em cada um e ordena pelo q50 | não | `CANAIS_DIFERENTES`, `MODALIDADES_DIFERENTES`, `NIVEIS_DIFERENTES` |

### Princípios

- **O número vem só do modelo.** Os campos numéricos são preenchidos antes do único nó com LLM. Um teste com um redator que inventa números (999999, "77%") confirma que o validador o reprova e que os números ficam iguais.
- **Uma lógica, dois acessos.** O servidor MCP é uma camada fina sobre `prever` e `ordenar`; função, grafo, API e MCP devolvem os mesmos números (verificado, ver [verificado_ao_vivo_offline_sintetico.md](verificado_ao_vivo_offline_sintetico.md)).
- **Uma modalidade, um modelo.** O vídeo usa o modelo completo; a imagem é a capa planejada e usa o modelo de imagem; o texto é a legenda planejada e usa o modelo de texto. Cada saída traz `input_modality`, `model_level`, `model_version`, `bundle_id` e a `coverage_estimate` do próprio modelo.
- **Falha explícita.** Gemini indisponível, cota esgotada, canal sem histórico e entrada inválida viram erro ou aviso com código. Nunca há previsão só com histórico sob o nome do modelo completo.
- **Mídia sem download.** Nenhum vídeo ou áudio do YouTube passa pelo disco. O Gemini recebe a URL pública. Só miniaturas são baixadas de `i.ytimg.com`.
- **Estatísticas do próprio vídeo nunca entram.** O vídeo consultado é retirado do histórico por `video_id`; sem o identificador, só a defasagem de 30 dias protege e a saída avisa (`identidade_do_proprio_video_ausente`).
- **Offline por padrão.** Nenhuma chamada de rede acontece sem cliente injetado; o servidor MCP só liga o modo online com variáveis de ambiente.

## Saída de `prever` (campos principais)

`predicted_metric` (curtidas previstas, q50), `interval_low` e `interval_high` (mínimo e máximo das três saídas), `interval_nominal_level` (0,8), `relative_to_channel`, `reference_likes` e `reference_source` (canal ou setor), `history` (origem, data de referência, número de posts), `warnings` (avisos tipados), `top_drivers` e `family_contributions` (SHAP), `explanation` e `explanation_source` (`llm` ou `template`), `comparables`, `extraction`, `model_version`, `model_level`, `bundle_id`, `coverage_estimate` (declaração do procedimento, com `garantia_de_cobertura: false` e `medida_no_artefato_servido: false`), `statistical_confidence` e `confidence_flag`/`confidence_reasons`. Para material já publicado: `retrospective: true`.

O fator `exp(phi)` do SHAP pertence à decomposição `1 + curtidas_previstas = (1 + referência) · exp(phi0) · Π exp(phi_j)` com a referência fixa. Não multiplica curtidas brutas, não é efeito de mudar o atributo isoladamente e não é causal.

## Pacote ativo

O agente não tem caminhos de modelo no código. `Servicos.padrao()` lê `data/models/pacote_servido=v1.json`, valida pasta e SHA-256 de cada arquivo dos três pacotes, do dataset de referência e do limiar, e só então serve. Manifesto ausente ou pacote que não confere é erro claro (`ErroManifesto`), sem volta silenciosa a modelos de desenvolvimento. A partida confere os hashes a cada vez (alguns segundos).

## Dados e treino

Coleta (YouTube Data API) → base curada → partição temporal → amostra de anotação → anotação Gemini e extração local de legenda e capa → atributos de histórico → dataset → LightGBM → avaliação → refit final e pacote. Comandos em [reproduzir_o_treino.md](reproduzir_o_treino.md).
