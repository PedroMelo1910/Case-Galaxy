# Demonstração

```bash
uv run casegalaxy-demo                       # roda os 6 criativos pelo agente servido, sem rede, e grava as respostas
uv run casegalaxy-demo --mostrar-salvas      # plano B: mostra as respostas salvas, sem carregar nenhum modelo
uv run casegalaxy-demo --verificar-ambiente  # somente leitura
```

Equivalente: `python -m src.demonstracao`. A demonstração **não cria cliente do Gemini nem do YouTube**: zero chamadas online. As respostas vão para `data/demonstracao/respostas_salvas/execucao_<UTC>.json`; o conjunto de referência, gerado uma vez com o pacote `final=v1`, é `respostas_final=v1.json`.

## O que é

Um **de-para retrospectivo** (D-F6): 2 conjuntos de 3 Shorts (beleza e automotivo), de canais **fora da base coletada** (interseção vazia de canais e de vídeos, conferida no manifesto), publicados entre 2026-09-01 e 2026-09-15. Cada vídeo passa por `prever` com a URL, a legenda, a capa e o histórico do canal até 30 dias antes da publicação (o próprio vídeo excluído). As anotações do Gemini desses seis vídeos estão num **cache separado** (`data/demonstracao/anotacoes`), que não toca o cache congelado da avaliação. Cada resposta mostra a versão do modelo, a faixa, a **cobertura declarada do procedimento** (64,2% para o modelo completo, com a ressalva de que não é garantida) e o **resultado real conhecido** (curtidas lidas em 18/10/2026, rotulado "de-para retrospectivo", só ilustrativo).

A demonstração também mostra `ordenar`: entre os criativos de canais diferentes o produto **recusa** (`CANAIS_DIFERENTES`, o ranking só vale dentro do canal), e uma ilustração com duas variantes de legenda do mesmo criativo, pelo modelo de texto. As variantes **não são vídeos distintos** e o resultado não prova ordenação.

## Regra de escolha (congelada antes de qualquer previsão)

`regra_de_escolha_v1.json`, SHA-256 `b042acd63b1a48571ced289585d573551724b05ad78397af16ba3cac1ef34fc1`: seis consultas `search.list` (uma página de 50; `carro`, `moto`, `lançamento carro`, `maquiagem`, `skincare`, `cabelo`; Shorts, Brasil, português, mais recentes primeiro) na janela de publicação; elegível = Short pela definição da base, público, sem transmissão, curtidas visíveis, pelo menos 30 dias, fora da base, sem título duplicado em outro canal e no máximo um por canal; **ordem pelo SHA-256 do `video_id`**, nunca por desempenho. Resultado congelado em `lista_de_para_v1.json` (SHA-256 `c90ffea5c73ef082c74e815e1f17cd2a26c42b2411beb31c47dafff8166f38c4`) às 21:46:02Z de 18/10/2026. A demonstração confere esse SHA-256 antes de rodar. Foram 299 candidatos (135 e 128 elegíveis), 653 unidades da YouTube Data API e 6 requisições (474 s de vídeo) do Gemini, sem repetições nem falhas.

| Setor | video_id | Canal | País na API | Publicado | Duração |
|---|---|---|---|---|---|
| beleza | `cC72PCGoNos` | Valeria Enríquez | ausente | 2026-09-15 | 123 s |
| beleza | `BJxXNcC43nM` | Kenzie Yolles | ausente | 2026-09-15 | 35 s |
| beleza | `v3CYJEM0zqc` | Super Bonita Makeup | ausente | 2026-09-15 | 105 s |
| automotivo | `dn2TvmAF1xg` | JSMN ZXR | US | 2026-09-15 | 11 s |
| automotivo | `Mg9al7WM0Mw` | Loja Carros | BR | 2026-09-10 | 128 s |
| automotivo | `mtaKSq3hJwM` | Rafael Ragonezi | BR | 2026-09-15 | 72 s |

## Ressalvas (a demonstração só ilustra a mecânica)

- **4 dos 6 canais sem confirmação de país ou idioma do Brasil** (1 com país "US", 3 sem país na API); parte do conteúdo está em inglês e em espanhol, fora do domínio de treino (canais do Brasil, em português). O idioma não foi verificado.
- **`channel_type` ausente** nos seis (canais fora da base, tipo não decidido): aviso `tipo_canal_ausente`.
- **Um canal prolífico (`dn2TvmAF1xg`) sem histórico utilizável**: nenhum post com curtidas até 30 dias antes dentro das 300 publicações mais recentes consultadas; o agente usa a referência do setor e avisa histórico curto.
- A regra congelada não foi alterada para melhorar a amostra; vídeo só seria trocado por falha técnica documentada, e nenhuma ocorreu.
- Uma observação por vídeo não mede o modelo; a evidência estatística é a [avaliação final](resultados.md).
- O histórico usa curtidas lidas em 18/10/2026 (por isso "retrospectivo"). Os seis vídeos já estavam publicados: a saída traz `retrospective: true` e o aviso `material_ja_publicado`.

## Prazos

Os dados do de-para (contexto, capas, anotações) vencem em 2026-11-17T21:45:43Z; as respostas salvas, derivadas do modelo, em 2026-11-03T23:46:26Z. Ver [dados_e_retencao.md](dados_e_retencao.md).
