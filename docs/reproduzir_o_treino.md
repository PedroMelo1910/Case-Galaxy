# Reproduzir o treino

Este caminho **não é necessário para experimentar o software** (a distribuição leva o pacote pronto). Reproduzir exige chaves, rede e vários dias de cota do Gemini, e gera **outra base** (um novo snapshot não é o coletado em 14/10/2026). A consulta única ao teste final de 18/10/2026 não pode ser repetida: o relatório de desenvolvimento pode.

Comandos a partir de `Entrega Galaxy/`, na ordem. Cada passo grava em `data/` (ignorado pelo Git) e confere o que leu. Use `uv run python -m src.<módulo> --help` para as opções.

| # | Passo | Comando (resumo) | Precisa de |
|---|---|---|---|
| 1 | Coleta | `python -m src.coletar_snapshot [--limite-unidades N] [--max-paginas N]` (lê `configs/canais.csv`, só os `aprovado`) | `YOUTUBE_API_KEY` |
| 2 | Curadoria | `python -m src.curadoria --snapshot data/raw/snapshot=<id>` | |
| 3 | Partição temporal | `python -m src.particoes --base data/processed/snapshot=<id>/curadoria=v1` | |
| 4 | Amostra a anotar | `python -m src.amostra_anotacao --base ... --semente galaxies-anotacao-2026-10-14 --n 1500 --redistribuir --descricao-orcamento "..."` (LIM: `--lim --n 100 --semente galaxies-lim-2026-10-14 --destino .../lim=v1`) | |
| 5 | Anotação com o Gemini | `python -m src.anotacao --base ... --amostra .../amostra=v2 --lote N --max-requisicoes R --max-segundos S --intervalo 10` | `GEMINI_API_KEY` e cota |
| 6 | Legenda e capa locais | `python -I -m src.extracao_legenda_capa --etapa {legenda,baixar,cor,rosto,ocr,clip,materializar} --base ... --amostra ...` (uma etapa por vez, retomável) | pesos dos codificadores; miniaturas de `i.ytimg.com` |
| 7 | Atributos de histórico | `python -m src.historico --base ... --particao ... --defasagem 30` | |
| 8 | Datasets | `python -m src.dataset_desenvolvimento --legenda-capa .../legenda_capa=v1`; reduzidos: `python -m src.dataset_modalidades --modalidade {imagem,texto}` | |
| 9 | Modelos | `python -m src.modelo_desenvolvimento --dataset ...` (q50); `python -m src.faixa_previsao --dataset ... --q50 ...` (q10 e q90); `python -m src.modelo_modalidades --dataset ... --referencia ...` | |
| 10 | Avaliação | `python -m src.avaliacao.relatorio --dataset ... --q50 ... --faixa ... --saida ...` (desenvolvimento); protocolo final: `python -m src.avaliacao.execucao_final {hashes,congelar,revisao,executar,retomar}` | o teste só se abre uma vez |
| 11 | Refit final e pacote | `python -m src.refit_dados`, `python -m src.refit_treino`, `python -m src.refit_verificacao --pacote <pasta>`; ativação: `python -m src.agente.manifesto_servido ativar-final ...` | |

Notas:

- **Cota do Gemini** (nível gratuito, limites conferidos pelo autor e não reconferidos): 15 requisições por minuto, 500 por dia, 28.800 s de vídeo por dia, contados pelo dia do Pacífico por premissa operacional (o Google não confirma o horário de renovação do limite de vídeo). Os 1.500 alvos e os 100 da LIM consumiram 1.464 requisições e cerca de 98.700 s em vários dias, com instabilidade do serviço (erros 503) em alguns. Pare em erro de cota; não troque de conta.
- **Retenção:** o snapshot e tudo o que deriva dele devem ser apagados até 30 dias depois da data real da coleta (ver [dados_e_retencao.md](dados_e_retencao.md)).
- **Só de desenvolvimento, não entregue:** os testes (`TESTES/`), as verificações (`VERIFICACOES/`: orçamento de cota, conciliação, ensaios, protocolo da avaliação final, scripts da demonstração e da distribuição) e os registros de decisão ficam fora deste repositório.
- Os manifestos gravam `revisao: null` para e5 e CLIP; para repetir o desenvolvimento use as revisões de [instalacao_e_modelos.md](instalacao_e_modelos.md).
