# Uso: função, MCP e modo online

## Função Python

```python
from src.agente.api import prever, ordenar
from src.agente.servicos import Servicos

servicos = Servicos.padrao()          # lê o manifesto de pacote ativo e confere os hashes; alguns segundos

resposta = prever({
    "creative": {"tipo": "texto", "texto": "Lançamento: novo sérum com vitamina C #skincare"},
    "planned_datetime": "2026-11-01T12:00:00Z",
    "sector": "beleza",
    "channel_id": "UC...",                 # canal da base, ou informe channel_stats
    "channel_stats": [                      # opcional: posts anteriores do canal
        {"published_at": "2026-08-01T10:00:00Z", "likes": 1200, "collected_at": "2026-10-18T21:00:00Z", "video_id": "AAAAAAAAAAA"},
    ],
}, servicos)
# {"success": True, "data": {...}}  ou  {"success": False, "error": {"code": ..., "message": ..., "details": ...}}
```

- `creative` é `{"tipo": "video", "url": "https://www.youtube.com/shorts/ID"}` ou `{"tipo": "video", "arquivo": "caminho"}`, `{"tipo": "imagem", "arquivo": "caminho"}` ou `{"tipo": "texto", "texto": "..."}`. Imagem é a capa planejada; texto é a legenda planejada.
- Informe `channel_id` (canal da base, ou com `channel_stats` ou `consultar_youtube`) ou só `channel_stats`. `channel_stats` aceita `video_id` em cada post e `own_video_id` no pedido: com a identidade, o próprio criativo é excluído do histórico, da referência e da cadência.
- `ordenar({"creatives": [pedido1, pedido2, ...]}, servicos)` aceita de 2 a 5 criativos do **mesmo canal e da mesma modalidade** e devolve o ranking pelo q50. Entre canais diferentes recusa com `CANAIS_DIFERENTES` (o ranking só é válido dentro do canal).
- Nenhuma operação levanta exceção para o chamador.

Códigos de erro: `ENTRADA_INVALIDA`, `MIDIA_INVALIDA`, `CANAIS_DIFERENTES`, `MODALIDADES_DIFERENTES`, `NIVEIS_DIFERENTES`, `HISTORICO_INDISPONIVEL`, `MODELO_INDISPONIVEL`, `COTA_ESGOTADA`, `ERRO_INTERNO`.

## Servidor MCP (stdio)

Ferramentas `prever_criativo` e `ordenar_criativos`, com os mesmos contratos. Exemplo de configuração de um cliente MCP (ajuste os caminhos):

```json
{ "mcpServers": { "casegalaxy": {
    "command": "uv",
    "args": ["run", "--project", "/caminho/para/Entrega Galaxy", "casegalaxy-mcp"],
    "env": { "CASEGALAXY_MCP_RAIZES": "/caminho/dos/arquivos/de/imagem/e/video" }
} } }
```

- **Fronteira de arquivos:** arquivos de imagem e vídeo só são lidos dentro das pastas listadas em `CASEGALAXY_MCP_RAIZES` (separadas por `:`); sem a variável, nenhum arquivo é aceito. Limites: imagem até 20 MB, vídeo até 100 MB, 180 s e vertical ou quadrado, texto até 10.000 caracteres, histórico fornecido até **200 posts** (a função não tem esse teto; por isso a demonstração usa a função).
- **Offline é o padrão.** Sem `CASEGALAXY_MCP_ONLINE=1`, nenhum cliente de rede é criado; as descrições das ferramentas dizem "Modo offline".
- **Modo online (opt-in):**

| Variável | Papel | Padrão |
|---|---|---|
| `CASEGALAXY_MCP_ONLINE` | `1` liga; vazio ou `0` desliga; outro valor é erro de partida | desligado |
| `CASEGALAXY_MCP_LIVRO` | **obrigatória** no modo online: arquivo do livro de consumo (uma linha JSON por requisição, gravada antes de ela sair) | n/a |
| `CASEGALAXY_MCP_MAX_REQUISICOES` | teto de requisições ao Gemini por execução do servidor | 30 |
| `CASEGALAXY_MCP_MAX_SEGUNDOS_VIDEO` | teto de segundos de vídeo por execução | 360 |
| `CASEGALAXY_MCP_MAX_UNIDADES_YOUTUBE` | teto de unidades da YouTube Data API por execução | 20 |
| `CASEGALAXY_MCP_INTERVALO_S` | intervalo mínimo entre requisições ao Gemini | 5 |

  Exige `YOUTUBE_API_KEY` e `GEMINI_API_KEY` no `.env` ou no ambiente (a partida falha nomeando só a variável que falta). Os tetos valem por execução do servidor e **não substituem a cota diária**, que é do usuário (nível gratuito do Gemini: 15 por minuto, 500 por dia, 28.800 s de vídeo por dia). Em cota diária esgotada o cliente não repete a chamada; o vídeo cai no modelo de imagem, com aviso. A rota por URL nova não grava a anotação em cache (pendência conhecida): consultar o mesmo vídeo de novo pode gastar cota de novo.
- **Upload de vídeo local** (Files API do Gemini): sonda local (contêiner, 180 s, vertical ou quadrado), cache por SHA-256 do arquivo (retenção de 30 dias, com a transcrição dentro), uma única tentativa de geração e exclusão do arquivo no Gemini. **Implementado e verificado offline com Files API simulada; NÃO verificado ao vivo** (depende de um vídeo local autorizado).

## Cuidado com o histórico de canais da base

Para um canal da base sem `channel_stats`, o histórico vem do snapshot de 2026-10-14 (`history.origem = historico_do_snapshot`). Os posts mais recentes do snapshot tinham pouco mais de 30 dias quando as curtidas foram lidas, então a referência do canal pode ficar baixa para uma data planejada bem posterior à coleta. Confira `reference_likes` e `history` na resposta; para uma referência atual, informe `channel_stats` ou use o modo online (consulta do YouTube).

## Rollback do pacote ativo

```bash
uv run casegalaxy-pacote-servido mostrar
uv run casegalaxy-pacote-servido reverter --motivo "texto"     # só se houver versão anterior em disco
```

## Interpretando a saída

- Use `relative_to_channel` e o ranking dentro do canal como saída principal; `predicted_metric` é uma estimativa central.
- `interval_low` a `interval_high` é uma faixa de **80% pretendidos**. O campo `coverage_estimate` diz quanto o procedimento de avaliação cobriu (64,2%, 63,1% ou 63,1%, conforme o modelo), que isso não foi medido no artefato servido e que não há garantia.
- `warnings` e `confidence_flag` importam: imagem, texto e contingência do vídeo sempre carregam "evidência limitada"; histórico curto usa a referência do setor.
