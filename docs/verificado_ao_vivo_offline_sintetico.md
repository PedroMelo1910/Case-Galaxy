# O que foi verificado: ao vivo, offline com artefatos reais e sintético

Cada linha diz o que foi realmente executado. **Declarar uma ausência não a atende.**

| Item | Ao vivo (API real) | Offline com artefatos reais | Sintético (testes) |
|---|---|---|---|
| Coleta da base (YouTube Data API) | **sim**, 14/10/2026 | n/a | sim |
| Anotação Gemini por URL pública (1.500 + 100 + 6 vídeos) | **sim**, 14 a 18/10/2026 | n/a | sim |
| Escolha e leitura do de-para (busca, `videos.list`, histórico) | **sim**, 18/10/2026 (653 unidades) | n/a | sim (a regra e o transporte têm testes) |
| Agente completo com histórico do YouTube e Gemini por URL | **sim, em 16/10/2026, com o pacote de desenvolvimento**; **não repetido com o pacote final `final=v1`** | n/a | sim |
| Explicação por LLM (redator) | **sim**, em 16/10/2026, em pacote de desenvolvimento; não repetida com `final=v1` | texto fixo (padrão offline) | sim (redator falso que inventa números é reprovado) |
| Função direta, grafo, API, MCP em memória e MCP stdio com `final=v1` | não | **sim** (mesmos números nas camadas; seis combinações de dois vídeos e três modalidades) | sim |
| `python -m src.demonstracao` com `final=v1` | não (zero chamadas) | **sim** (6 criativos, respostas salvas) | sim (dublês) |
| Recarga dos três pacotes em processo novo, TreeSHAP aditivo | n/a | **sim** (previsões idênticas bit a bit; soma das contribuições até 6e-15) | sim |
| Instalação a partir da distribuição em pasta isolada (cópia da árvore sem `data/` nem `.venv`, `uv sync --locked --offline`, `.env` novo, `instalar`, `casegalaxy-demo`, `casegalaxy-mcp`) | n/a | **sim, em 19/10/2026, usando o cache do `uv`, o cache de modelos do Hugging Face, `~/.cache/casegalaxy` e as bibliotecas do sistema desta máquina: NÃO é verificação em máquina limpa.** Respostas idênticas às salvas; servidor MCP respondeu `initialize`, `tools/list` e um `prever_criativo` de texto de canal da base | sim (construtor) |
| Upload de vídeo local pela Files API | **NÃO** (aguarda vídeo local autorizado) | sim, com a Files API simulada | sim |
| Modo online do MCP (opt-in) | **NÃO** com `final=v1` | partida e tetos | sim |
| CLIP, EasyOCR e MediaPipe reais no servidor MCP real | n/a | demonstração (função) sim; **servidor MCP real com esses extratores: não verificado** | sim (extratores falsos) |
| Teste final (consulta única) | n/a | **sim**, 18/10/2026 | n/a |

Rotulagem: "ao vivo" significa chamada real à API; "offline com artefatos reais" usa os modelos, a base, o histórico e as anotações verdadeiras, sem rede; "sintético" usa dados e servidores inventados (nunca valem como evidência de desempenho). A demonstração e as verificações offline não fazem nenhuma chamada de rede.
