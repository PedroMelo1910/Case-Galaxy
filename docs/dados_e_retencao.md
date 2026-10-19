# Dados, retenção, termos e licenças

## Base

- Fonte: YouTube Data API v3, coleta única (snapshot `20261014T021437Z`, 2026-10-14T02:14:37Z). Lista de canais em `configs/canais.csv` (30 aprovados pela identidade; 7 candidatos e 6 rejeitados não são coletados). Critério de identidade: o site oficial liga para o canal, ou fonte secundária confiável com link nos dois sentidos.
- Canais do Brasil (país `BR` na API), com três exceções documentadas, só quando o país estava ausente e havia atuação brasileira comprovada. 16 canais elegíveis: 12 automotivo e 4 beleza.
- Short: até 180 s e player vertical ou quadrado. Alvos: publicados nos 365 dias até a coleta, menos os 30 últimos. Amostra de anotação: 1.500 alvos (automotivo 1.227, beleza 273) e 100 da LIM.
- Rótulo: curtidas lidas na coleta. Inscritos e estatísticas do canal nunca entram como atributo.
- Limites: 16 canais (a meta do briefing era de 60 ou mais), 82% automotivo, beleza com 4 canais, uma coleta sem recoleta.

## Retenção (decisão conservadora de 18/10/2026)

| Dado | Prazo (`apagar_ate_utc`) |
|---|---|
| tudo o que deriva das anotações Gemini dos alvos: dataset de referência (comparáveis), pacotes de modelo, manifesto ativo, caches, respostas salvas | **2026-11-03T23:46:26Z** (o prazo mais próximo) |
| base curada, legenda e capa, LIM | 2026-11-13T02:14:37Z (30 dias da coleta) |
| dados do de-para (contexto, capas, anotações, resultado conhecido) | 2026-11-17T21:45:43Z (30 dias da leitura de 18/10) |

- A regra do projeto (D-B9) é apagar o snapshot e tudo o que deriva dele até 30 dias depois da data real da coleta. Os pacotes de modelo e as métricas derivadas dependem de esclarecimento de plataforma (N9): enquanto não houver, seguem o prazo mais curto.
- Podem ser mantidos sem prazo: código, configuração, rubrica, manifestos sem dados, relatórios agregados e esta documentação.
- **O autor pode estender os prazos se a regra de origem permitir. Nenhuma permissão indefinida é presumida.** A distribuição registra os prazos no `MANIFEST.json`, recusa instalar grupo vencido e lista (ou apaga, com `--apagar`) o que venceu (ver [instalacao_e_modelos.md](instalacao_e_modelos.md)).
- O cache bruto das 1.500 anotações dos alvos **não** vai na distribuição (não é necessário para inferir).
- Os dados do de-para foram lidos da API em 18/10/2026 (653 unidades) e anotados com o Gemini (6 requisições).

## O que pode ser compartilhado

| | |
|---|---|
| Código, configuração, rubrica, documentação, relatórios agregados | sim, sem prazo |
| Pacotes de modelo, dataset de referência, base curada, caches, anotações do Gemini, capas, histórico e respostas salvas | só para avaliação por quem recebe o arquivo do autor, e **até os prazos acima**; não é publicação aberta; a redistribuição depois do prazo não é presumida |
| Pesos de e5-small, CLIP, EasyOCR e MediaPipe | **não vão na distribuição**; cada pessoa baixa da fonte oficial |

## Termos das plataformas (questões pendentes, declaradas)

- YouTube API Services Developer Policies: não baixar nem armazenar o audiovisual (III.E.1.1; o projeto não baixa), dados não autorizados por até 30 dias (III.E.4.d), não criar dados ou métricas derivadas (III.E.4.h) e agregação só dentro do mesmo dono (III.E.2.1). Estas duas últimas são **questões de plataforma pendentes** (N9). Os Termos de Serviço do YouTube proíbem o download e o acesso automatizado sem autorização.
- Gemini API, nível gratuito: o projeto envia **só conteúdo público ou não confidencial** (URLs públicas de Shorts; um criativo local, se o autor o autorizar). Os termos de uso das saídas e o tratamento de dados do nível gratuito **não foram reconferidos** para esta entrega; não se presume permissão de redistribuição indefinida das saídas, que seguem os prazos acima.
- O uso de URLs do YouTube no Gemini e o armazenamento de miniaturas são questões pendentes (N10).

## Licenças de terceiros

Lidas do metadado dos pacotes instalados (versões do `uv.lock`): LightGBM 4.7.0 (MIT), scikit-learn 1.9.1 (BSD-3-Clause), LangGraph 1.2.15 (MIT), MCP SDK 2.3.0 (MIT), PyTorch 2.14.1 CPU e torchvision (BSD e outras, ver o pacote), Transformers 5.19.0 (Apache-2.0), EasyOCR 1.7.2 (código, Apache-2.0), MediaPipe 1.1.1 (código, Apache-2.0), OpenCV headless 5.0.0.93 (Apache-2.0; o wheel traz licenças de terceiros), NumPy, PyArrow, Pydantic, httpx e Pillow (BSD, Apache, MIT).

**Licenças dos pesos: NÃO CONFIRMADAS.** Onde conferir antes de qualquer redistribuição:

| Pesos | Onde conferir | Situação |
|---|---|---|
| `intfloat/multilingual-e5-small` | cartão do modelo em huggingface.co/intfloat/multilingual-e5-small | não confirmada (a consulta desta entrega veio truncada; a lembrança do autor é MIT) |
| `openai/clip-vit-base-patch32` | cartão em huggingface.co/openai/clip-vit-base-patch32 e repositório `openai/CLIP` | não confirmada (o cartão não mostra licença no texto visível; o código do CLIP é MIT) |
| EasyOCR `craft_mlt_25k.pth`, `latin_g2.pth` | repositório `JaidedAI/EasyOCR` (modelos publicados nas releases) | não confirmada individualmente |
| MediaPipe `blaze_face_short_range.tflite` | documentação de modelos do MediaPipe (ai.google.dev/edge/mediapipe) | não confirmada |

O levantamento completo, com a mesma situação, acompanha a distribuição (`licencas_terceiros.md`).
