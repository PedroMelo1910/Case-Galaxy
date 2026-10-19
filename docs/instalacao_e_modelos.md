# Instalação e obtenção dos modelos

Os comandos valem a partir da pasta `Entrega Galaxy/`. Nenhum exige caminho absoluto do autor.

## 1. Ambiente

```bash
uv sync --locked      # Python 3.12 (.python-version); torch e torchvision pelo índice CPU do pyproject
```

- O `.venv` ocupa cerca de 1,9 GB. O `uv` baixa as dependências de `pypi.org` e de `download.pytorch.org/whl/cpu`.
- **Bibliotecas do sistema:** `libgomp.so.1` (LightGBM) e `libGLESv2.so.2` e `libEGL.so.1` (MediaPipe). Em Ubuntu/Debian provavelmente `libgomp1`, `libgles2` e `libegl1` (nomes **não verificados** em máquina limpa). Se não puder instalá-las, aponte `LD_LIBRARY_PATH` para uma pasta que as contenha. `uv run casegalaxy-demo --verificar-ambiente` testa as três.
- Chaves: `cp .env.example .env`. Offline não precisam de valor. Nunca registre o `.env` no Git.

## 2. Distribuição de modelos e dados de serviço

Fora do Git. Um arquivo `casegalaxy_distribuicao_v1.tar.gz` (cerca de 12,5 MB) com `.sha256`, fornecido pelo autor. Contém, **sem pesos de terceiros e sem o cache bruto das anotações dos alvos**:

| Grupo | O que é | Prazo (`apagar_ate_utc`) |
|---|---|---|
| modelos finais | `data/models/final=v1/` (vídeo, imagem, texto; q10, q50, q90; pré-processamento; limiar do vídeo) | 2026-11-03T23:46:26Z |
| manifesto do pacote ativo | `data/models/pacote_servido=v1.json` | 2026-11-03T23:46:26Z |
| coleção de referência dos comparáveis | dataset do refit final (1.500 linhas, sem LIM) e `REGISTRO_REFIT.json` | 2026-11-03T23:46:26Z |
| caches por conteúdo | `data/agente/cache` (embeddings de legenda, capa e transcrição) | 2026-11-03T23:46:26Z |
| base curada | `data/processed/.../curadoria=v1` (canais do histórico offline) | 2026-11-13T02:14:37Z |
| de-para | contexto, capas e anotações dos 6 criativos, lista congelada, resultado conhecido | 2026-11-17T21:45:43Z (respostas salvas: 2026-11-03T23:46:26Z) |

```bash
tar -xzf casegalaxy_distribuicao_v1.tar.gz casegalaxy_distribuicao_v1/verificar_distribuicao.py
uv run python casegalaxy_distribuicao_v1/verificar_distribuicao.py verificar casegalaxy_distribuicao_v1.tar.gz
uv run python casegalaxy_distribuicao_v1/verificar_distribuicao.py instalar casegalaxy_distribuicao_v1.tar.gz --destino .
# depois dos prazos:
uv run python casegalaxy_distribuicao_v1/verificar_distribuicao.py vencidos . --manifesto data/MANIFEST_DISTRIBUICAO.json --apagar
```

- `verificar` desempacota numa pasta temporária e confere o SHA-256 de cada arquivo contra o `MANIFEST.json`. `instalar` recusa grupo com prazo vencido, não sobrescreve arquivo diferente sem `--forcar` e guarda o manifesto em `data/MANIFEST_DISTRIBUICAO.json`, que `vencidos` usa para listar (ou, com `--apagar`, remover) os grupos cujo prazo passou.
- O manifesto ativo lista uma versão anterior (`desenvolvimento=v1-servido-ate-18-10`) para rollback, mas a distribuição **não** leva esses pacotes: `casegalaxy-pacote-servido reverter` falha com erro claro fora da máquina do autor.
- **Retenção:** ver [dados_e_retencao.md](dados_e_retencao.md). Depois de 2026-11-03T23:46:26Z, os grupos desse prazo devem ser apagados ou ter extensão formalmente autorizada.

## 3. Pesos dos codificadores (não vão na distribuição)

O runtime carrega estes pesos, todos em CPU. As **licenças dos pesos NÃO foram confirmadas** (ver [dados_e_retencao.md](dados_e_retencao.md)); por isso cada pessoa os baixa da fonte oficial.

| Componente | Fonte | Onde fica | Tamanho | Revisão |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | Hugging Face | `$HF_HUB_CACHE`, ou `$HF_HOME/hub`, ou `~/.cache/huggingface/hub` | 471 MB | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `openai/clip-vit-base-patch32` | Hugging Face | idem | 605 MB (`pytorch_model.bin`) | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (a `refs/main` usada no desenvolvimento) |
| EasyOCR `craft_mlt_25k.pth` e `latin_g2.pth` | baixados pelo próprio EasyOCR | `~/.cache/casegalaxy/easyocr` | 83 MB e 15 MB | SHA-256 `4a5efbfb…db17` e `aaa95be1…8c55` |
| MediaPipe `blaze_face_short_range.tflite` | `storage.googleapis.com/mediapipe-models/...` | `~/.cache/casegalaxy/mediapipe` | 230 KB | SHA-256 `b4578f35…0152f` |

```bash
uv run casegalaxy-demo --verificar-ambiente      # somente leitura; tamanho e SHA-256 de cada arquivo
uv run casegalaxy-demo --baixar-codificadores    # opt-in; imprime os tamanhos antes; só o que falta
```

- Os manifestos de treino gravam `revisao: null` para e5 e CLIP: a reprodutibilidade depende dessas revisões. `--baixar-codificadores` usa as fixas acima e cria `refs/main` quando falta, para o runtime offline achar o snapshot.
- Falta de peso: a demonstração para com uma mensagem que lista cada arquivo ausente, o tamanho e o comando. As demais portas de entrada (`casegalaxy-mcp`, função `prever`) **não** têm essa mensagem própria: mostram o erro da biblioteca (`OSError` do Hugging Face, `ConnectError` do MediaPipe, `URLError` do EasyOCR; testado com rede cortada). O servidor MCP força `HF_HUB_OFFLINE=1`; MediaPipe e EasyOCR não têm essa bandeira e tentam baixar o que falta.
- O cache do Hugging Face desta máquina tem também `Systran/faster-whisper-small` (464 MB), **que o produto não usa**.
