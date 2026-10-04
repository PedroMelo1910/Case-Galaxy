# CaseGalaxy — agente preditivo de curtidas em YouTube Shorts

Projeto individual do Inteli Academy × Galaxies. Entrega final em 23/10/2026.

> **Estado em 04/10/2026:** só a estrutura inicial do ambiente está pronta: o pacote `casegalaxy` é importável, mas ainda não tem módulos. Coleta, modelo, agente e servidor MCP não existem. Este README será completado ao longo do projeto, com as seções exigidas pelo TAPI §10.

## Requisitos

- WSL 2 (Ubuntu). Todos os comandos rodam no WSL, dentro de `/mnt/c/Users/admin/Documents/Inteli Academy/CaseGalaxy/Entrega Galaxy`. A pasta não é copiada para o disco Linux.
- [`uv`](https://docs.astral.sh/uv/) 0.12 ou mais recente. O `uv` instala o Python 3.12.
- Git, usado só pelo WSL.

## Instalar

```bash
cd "/mnt/c/Users/admin/Documents/Inteli Academy/CaseGalaxy/Entrega Galaxy"
uv python install 3.12   # só na primeira vez
uv sync                  # cria .venv/ e instala o pacote
```

## Verificar o ambiente

```bash
uv run python -c "import sys, casegalaxy; print(casegalaxy.__version__, sys.version.split()[0])"
```

A saída esperada é a versão do pacote seguida de `3.12.x`.

## Chaves de API

As chaves ficam no arquivo `.env`, na raiz. O arquivo já existe, vazio, e o Git o ignora; `.env.example` declara os nomes. Nesta estrutura inicial, nenhum código lê as chaves ainda.

| Variável | Onde obter |
|---|---|
| `YOUTUBE_API_KEY` | Google Cloud Console → ativar a YouTube Data API v3 no projeto → Credenciais → Criar credenciais → Chave de API |
| `GEMINI_API_KEY` | Google AI Studio → Get API key |

Para preencher:

1. Abra o arquivo num editor local, por exemplo `nano .env` no WSL.
2. Cole cada chave logo depois do `=`, sem aspas e sem espaços.
3. Salve e confira só a presença, sem exibir os valores:

```bash
for v in YOUTUBE_API_KEY GEMINI_API_KEY; do
  grep -qE "^${v}=.+" .env && echo "$v: preenchida" || echo "$v: AUSENTE"
done
git check-ignore -q .env && echo ".env ignorado pelo Git"
```

Cuidados:
- Em `/mnt/c`, o WSL não aplica permissões Unix (`chmod 600` não tem efeito). Qualquer programa do seu usuário do Windows consegue ler o `.env`.
- No Console, restrinja a chave do YouTube à YouTube Data API v3.
- Não defina também `GOOGLE_API_KEY` no ambiente: o SDK oficial do Gemini lê essa variável além de `GEMINI_API_KEY`.
- Se uma chave vazar (por exemplo, em um log ou commit), revogue-a e gere outra.

## Git

Remoto `origin`: <https://github.com/PedroMelo1910/Case-Galaxy>.
