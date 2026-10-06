"""Snapshot de metadados da YouTube Data API: registros, respostas brutas e proveniência.

Grava em `data/raw/snapshot=<id>/`, fora do Git. Um snapshot nunca é sobrescrito. Os dados
valem por no máximo 30 dias a partir da coleta; `proveniencia.json` registra a data-limite,
que vale para a pasta inteira e para tudo o que for derivado dela.
"""

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Collection, Iterable
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

from src import coleta
from src.config import RAIZ_PROJETO
from src.contratos import (
    DIAS_MATURIDADE,
    DURACAO_MAXIMA_SHORT_S,
    Formato,
    VideoColetado,
)
from src.lista_canais import ListaCanais
from src.youtube_api import (
    ALTURA_MAXIMA_PLAYER,
    CUSTO_POR_CHAMADA,
    MAX_IDS_POR_CHAMADA,
    PARTES_CANAL,
    PARTES_PLAYLIST,
    PARTES_VIDEO,
    URL_BASE,
)

DIAS_RETENCAO = 30
DIRETORIO_PADRAO = RAIZ_PROJETO / "data" / "raw"
TIPOS = ("piloto", "coleta")
ARQUIVO_MANIFESTO = "MANIFESTO_SHA256.txt"
DEPENDENCIAS_REGISTRADAS = ("httpx", "pydantic", "pydantic-settings")


def prazo_de_retencao(coletado_em: datetime) -> datetime:
    return coletado_em + timedelta(days=DIAS_RETENCAO)


def _iso(instante: datetime) -> str:
    return instante.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relativo(caminho: Path) -> str:
    """Caminho relativo ao projeto; fora dele, só o nome do arquivo (não expõe o caminho local)."""
    try:
        return Path(caminho).resolve().relative_to(RAIZ_PROJETO).as_posix()
    except ValueError:
        return Path(caminho).name


def _estado_do_git() -> dict:
    def rodar(*argumentos: str) -> str | None:
        saida = subprocess.run(
            ["git", *argumentos], cwd=RAIZ_PROJETO, capture_output=True, text=True, timeout=15, check=False
        )
        return saida.stdout.strip() if saida.returncode == 0 else None

    try:
        commit, status = rodar("rev-parse", "--short", "HEAD"), rodar("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return {"commit_git": None, "alteracoes_locais": None}
    return {"commit_git": commit, "alteracoes_locais": None if status is None else bool(status)}


def _sha256(arquivo: Path) -> str:
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()


def estado_do_codigo() -> dict:
    """Identifica o código e o ambiente mesmo sem commit: SHA-256 dos arquivos e versões instaladas."""
    arquivos = [*sorted((RAIZ_PROJETO / "src").glob("*.py")), RAIZ_PROJETO / "pyproject.toml", RAIZ_PROJETO / "uv.lock"]
    return {
        "pacote": version("casegalaxy"),
        **_estado_do_git(),
        "python": platform.python_version(),
        "dependencias": {nome: version(nome) for nome in DEPENDENCIAS_REGISTRADAS},
        "arquivos_sha256": {a.relative_to(RAIZ_PROJETO).as_posix(): _sha256(a) for a in arquivos if a.is_file()},
    }


def _indicadores(videos: list[VideoColetado]) -> dict:
    """Curtidas ocultas, ritmo de publicação e primeiro Short entre os vídeos lidos do canal."""
    datas = sorted(v.published_at for v in videos)
    shorts = [v.published_at for v in videos if v.formato is Formato.SHORT]
    intervalo_dias = max((datas[-1] - datas[0]).days, 1) if datas else None
    return {
        "curtidas_ocultas": sum(v.curtidas is None for v in videos),
        "videos_por_dia": round(len(datas) / intervalo_dias, 3) if datas else None,
        "primeiro_short_utc": _iso(min(shorts)) if shorts else None,
    }


def _situacao_dos_canais(resultado: coleta.ResultadoColeta) -> list[dict]:
    situacao = coleta.avaliar_canais(resultado)
    linhas = []
    for canal in resultado.canais:
        lido = resultado.uploads_por_canal.get(canal.channel_id)
        motivo = situacao[canal.channel_id]
        mais_antiga = lido.data_mais_antiga_lida if lido else None
        linhas.append(
            {
                "channel_id": canal.channel_id,
                "setor": str(canal.setor),
                "tipo_canal": str(canal.tipo_canal),
                "pais_api": canal.pais,
                "situacao": str(motivo) if motivo else "elegivel",
                "paginas_lidas": lido.paginas_lidas if lido else 0,
                "itens_lidos": lido.itens_lidos if lido else 0,
                "itens_anteriores_a_janela": lido.itens_anteriores_a_janela if lido else 0,
                "data_mais_antiga_lida_utc": _iso(mais_antiga) if mais_antiga else None,
                "videos_listados_na_janela": len(lido.ids) if lido else 0,
                "amostra_parcial": bool(lido and lido.interrompida),
                "falha": resultado.falhas_por_canal.get(canal.channel_id),
                **_indicadores([v for v in resultado.videos if v.channel_id == canal.channel_id]),
            }
        )
    return linhas


def proveniencia(
    resultado: coleta.ResultadoColeta,
    *,
    lista: ListaCanais,
    tipo: str,
    status_permitidos: Collection[str] | None = None,
) -> dict:
    """Origem, parâmetros, versão da lista, código, cota, contagens, falhas e retenção do snapshot.

    `parcial`: alguma leitura de canal foi cortada pelo limite de páginas. `com_falhas`: algum canal ou
    vídeo não pôde ser lido. `completo`: nenhum dos dois.
    """
    janela = coleta.Janela(resultado.coletado_em)
    status_da_lista = {canal.channel_id: str(canal.status) for canal in lista.canais}
    parcial = bool(resultado.canais_parciais)
    com_falhas = bool(
        resultado.falhas_por_canal
        or resultado.canais_nao_encontrados
        or resultado.videos_indisponiveis
        or resultado.videos_invalidos
    )
    return {
        "snapshot_id": resultado.snapshot_id,
        "tipo": tipo,
        "parcial": parcial,
        "com_falhas": com_falhas,
        "completo": not (parcial or com_falhas),
        "coletado_em_utc": _iso(resultado.coletado_em),
        "terminado_em_utc": _iso(resultado.terminado_em) if resultado.terminado_em else None,
        "fonte": {
            "api": "YouTube Data API v3",
            "url_base": URL_BASE,
            "autenticacao": "chave de API enviada em cabeçalho; o valor nunca é registrado",
        },
        "metodos": {
            "channels.list": {"part": PARTES_CANAL},
            "playlistItems.list": {"part": PARTES_PLAYLIST, "maxResults": MAX_IDS_POR_CHAMADA},
            "videos.list": {"part": PARTES_VIDEO, "maxHeight": ALTURA_MAXIMA_PLAYER},
        },
        "parametros": {
            "dias_maturidade": DIAS_MATURIDADE,
            "dias_janela_alvos": coleta.DIAS_JANELA_ALVOS,
            "dias_historico_extra": coleta.DIAS_HISTORICO_EXTRA,
            "duracao_maxima_short_s": DURACAO_MAXIMA_SHORT_S,
            "tolerancia_quadrado": coleta.TOLERANCIA_QUADRADO,
            "minimo_shorts_por_canal": coleta.MINIMO_SHORTS_POR_CANAL,
            "pais_exigido": coleta.PAIS_EXIGIDO,
            "max_paginas_por_canal": resultado.max_paginas_por_canal,
            "limite_unidades": resultado.limite_unidades,
        },
        "janela_publicacao_utc": {
            "inicio_historico": _iso(janela.inicio_historico),
            "inicio_alvos": _iso(janela.inicio_alvos),
            "fim_alvos": _iso(janela.fim_alvos),
        },
        "lista_canais": {
            "arquivo": _relativo(lista.arquivo),
            "sha256": lista.sha256,
            "canais_no_arquivo": len(lista.canais),
            "canais_usados": list(resultado.canais_solicitados),
            "status_permitidos": sorted(status_permitidos) if status_permitidos is not None else None,
            "status_por_canal": {i: status_da_lista.get(i) for i in resultado.canais_solicitados},
        },
        "codigo": estado_do_codigo(),
        "cota": {"unidades_consumidas": resultado.unidades_cota, "custo_por_chamada": CUSTO_POR_CHAMADA},
        "contagens": resultado.resumo(),
        "canais": _situacao_dos_canais(resultado),
        "falhas": {
            "canais_nao_encontrados": list(resultado.canais_nao_encontrados),
            "falhas_por_canal": dict(resultado.falhas_por_canal),
            "videos_indisponiveis": len(resultado.videos_indisponiveis),
            "videos_invalidos": len(resultado.videos_invalidos),
            "videos_invalidos_motivos": dict(resultado.motivos_invalidos),
            "itens_playlist_sem_data": resultado.itens_playlist_sem_data,
        },
        "retencao": {
            "dias": DIAS_RETENCAO,
            "apagar_ate_utc": _iso(prazo_de_retencao(resultado.coletado_em)),
            "abrange": "esta pasta e todo dado derivado dela (tabelas, atributos, anotações, thumbnails)",
        },
    }


def _jsonl(linhas: Iterable[str]) -> bytes:
    return "".join(f"{linha}\n" for linha in linhas).encode("utf-8")


def _json(conteudo: object) -> bytes:
    return json.dumps(conteudo, ensure_ascii=False, indent=1).encode("utf-8")


def _conteudo_do_snapshot(
    resultado: coleta.ResultadoColeta, lista: ListaCanais, tipo: str, status_permitidos: Collection[str] | None
) -> dict[str, bytes]:
    """Todos os arquivos, já serializados, com o manifesto SHA-256 (formato do `sha256sum`) por último."""
    arquivos = {
        "canais.jsonl": _jsonl(canal.model_dump_json() for canal in resultado.canais),
        "videos.jsonl": _jsonl(video.model_dump_json() for video in resultado.videos),
        "bruto_canais.json": _json(list(resultado.brutos_canais)),
        "bruto_videos.json": _json(list(resultado.brutos_videos)),
        "proveniencia.json": _json(
            proveniencia(resultado, lista=lista, tipo=tipo, status_permitidos=status_permitidos)
        ),
    }
    manifesto = "".join(f"{hashlib.sha256(dados).hexdigest()}  {nome}\n" for nome, dados in sorted(arquivos.items()))
    return {**arquivos, ARQUIVO_MANIFESTO: manifesto.encode("utf-8")}


def salvar_snapshot(
    resultado: coleta.ResultadoColeta,
    *,
    lista: ListaCanais,
    destino: Path = DIRETORIO_PADRAO,
    tipo: str = "coleta",
    status_permitidos: Collection[str] | None = None,
) -> Path:
    """Grava o snapshot numa pasta nova e recusa sobrescrever.

    O conteúdo é montado inteiro em memória, gravado numa pasta temporária e só então renomeado para o
    nome final: uma falha no meio não deixa snapshot parcial com aparência de completo.
    """
    if tipo not in TIPOS:
        raise ValueError(f"tipo de snapshot inválido: {tipo!r}; use um de {TIPOS}")
    pasta = Path(destino) / f"snapshot={resultado.snapshot_id}"
    if pasta.exists():
        raise FileExistsError(f"snapshot já existe e não será sobrescrito: {pasta.name}")
    arquivos = _conteudo_do_snapshot(resultado, lista, tipo, status_permitidos)
    Path(destino).mkdir(parents=True, exist_ok=True)
    temporaria = Path(tempfile.mkdtemp(prefix=f".{pasta.name}.", suffix=".parcial", dir=destino))
    try:
        for nome, dados in arquivos.items():
            (temporaria / nome).write_bytes(dados)
        temporaria.rename(pasta)
    except OSError as erro:
        shutil.rmtree(temporaria, ignore_errors=True)
        if pasta.exists():
            raise FileExistsError(f"snapshot já existe e não será sobrescrito: {pasta.name}") from erro
        raise
    return pasta
