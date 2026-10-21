"""Módulo consolidado. Reúne:
- Lista curada de canais em CSV: leitura, validação e versão (SHA-256 do arquivo).
- Snapshot de metadados da YouTube Data API: registros, respostas brutas e proveniência.
- Coleta metadados dos canais da lista na YouTube Data API e grava um snapshot."""

import argparse
import csv
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator, StringConstraints, ValidationError

from src.config import ChaveAusente, RAIZ_PROJETO
from src.dados import coleta
from src.dados.coleta import (
    CanalAlvo,
    coletar,
    DIAS_MATURIDADE,
    DURACAO_MAXIMA_SHORT_S,
    Formato,
    IdCanal,
    Setor,
    TipoCanal,
    VideoColetado,
)
from src.integracoes.youtube import (
    ALTURA_MAXIMA_PLAYER,
    ClienteYouTube,
    CUSTO_POR_CHAMADA,
    ErroYouTube,
    MAX_IDS_POR_CHAMADA,
    PARTES_CANAL,
    PARTES_PLAYLIST,
    PARTES_VIDEO,
    URL_BASE,
)
from src.proveniencia import iso_utc as _iso, sha256_arquivo as _sha256


# ──── Lista curada de canais em CSV: leitura, validação e versão (SHA-256 do arquivo).
COLUNAS = ("channel_id", "handle", "nome", "url", "setor", "tipo_canal", "status", "evidencia", "incerteza")
Handle = Annotated[str, StringConstraints(pattern=r"^@[\w.\-]+$")]


class StatusConferencia(StrEnum):
    CANDIDATO = "candidato"
    APROVADO = "aprovado"
    REJEITADO = "rejeitado"


class CanalDaLista(BaseModel):
    """Uma linha da lista: identidade do canal, classificação e evidência da conferência."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel_id: IdCanal
    handle: Handle
    nome: str = Field(min_length=1)
    url: str
    setor: Setor
    tipo_canal: TipoCanal
    status: StatusConferencia
    evidencia: str
    incerteza: str

    @model_validator(mode="after")
    def _url_do_canal(self) -> "CanalDaLista":
        if self.url != f"https://www.youtube.com/channel/{self.channel_id}":
            raise ValueError("url deve ser https://www.youtube.com/channel/<channel_id>")
        if self.status is StatusConferencia.APROVADO and not self.evidencia:
            raise ValueError("canal aprovado precisa de evidencia")
        return self

    def como_alvo(self) -> CanalAlvo:
        return CanalAlvo(channel_id=self.channel_id, setor=self.setor, tipo_canal=self.tipo_canal)


@dataclass(frozen=True)
class ListaCanais:
    arquivo: Path
    sha256: str
    canais: tuple[CanalDaLista, ...]

    def selecionar(
        self,
        ids: Iterable[str] | None = None,
        *,
        status: Collection[StatusConferencia] = (StatusConferencia.APROVADO,),
    ) -> list[CanalAlvo]:
        """Sem `ids`, todos os canais com status permitido; com `ids`, exatamente esses, na ordem dada."""
        if ids is None:
            return [canal.como_alvo() for canal in self.canais if canal.status in status]
        por_id = {canal.channel_id: canal for canal in self.canais}
        selecionados = []
        for channel_id in ids:
            canal = por_id.get(channel_id)
            if canal is None:
                raise ValueError(f"canal {channel_id} não está na lista {self.arquivo.name}")
            if canal.status not in status:
                permitidos = sorted(str(s) for s in status)
                raise ValueError(f"canal {channel_id} tem status '{canal.status}'; permitidos: {permitidos}")
            selecionados.append(canal.como_alvo())
        return selecionados


def ler_lista(caminho: Path) -> ListaCanais:
    """Lê o CSV (UTF-8, com ou sem BOM), valida cada linha e recusa canais repetidos."""
    caminho = Path(caminho)
    conteudo = caminho.read_bytes()
    leitor = csv.DictReader(io.StringIO(conteudo.decode("utf-8-sig"), newline=""))
    if tuple(leitor.fieldnames or ()) != COLUNAS:
        raise ValueError(f"{caminho.name}: colunas esperadas {list(COLUNAS)}, encontradas {leitor.fieldnames}")
    canais: list[CanalDaLista] = []
    vistos: set[str] = set()
    handles: set[str] = set()
    for numero, linha in enumerate(leitor, start=2):
        if None in linha:
            raise ValueError(f"{caminho.name}, linha {numero}: há mais campos que colunas")
        if None in linha.values():
            raise ValueError(f"{caminho.name}, linha {numero}: faltam campos")
        try:
            canal = CanalDaLista.model_validate({chave: valor.strip() for chave, valor in linha.items()})
        except ValidationError as erro:
            raise ValueError(f"{caminho.name}, linha {numero}: {erro}") from None
        if canal.channel_id in vistos:
            raise ValueError(f"{caminho.name}, linha {numero}: canal repetido {canal.channel_id}")
        if canal.handle.lower() in handles:
            raise ValueError(f"{caminho.name}, linha {numero}: handle repetido {canal.handle}")
        vistos.add(canal.channel_id)
        handles.add(canal.handle.lower())
        canais.append(canal)
    return ListaCanais(arquivo=caminho, sha256=hashlib.sha256(conteudo).hexdigest(), canais=tuple(canais))


# ──── Snapshot de metadados da YouTube Data API: registros, respostas brutas e proveniência.
DIAS_RETENCAO = 30
DIRETORIO_PADRAO = RAIZ_PROJETO / "data" / "raw"
TIPOS = ("piloto", "coleta")
ARQUIVO_MANIFESTO = "MANIFESTO_SHA256.txt"
DEPENDENCIAS_REGISTRADAS = ("httpx", "pydantic", "pydantic-settings")


def prazo_de_retencao(coletado_em: datetime) -> datetime:
    return coletado_em + timedelta(days=DIAS_RETENCAO)


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


def estado_do_codigo() -> dict:
    """Identifica o código e o ambiente mesmo sem commit: SHA-256 dos arquivos e versões instaladas."""
    codigo = sorted(p for p in (RAIZ_PROJETO / "src").rglob("*.py") if "__pycache__" not in p.parts)
    arquivos = [*codigo, RAIZ_PROJETO / "pyproject.toml", RAIZ_PROJETO / "uv.lock"]
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


def _excecao_pais_aplicada(resultado: coleta.ResultadoColeta, linhas: list[dict], lista: ListaCanais) -> list[dict]:
    """Canais cujo país ausente na API foi aceito pela exceção: país observado, situação final e evidência da lista."""
    situacao = {linha["channel_id"]: linha["situacao"] for linha in linhas}
    evidencia = {canal.channel_id: canal.evidencia for canal in lista.canais}
    return [
        {
            "channel_id": canal.channel_id,
            "pais_api": canal.pais,
            "situacao": situacao[canal.channel_id],
            "evidencia": evidencia.get(canal.channel_id),
        }
        for canal in resultado.canais
        if coleta.pais_aceito_por_excecao(canal)
    ]


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
    canais = _situacao_dos_canais(resultado)
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
            "canais_excecao_pais": sorted(coleta.CANAIS_EXCECAO_PAIS),
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
        "canais": canais,
        "excecao_pais_aplicada": _excecao_pais_aplicada(resultado, canais, lista),
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


# ──── Coleta metadados dos canais da lista na YouTube Data API e grava um snapshot.
LISTA_PADRAO = RAIZ_PROJETO / "configs" / "canais.csv"


def _inteiro_positivo(texto: str) -> int:
    try:
        valor = int(texto)
    except ValueError:
        raise argparse.ArgumentTypeError(f"esperado inteiro, recebido {texto!r}") from None
    if valor < 1:
        raise argparse.ArgumentTypeError(f"deve ser pelo menos 1, recebido {valor}")
    return valor


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.dados.snapshot", description=__doc__.splitlines()[0])
    parser.add_argument("--lista", type=Path, default=LISTA_PADRAO, help="CSV da lista de canais")
    parser.add_argument("--canais", help="IDs separados por vírgula; sem isso, todos os canais aprovados")
    parser.add_argument(
        "--incluir-candidatos", action="store_true", help="aceita canais ainda não aprovados (exige --piloto)"
    )
    parser.add_argument("--max-paginas", type=_inteiro_positivo, help="limite de páginas de uploads por canal")
    parser.add_argument("--limite-unidades", type=_inteiro_positivo, help="teto de unidades de cota desta execução")
    parser.add_argument("--piloto", action="store_true", help="marca o snapshot como piloto")
    parser.add_argument("--destino", type=Path, default=DIRETORIO_PADRAO, help="pasta dos snapshots")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, *, cliente: ClienteYouTube | None = None, coletado_em: datetime | None = None
) -> int:
    """0 = snapshot gravado; 2 = erro antes da coleta; 3 = coleta interrompida; 4 = falha ao gravar."""
    args = _argumentos(argv)
    status = {StatusConferencia.APROVADO}
    if args.incluir_candidatos:
        status.add(StatusConferencia.CANDIDATO)
    try:
        if args.incluir_candidatos and not args.piloto:
            raise ValueError("--incluir-candidatos só é aceito com --piloto: candidatos não entram em coleta")
        lista = ler_lista(args.lista)
        ids = [i.strip() for i in args.canais.split(",") if i.strip()] if args.canais else None
        alvos = lista.selecionar(ids, status=status)
        if not alvos:
            raise ValueError("nenhum canal selecionado")
        cliente = cliente or ClienteYouTube.a_partir_da_configuracao(limite_unidades=args.limite_unidades)
    except (OSError, ValueError, ChaveAusente) as erro:
        print(f"Erro antes da coleta: {erro}", file=sys.stderr)
        return 2
    tipo = "piloto" if args.piloto else "coleta"
    try:
        with cliente:
            resultado = coletar(cliente, alvos, coletado_em=coletado_em, max_paginas_por_canal=args.max_paginas)
    except ErroYouTube as erro:
        print(f"Coleta interrompida [{erro.codigo}]: {erro}. Nenhum snapshot foi gravado.", file=sys.stderr)
        return 3
    except (KeyError, TypeError, ValueError) as erro:
        print(f"Coleta interrompida por dado inesperado ({type(erro).__name__}). Nenhum snapshot foi gravado.",
              file=sys.stderr)
        return 3
    if resultado.videos_invalidos:
        print(f"Aviso: {len(resultado.videos_invalidos)} vídeo(s) inválido(s); motivos em proveniencia.json.",
              file=sys.stderr)
    try:
        pasta = salvar_snapshot(
            resultado, lista=lista, destino=args.destino, tipo=tipo, status_permitidos=[str(s) for s in status]
        )
    except OSError as erro:
        print(f"Falha ao gravar o snapshot ({type(erro).__name__}: {erro}).", file=sys.stderr)
        return 4
    print(json.dumps({"tipo": tipo, "pasta": str(pasta), **resultado.resumo()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
