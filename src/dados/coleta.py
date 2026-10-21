"""Módulo consolidado. Reúne:
- Contratos de dados da coleta do YouTube e da entrada da extração multimodal.
- Coleta de metadados de Shorts pela YouTube Data API e preparação da entrada da extração."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator, StringConstraints, ValidationError

from src.integracoes.youtube import (
    ApiDesativada,
    ChaveInvalida,
    ClienteYouTube,
    CotaEsgotada,
    ErroYouTube,
    LimiteDeUnidades,
    RespostaInvalida,
)


# ──── Contratos de dados da coleta do YouTube e da entrada da extração multimodal.
DURACAO_MAXIMA_SHORT_S = 180
DIAS_MATURIDADE = 30

IdVideo = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{11}$")]
IdCanal = Annotated[str, StringConstraints(pattern=r"^UC[A-Za-z0-9_-]{22}$")]
IdSnapshot = Annotated[str, StringConstraints(pattern=r"^\d{8}T\d{6}Z$")]


def url_do_video(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _exigir_observacao_apos_publicacao(publicado: datetime, observado: datetime) -> None:
    if observado < publicado:
        raise ValueError("collected_at não pode ser anterior a published_at")


class Setor(StrEnum):
    BELEZA = "beleza"
    AUTOMOTIVO = "automotivo"
    MODA = "moda"


class TipoCanal(StrEnum):
    MARCA = "marca"
    CRIADOR = "criador"


class Formato(StrEnum):
    """Classificação pela duração e pela proporção do player informadas pela API."""

    SHORT = "short"
    LONGO = "longo"
    HORIZONTAL = "horizontal"
    PROPORCAO_DESCONHECIDA = "proporcao_desconhecida"
    SEM_DURACAO = "sem_duracao"


class Papel(StrEnum):
    """Uso do vídeo, pela data de publicação em relação à coleta."""

    ALVO = "alvo"
    SO_HISTORICO = "so_historico"
    AMOSTRA_LIM = "amostra_lim"
    FORA_DA_JANELA = "fora_da_janela"


class MotivoExclusao(StrEnum):
    """Primeiro motivo que impede o uso do vídeo no papel atribuído (um por vídeo)."""

    TRANSMISSAO = "transmissao_ao_vivo_ou_agendada"
    SEM_DURACAO = "sem_duracao"
    NAO_E_SHORT = "nao_e_short"
    PROPORCAO_DESCONHECIDA = "proporcao_desconhecida"
    NAO_PUBLICO = "nao_publico"
    FORA_DA_JANELA = "fora_da_janela"
    CURTIDAS_OCULTAS = "curtidas_ocultas"


class MotivoExclusaoCanal(StrEnum):
    PAIS_NAO_INFORMADO = "pais_nao_informado"
    PAIS_DIFERENTE = "pais_diferente"
    POUCOS_SHORTS = "poucos_shorts_alvo"
    AMOSTRA_PARCIAL = "amostra_parcial_indeterminado"  # paginação parou no limite: contagem incompleta
    FALHA_NA_COLETA = "falha_na_coleta"  # uploads do canal não puderam ser lidos


class _Registro(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CanalAlvo(_Registro):
    """Canal da lista curada; setor e tipo são atribuídos por canal."""

    channel_id: IdCanal
    setor: Setor
    tipo_canal: TipoCanal


class CanalColetado(_Registro):
    """Dados do canal lidos em `channels.list`."""

    channel_id: IdCanal
    setor: Setor
    tipo_canal: TipoCanal
    snapshot_id: IdSnapshot
    collected_at: AwareDatetime
    titulo: str
    pais: str | None
    playlist_uploads: str | None
    inscritos: int | None = Field(description="Valor atual na coleta; nunca entra como atributo do modelo")
    total_videos: int | None


class VideoColetado(_Registro):
    """Metadados de um vídeo lidos em `videos.list`, já classificados."""

    video_id: IdVideo
    channel_id: IdCanal
    snapshot_id: IdSnapshot
    published_at: AwareDatetime
    collected_at: AwareDatetime
    titulo: str
    descricao: str
    tags: tuple[str, ...]
    categoria_id: str | None
    idioma_padrao: str | None
    idioma_audio_padrao: str | None
    duracao_s: int | None
    largura_player: int | None
    altura_player: int | None
    formato: Formato
    privacidade: str
    transmissao: str
    curtidas: int | None = Field(description="None quando o canal oculta as curtidas")
    visualizacoes: int | None = Field(description="Descritivo; nunca entra como atributo do modelo")
    comentarios: int | None = Field(description="Descritivo; nunca entra como atributo do modelo")
    thumbnail_url: str | None
    papel: Papel
    motivo_exclusao: MotivoExclusao | None

    @property
    def elegivel(self) -> bool:
        return self.motivo_exclusao is None

    @model_validator(mode="after")
    def _datas_coerentes(self) -> "VideoColetado":
        _exigir_observacao_apos_publicacao(self.published_at, self.collected_at)
        return self


class ItemExtracao(_Registro):
    """Entrada da extração multimodal de um vídeo da base.

    A chave do cache da extração é o `video_id`. Estatísticas do vídeo e do canal
    (curtidas, visualizações, comentários, inscritos) ficam fora deste contrato.
    Com `thumbnail_url` ausente, a extração registra aviso e segue sem thumbnail.
    """

    video_id: IdVideo
    channel_id: IdCanal
    snapshot_id: IdSnapshot
    papel: Papel
    setor: Setor
    url_video: str
    thumbnail_url: str | None
    duracao_s: int = Field(gt=0, le=DURACAO_MAXIMA_SHORT_S)
    published_at: AwareDatetime
    collected_at: AwareDatetime

    @model_validator(mode="after")
    def _coerencia(self) -> "ItemExtracao":
        if self.papel not in (Papel.ALVO, Papel.AMOSTRA_LIM):
            raise ValueError("papel deve ser 'alvo' ou 'amostra_lim'")
        _exigir_observacao_apos_publicacao(self.published_at, self.collected_at)
        if self.url_video != url_do_video(self.video_id):
            raise ValueError("url_video não corresponde ao video_id")
        maduro = self.collected_at - self.published_at >= timedelta(days=DIAS_MATURIDADE)
        if (self.papel is Papel.ALVO) != maduro:
            raise ValueError(f"alvo exige pelo menos {DIAS_MATURIDADE} dias na coleta; amostra_lim, menos")
        return self


# ──── Coleta de metadados de Shorts pela YouTube Data API e preparação da entrada da extração.
# Erros que valem para a coleta inteira; os demais ficam registrados no canal em que ocorreram.
ERROS_FATAIS = (CotaEsgotada, ChaveInvalida, ApiDesativada, LimiteDeUnidades)

TOLERANCIA_QUADRADO = 0.02  # largura/altura até 1,02 conta como quadrado (arredondamento do player)
DIAS_JANELA_ALVOS = 365  # alvos publicados nos 12 meses até a coleta, menos os últimos 30 dias
DIAS_HISTORICO_EXTRA = 183  # 6 meses anteriores aos alvos, só para histórico do canal
MINIMO_SHORTS_POR_CANAL = 20
PAIS_EXIGIDO = "BR"
# Canais aprovados que a API publica sem país, com atuação brasileira comprovada por fonte externa (coluna
# `evidencia` de configs/canais.csv). Valem só com o país ausente: país informado diferente de BR continua excluindo.
CANAIS_EXCECAO_PAIS = frozenset(
    {
        "UCtDew4iI18GOrvPXrvhXzhw",  # Peugeot Brasil
        "UCsitysA6ZME95b0j5uoAe9Q",  # Toyota do Brasil
        "UCcHR0RvLYk33ppkjBw3UaEA",  # Mari Maria
    }
)
PRIORIDADE_THUMBNAIL = ("maxres", "standard", "high", "medium", "default")
FORMATO_SNAPSHOT = "%Y%m%dT%H%M%SZ"

_DURACAO_ISO = re.compile(
    r"^P(?:(?P<dias>\d+)D)?(?:T(?=\d)(?:(?P<horas>\d+)H)?(?:(?P<minutos>\d+)M)?(?:(?P<segundos>\d+)S)?)?$"
)


def duracao_em_segundos(iso: str | None) -> int | None:
    """Converte a duração ISO 8601 da API (ex.: PT1M5S); devolve None se não reconhecer."""
    achado = _DURACAO_ISO.match(iso or "")
    if achado is None or all(valor is None for valor in achado.groupdict().values()):
        return None
    dias, horas, minutos, segundos = (int(achado[nome] or 0) for nome in ("dias", "horas", "minutos", "segundos"))
    return ((dias * 24 + horas) * 60 + minutos) * 60 + segundos


def classificar_formato(duracao_s: int | None, largura: int | None, altura: int | None) -> Formato:
    if not duracao_s:
        return Formato.SEM_DURACAO
    if duracao_s > DURACAO_MAXIMA_SHORT_S:
        return Formato.LONGO
    if not largura or not altura:
        return Formato.PROPORCAO_DESCONHECIDA
    if largura / altura > 1 + TOLERANCIA_QUADRADO:
        return Formato.HORIZONTAL
    return Formato.SHORT


@dataclass(frozen=True)
class Janela:
    """Janelas de publicação relativas ao instante da coleta."""

    coletado_em: datetime

    def __post_init__(self) -> None:
        if self.coletado_em.tzinfo is None:
            raise ValueError("coletado_em precisa de fuso horário")

    @property
    def fim_alvos(self) -> datetime:
        return self.coletado_em - timedelta(days=DIAS_MATURIDADE)

    @property
    def inicio_alvos(self) -> datetime:
        return self.coletado_em - timedelta(days=DIAS_JANELA_ALVOS)

    @property
    def inicio_historico(self) -> datetime:
        return self.inicio_alvos - timedelta(days=DIAS_HISTORICO_EXTRA)

    def papel(self, publicado_em: datetime) -> Papel:
        if publicado_em > self.fim_alvos:
            return Papel.AMOSTRA_LIM
        if publicado_em >= self.inicio_alvos:
            return Papel.ALVO
        if publicado_em >= self.inicio_historico:
            return Papel.SO_HISTORICO
        return Papel.FORA_DA_JANELA


def motivo_exclusao(
    *, formato: Formato, transmissao: str, privacidade: str, papel: Papel, curtidas: int | None
) -> MotivoExclusao | None:
    """Primeiro motivo aplicável, nesta ordem; None quando o vídeo serve ao seu papel."""
    if transmissao != "none":
        return MotivoExclusao.TRANSMISSAO
    if formato is Formato.SEM_DURACAO:
        return MotivoExclusao.SEM_DURACAO
    if formato in (Formato.LONGO, Formato.HORIZONTAL):
        return MotivoExclusao.NAO_E_SHORT
    if formato is Formato.PROPORCAO_DESCONHECIDA:
        return MotivoExclusao.PROPORCAO_DESCONHECIDA
    if privacidade != "public":
        return MotivoExclusao.NAO_PUBLICO
    if papel is Papel.FORA_DA_JANELA:
        return MotivoExclusao.FORA_DA_JANELA
    if curtidas is None and papel in (Papel.ALVO, Papel.SO_HISTORICO):
        return MotivoExclusao.CURTIDAS_OCULTAS
    return None


def _instante(texto: str) -> datetime:
    instante = datetime.fromisoformat(texto)
    if instante.tzinfo is None:
        raise ValueError(f"data sem fuso horário: {texto}")
    return instante.astimezone(UTC)


def _instante_ou_none(texto: object) -> datetime | None:
    if not isinstance(texto, str) or not texto:
        return None
    try:
        return _instante(texto)
    except ValueError:
        return None


def _inteiro(valor: str | int | None) -> int | None:
    return None if valor is None else int(valor)


def _thumbnail(thumbnails: dict) -> str | None:
    for nome in PRIORIDADE_THUMBNAIL:
        url = thumbnails.get(nome, {}).get("url")
        if url:
            return url
    return None


def video_de_recurso(recurso: dict, *, janela: Janela, snapshot_id: str) -> VideoColetado:
    """Converte um item de `videos.list` em registro classificado."""
    snippet = recurso.get("snippet", {})
    estatisticas = recurso.get("statistics", {})
    player = recurso.get("player", {})
    publicado_em = _instante(snippet["publishedAt"])
    duracao = duracao_em_segundos(recurso.get("contentDetails", {}).get("duration"))
    largura, altura = _inteiro(player.get("embedWidth")), _inteiro(player.get("embedHeight"))
    formato = classificar_formato(duracao, largura, altura)
    transmissao = snippet.get("liveBroadcastContent", "none")
    privacidade = recurso.get("status", {}).get("privacyStatus", "desconhecida")
    curtidas = _inteiro(estatisticas.get("likeCount"))
    papel = janela.papel(publicado_em)
    return VideoColetado(
        video_id=recurso["id"],
        channel_id=snippet["channelId"],
        snapshot_id=snapshot_id,
        published_at=publicado_em,
        collected_at=janela.coletado_em,
        titulo=snippet.get("title", ""),
        descricao=snippet.get("description", ""),
        tags=tuple(snippet.get("tags", ())),
        categoria_id=snippet.get("categoryId"),
        idioma_padrao=snippet.get("defaultLanguage"),
        idioma_audio_padrao=snippet.get("defaultAudioLanguage"),
        duracao_s=duracao,
        largura_player=largura,
        altura_player=altura,
        formato=formato,
        privacidade=privacidade,
        transmissao=transmissao,
        curtidas=curtidas,
        visualizacoes=_inteiro(estatisticas.get("viewCount")),
        comentarios=_inteiro(estatisticas.get("commentCount")),
        thumbnail_url=_thumbnail(snippet.get("thumbnails", {})),
        papel=papel,
        motivo_exclusao=motivo_exclusao(
            formato=formato, transmissao=transmissao, privacidade=privacidade, papel=papel, curtidas=curtidas
        ),
    )


def canal_de_recurso(recurso: dict, alvo: CanalAlvo, *, coletado_em: datetime, snapshot_id: str) -> CanalColetado:
    """Converte um item de `channels.list`; setor e tipo vêm da lista curada."""
    snippet = recurso.get("snippet", {})
    estatisticas = recurso.get("statistics", {})
    uploads = recurso.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads") or None
    inscritos = None if estatisticas.get("hiddenSubscriberCount") else _inteiro(estatisticas.get("subscriberCount"))
    return CanalColetado(
        channel_id=recurso["id"],
        setor=alvo.setor,
        tipo_canal=alvo.tipo_canal,
        snapshot_id=snapshot_id,
        collected_at=coletado_em,
        titulo=snippet.get("title", ""),
        pais=snippet.get("country"),
        playlist_uploads=uploads,
        inscritos=inscritos,
        total_videos=_inteiro(estatisticas.get("videoCount")),
    )


@dataclass(frozen=True)
class UploadsDoCanal:
    """Resultado da paginação dos uploads de um canal."""

    ids: tuple[str, ...]
    itens_sem_data: int
    paginas_lidas: int
    interrompida: bool  # parou no limite de páginas e a última página lida estava toda dentro da janela
    itens_lidos: int = 0
    itens_anteriores_a_janela: int = 0
    data_mais_antiga_lida: datetime | None = None


def percorrer_uploads(
    cliente: ClienteYouTube, playlist_id: str, desde: datetime, *, max_paginas: int | None = None
) -> UploadsDoCanal:
    """IDs publicados a partir de `desde`, itens sem data válida (privados, apagados ou malformados) e se a
    leitura ficou parcial.

    Pagina até o fim, até uma página inteira ser anterior a `desde` ou até `max_paginas`. Cada item é
    filtrado pela própria data. No limite de páginas, a leitura só conta como parcial se a última página
    não tiver nenhum item anterior a `desde`; isso supõe a ordem do mais novo para o mais antigo
    observada na playlist de uploads, que a documentação da API não garante.
    """
    if max_paginas is not None and max_paginas < 1:
        raise ValueError("max_paginas deve ser pelo menos 1")
    ids: list[str] = []
    sem_data = paginas = lidos = anteriores = 0
    mais_antiga: datetime | None = None
    token: str | None = None
    tokens_vistos: set[str] = set()
    while True:
        pagina = cliente.pagina_uploads(playlist_id, token)
        paginas += 1
        itens = pagina.get("items", [])
        if not isinstance(itens, list):
            raise RespostaInvalida(f"playlistItems: campo 'items' inválido na playlist {playlist_id}")
        datas: list[datetime] = []
        for item in itens:
            lidos += 1
            detalhes = item.get("contentDetails", {}) if isinstance(item, dict) else {}
            video_id, instante = detalhes.get("videoId"), _instante_ou_none(detalhes.get("videoPublishedAt"))
            if not video_id or instante is None:
                sem_data += 1
                continue
            datas.append(instante)
            if instante >= desde:
                ids.append(video_id)
            else:
                anteriores += 1
        if datas:
            mais_antiga = min([*datas, mais_antiga] if mais_antiga else datas)
        token = pagina.get("nextPageToken")
        pagina_toda_antiga = bool(datas) and max(datas) < desde
        passou_do_inicio = bool(datas) and min(datas) < desde
        no_limite = max_paginas is not None and paginas >= max_paginas
        if not token or pagina_toda_antiga or no_limite:
            interrompida = bool(token) and no_limite and not passou_do_inicio
            return UploadsDoCanal(tuple(ids), sem_data, paginas, interrompida, lidos, anteriores, mais_antiga)
        if token in tokens_vistos:
            raise RespostaInvalida(f"playlistItems: token de página repetido na playlist {playlist_id}")
        tokens_vistos.add(token)


def listar_uploads(cliente: ClienteYouTube, playlist_id: str, desde: datetime) -> tuple[list[str], int]:
    """Atalho sem limite de páginas: IDs dentro da janela e número de itens sem data."""
    uploads = percorrer_uploads(cliente, playlist_id, desde)
    return list(uploads.ids), uploads.itens_sem_data


@dataclass(frozen=True)
class ResultadoColeta:
    snapshot_id: str
    coletado_em: datetime
    canais: tuple[CanalColetado, ...]
    videos: tuple[VideoColetado, ...]
    canais_nao_encontrados: tuple[str, ...]
    videos_indisponiveis: tuple[str, ...]  # listados na playlist, mas ausentes em videos.list
    videos_invalidos: tuple[str, ...]  # resposta sem campos obrigatórios
    itens_playlist_sem_data: int
    unidades_cota: int
    motivos_invalidos: dict[str, str] = field(default_factory=dict)  # video_id → motivo
    canais_solicitados: tuple[str, ...] = ()
    uploads_por_canal: dict[str, UploadsDoCanal] = field(default_factory=dict)
    falhas_por_canal: dict[str, str] = field(default_factory=dict)  # código do erro ao ler os uploads
    brutos_canais: tuple[dict, ...] = ()  # itens de channels.list como vieram da API
    brutos_videos: tuple[dict, ...] = ()  # itens de videos.list como vieram da API
    limite_unidades: int | None = None
    max_paginas_por_canal: int | None = None
    terminado_em: datetime | None = None

    @property
    def canais_parciais(self) -> tuple[str, ...]:
        return tuple(canal for canal, uploads in self.uploads_por_canal.items() if uploads.interrompida)

    def resumo(self) -> dict:
        elegiveis = Counter(str(v.papel) for v in self.videos if v.elegivel)
        exclusoes = Counter(str(v.motivo_exclusao) for v in self.videos if not v.elegivel)
        curtos = [v for v in self.videos if v.duracao_s and v.duracao_s <= DURACAO_MAXIMA_SHORT_S]
        desconhecidos = sum(v.formato is Formato.PROPORCAO_DESCONHECIDA for v in curtos)
        return {
            "snapshot_id": self.snapshot_id,
            "canais_coletados": len(self.canais),
            "canais_nao_encontrados": len(self.canais_nao_encontrados),
            "videos": len(self.videos),
            "elegiveis_por_papel": dict(elegiveis),
            "exclusoes_por_motivo": dict(exclusoes),
            "videos_indisponiveis": len(self.videos_indisponiveis),
            "videos_invalidos": len(self.videos_invalidos),
            "itens_playlist_sem_data": self.itens_playlist_sem_data,
            "fracao_proporcao_desconhecida_curtos": desconhecidos / len(curtos) if curtos else None,
            "canais_com_amostra_parcial": len(self.canais_parciais),
            "canais_com_falha": len(self.falhas_por_canal),
            "unidades_cota": self.unidades_cota,
        }


def _sem_repeticao(canais_alvo: Sequence[CanalAlvo]) -> dict[str, CanalAlvo]:
    por_id: dict[str, CanalAlvo] = {}
    for alvo in canais_alvo:
        if alvo.channel_id in por_id:
            raise ValueError(f"canal repetido na lista: {alvo.channel_id}")
        por_id[alvo.channel_id] = alvo
    return por_id


def _motivo(erro: Exception) -> str:
    if isinstance(erro, ValidationError):
        campos = ".".join(str(parte) for parte in erro.errors()[0]["loc"]) if erro.errors() else "?"
        return f"ValidationError: {campos}"
    if isinstance(erro, KeyError):
        return f"KeyError: {erro.args[0] if erro.args else '?'}"
    return type(erro).__name__


def _converter_videos(
    recursos: list[dict], janela: Janela, snapshot_id: str
) -> tuple[list[VideoColetado], dict[str, str]]:
    """Registros válidos e, para os demais, o motivo da recusa (por video_id)."""
    videos: list[VideoColetado] = []
    invalidos: dict[str, str] = {}
    for recurso in recursos:
        try:
            videos.append(video_de_recurso(recurso, janela=janela, snapshot_id=snapshot_id))
        except (KeyError, TypeError, ValueError) as erro:
            invalidos[str(recurso.get("id", "?"))] = _motivo(erro)
    return videos, invalidos


def _ler_canais(
    cliente: ClienteYouTube,
    recursos: list[dict],
    por_id: dict[str, CanalAlvo],
    janela: Janela,
    snapshot_id: str,
    max_paginas: int | None,
) -> tuple[list[CanalColetado], dict[str, UploadsDoCanal], dict[str, str]]:
    """Canais, uploads dentro da janela e falhas por canal. Só erros de ERROS_FATAIS interrompem."""
    canais: list[CanalColetado] = []
    uploads: dict[str, UploadsDoCanal] = {}
    falhas: dict[str, str] = {}
    for recurso in recursos:
        channel_id = str(recurso.get("id"))
        try:
            alvo = por_id[channel_id]
            canal = canal_de_recurso(recurso, alvo, coletado_em=janela.coletado_em, snapshot_id=snapshot_id)
        except (KeyError, TypeError, ValueError):
            falhas[channel_id] = "CANAL_INVALIDO"
            continue
        canais.append(canal)
        if not canal.playlist_uploads:
            falhas[channel_id] = "SEM_PLAYLIST_UPLOADS"
            continue
        try:
            uploads[channel_id] = percorrer_uploads(
                cliente, canal.playlist_uploads, janela.inicio_historico, max_paginas=max_paginas
            )
        except ERROS_FATAIS:
            raise
        except ErroYouTube as erro:
            falhas[channel_id] = erro.codigo
    return canais, uploads, falhas


def coletar(
    cliente: ClienteYouTube,
    canais_alvo: Sequence[CanalAlvo],
    *,
    coletado_em: datetime | None = None,
    max_paginas_por_canal: int | None = None,
) -> ResultadoColeta:
    """Coleta canais, uploads dentro da janela e metadados dos vídeos.

    `coletado_em` (início da execução) é o instante de observação registrado em todos os vídeos.
    Falhas ao ler um canal ficam registradas e a coleta segue; cota esgotada, chave inválida, API
    desativada e teto de unidades interrompem tudo.
    """
    por_id = _sem_repeticao(canais_alvo)
    unidades_antes = cliente.unidades_usadas
    janela = Janela((coletado_em or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0))
    snapshot_id = janela.coletado_em.strftime(FORMATO_SNAPSHOT)

    recursos_canais = cliente.canais(list(por_id))
    canais, uploads, falhas = _ler_canais(
        cliente, recursos_canais, por_id, janela, snapshot_id, max_paginas_por_canal
    )
    ids_videos = list(dict.fromkeys(i for lido in uploads.values() for i in lido.ids))
    recursos_videos = cliente.videos(ids_videos)
    videos, invalidos = _converter_videos(recursos_videos, janela, snapshot_id)
    recebidos = {v.video_id for v in videos} | set(invalidos)
    encontrados = {c.channel_id for c in canais} | set(falhas)
    return ResultadoColeta(
        snapshot_id=snapshot_id,
        coletado_em=janela.coletado_em,
        canais=tuple(canais),
        videos=tuple(videos),
        canais_nao_encontrados=tuple(i for i in por_id if i not in encontrados),
        videos_indisponiveis=tuple(i for i in ids_videos if i not in recebidos),
        videos_invalidos=tuple(invalidos),
        itens_playlist_sem_data=sum(lido.itens_sem_data for lido in uploads.values()),
        unidades_cota=cliente.unidades_usadas - unidades_antes,
        motivos_invalidos=invalidos,
        canais_solicitados=tuple(por_id),
        uploads_por_canal=uploads,
        falhas_por_canal=falhas,
        brutos_canais=tuple(recursos_canais),
        brutos_videos=tuple(recursos_videos),
        limite_unidades=cliente.limite_unidades,
        max_paginas_por_canal=max_paginas_por_canal,
        terminado_em=datetime.now(UTC).replace(microsecond=0),
    )


def pais_aceito_por_excecao(canal: CanalColetado) -> bool:
    """Verdadeiro quando a API não informa país e o canal está em CANAIS_EXCECAO_PAIS.

    Só o critério de país fica atendido; os demais critérios de elegibilidade continuam valendo. O país observado
    continua registrado como ausente.
    """
    return canal.pais is None and canal.channel_id in CANAIS_EXCECAO_PAIS


def avaliar_canais(resultado: ResultadoColeta) -> dict[str, MotivoExclusaoCanal | None]:
    """Canal elegível: país BR informado na API (ou ausente, só em CANAIS_EXCECAO_PAIS) e pelo menos 20 Shorts
    alvo elegíveis.

    Amostra parcial ou falha na leitura dos uploads não permitem concluir que faltam Shorts.
    """
    alvos = Counter(v.channel_id for v in resultado.videos if v.elegivel and v.papel is Papel.ALVO)
    parciais = set(resultado.canais_parciais)
    situacao: dict[str, MotivoExclusaoCanal | None] = {}
    for canal in resultado.canais:
        if canal.pais is None and not pais_aceito_por_excecao(canal):
            situacao[canal.channel_id] = MotivoExclusaoCanal.PAIS_NAO_INFORMADO
        elif canal.pais is not None and canal.pais != PAIS_EXIGIDO:
            situacao[canal.channel_id] = MotivoExclusaoCanal.PAIS_DIFERENTE
        elif canal.channel_id in resultado.falhas_por_canal:
            situacao[canal.channel_id] = MotivoExclusaoCanal.FALHA_NA_COLETA
        elif alvos[canal.channel_id] >= MINIMO_SHORTS_POR_CANAL:
            situacao[canal.channel_id] = None
        elif canal.channel_id in parciais:
            situacao[canal.channel_id] = MotivoExclusaoCanal.AMOSTRA_PARCIAL
        else:
            situacao[canal.channel_id] = MotivoExclusaoCanal.POUCOS_SHORTS
    return situacao


def itens_para_extracao(resultado: ResultadoColeta) -> list[ItemExtracao]:
    """Alvos e amostra LIM elegíveis de canais elegíveis, no contrato da extração."""
    situacao = avaliar_canais(resultado)
    setor = {c.channel_id: c.setor for c in resultado.canais}
    return [
        ItemExtracao(
            video_id=v.video_id,
            channel_id=v.channel_id,
            snapshot_id=v.snapshot_id,
            papel=v.papel,
            setor=setor[v.channel_id],
            url_video=url_do_video(v.video_id),
            thumbnail_url=v.thumbnail_url,
            duracao_s=v.duracao_s,
            published_at=v.published_at,
            collected_at=v.collected_at,
        )
        for v in resultado.videos
        if v.elegivel
        and v.duracao_s is not None
        and v.papel in (Papel.ALVO, Papel.AMOSTRA_LIM)
        and situacao.get(v.channel_id) is None
    ]
