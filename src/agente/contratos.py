"""Contratos tipados de entrada e saída do agente e códigos de erro (ARQUITETURA §9)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qs, urlparse

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

MAX_CRIATIVOS_ORDENAR = 5
MIN_CRIATIVOS_ORDENAR = 2


_ID_VIDEO = re.compile(r"^[A-Za-z0-9_-]{11}$")
PADRAO_ID_VIDEO = r"^[A-Za-z0-9_-]{11}$"  # mesmo formato, para o esquema público (JSON Schema)
_HOSTS_YOUTUBE = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


def video_id_da_url(url: str | None) -> str | None:
    """ID de 11 caracteres de uma URL pública do YouTube (watch, shorts, embed ou youtu.be); senão None."""
    if not url:
        return None
    partes = urlparse(url.strip())
    if partes.scheme != "https":
        return None
    candidato = None
    if partes.hostname == "youtu.be":
        candidato = partes.path.lstrip("/").split("/")[0]
    elif partes.hostname in _HOSTS_YOUTUBE:
        if partes.path == "/watch":
            candidato = (parse_qs(partes.query).get("v") or [None])[0]
        else:
            m = re.match(r"^/(?:shorts|embed)/([^/?#]+)", partes.path)
            candidato = m.group(1) if m else None
    return candidato if candidato and _ID_VIDEO.fullmatch(candidato) else None


class Codigo(StrEnum):
    ENTRADA_INVALIDA = "ENTRADA_INVALIDA"
    MIDIA_INVALIDA = "MIDIA_INVALIDA"
    CANAIS_DIFERENTES = "CANAIS_DIFERENTES"
    MODALIDADES_DIFERENTES = "MODALIDADES_DIFERENTES"
    NIVEIS_DIFERENTES = "NIVEIS_DIFERENTES"
    HISTORICO_INDISPONIVEL = "HISTORICO_INDISPONIVEL"
    MODELO_INDISPONIVEL = "MODELO_INDISPONIVEL"
    COTA_ESGOTADA = "COTA_ESGOTADA"
    ERRO_INTERNO = "ERRO_INTERNO"


class ErroAgente(Exception):
    """Falha prevista do fluxo: vira `{success: false, error}` sem rastro de pilha."""

    def __init__(self, codigo: Codigo, mensagem: str, detalhes: Mapping[str, object] | None = None) -> None:
        super().__init__(mensagem)
        self.codigo, self.mensagem, self.detalhes = codigo, mensagem, dict(detalhes or {})

    def como_dict(self) -> dict:
        return {"code": str(self.codigo), "message": self.mensagem, "details": self.detalhes}


class _Modelo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Legenda(_Modelo):
    """Legenda planejada: título e descrição, como na legenda publicada usada no treino."""

    title: str | None = None
    description: str | None = None


class CriativoVideo(_Modelo):
    tipo: Literal["video"]
    url: str | None = None
    arquivo: Path | None = None

    @model_validator(mode="after")
    def _uma_origem(self) -> CriativoVideo:
        if (self.url is None) == (self.arquivo is None):
            raise ValueError("vídeo exige exatamente um entre `url` e `arquivo`")
        if self.url is not None and video_id_da_url(self.url) is None:
            raise ValueError("`url` deve ser um link https público do YouTube com ID de vídeo válido")
        return self


class CriativoImagem(_Modelo):
    tipo: Literal["imagem"]
    arquivo: Path


class CriativoTexto(_Modelo):
    tipo: Literal["texto"]
    texto: str = Field(min_length=1)


Criativo = Annotated[CriativoVideo | CriativoImagem | CriativoTexto, Field(discriminator="tipo")]


class PostFornecido(_Modelo):
    """Post anterior informado pelo chamador. `collected_at` diz quando as curtidas foram lidas.

    `likes` ausente (None) conta só para a cadência de publicação; não vira zero nem entra na referência.

    `video_id` (opcional, ID de 11 caracteres do YouTube) identifica o post. Com ele o agente exclui o próprio criativo
    do histórico, da referência e da cadência antes dos cálculos, qualquer que seja a data planejada (pareado com a
    URL do vídeo ou com `own_video_id` do pedido). Sem ele, só a defasagem de 30 dias protege, e o agente avisa.
    """

    published_at: AwareDatetime
    likes: int | None = Field(default=None, ge=0)
    collected_at: AwareDatetime
    video_id: str | None = Field(
        default=None, pattern=PADRAO_ID_VIDEO,
        description="ID de 11 caracteres do vídeo no YouTube (opcional). Permite excluir o próprio criativo do "
                    "histórico quando ele aparece na lista.")

    @model_validator(mode="after")
    def _ordem(self) -> PostFornecido:
        if self.collected_at < self.published_at:
            raise ValueError("collected_at anterior a published_at")
        return self


class PostConsultado(PostFornecido):
    """Uso interno: post devolvido por um consultor do YouTube, com o id real do vídeo.

    Mantido por compatibilidade: o id agora também existe (opcional) em `PostFornecido`. Aqui o formato não é
    validado, pois os consultores e os dublês de teste podem usar ids curtos.
    """

    video_id: str | None = None


class PedidoPrever(_Modelo):
    creative: Criativo
    caption: Legenda | None = None
    tags: tuple[str, ...] = ()
    thumbnail: Path | None = None
    planned_datetime: AwareDatetime
    sector: str
    channel_id: str | None = None
    channel_stats: tuple[PostFornecido, ...] | None = None
    channel_type: Literal["marca", "criador"] | None = None
    duration_s: int | None = Field(default=None, gt=0)
    consultar_youtube: bool = False
    own_video_id: str | None = Field(
        default=None, pattern=PADRAO_ID_VIDEO,
        description="ID de 11 caracteres do YouTube do próprio criativo, quando ele já existe no canal (opcional). "
                    "Vem da URL em vídeos com `url`; para imagem, texto e vídeo em arquivo, informe-o para que o "
                    "agente exclua o criativo do histórico antes da referência e da cadência.")

    @model_validator(mode="after")
    def _canal(self) -> PedidoPrever:
        if self.channel_id is None and self.channel_stats is None:
            raise ValueError("informe `channel_id` ou `channel_stats`")
        return self

    @model_validator(mode="after")
    def _identidade(self) -> PedidoPrever:
        da_url = video_id_da_url(getattr(self.creative, "url", None))
        if self.own_video_id is not None and da_url is not None and da_url != self.own_video_id:
            raise ValueError("`own_video_id` diverge do ID da `url` do vídeo")
        ids = [p.video_id for p in self.channel_stats or () if p.video_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("`channel_stats` repete um `video_id`")
        return self

    @property
    def modalidade(self) -> str:
        return self.creative.tipo


def id_do_proprio_video(pedido: PedidoPrever) -> str | None:
    """ID do próprio criativo no YouTube: o da URL do vídeo ou `own_video_id`; None se o pedido não o identifica."""
    return video_id_da_url(getattr(pedido.creative, "url", None)) or pedido.own_video_id


class PedidoOrdenar(_Modelo):
    creatives: tuple[PedidoPrever, ...]


ORIGEM_SNAPSHOT = "historico_do_snapshot"
ORIGEM_FORNECIDO = "historico_fornecido"
ORIGEM_CONSULTA_ATUAL = "consulta_atual"


def envelope_ok(dados: dict) -> dict:
    return {"success": True, "data": dados}


def envelope_erro(erro: ErroAgente) -> dict:
    return {"success": False, "error": erro.como_dict()}
