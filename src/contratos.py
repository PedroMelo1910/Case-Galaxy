"""Contratos de dados da coleta do YouTube e da entrada da extração multimodal.

Datas sempre com fuso horário. `published_at` é a publicação do vídeo (p) e
`collected_at` é a observação (o): o início da execução da coleta, usado como instante de
observação de todos os registros dela (as leituras ocorrem entre esse instante e o fim da execução).
"""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

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
