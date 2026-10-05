"""Lista curada de canais em CSV: leitura, validação e versão (SHA-256 do arquivo)."""

import csv
import hashlib
import io
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from src.contratos import CanalAlvo, IdCanal, Setor, TipoCanal

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
