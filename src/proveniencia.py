"""Proveniência: SHA-256 de bytes, arquivos, JSON canônico e listas de IDs, e carimbos de tempo em UTC.

Primitivas puras, sem estado e sem dependência do restante do projeto. Os hashes gravados em manifestos, protocolos, caches e
pacotes são calculados por estas funções, de modo que o mesmo conteúdo gera o mesmo SHA-256 em qualquer módulo.
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"


def sha256_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def sha256_arquivo(arquivo: str | Path) -> str:
    return hashlib.sha256(Path(arquivo).read_bytes()).hexdigest()


def sha256_json(conteudo: object) -> str:
    """SHA-256 do JSON canônico (chaves ordenadas, separadores compactos, UTF-8)."""
    return hashlib.sha256(json.dumps(conteudo, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    """SHA-256 de uma lista de identificadores, um por linha, na ordem recebida."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def iso_utc(instante: datetime) -> str:
    return instante.astimezone(UTC).strftime(FORMATO_UTC)


def agora_utc() -> str:
    return datetime.now(UTC).strftime(FORMATO_UTC)
