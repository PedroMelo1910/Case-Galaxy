"""Cache por conteúdo do vídeo + identidade da configuração, retomável e com escrita atômica."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, UTC
from pathlib import Path
from typing import BinaryIO

import numpy as np


# ──── Cache por conteúdo do vídeo + identidade da configuração, retomável e com escrita atômica.
VARIAVEL_RAIZ = "CASEGALAXY_CACHE_AUDIOVISUAL"
RAIZ_PADRAO = Path(os.environ.get(VARIAVEL_RAIZ) or
                   Path(__file__).resolve().parents[2] / "data" / "extracao_audiovisual" / "cache")
BLOCO_HASH_BYTES = 1_048_576


def sha256_arquivo(caminho: Path) -> str:
    """SHA-256 do conteúdo, lido em blocos (vídeos podem ter dezenas de MB)."""
    resumo = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        while bloco := arquivo.read(BLOCO_HASH_BYTES):
            resumo.update(bloco)
    return resumo.hexdigest()


class CacheExtracao:
    """Cache de uma extração: um diretório por (conteúdo, identidade) e um arquivo por etapa."""

    def __init__(self, raiz: Path, sha256_conteudo: str, identidade: str) -> None:
        self.sha256_conteudo, self.identidade = sha256_conteudo, identidade
        self.diretorio = Path(raiz) / sha256_conteudo / identidade

    def _gravar_atomico(self, destino: Path, escrever: Callable[[BinaryIO], object]) -> None:
        """Escreve num temporário no mesmo diretório e troca com `os.replace`; falha não deixa arquivo parcial."""
        self.diretorio.mkdir(parents=True, exist_ok=True)
        descritor, nome = tempfile.mkstemp(dir=self.diretorio, prefix=destino.name + ".", suffix=".tmp")
        try:
            with os.fdopen(descritor, "wb") as arquivo:
                escrever(arquivo)
                arquivo.flush()
                os.fsync(arquivo.fileno())
            os.replace(nome, destino)
        finally:
            Path(nome).unlink(missing_ok=True)

    def gravar(self, etapa: str, dados: Mapping) -> None:
        envelope = {"etapa": etapa, "identidade": self.identidade, "sha256_conteudo": self.sha256_conteudo,
                    "gerado_em": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "dados": dict(dados)}
        corpo = json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")  # falha antes de abrir arquivo
        self._gravar_atomico(self.diretorio / f"{etapa}.json", lambda arquivo: arquivo.write(corpo))

    def ler(self, etapa: str) -> dict | None:
        """Dados da etapa, ou None se não existe, está corrompida ou é de outra identidade/conteúdo."""
        try:
            envelope = json.loads((self.diretorio / f"{etapa}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        valido = (isinstance(envelope, dict) and envelope.get("identidade") == self.identidade
                  and envelope.get("sha256_conteudo") == self.sha256_conteudo and envelope.get("etapa") == etapa
                  and isinstance(envelope.get("dados"), dict))
        return envelope["dados"] if valido else None

    def gravar_array(self, etapa: str, matriz: np.ndarray) -> None:
        self._gravar_atomico(self.diretorio / f"{etapa}.npy",
                             lambda arquivo: np.save(arquivo, matriz, allow_pickle=False))

    def ler_array(self, etapa: str) -> np.ndarray | None:
        try:
            return np.load(self.diretorio / f"{etapa}.npy", allow_pickle=False)
        except (OSError, ValueError, EOFError):  # arquivo vazio ou truncado é miss, nunca erro
            return None

    def diretorio_quadros(self) -> Path:
        return self.diretorio / "quadros"

    def apagar_quadros(self) -> None:
        shutil.rmtree(self.diretorio_quadros(), ignore_errors=True)
