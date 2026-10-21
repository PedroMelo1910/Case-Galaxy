"""Gravação do vídeo enviado em um diretório temporário controlado, removido ao fim da análise."""

import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.interface_web.validacao import EXTENSOES_VIDEO


# ──── Gravação do vídeo enviado em um diretório temporário controlado, removido ao fim da análise.
NOME_DA_PASTA = "casegalaxy_interface"
IDADE_MAXIMA_DE_RESTOS_S = 3600.0


def pasta_base() -> Path:
    """Raiz controlada (criada se preciso, caminho já resolvido)."""
    base = (Path(tempfile.gettempdir()) / NOME_DA_PASTA).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def limpar_restos(base: Path | None = None, idade_maxima_s: float = IDADE_MAXIMA_DE_RESTOS_S) -> int:
    """Remove subpastas antigas deixadas por uma execução interrompida; devolve quantas removeu."""
    base = pasta_base() if base is None else base
    removidas = 0
    for filho in base.iterdir():
        try:
            if filho.is_dir() and time.time() - filho.stat().st_mtime > idade_maxima_s:
                shutil.rmtree(filho, ignore_errors=True)
                removidas += 1
        except OSError:
            continue
    return removidas


@contextmanager
def arquivo_temporario(conteudo: bytes, extensao: str, base: Path | None = None) -> Iterator[Path]:
    """Grava `conteudo` em `<base>/<id>/video.<ext>` e remove a subpasta ao sair, mesmo em caso de erro."""
    base = pasta_base() if base is None else Path(base)
    extensao = extensao.lower().lstrip(".")
    pasta = base / uuid.uuid4().hex
    pasta.mkdir(parents=True)
    try:
        caminho = pasta / f"video.{extensao if extensao in EXTENSOES_VIDEO else 'mp4'}"
        caminho.write_bytes(conteudo)
        yield caminho.resolve()
    finally:
        shutil.rmtree(pasta, ignore_errors=True)
