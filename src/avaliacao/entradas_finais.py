"""Hash das entradas que definem as features do teste (protocolo final §10 d; revisão ML, C1).

O dataset congelado cobre só as linhas de desenvolvimento. As features dos vídeos de teste e da LIM nascem, na hora da
consulta, de entradas que o protocolo precisa fixar: as anotações do Gemini desses IDs, as tabelas de legenda e capa, o
cache de embeddings da transcrição, a tabela restrita de inscritos (baseline 4) e o `uv.lock`. Aqui só se calcula SHA-256 de bytes; nenhum conteúdo é interpretado e
nenhum rótulo é lido.
"""

import hashlib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path


def _sha(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def hash_do_arquivo(arquivo: Path) -> str:
    resumo = hashlib.sha256()
    with open(arquivo, "rb") as entrada:
        for bloco in iter(lambda: entrada.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def hash_da_pasta(pasta: Path) -> str:
    """SHA-256 do conjunto (caminho relativo, hash do conteúdo) de todos os arquivos da pasta, em ordem."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise FileNotFoundError(f"pasta de entrada ausente: {pasta}")
    linhas = [f"{arquivo.relative_to(pasta).as_posix()}\t{hash_do_arquivo(arquivo)}"
              for arquivo in sorted(p for p in pasta.rglob("*") if p.is_file())]
    return _sha("\n".join(linhas).encode("utf-8"))


def hash_das_anotacoes(pasta_cache: Path, ids: Collection[str]) -> dict:
    """Hash dos arquivos `<id>.json` dos IDs pedidos; os ausentes ficam listados (e mudam o hash)."""
    pasta_cache = Path(pasta_cache)
    linhas, ausentes = [], []
    for video_id in sorted(set(ids)):
        arquivo = pasta_cache / f"{video_id}.json"
        if arquivo.is_file():
            linhas.append(f"{video_id}\t{hash_do_arquivo(arquivo)}")
        else:
            ausentes.append(video_id)
            linhas.append(f"{video_id}\tAUSENTE")
    return {"sha256": _sha("\n".join(linhas).encode("utf-8")), "n": len(linhas) - len(ausentes), "ausentes": ausentes}


def hashes_das_entradas(*, caches_de_anotacao: Sequence[tuple[Path, Collection[str]]], pasta_legenda_capa: Path,
                        cache_embeddings: Path, tabela_inscritos: Path, uv_lock: Path) -> Mapping[str, str]:
    """As entradas do protocolo, na forma esperada por `abrir_consulta_ao_teste`."""
    partes = [hash_das_anotacoes(pasta, ids)["sha256"] for pasta, ids in caches_de_anotacao]
    return {"anotacoes_teste_e_lim": _sha("\n".join(partes).encode("utf-8")),
            "legenda_capa": hash_da_pasta(pasta_legenda_capa), "embeddings": hash_da_pasta(cache_embeddings),
            "inscritos": hash_do_arquivo(tabela_inscritos), "uv_lock": hash_do_arquivo(uv_lock)}
