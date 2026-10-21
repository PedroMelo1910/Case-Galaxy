"""Partições temporais dos vídeos-alvo (D-D2): treino, validação e teste em 3 sub-blocos, pela ordem de publicação."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path

from src.extracao.rubrica import gravar_json_atomico
from src.dados import curadoria
from src.dados.snapshot import ARQUIVO_MANIFESTO
from src.proveniencia import iso_utc as _iso, sha256_arquivo as _sha256_arquivo, sha256_json as _sha256_json


# ──── Partições temporais dos vídeos-alvo (D-D2): treino, validação e teste em 3 sub-blocos, pela ordem de
VERSAO = "1"
BLOCOS = ("treino", "validacao", "teste_1", "teste_2", "teste_3")
INTERVALO_DIAS = 30
ARQUIVO = "particao.json"
REGRAS = {
    "populacao": "vídeos com papel alvo na base curada; só histórico e amostra LIM ficam fora",
    "ordem": "published_at crescente; empate desfeito pelo video_id",
    "fracoes": {"treino": 0.55, "validacao": 0.15, "teste": 0.30},
    "arredondamento": "cortes inteiros: treino ⌊55n/100⌋, validação até ⌊70n/100⌋; teste dividido em ⌊m/3⌋ e "
    "⌊2m/3⌋, com m vídeos de teste",
    "empates": "um corte entre vídeos publicados no mesmo instante avança até o instante mudar; o empate fica no "
    "bloco anterior",
    "intervalo_dias": INTERVALO_DIAS,
    "uso_do_intervalo": "para avaliar um bloco, o treino só usa vídeos publicados até treino_ate_utc (30 dias antes "
    "do primeiro vídeo do bloco)",
}
NAO_INCLUIDO = [
    "sorteios de canais não vistos (D-D2), definidos no pacote de avaliação",
    "detecção de repostagens: IDs únicos não provam que o mesmo criativo não aparece em blocos diferentes",
]


class ErroParticao(ValueError):
    """Partição inconsistente com a base ou com o manifesto congelado."""


@dataclass(frozen=True)
class AlvoNoBloco:
    video_id: str
    published_at: datetime
    bloco: str


def cortes(n: int) -> tuple[int, ...]:
    """Índices de início de cada bloco, mais o fim, para n vídeos em ordem de publicação."""
    if n < 0:
        raise ValueError("n não pode ser negativo")
    fim_treino, fim_validacao = 55 * n // 100, 70 * n // 100
    teste = n - fim_validacao
    return 0, fim_treino, fim_validacao, fim_validacao + teste // 3, fim_validacao + 2 * teste // 3, n


def particionar(alvos: Sequence[tuple[str, datetime]]) -> list[AlvoNoBloco]:
    """Atribui cada (video_id, published_at) a um bloco, em ordem cronológica, sem separar empates de instante."""
    ordenados = sorted(alvos, key=lambda alvo: (alvo[1], alvo[0]))
    if len({video_id for video_id, _ in ordenados}) != len(ordenados):
        raise ErroParticao("vídeo repetido entre os alvos")
    limites = list(cortes(len(ordenados)))
    for i in range(1, len(limites) - 1):
        limites[i] = max(limites[i], limites[i - 1])
        while 0 < limites[i] < len(ordenados) and ordenados[limites[i]][1] == ordenados[limites[i] - 1][1]:
            limites[i] += 1
    return [
        AlvoNoBloco(video_id, publicado, bloco)
        for bloco, inicio, fim in zip(BLOCOS, limites, limites[1:], strict=False)
        for video_id, publicado in ordenados[inicio:fim]
    ]


def _resumo_dos_blocos(atribuidos: list[AlvoNoBloco]) -> list[dict]:
    resumo, inicio = [], 0
    for bloco in BLOCOS:
        membros = [a for a in atribuidos if a.bloco == bloco]
        primeira = membros[0].published_at if membros else None
        resumo.append(
            {
                "bloco": bloco,
                "n": len(membros),
                "inicio_indice": inicio,
                "primeira_publicacao_utc": _iso(primeira) if primeira else None,
                "ultima_publicacao_utc": _iso(membros[-1].published_at) if membros else None,
                "treino_ate_utc": (
                    _iso(primeira - timedelta(days=INTERVALO_DIAS)) if primeira and bloco != "treino" else None
                ),
            }
        )
        inicio += len(membros)
    return resumo


def montar_particao(pasta_base: Path) -> dict:
    """Manifesto da partição dos alvos de uma base curada, com regras, cortes, hashes de entrada e retenção."""
    pasta_base = Path(pasta_base)
    videos = curadoria.ler_base(pasta_base)["videos.parquet"].select(["video_id", "papel", "published_at"])
    alvos = [(v["video_id"], v["published_at"]) for v in videos.to_pylist() if v["papel"] == "alvo"]
    if not alvos:
        raise ErroParticao("a base não tem vídeos-alvo")
    prov = json.loads((pasta_base / curadoria.PROVENIENCIA).read_text(encoding="utf-8"))
    atribuidos = particionar(alvos)
    atribuicao = {a.video_id: a.bloco for a in atribuidos}
    return {
        "versao_particao": VERSAO,
        "base": {
            "pasta": f"{pasta_base.parent.name}/{pasta_base.name}",
            "snapshot_id": prov["origem"]["snapshot_id"],
            "versao_curadoria": prov["versao_curadoria"],
            "sha256_manifesto": _sha256_arquivo(pasta_base / ARQUIVO_MANIFESTO),
        },
        "regras": REGRAS,
        "blocos": _resumo_dos_blocos(atribuidos),
        "atribuicao": atribuicao,
        "sha256_atribuicao": _sha256_json(atribuicao),
        "retencao": {
            "apagar_ate_utc": prov["retencao"]["apagar_ate_utc"],
            "herdada_de": f"snapshot={prov['origem']['snapshot_id']}",
        },
        "nao_incluido": NAO_INCLUIDO,
    }


def congelar_particao(pasta_base: Path, *, destino: Path | None = None) -> Path:
    """Grava o manifesto uma única vez. Se já existe, confere que é igual ao recalculado; nunca sobrescreve."""
    arquivo = Path(destino) if destino else Path(pasta_base).parent / f"particao=v{VERSAO}" / ARQUIVO
    manifesto = montar_particao(pasta_base)
    if arquivo.exists():
        registrado = json.loads(arquivo.read_text(encoding="utf-8"))
        registrado.pop("congelada_em_utc", None)
        if registrado != manifesto:
            raise ErroParticao(f"{arquivo.parent.name}/{arquivo.name}: difere da partição recalculada; não será "
                               "sobrescrita")
        return arquivo
    gravar_json_atomico(arquivo, manifesto | {"congelada_em_utc": _iso(datetime.now(UTC))})
    return arquivo


def ler_particao(arquivo: Path, *, pasta_base: Path | None = None) -> dict:
    """Manifesto congelado, conferido pelo hash da atribuição e, se dada, pela base de origem."""
    conteudo = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    if conteudo.get("versao_particao") != VERSAO:
        raise ErroParticao(f"versão de partição desconhecida: {conteudo.get('versao_particao')!r}")
    if _sha256_json(conteudo["atribuicao"]) != conteudo["sha256_atribuicao"]:
        raise ErroParticao("atribuição não confere com o hash registrado")
    if pasta_base is not None and _sha256_arquivo(Path(pasta_base) / ARQUIVO_MANIFESTO) != conteudo["base"][
        "sha256_manifesto"
    ]:
        raise ErroParticao("a base atual difere da base registrada na partição")
    return conteudo


def main(argv: Sequence[str] | None = None) -> int:
    """0 = partição congelada ou conferida; 2 = base ou partição inválida."""
    parser = argparse.ArgumentParser(prog="python -m src.dados.particoes", description="Congela a partição temporal.")
    parser.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n> da base curada")
    parser.add_argument("--destino", type=Path, help="arquivo do manifesto (padrão: ao lado da base)")
    args = parser.parse_args(argv)
    try:
        arquivo = congelar_particao(args.base, destino=args.destino)
        conteudo = ler_particao(arquivo, pasta_base=args.base)
    except (OSError, ValueError, KeyError) as erro:
        print(f"Partição recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({"arquivo": str(arquivo), "blocos": conteudo["blocos"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
