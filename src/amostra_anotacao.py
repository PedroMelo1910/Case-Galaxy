"""Amostra estratificada para anotação pelo Gemini (D-E2), sorteada e congelada antes de anotar.

População: vídeos com papel `alvo` da base curada, cada um com o bloco da partição temporal congelada. Vídeos só de
histórico e da amostra LIM não entram. Estrato = bloco × setor × `tipo_canal`.

Regras:
- N vem da fórmula aprovada (REGISTRO §8.3): N = ⌊(1 − r) × ΣR_d / (1 + q)⌋, com a reserva r = 10% aplicada uma vez;
- cota do estrato proporcional ao número de alvos (maiores restos; empate pela ordem bloco, setor, tipo);
- dentro do estrato, cota de cada canal proporcional aos seus alvos, com teto ⌈2 × cota do estrato / canais do
  estrato⌉; o excedente vai para os canais abaixo do teto e, se o teto impedir completar o estrato, a falta é
  registrada (nada é empurrado para outro estrato);
- dentro do canal, vídeos na ordem de sha256(semente:video_id). A seleção só lê IDs, canal, setor, tipo e bloco:
  nunca curtidas, visualizações, comentários, conteúdo ou sucesso da extração;
- ordem de execução intercalada pela posição fracionária (2k+1)/(2n), primeiro entre canais do estrato e depois
  entre estratos: parar em qualquer ponto deixa uma amostra perto da proporção planejada.

Exemplo:
    uv run python -m src.amostra_anotacao --base data/processed/snapshot=<id>/curadoria=v1 --semente <texto> \\
        --requisicoes-por-dia 100,434,434 --descricao-orcamento "<origem de cada R_d>"
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from src import curadoria, particoes
from src.cache_gemini import gravar_json_atomico
from src.config import RAIZ_PROJETO

VERSAO = "1"
FATOR_TETO = 2
RESERVA = Fraction(1, 10)
DIRETORIO_PADRAO = RAIZ_PROJETO / "data" / "annotations" / "gemini"
ARQUIVO = "amostra.json"
REGRAS = {
    "populacao": "vídeos com papel alvo na base curada; só histórico e amostra LIM ficam fora",
    "estratos": "bloco da partição × setor × tipo_canal",
    "alocacao": "cota do estrato proporcional ao número de alvos, pelos maiores restos",
    "teto_por_canal": "⌈2 × cota do estrato / canais do estrato⌉; excedente para os canais abaixo do teto; falta "
    "registrada quando o teto impede completar o estrato",
    "sorteio": "dentro do canal, ordem de sha256(semente:video_id)",
    "ordem_de_execucao": "posição fracionária (2k+1)/(2n), entre canais no estrato e depois entre estratos",
    "sem_resultados": "a seleção não lê curtidas, visualizações, comentários, conteúdo nem sucesso da extração",
}


class ErroAmostra(ValueError):
    """Amostra inconsistente com a base, com a partição ou com a lista congelada."""


@dataclass(frozen=True)
class Candidato:
    video_id: str
    channel_id: str
    bloco: str
    setor: str
    tipo_canal: str

    @property
    def estrato(self) -> tuple[str, str, str]:
        return self.bloco, self.setor, self.tipo_canal


@dataclass(frozen=True)
class Selecao:
    lista: tuple[Candidato, ...]
    estratos: tuple[dict, ...]


def orcamento_n(
    requisicoes_por_dia: Sequence[int], *, reserva: Fraction = RESERVA, fracao_dupla: Fraction = Fraction(0)
) -> int:
    """N = ⌊(1 − r) × R / (1 + q)⌋, com R = soma das requisições possíveis por dia de anotação."""
    if any(r < 0 for r in requisicoes_por_dia) or not 0 <= reserva < 1 or fracao_dupla < 0:
        raise ValueError("requisições por dia não negativas, 0 <= reserva < 1 e q >= 0")
    return math.floor((1 - Fraction(reserva)) * sum(requisicoes_por_dia) / (1 + Fraction(fracao_dupla)))


def _sorteio(semente: str, video_id: str) -> str:
    return hashlib.sha256(f"{semente}:{video_id}".encode()).hexdigest()


def _sha256_json(conteudo: object) -> str:
    return hashlib.sha256(json.dumps(conteudo, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _maiores_restos[K](total: int, pesos: dict[K, int]) -> dict[K, int]:
    """Divide `total` proporcionalmente aos pesos; sobras pelos maiores restos, empate pela ordem das chaves."""
    soma = sum(pesos.values())
    if soma == 0:
        return dict.fromkeys(pesos, 0)
    partes = {chave: total * peso // soma for chave, peso in pesos.items()}
    sobras = sorted(pesos, key=lambda chave: -(total * pesos[chave] % soma))
    for chave in sobras[: total - sum(partes.values())]:
        partes[chave] += 1
    return partes


def _cotas_dos_canais(cota: int, disponiveis: dict[str, int]) -> tuple[dict[str, int], int, int]:
    """Cotas por canal com teto; devolve as cotas, a falta por teto e o teto."""
    teto = math.ceil(FATOR_TETO * cota / len(disponiveis))
    limite = {canal: min(total, teto) for canal, total in disponiveis.items()}
    cotas = dict.fromkeys(disponiveis, 0)
    restante = cota
    while restante > 0:
        abertos = {canal: disponiveis[canal] for canal in disponiveis if cotas[canal] < limite[canal]}
        if not abertos:
            break
        for canal, parte in _maiores_restos(restante, abertos).items():
            dado = min(parte, limite[canal] - cotas[canal])
            cotas[canal] += dado
            restante -= dado
    return cotas, restante, teto


def _intercalar[K, T](grupos: dict[K, list[T]]) -> list[T]:
    """Ordena pela posição fracionária (2k+1)/(2n) de cada item no seu grupo; empate pela ordem dos grupos."""
    posicoes = [
        (Fraction(2 * k + 1, 2 * len(itens)), indice, k, item)
        for indice, itens in enumerate(grupos.values())
        for k, item in enumerate(itens)
    ]
    return [item for *_, item in sorted(posicoes, key=lambda p: p[:3])]


def _ordem_do_estrato(estrato: tuple[str, str, str]) -> tuple:
    return particoes.BLOCOS.index(estrato[0]), estrato[1], estrato[2]


def selecionar(candidatos: Sequence[Candidato], *, n: int, semente: str) -> Selecao:
    """Sorteio estratificado com teto por canal e ordem de execução intercalada."""
    if len({c.video_id for c in candidatos}) != len(candidatos):
        raise ErroAmostra("vídeo repetido entre os candidatos")
    if not 0 <= n <= len(candidatos):
        raise ErroAmostra(f"n deve estar entre 0 e {len(candidatos)} candidatos")
    grupos: dict[tuple, dict[str, list[Candidato]]] = defaultdict(lambda: defaultdict(list))
    for candidato in candidatos:
        grupos[candidato.estrato][candidato.channel_id].append(candidato)
    ordem = sorted(grupos, key=_ordem_do_estrato)
    cotas = _maiores_restos(n, {e: sum(len(v) for v in grupos[e].values()) for e in ordem})
    sequencias, estratos = {}, []
    for estrato in ordem:
        canais = {c: sorted(grupos[estrato][c], key=lambda x: _sorteio(semente, x.video_id)) for c in sorted(
            grupos[estrato])}
        por_canal, falta, teto = _cotas_dos_canais(cotas[estrato], {c: len(v) for c, v in canais.items()})
        sequencias[estrato] = _intercalar({c: canais[c][: por_canal[c]] for c in canais if por_canal[c]})
        estratos.append(
            {
                "bloco": estrato[0],
                "setor": estrato[1],
                "tipo_canal": estrato[2],
                "populacao": sum(len(v) for v in canais.values()),
                "canais": len(canais),
                "cota": cotas[estrato],
                "teto_por_canal": teto,
                "selecionados": len(sequencias[estrato]),
                "falta_por_teto": falta,
                "por_canal": {c: {"populacao": len(canais[c]), "selecionados": por_canal[c]} for c in canais},
            }
        )
    lista = _intercalar({e: sequencias[e] for e in ordem if sequencias[e]})
    return Selecao(lista=tuple(lista), estratos=tuple(estratos))


def candidatos(pasta_base: Path, particao: dict) -> list[Candidato]:
    """Alvos da base com o bloco da partição; recusa partição que não cubra exatamente os alvos."""
    tabelas = curadoria.ler_base(pasta_base)
    canais = {c["channel_id"]: c for c in tabelas["canais.parquet"].select(["channel_id", "setor", "tipo_canal"])
              .to_pylist()}
    alvos = [v for v in tabelas["videos.parquet"].select(["video_id", "channel_id", "papel"]).to_pylist()
             if v["papel"] == "alvo"]
    atribuicao = particao["atribuicao"]
    if {v["video_id"] for v in alvos} != set(atribuicao):
        raise ErroAmostra("a partição não cobre exatamente os alvos da base")
    return [
        Candidato(v["video_id"], v["channel_id"], atribuicao[v["video_id"]], canais[v["channel_id"]]["setor"],
                  canais[v["channel_id"]]["tipo_canal"])
        for v in alvos
    ]


def _duracao_media_alvos(pasta_base: Path) -> float:
    videos = curadoria.ler_base(pasta_base)["videos.parquet"].select(["papel", "duracao_s"]).to_pylist()
    duracoes = [v["duracao_s"] for v in videos if v["papel"] == "alvo"]
    return round(sum(duracoes) / len(duracoes), 1)


def montar_amostra(
    pasta_base: Path,
    arquivo_particao: Path,
    *,
    semente: str,
    requisicoes_por_dia: Sequence[int],
    descricao_orcamento: str,
    reserva: Fraction = RESERVA,
    fracao_dupla: Fraction = Fraction(0),
) -> dict:
    """Manifesto da amostra: semente, orçamento, partição e base de origem, estratos, lista ordenada e retenção."""
    particao = particoes.ler_particao(arquivo_particao, pasta_base=pasta_base)
    n = orcamento_n(requisicoes_por_dia, reserva=reserva, fracao_dupla=fracao_dupla)
    selecao = selecionar(candidatos(pasta_base, particao), n=n, semente=semente)
    lista = [
        {"ordem": ordem, "video_id": c.video_id, "channel_id": c.channel_id, "bloco": c.bloco, "setor": c.setor,
         "tipo_canal": c.tipo_canal}
        for ordem, c in enumerate(selecao.lista, start=1)
    ]
    return {
        "versao_amostra": VERSAO,
        "semente": semente,
        "orcamento": {
            "formula": "N = ⌊(1 − r) × ΣR_d / (1 + q)⌋ (REGISTRO §8.3)",
            "requisicoes_por_dia": list(requisicoes_por_dia),
            "reserva_r": str(reserva),
            "fracao_dupla_q": str(fracao_dupla),
            "n_planejado": n,
            "descricao": descricao_orcamento,
            "duracao_media_alvos_s": _duracao_media_alvos(pasta_base),
        },
        "base": particao["base"],
        "particao": {
            "arquivo": f"{Path(arquivo_particao).parent.name}/{Path(arquivo_particao).name}",
            "sha256_arquivo": hashlib.sha256(Path(arquivo_particao).read_bytes()).hexdigest(),
            "sha256_atribuicao": particao["sha256_atribuicao"],
        },
        "regras": REGRAS,
        "estratos": list(selecao.estratos),
        "selecionados": len(lista),
        "falta_por_teto": sum(e["falta_por_teto"] for e in selecao.estratos),
        "lista": lista,
        "sha256_lista": _sha256_json(lista),
        "retencao": particao["retencao"],
    }


def congelar_amostra(pasta_base: Path, arquivo_particao: Path, *, destino: Path, **parametros) -> Path:
    """Grava a amostra uma única vez em `destino`; se já existe, confere que é igual e nunca a substitui."""
    arquivo = Path(destino) / ARQUIVO
    amostra = montar_amostra(pasta_base, arquivo_particao, **parametros)
    if arquivo.exists():
        registrada = json.loads(arquivo.read_text(encoding="utf-8"))
        registrada.pop("congelada_em_utc", None)
        if registrada != amostra:
            raise ErroAmostra(f"{arquivo.parent.name}: já existe outra amostra congelada; não será substituída")
        return arquivo
    gravar_json_atomico(arquivo, amostra | {"congelada_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return arquivo


def ler_amostra(arquivo: Path, *, pasta_base: Path | None = None, arquivo_particao: Path | None = None) -> dict:
    """Amostra congelada, conferida pelo hash da lista e, se dados, pela base e pela partição de origem."""
    amostra = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    if amostra.get("versao_amostra") != VERSAO:
        raise ErroAmostra(f"versão de amostra desconhecida: {amostra.get('versao_amostra')!r}")
    if _sha256_json(amostra["lista"]) != amostra["sha256_lista"]:
        raise ErroAmostra("a lista não confere com o hash registrado")
    if arquivo_particao is not None:
        particao = particoes.ler_particao(arquivo_particao, pasta_base=pasta_base)
        if particao["sha256_atribuicao"] != amostra["particao"]["sha256_atribuicao"]:
            raise ErroAmostra("a partição atual difere da partição registrada na amostra")
    return amostra


def main(argv: Sequence[str] | None = None) -> int:
    """0 = amostra congelada ou conferida; 2 = entrada inválida."""
    parser = argparse.ArgumentParser(prog="python -m src.amostra_anotacao", description="Congela a amostra.")
    parser.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n> da base curada")
    parser.add_argument("--particao", type=Path, help="manifesto da partição (padrão: ao lado da base)")
    parser.add_argument("--semente", required=True)
    parser.add_argument("--requisicoes-por-dia", required=True, help="R_d separados por vírgula")
    parser.add_argument("--descricao-orcamento", required=True, help="de onde veio cada R_d (RPD, d̄, lote)")
    parser.add_argument("--destino", type=Path, help="pasta da anotação (padrão: data/annotations/gemini/...)")
    args = parser.parse_args(argv)
    particao = args.particao or args.base.parent / f"particao=v{particoes.VERSAO}" / particoes.ARQUIVO
    try:
        snapshot_id = particoes.ler_particao(particao)["base"]["snapshot_id"]
        destino = args.destino or DIRETORIO_PADRAO / f"snapshot={snapshot_id}" / f"amostra=v{VERSAO}"
        requisicoes = [int(r) for r in args.requisicoes_por_dia.split(",")]
        arquivo = congelar_amostra(args.base, particao, destino=destino, semente=args.semente,
                                   requisicoes_por_dia=requisicoes, descricao_orcamento=args.descricao_orcamento)
        amostra = ler_amostra(arquivo, pasta_base=args.base, arquivo_particao=particao)
    except (OSError, ValueError, KeyError) as erro:
        print(f"Amostra recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    resumo = {"arquivo": str(arquivo), "n_planejado": amostra["orcamento"]["n_planejado"],
              "selecionados": amostra["selecionados"], "falta_por_teto": amostra["falta_por_teto"]}
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
