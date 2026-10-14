"""Amostras para anotação pelo Gemini (D-E2, D-E5), sorteadas e congeladas antes de anotar.

Alvos: vídeos com papel `alvo` da base curada, cada um com o bloco da partição temporal congelada. Estrato =
bloco × setor × `tipo_canal`. LIM: vídeos com papel `amostra_lim`, numa amostra separada (estrato = setor ×
`tipo_canal`), fora de treino, validação, teste e comparáveis. Vídeos só de histórico nunca entram.

Regras:
- N vem da fórmula aprovada (REGISTRO §8.3), N = ⌊(1 − r) × ΣR_d / (1 + q)⌋, ou de uma decisão explícita de Pedro;
- cota do estrato proporcional ao número de vídeos (maiores restos; empate pela ordem bloco, setor, tipo);
- dentro do estrato, cota de cada canal proporcional aos seus vídeos, com teto ⌈2 × cota do estrato / canais do
  estrato⌉; o excedente vai para os canais abaixo do teto e a falta por teto é registrada;
- com redistribuição (decisão de Pedro, 14/10), as vagas que os tetos deixaram sem vídeo vão, uma a uma, para
  estratos sem falta que ainda aceitam mais um vídeo com o teto recalculado, escolhidos pelo menor (vagas recebidas
  + 1) / (população − cota proporcional), com desempate pela ordem do estrato;
- dentro do canal, vídeos na ordem de sha256(semente:video_id). A seleção só lê IDs, canal, setor, tipo e bloco:
  nunca curtidas, visualizações, comentários, conteúdo ou sucesso da extração;
- ordem de execução intercalada pela posição fracionária (2k+1)/(2n), entre canais do estrato e depois entre
  estratos. Os lotes de execução são fatias consecutivas dessa ordem, sem os vídeos já anotados.

Exemplo (amostra ampliada):
    uv run python -m src.amostra_anotacao --base <curadoria=v1> --semente <da v1> --n 1500 --redistribuir \\
        --anterior <amostra=v1> --destino <amostra=v2> --descricao-orcamento "<decisão>"
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from src import curadoria, particoes
from src.cache_gemini import gravar_json_atomico
from src.config import RAIZ_PROJETO
from src.snapshot import ARQUIVO_MANIFESTO

VERSAO = "1"
FATOR_TETO = 2
RESERVA = Fraction(1, 10)
DIRETORIO_PADRAO = RAIZ_PROJETO / "data" / "annotations" / "gemini"
ARQUIVO = "amostra.json"
ARQUIVO_LOTES = "lotes.json"
PAPEL_ALVO, PAPEL_LIM = "alvo", "amostra_lim"
ORDEM_BLOCOS = {bloco: i for i, bloco in enumerate((*particoes.BLOCOS, PAPEL_LIM))}
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
REGRA_REDISTRIBUICAO = (
    "vagas deixadas pelos tetos vão, uma a uma, para estratos sem falta que aceitam mais um vídeo com o teto "
    "recalculado; escolha pelo menor (vagas recebidas + 1) / (população − cota proporcional), desempate pela ordem "
    "do estrato; mesma semente e mesmos desempates"
)
USO_LIM = (
    "só o limiar de largura do intervalo do N4 servido; fora de treino, validação, teste, comparáveis, seleção de "
    "atributos, ajuste de hiperparâmetros e estimativa de cobertura com rótulo"
)


class ErroAmostra(ValueError):
    """Amostra inconsistente com a base, com a partição, com a amostra anterior ou com a lista congelada."""


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


def requisicoes_para(n: int, *, reserva: Fraction = RESERVA) -> int:
    """Menor R com ⌊(1 − r) × R⌋ = n: o teto de requisições de um lote de n vídeos, com a reserva aplicada uma vez."""
    return math.ceil(n / (1 - Fraction(reserva)))


def _sorteio(semente: str, video_id: str) -> str:
    return hashlib.sha256(f"{semente}:{video_id}".encode()).hexdigest()


def _sha256_json(conteudo: object) -> str:
    return hashlib.sha256(json.dumps(conteudo, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_arquivo(arquivo: Path) -> str:
    return hashlib.sha256(Path(arquivo).read_bytes()).hexdigest()


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
    return ORDEM_BLOCOS[estrato[0]], estrato[1], estrato[2]


def _redistribuir(
    proporcionais: dict[tuple, int], disponiveis: dict[tuple, dict[str, int]]
) -> tuple[dict[tuple, int], Counter]:
    """Reparte as vagas que os tetos deixaram sem vídeo entre estratos sem falta, com o teto recalculado."""
    faltas = {e: _cotas_dos_canais(q, disponiveis[e])[1] for e, q in proporcionais.items()}
    aceitam = [e for e in proporcionais if faltas[e] == 0]
    restante = {e: sum(disponiveis[e].values()) - proporcionais[e] for e in aceitam}
    cotas, extras = dict(proporcionais), Counter()
    for _ in range(sum(faltas.values())):
        viaveis = [e for e in aceitam if restante[e] > 0 and _cotas_dos_canais(cotas[e] + 1, disponiveis[e])[1] == 0]
        if not viaveis:
            raise ErroAmostra("sem capacidade para completar n respeitando os tetos por canal")
        escolhido = min(viaveis, key=lambda e: (Fraction(extras[e] + 1, restante[e]), _ordem_do_estrato(e)))
        cotas[escolhido] += 1
        extras[escolhido] += 1
    return cotas, extras


def selecionar(candidatos: Sequence[Candidato], *, n: int, semente: str, redistribuir: bool = False) -> Selecao:
    """Sorteio estratificado com teto por canal e ordem de execução intercalada; redistribuição opcional."""
    if len({c.video_id for c in candidatos}) != len(candidatos):
        raise ErroAmostra("vídeo repetido entre os candidatos")
    if not 0 <= n <= len(candidatos):
        raise ErroAmostra(f"n deve estar entre 0 e {len(candidatos)} candidatos")
    grupos: dict[tuple, dict[str, list[Candidato]]] = defaultdict(lambda: defaultdict(list))
    for candidato in candidatos:
        grupos[candidato.estrato][candidato.channel_id].append(candidato)
    ordem = sorted(grupos, key=_ordem_do_estrato)
    disponiveis = {e: {c: len(grupos[e][c]) for c in sorted(grupos[e])} for e in ordem}
    proporcionais = _maiores_restos(n, {e: sum(disponiveis[e].values()) for e in ordem})
    cotas, extras = _redistribuir(proporcionais, disponiveis) if redistribuir else (proporcionais, Counter())
    sequencias, estratos = {}, []
    for estrato in ordem:
        canais = {c: sorted(grupos[estrato][c], key=lambda x: _sorteio(semente, x.video_id)) for c in disponiveis[
            estrato]}
        por_canal, falta, teto = _cotas_dos_canais(cotas[estrato], disponiveis[estrato])
        sequencias[estrato] = _intercalar({c: canais[c][: por_canal[c]] for c in canais if por_canal[c]})
        relatorio = {
            "bloco": estrato[0],
            "setor": estrato[1],
            "tipo_canal": estrato[2],
            "populacao": sum(disponiveis[estrato].values()),
            "canais": len(canais),
            "cota": cotas[estrato],
            "teto_por_canal": teto,
            "selecionados": len(sequencias[estrato]),
            "falta_por_teto": falta,
            "por_canal": {c: {"populacao": len(canais[c]), "selecionados": por_canal[c]} for c in canais},
        }
        if redistribuir:
            _, falta_original, teto_original = _cotas_dos_canais(proporcionais[estrato], disponiveis[estrato])
            relatorio |= {"cota_proporcional": proporcionais[estrato], "teto_proporcional": teto_original,
                          "falta_sem_redistribuicao": falta_original, "vagas_redistribuidas": extras[estrato]}
        estratos.append(relatorio)
    lista = _intercalar({e: sequencias[e] for e in ordem if sequencias[e]})
    return Selecao(lista=tuple(lista), estratos=tuple(estratos))


def _tabelas_sem_resultados(pasta_base: Path) -> tuple[dict, list[dict]]:
    tabelas = curadoria.ler_base(pasta_base)
    canais = {c["channel_id"]: c for c in tabelas["canais.parquet"].select(["channel_id", "setor", "tipo_canal"])
              .to_pylist()}
    videos = tabelas["videos.parquet"].select(["video_id", "channel_id", "papel"]).to_pylist()
    return canais, videos


def candidatos(pasta_base: Path, particao: dict) -> list[Candidato]:
    """Alvos da base com o bloco da partição; recusa partição que não cubra exatamente os alvos."""
    canais, videos = _tabelas_sem_resultados(pasta_base)
    alvos = [v for v in videos if v["papel"] == PAPEL_ALVO]
    atribuicao = particao["atribuicao"]
    if {v["video_id"] for v in alvos} != set(atribuicao):
        raise ErroAmostra("a partição não cobre exatamente os alvos da base")
    return [
        Candidato(v["video_id"], v["channel_id"], atribuicao[v["video_id"]], canais[v["channel_id"]]["setor"],
                  canais[v["channel_id"]]["tipo_canal"])
        for v in alvos
    ]


def candidatos_lim(pasta_base: Path) -> list[Candidato]:
    """Vídeos da amostra LIM da base (menos de 30 dias na coleta), num pseudobloco próprio, fora da partição."""
    canais, videos = _tabelas_sem_resultados(pasta_base)
    return [
        Candidato(v["video_id"], v["channel_id"], PAPEL_LIM, canais[v["channel_id"]]["setor"],
                  canais[v["channel_id"]]["tipo_canal"])
        for v in videos
        if v["papel"] == PAPEL_LIM
    ]


def _duracoes(pasta_base: Path) -> dict[str, int]:
    videos = curadoria.ler_base(pasta_base)["videos.parquet"].select(["video_id", "duracao_s"]).to_pylist()
    return {v["video_id"]: v["duracao_s"] for v in videos}


def _lista(selecao: Selecao) -> list[dict]:
    return [
        {"ordem": ordem, "video_id": c.video_id, "channel_id": c.channel_id, "bloco": c.bloco, "setor": c.setor,
         "tipo_canal": c.tipo_canal}
        for ordem, c in enumerate(selecao.lista, start=1)
    ]


def _orcamento(n: int, requisicoes_por_dia, reserva, fracao_dupla, descricao: str, duracao_media: float) -> dict:
    if requisicoes_por_dia is None:
        return {"n_planejado": n, "regra": "N por decisão explícita de Pedro", "descricao": descricao,
                "duracao_media_populacao_s": duracao_media}
    return {
        "formula": "N = ⌊(1 − r) × ΣR_d / (1 + q)⌋ (REGISTRO §8.3)",
        "requisicoes_por_dia": list(requisicoes_por_dia),
        "reserva_r": str(reserva),
        "fracao_dupla_q": str(fracao_dupla),
        "n_planejado": n,
        "descricao": descricao,
        "duracao_media_alvos_s": duracao_media,
    }


def _amostra_anterior(arquivo: Path, lista: list[dict], sha256_base: str) -> dict:
    anterior = ler_amostra(arquivo)
    if anterior["base"]["sha256_manifesto"] != sha256_base:
        raise ErroAmostra("a amostra anterior veio de outra base")
    faltam = {e["video_id"] for e in anterior["lista"]} - {e["video_id"] for e in lista}
    if faltam:
        raise ErroAmostra(f"a nova amostra não contém {len(faltam)} vídeos da amostra anterior")
    return {"arquivo": f"{Path(arquivo).parent.name}/{Path(arquivo).name}", "sha256_arquivo": _sha256_arquivo(arquivo),
            "sha256_lista": anterior["sha256_lista"], "selecionados": anterior["selecionados"],
            "contida_na_nova": True}


def montar_amostra(
    pasta_base: Path,
    arquivo_particao: Path,
    *,
    semente: str,
    descricao_orcamento: str,
    requisicoes_por_dia: Sequence[int] | None = None,
    n: int | None = None,
    reserva: Fraction = RESERVA,
    fracao_dupla: Fraction = Fraction(0),
    redistribuir: bool = False,
    amostra_anterior: Path | None = None,
) -> dict:
    """Manifesto da amostra de alvos: semente, orçamento, partição e base de origem, estratos, lista e retenção."""
    if (requisicoes_por_dia is None) == (n is None):
        raise ErroAmostra("informe requisições por dia (fórmula) ou n (decisão explícita), e só um deles")
    particao = particoes.ler_particao(arquivo_particao, pasta_base=pasta_base)
    n = n if n is not None else orcamento_n(requisicoes_por_dia, reserva=reserva, fracao_dupla=fracao_dupla)
    lista_candidatos = candidatos(pasta_base, particao)
    if not lista_candidatos:
        raise ErroAmostra("a base não tem alvos")
    selecao = selecionar(lista_candidatos, n=n, semente=semente, redistribuir=redistribuir)
    lista = _lista(selecao)
    duracao = _duracoes(pasta_base)
    media = round(sum(duracao[c.video_id] for c in lista_candidatos) / len(lista_candidatos), 1)
    amostra = {
        "versao_amostra": VERSAO,
        "semente": semente,
        "orcamento": _orcamento(n, requisicoes_por_dia, reserva, fracao_dupla, descricao_orcamento, media),
        "base": particao["base"],
        "particao": {
            "arquivo": f"{Path(arquivo_particao).parent.name}/{Path(arquivo_particao).name}",
            "sha256_arquivo": _sha256_arquivo(arquivo_particao),
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
    if redistribuir:
        amostra["regra_redistribuicao"] = REGRA_REDISTRIBUICAO
        amostra["vagas_redistribuidas"] = sum(e["vagas_redistribuidas"] for e in selecao.estratos)
        amostra["composicao"] = {
            "por_bloco": dict(Counter(e["bloco"] for e in lista)),
            "por_setor": dict(Counter(e["setor"] for e in lista)),
            "por_tipo_canal": dict(Counter(e["tipo_canal"] for e in lista)),
            "por_setor_tipo": dict(sorted(Counter(f"{e['setor']}/{e['tipo_canal']}" for e in lista).items())),
            "observacao": "com vagas redistribuídas a amostra deixa de ser autoponderada: estimativas para a "
            "população devem ponderar por estrato",
        }
    if amostra_anterior is not None:
        amostra["amostra_anterior"] = _amostra_anterior(amostra_anterior, lista, particao["base"]["sha256_manifesto"])
    return amostra


def montar_amostra_lim(pasta_base: Path, *, semente: str, n: int, descricao_orcamento: str) -> dict:
    """Manifesto da amostra LIM separada: mesmas regras de estrato e teto, sem partição e com uso restrito."""
    pasta_base = Path(pasta_base)
    lista_candidatos = candidatos_lim(pasta_base)
    if not lista_candidatos:
        raise ErroAmostra("a base não tem vídeos da amostra LIM")
    selecao = selecionar(lista_candidatos, n=n, semente=semente, redistribuir=True)
    lista = _lista(selecao)
    prov = json.loads((pasta_base / curadoria.PROVENIENCIA).read_text(encoding="utf-8"))
    duracao = _duracoes(pasta_base)
    return {
        "versao_amostra": VERSAO,
        "papel": PAPEL_LIM,
        "uso": USO_LIM,
        "ressalvas": [
            "poucos canais e vídeos do mesmo canal não são independentes: o limiar deve ser relatado com bootstrap "
            "por canal",
            "vídeos com menos de 30 dias na coleta ainda estão amadurecendo; o limiar herda essa diferença em "
            "relação aos alvos",
        ],
        "semente": semente,
        "orcamento": _orcamento(n, None, None, None, descricao_orcamento,
                                round(sum(duracao[c.video_id] for c in lista_candidatos) / len(lista_candidatos), 1)),
        "base": {
            "pasta": f"{pasta_base.parent.name}/{pasta_base.name}",
            "snapshot_id": prov["origem"]["snapshot_id"],
            "versao_curadoria": prov["versao_curadoria"],
            "sha256_manifesto": _sha256_arquivo(pasta_base / ARQUIVO_MANIFESTO),
        },
        "particao": None,
        "regras": REGRAS | {"populacao": "vídeos com papel amostra_lim na base curada (menos de 30 dias na coleta)",
                            "estratos": "setor × tipo_canal"},
        "regra_redistribuicao": REGRA_REDISTRIBUICAO,
        "estratos": list(selecao.estratos),
        "selecionados": len(lista),
        "populacao": len(lista_candidatos),
        "falta_por_teto": sum(e["falta_por_teto"] for e in selecao.estratos),
        "lista": lista,
        "sha256_lista": _sha256_json(lista),
        "retencao": {"apagar_ate_utc": prov["retencao"]["apagar_ate_utc"],
                     "herdada_de": f"snapshot={prov['origem']['snapshot_id']}"},
    }


def _congelar(arquivo: Path, conteudo: dict, campo_instante: str) -> Path:
    """Grava uma única vez; se já existe, confere que é igual (sem o instante) e nunca substitui."""
    if arquivo.exists():
        registrado = json.loads(arquivo.read_text(encoding="utf-8"))
        registrado.pop(campo_instante, None)
        if registrado != conteudo:
            raise ErroAmostra(f"{arquivo.parent.name}/{arquivo.name}: já existe outra versão congelada; não será "
                              "substituída")
        return arquivo
    instante = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    gravar_json_atomico(arquivo, conteudo | {campo_instante: instante})
    return arquivo


def congelar_amostra(pasta_base: Path, arquivo_particao: Path, *, destino: Path, **parametros) -> Path:
    """Congela a amostra de alvos em `destino`, uma única vez."""
    return _congelar(Path(destino) / ARQUIVO, montar_amostra(pasta_base, arquivo_particao, **parametros),
                     "congelada_em_utc")


def congelar_amostra_lim(pasta_base: Path, *, destino: Path, **parametros) -> Path:
    """Congela a amostra LIM em `destino`, uma única vez."""
    return _congelar(Path(destino) / ARQUIVO, montar_amostra_lim(pasta_base, **parametros), "congelada_em_utc")


def ler_amostra(arquivo: Path, *, pasta_base: Path | None = None, arquivo_particao: Path | None = None) -> dict:
    """Amostra congelada, conferida pelo hash da lista e, se dados, pela base e pela partição de origem."""
    amostra = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    if amostra.get("versao_amostra") != VERSAO:
        raise ErroAmostra(f"versão de amostra desconhecida: {amostra.get('versao_amostra')!r}")
    if _sha256_json(amostra["lista"]) != amostra["sha256_lista"]:
        raise ErroAmostra("a lista não confere com o hash registrado")
    if pasta_base is not None and _sha256_arquivo(Path(pasta_base) / ARQUIVO_MANIFESTO) != amostra["base"][
        "sha256_manifesto"
    ]:
        raise ErroAmostra("a base atual difere da base registrada na amostra")
    if arquivo_particao is not None:
        particao = particoes.ler_particao(arquivo_particao, pasta_base=pasta_base)
        if particao["sha256_atribuicao"] != amostra["particao"]["sha256_atribuicao"]:
            raise ErroAmostra("a partição atual difere da partição registrada na amostra")
    return amostra


def montar_lotes(pasta_base: Path, amostra: dict, ja_anotados: set[str], *, tamanho: int, limites: dict) -> dict:
    """Fatias consecutivas da ordem congelada, só com vídeos ainda sem anotação; recusa lote que não caiba no teto."""
    duracao = _duracoes(pasta_base)
    pendentes = [e["video_id"] for e in amostra["lista"] if e["video_id"] not in ja_anotados]
    reserva = limites["max_requisicoes"] - tamanho
    lotes = []
    for numero, inicio in enumerate(range(0, len(pendentes), tamanho), start=1):
        ids = pendentes[inicio: inicio + tamanho]
        segundos = sorted((duracao[i] for i in ids), reverse=True)
        pior_caso = sum(segundos) + sum(segundos[: max(reserva, 0)])
        if pior_caso > limites["max_segundos_video"]:
            raise ErroAmostra(f"lote {numero}: {pior_caso} s no pior caso passam do teto de "
                              f"{limites['max_segundos_video']} s")
        lotes.append({"lote": numero, "n": len(ids), "video_ids": ids, "sha256": _sha256_json(ids),
                      "segundos_primeira_passada": sum(segundos), "segundos_pior_caso_com_reserva": pior_caso})
    anotados = sorted(ja_anotados & {e["video_id"] for e in amostra["lista"]})
    return {
        "versao_lotes": VERSAO,
        "papel": amostra.get("papel", PAPEL_ALVO),
        "amostra_sha256_lista": amostra["sha256_lista"],
        "ja_anotados": {"n": len(anotados), "sha256": _sha256_json(anotados)},
        "tamanho": tamanho,
        "limites_por_lote": limites,
        "regra": "fatias consecutivas da ordem congelada, sem os vídeos já anotados; um lote só anota os próprios "
        "vídeos, mesmo que sobre teto de requisições; vídeo que falhar continua no seu lote, sem substituto",
        "lotes": lotes,
    }


def congelar_lotes(pasta: Path, conteudo: dict) -> Path:
    """Congela os lotes ao lado da amostra, uma única vez."""
    return _congelar(Path(pasta) / ARQUIVO_LOTES, conteudo, "congelados_em_utc")


def ler_lotes(pasta: Path, amostra: dict) -> dict | None:
    """Lotes congelados da amostra, conferidos pelos hashes; None se a amostra não tem lotes."""
    arquivo = Path(pasta) / ARQUIVO_LOTES
    if not arquivo.is_file():
        return None
    lotes = json.loads(arquivo.read_text(encoding="utf-8"))
    if lotes["amostra_sha256_lista"] != amostra["sha256_lista"]:
        raise ErroAmostra("os lotes foram congelados para outra lista")
    if any(_sha256_json(lote["video_ids"]) != lote["sha256"] for lote in lotes["lotes"]):
        raise ErroAmostra("a lista de um lote não confere com o hash registrado")
    ids = [v for lote in lotes["lotes"] for v in lote["video_ids"]]
    if len(ids) != len(set(ids)) or not set(ids) <= {e["video_id"] for e in amostra["lista"]}:
        raise ErroAmostra("lotes com vídeo repetido ou fora da lista")
    return lotes


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.amostra_anotacao", description="Congela uma amostra.")
    parser.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n> da base curada")
    parser.add_argument("--particao", type=Path, help="manifesto da partição (padrão: ao lado da base)")
    parser.add_argument("--semente", required=True)
    parser.add_argument("--requisicoes-por-dia", help="R_d separados por vírgula (fórmula de N)")
    parser.add_argument("--n", type=int, help="N por decisão explícita, no lugar da fórmula")
    parser.add_argument("--redistribuir", action="store_true", help="redistribui as vagas deixadas pelos tetos")
    parser.add_argument("--anterior", type=Path, help="pasta da amostra anterior, que deve ficar contida na nova")
    parser.add_argument("--lim", action="store_true", help="amostra LIM separada (exige --n e --destino)")
    parser.add_argument("--descricao-orcamento", required=True, help="origem de N (decisão, RPD, d̄, lote)")
    parser.add_argument("--destino", type=Path, help="pasta da amostra (padrão: data/annotations/gemini/...)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """0 = amostra congelada ou conferida; 2 = entrada inválida."""
    args = _argumentos(argv)
    try:
        if args.lim:
            if args.n is None or args.destino is None or args.anterior or args.redistribuir:
                raise ErroAmostra("a amostra LIM exige --n e --destino e não aceita --anterior nem --redistribuir")
            arquivo = congelar_amostra_lim(args.base, destino=args.destino, semente=args.semente, n=args.n,
                                           descricao_orcamento=args.descricao_orcamento)
            amostra = ler_amostra(arquivo, pasta_base=args.base)
        else:
            particao = args.particao or args.base.parent / f"particao=v{particoes.VERSAO}" / particoes.ARQUIVO
            snapshot_id = particoes.ler_particao(particao)["base"]["snapshot_id"]
            destino = args.destino or DIRETORIO_PADRAO / f"snapshot={snapshot_id}" / f"amostra=v{VERSAO}"
            requisicoes = [int(r) for r in args.requisicoes_por_dia.split(",")] if args.requisicoes_por_dia else None
            anterior = args.anterior / ARQUIVO if args.anterior else None
            arquivo = congelar_amostra(args.base, particao, destino=destino, semente=args.semente,
                                       requisicoes_por_dia=requisicoes, n=args.n, redistribuir=args.redistribuir,
                                       amostra_anterior=anterior, descricao_orcamento=args.descricao_orcamento)
            amostra = ler_amostra(arquivo, pasta_base=args.base, arquivo_particao=particao)
    except (OSError, ValueError, KeyError) as erro:
        print(f"Amostra recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    resumo = {"arquivo": str(arquivo), "n_planejado": amostra["orcamento"]["n_planejado"],
              "selecionados": amostra["selecionados"], "falta_por_teto": amostra["falta_por_teto"],
              "vagas_redistribuidas": amostra.get("vagas_redistribuidas")}
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
