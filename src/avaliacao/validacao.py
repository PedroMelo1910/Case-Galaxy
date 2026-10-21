"""Módulo consolidado. Reúne:
- Treino e previsão de uma dobra (protocolo final §2 e §3).
- Dobras cronológicas do experimento audiovisual local.
- Sorteio dos canais retidos para a avaliação de canais não vistos (D-D2, D-C4; protocolo final §4).
- Histórico, referências e fallback de setor recalculados POR DOBRA, com auditoria independente.
- Portões de população do experimento audiovisual: piso, exclusões, duplicatas, sonda de canal e coerência das visões."""

import math
import statistics
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np

from src.avaliacao import experimentos as af, metricas as fx
from src.dados import historico as hs
from src.dados.coleta import Papel
from src.extracao import pipeline as col
from src.modelos import treino as fp, treino as md
from src.proveniencia import sha256_ids


# ──── Treino e previsão de uma dobra (protocolo final §2 e §3).
ALFA_CENTRAL = 0.5


@dataclass(frozen=True)
class Dobra:
    treino: list[dict]
    avaliacao: list[dict]
    ignoradas: dict  # usos que não entram (intervalo excluído, canal retido), contados


def separar(linhas: Sequence[Mapping]) -> Dobra:
    usos = Counter(str(l.get("uso_no_contexto")) for l in linhas)
    return Dobra([dict(l) for l in linhas if l.get("uso_no_contexto") == md.USO_TREINO],
                 [dict(l) for l in linhas if l.get("uso_no_contexto") == md.USO_AVALIACAO],
                 {u: n for u, n in sorted(usos.items()) if u not in (md.USO_TREINO, md.USO_AVALIACAO)})


def treinar_q50(treino: Sequence[Mapping], espec: md.EspecificacaoAtributos,
                parametros: Mapping) -> md.ModeloTreinado:
    return md.treinar(treino, espec, parametros=dict(parametros) | {"alpha": ALFA_CENTRAL})


def treinar_quantis(treino: Sequence[Mapping], espec: md.EspecificacaoAtributos,
                    parametros: Mapping) -> dict[str, md.ModeloTreinado]:
    q50 = treinar_q50(treino, espec, parametros)
    return {"q50": q50} | fp.treinar_companheiros(treino, q50)


def prever_central(modelo: md.ModeloTreinado, avaliacao: Sequence[Mapping]) -> dict:
    relativo = np.asarray(modelo.prever_relativo(avaliacao), dtype=float)
    central, cortadas = md.reconstruir_log(avaliacao, relativo)
    return {"relativo": relativo, "central_log": central, "cortadas_em_zero": cortadas}


def prever_com_faixa(modelos: Mapping[str, md.ModeloTreinado], avaliacao: Sequence[Mapping]) -> dict:
    relativos = {q: np.asarray(modelos[q].prever_relativo(avaliacao), dtype=float) for q in ("q10", "q50", "q90")}
    faixa = fp.prever_faixa(avaliacao, relativos["q10"], relativos["q50"], relativos["q90"])
    return faixa | {"relativo": relativos["q50"]}


def bordas_de_faixa(treino: Sequence[Mapping]) -> list[float]:
    """Tercis do alvo relativo do treino da dobra, usados no F1 por faixas (protocolo §7)."""
    return fx.tercis_do_treino([l["y_rel"] for l in treino])


# ──── Dobras cronológicas do experimento audiovisual local.
FRACAO_INICIAL = 0.4
N_JANELAS = 2
INTERVALO_DIAS = 30


class ErroDobras(ValueError):
    """Estrutura de dobras impossível ou violando uma afirmação obrigatória."""


def _ordem(linha: Mapping) -> tuple[datetime, str]:
    return linha["published_at"], linha["video_id"]


@dataclass(frozen=True)
class Janela:
    indice: int  # 1-based
    ids_avaliacao: tuple[str, ...]
    primeira_publicacao: datetime
    corte_de_treino: datetime  # primeira publicação da janela menos o intervalo


@dataclass(frozen=True)
class Estrutura:
    ids_ordenados: tuple[str, ...]
    ids_treino_inicial: tuple[str, ...]
    janelas: tuple[Janela, ...]
    intervalo_dias: int
    fracao_inicial: float


def construir(linhas: Sequence[Mapping], *, fracao_inicial: float = FRACAO_INICIAL, n_janelas: int = N_JANELAS,
              intervalo_dias: int = INTERVALO_DIAS) -> Estrutura:
    """Estrutura de dobras a partir de datas e IDs. Datas ingênuas e IDs repetidos são erros."""
    if not linhas:
        raise ErroDobras("nenhuma linha elegível")
    if any(l["published_at"].tzinfo is None for l in linhas):
        raise ErroDobras("publicação sem fuso horário")
    ids = [l["video_id"] for l in linhas]
    if len(set(ids)) != len(ids):
        raise ErroDobras("video_id repetido nas linhas elegíveis")
    if not 0 < fracao_inicial < 1 or n_janelas < 1:
        raise ErroDobras("fração inicial em (0, 1) e ao menos uma janela")
    ordenadas = sorted(linhas, key=_ordem)
    n = len(ordenadas)
    n_inicial = int(fracao_inicial * n)
    resto = ordenadas[n_inicial:]
    tamanho = len(resto) // n_janelas
    if n_inicial == 0 or tamanho == 0:
        raise ErroDobras(f"{n} linhas são poucas para {n_janelas} janelas com treino inicial de {fracao_inicial:.0%}")
    janelas = []
    for k in range(n_janelas):
        fim = len(resto) if k == n_janelas - 1 else (k + 1) * tamanho
        fatia = resto[k * tamanho:fim]
        primeira = fatia[0]["published_at"]
        janelas.append(Janela(k + 1, tuple(l["video_id"] for l in fatia), primeira,
                              primeira - timedelta(days=intervalo_dias)))
    return Estrutura(tuple(l["video_id"] for l in ordenadas), tuple(l["video_id"] for l in ordenadas[:n_inicial]),
                     tuple(janelas), intervalo_dias, fracao_inicial)


def ids_de_treino(estrutura: Estrutura, k: int, linhas: Sequence[Mapping]) -> tuple[str, ...]:
    """IDs de treino da dobra k: linhas anteriores à janela k publicadas até o corte de treino."""
    janela = estrutura.janelas[k - 1]
    ate_aqui = set(estrutura.ids_treino_inicial)
    for anterior in estrutura.janelas[:k - 1]:
        ate_aqui |= set(anterior.ids_avaliacao)
    por_id = {l["video_id"]: l for l in linhas}
    return tuple(sorted((v for v in ate_aqui if por_id[v]["published_at"] <= janela.corte_de_treino),
                        key=lambda v: _ordem(por_id[v])))


def afirmacoes(estrutura: Estrutura, linhas: Sequence[Mapping], ids_congelados: Sequence[str]) -> dict:
    """Confere as afirmações obrigatórias; devolve cada uma com o resultado e levanta se alguma falhar."""
    por_id = {l["video_id"]: l for l in linhas}
    resultado = {"linhas_conferem_com_a_lista_congelada": set(ids_congelados) == set(por_id)
                 and len(ids_congelados) == len(set(ids_congelados))}
    disjuntos, folga, so_anteriores = [], [], []
    for janela in estrutura.janelas:
        treino = ids_de_treino(estrutura, janela.indice, linhas)
        avaliacao = set(janela.ids_avaliacao)
        disjuntos.append(not (set(treino) & avaliacao))
        maior = max((por_id[v]["published_at"] for v in treino), default=None)
        menor = min(por_id[v]["published_at"] for v in avaliacao)
        folga.append(maior is None or maior + timedelta(days=estrutura.intervalo_dias) <= menor)
        permitido = set(estrutura.ids_treino_inicial)
        for anterior in estrutura.janelas[:janela.indice - 1]:
            permitido |= set(anterior.ids_avaliacao)
        so_anteriores.append(set(treino) <= permitido)
    resultado |= {"treino_e_avaliacao_disjuntos": all(disjuntos),
                  "treino_mais_intervalo_ate_a_primeira_publicacao_da_janela": all(folga),
                  "treino_so_com_linhas_anteriores_a_janela": all(so_anteriores)}
    falhas = sorted(k for k, ok in resultado.items() if not ok)
    if falhas:
        raise ErroDobras(f"afirmações obrigatórias violadas: {falhas}")
    return resultado


def descrever(estrutura: Estrutura, linhas: Sequence[Mapping]) -> dict:
    """Datas, contagens e hashes de IDs de cada dobra (para o protocolo e o relatório)."""
    por_id = {l["video_id"]: l for l in linhas}
    saida = {"intervalo_dias": estrutura.intervalo_dias, "fracao_inicial": estrutura.fracao_inicial,
             "n_linhas": len(estrutura.ids_ordenados), "sha256_ids_ordenados": sha256_ids(estrutura.ids_ordenados),
             "dobras": []}
    for janela in estrutura.janelas:
        treino = ids_de_treino(estrutura, janela.indice, linhas)
        saida["dobras"].append({
            "dobra": janela.indice, "n_treino": len(treino), "n_avaliacao": len(janela.ids_avaliacao),
            "corte_de_treino_utc": janela.corte_de_treino.isoformat(),
            "primeira_publicacao_da_janela_utc": janela.primeira_publicacao.isoformat(),
            "ultima_publicacao_da_janela_utc": max(por_id[v]["published_at"] for v in janela.ids_avaliacao).isoformat(),
            "sha256_ids_treino": sha256_ids(treino), "sha256_ids_avaliacao": sha256_ids(janela.ids_avaliacao)})
    return saida


# ──── Sorteio dos canais retidos para a avaliação de canais não vistos (D-D2, D-C4; protocolo final §4).
FRACAO = 0.30
SEMENTES = (20261015, 20261016, 20261017)


def _arredondar_meio_para_cima(valor: float) -> int:
    return int(math.floor(valor + 0.5))


def alocacao_por_estrato(canais: Mapping[str, tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Quantos canais cada estrato (setor, tipo_canal) devolve ao sorteio."""
    tamanhos = Counter(tuple(estrato) for estrato in canais.values())
    alocacao = {}
    for estrato, n in tamanhos.items():
        k = _arredondar_meio_para_cima(FRACAO * n)
        alocacao[estrato] = min(max(k, 1), n - 1) if n >= 2 else 0
    return alocacao


def sortear(canais: Mapping[str, tuple[str, str]], semente: int) -> list[str]:
    """Canais retidos de um sorteio, em ordem alfabética."""
    alocacao = alocacao_por_estrato(canais)
    rng = np.random.default_rng(semente)
    retidos: list[str] = []
    for estrato in sorted(alocacao):
        ids = sorted(c for c, e in canais.items() if tuple(e) == estrato)
        if alocacao[estrato]:
            retidos += [str(c) for c in rng.choice(ids, size=alocacao[estrato], replace=False)]
    return sorted(retidos)


def tres_sorteios(canais: Mapping[str, tuple[str, str]]) -> list[dict]:
    """Os três sorteios do protocolo, com semente e canais retidos."""
    return [{"semente": semente, "canais_retidos": sortear(canais, semente)} for semente in SEMENTES]


# ──── Histórico, referências e fallback de setor recalculados POR DOBRA, com auditoria independente.
TOLERANCIA = 1e-9
COLUNAS_COMPARADAS = ("ref_log", "pred_media_aritmetica", "pred_mediana", "pred_mme_log", "hist_n_posts",
                      "hist_ref_log_mediana", "hist_log_media", "hist_mme_log", "hist_ref_setor", "y_rel",
                      "estado_referencia", "n_posts_setor")


class ErroHistoricoDobra(ValueError):
    """Linha, post ou contexto incompatível com o recálculo por dobra."""


def mascarar_rotulos(posts: Iterable[hs.Post], atribuicao: Mapping[str, str], blocos: Collection[str]
                     ) -> list[hs.Post]:
    """Cópia dos posts com `curtidas = None` nos vídeos dos blocos dados (ex.: blocos de teste).

    Sem curtidas o post não entra no histórico nem no fallback (`historico` exige curtidas conhecidas), de modo que o
    desenvolvimento não pode depender de rótulos de teste, mesmo que a base em memória os contenha."""
    proibidos = set(blocos)
    return [replace(p, curtidas=None) if atribuicao.get(p.video_id) in proibidos else p for p in posts]


def mascarar_por_data_e_bloco(posts: Iterable[hs.Post], atribuicao: Mapping[str, str], limite_utc: datetime,
                              blocos_permitidos: Collection[str] = ("treino", "validacao")) -> list[hs.Post]:
    """Anula as curtidas de todo post publicado depois de `limite_utc` (fim da última janela de desenvolvimento) e de
    todo post atribuído a um bloco fora de `blocos_permitidos`.

    A regra por data independe de ID e de partição: um substituto que não está na partição original, ou um post só de
    histórico posterior ao desenvolvimento, não escapa. Nenhuma linha de desenvolvimento muda, pois o histórico e o
    fallback usam só posts anteriores à publicação do alvo menos a defasagem."""
    permitidos = set(blocos_permitidos)
    saida = []
    for p in posts:
        bloco = atribuicao.get(p.video_id)
        if p.published_at > limite_utc or (bloco is not None and bloco not in permitidos):
            saida.append(replace(p, curtidas=None))
        else:
            saida.append(p)
    return saida


def contexto_da_dobra(nome: str, corte_de_treino: datetime, canais_excluidos: Collection[str] = ()
                      ) -> hs.ContextoTreino:
    return hs.ContextoTreino(nome, corte_de_treino, frozenset(canais_excluidos))


def recalcular_linhas(posts: Sequence[hs.Post], canais: Mapping[str, hs.Canal], linhas_base: Sequence[Mapping],
                      ctx: hs.ContextoTreino, *, defasagem_dias: int = hs.DEFASAGEM_PADRAO) -> list[dict]:
    """Linhas base com todas as colunas do N1 (atributos, alvo, referências, auditoria) recalculadas no contexto.

    As colunas que não são do N1 (legenda, capa, Gemini, locais) permanecem como vieram. Identidade (canal,
    publicação) e `y_log` do alvo têm de coincidir com a linha base."""
    por_id = {p.video_id: p for p in posts}
    alvos = []
    for linha in linhas_base:
        post = por_id.get(linha["video_id"])
        if post is None or post.papel != Papel.ALVO.value:
            raise ErroHistoricoDobra(f"{linha['video_id']}: não é um alvo da base de posts")
        alvos.append(post)
    novas = {l["video_id"]: l for l in hs.calcular(posts, canais, alvos, ctx, defasagem_dias=defasagem_dias)}
    saida = []
    for base in linhas_base:
        nova = novas[base["video_id"]]
        if nova["channel_id"] != base["channel_id"] or nova["published_at"] != base["published_at"]:
            raise ErroHistoricoDobra(f"{base['video_id']}: canal ou publicação diferem da base de posts")
        if base.get("y_log") is not None and nova["y_log"] is not None and abs(base["y_log"] - nova["y_log"]) > TOLERANCIA:
            raise ErroHistoricoDobra(f"{base['video_id']}: o alvo da linha difere do da base de posts")
        saida.append({**base, **nova})
    return saida


# --- auditoria independente ---------------------------------------------------------------------------------------


def _historico_direto(posts: Sequence[hs.Post], alvo: hs.Post, defasagem: int) -> list[hs.Post]:
    limite = alvo.published_at - timedelta(days=defasagem)
    candidatos = [p for p in posts if p.channel_id == alvo.channel_id and p.papel in hs.PAPEIS_HISTORICO
                  and p.curtidas is not None and p.video_id != alvo.video_id and p.published_at <= limite]
    return sorted(candidatos, key=lambda p: (p.published_at, p.video_id))[-hs.MAX_POSTS:]


def _setor_direto(posts: Sequence[hs.Post], canais: Mapping[str, hs.Canal], alvo: hs.Post, setor: str,
                  limite: datetime, excluidos: Collection[str]) -> list[int]:
    return [p.curtidas for p in posts if p.channel_id in canais and canais[p.channel_id].setor == setor
            and p.channel_id not in excluidos and p.papel in hs.PAPEIS_HISTORICO and p.curtidas is not None
            and p.video_id != alvo.video_id and p.published_at <= limite]


def _mme(curtidas: Sequence[int], meia_vida: int = hs.MEIA_VIDA_POSTS) -> float:
    ultimo = len(curtidas) - 1
    pesos = [0.5 ** ((ultimo - i) / meia_vida) for i in range(len(curtidas))]
    return sum(w * math.log1p(c) for w, c in zip(pesos, curtidas, strict=True)) / sum(pesos)


def _perto(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOLERANCIA


def auditar_dobra(posts: Sequence[hs.Post], canais: Mapping[str, hs.Canal], linhas: Sequence[Mapping],
                  ctx: hs.ContextoTreino, *, defasagem_dias: int = hs.DEFASAGEM_PADRAO) -> dict:
    """Recalcula cada linha por um caminho independente e lista as violações das regras da dobra (vazio = aprovado)."""
    por_id = {p.video_id: p for p in posts}
    defasagem = timedelta(days=defasagem_dias)
    violacoes: dict[str, list[str]] = {nome: [] for nome in (
        "alvo_no_proprio_historico", "historico_alem_da_defasagem", "referencia_diferente_do_calculo_direto",
        "fallback_com_post_apos_o_corte_da_dobra", "fallback_alem_da_defasagem", "fallback_diferente_do_calculo_direto",
        "cadencia_com_post_posterior")}
    n_fallback = 0
    for linha in linhas:
        v = linha["video_id"]
        alvo = por_id[v]
        limite = alvo.published_at - defasagem
        historico = _historico_direto(posts, alvo, defasagem_dias)
        if linha.get("hist_ultimo_utc") is not None and linha["hist_ultimo_utc"] > limite:
            violacoes["historico_alem_da_defasagem"].append(v)
        if linha.get("hist_ultimo_utc") is not None and linha["hist_ultimo_utc"] >= alvo.published_at:
            violacoes["alvo_no_proprio_historico"].append(v)
        if not _perto(linha.get("limite_historico_utc").timestamp() if linha.get("limite_historico_utc") else None,
                      limite.timestamp()):
            violacoes["historico_alem_da_defasagem"].append(v)
        if linha["hist_n_posts"] != len(historico):
            violacoes["referencia_diferente_do_calculo_direto"].append(v)
        if len(historico) >= hs.MIN_POSTS:
            curtidas = [p.curtidas for p in historico]
            esperado = (math.log1p(statistics.median(curtidas)), math.log1p(statistics.fmean(curtidas)), _mme(curtidas))
            obtido = (linha["ref_log"], linha["pred_media_aritmetica"], linha["pred_mme_log"])
            if (linha["estado_referencia"] != "canal" or not _perto(linha["ref_log"], esperado[0])
                    or not _perto(linha["pred_media_aritmetica"], esperado[1])
                    or not _perto(obtido[2], esperado[2])):
                violacoes["referencia_diferente_do_calculo_direto"].append(v)
            continue
        n_fallback += 1
        limite_setor = min(ctx.treino_ate_utc, limite)
        gravado = linha.get("limite_setor_utc")
        if gravado is not None and gravado > ctx.treino_ate_utc:
            violacoes["fallback_com_post_apos_o_corte_da_dobra"].append(v)
        if gravado is not None and gravado > limite:
            violacoes["fallback_alem_da_defasagem"].append(v)
        curtidas = _setor_direto(posts, canais, alvo, linha["setor"], limite_setor, ctx.canais_excluidos)
        if not curtidas:
            ok = linha["estado_referencia"] == "indisponivel" and linha["ref_log"] is None
        else:
            referencia = math.log1p(statistics.median(curtidas))
            ok = (linha["estado_referencia"] == "setor" and _perto(linha["ref_log"], referencia)
                  and linha["n_posts_setor"] == len(curtidas))
        if not ok:
            violacoes["fallback_diferente_do_calculo_direto"].append(v)
        if linha.get("n_posts_setor") is not None and linha["n_posts_setor"] > len(curtidas):
            violacoes["fallback_com_post_apos_o_corte_da_dobra"].append(v)
    # Cadência: só publicações anteriores à do alvo (sem a defasagem), por definição do N1.
    for linha in linhas:
        alvo = por_id[linha["video_id"]]
        anteriores = [p.published_at for p in posts if p.channel_id == alvo.channel_id and p.papel in hs.PAPEIS_HISTORICO
                      and p.video_id != alvo.video_id and p.published_at < alvo.published_at]
        esperado = ((alvo.published_at - max(anteriores)).total_seconds() / 86400) if anteriores else None
        if not _perto(linha.get("dias_desde_post_anterior"), esperado):
            violacoes["cadencia_com_post_posterior"].append(linha["video_id"])
    violacoes = {k: sorted(set(v)) for k, v in violacoes.items()}
    return {"n_linhas": len(linhas), "n_com_fallback_de_setor": n_fallback, "violacoes": violacoes,
            "n_violacoes": sum(len(v) for v in violacoes.values()),
            "aprovado": not any(violacoes.values()), "corte_de_treino_utc": ctx.treino_ate_utc.isoformat(),
            "defasagem_dias": defasagem_dias}


def comparar_com_originais(originais: Sequence[Mapping], recalculadas: Sequence[Mapping],
                           ids_avaliacao: Collection[str]) -> dict:
    """Quantas linhas e colunas das colunas originais diferem das recalculadas na dobra, em treino e em avaliação."""
    antigas = {l["video_id"]: l for l in originais}
    avaliacao = set(ids_avaliacao)
    alteradas = {"treino": set(), "avaliacao": set()}
    por_coluna: dict[str, int] = {}
    faltando = []
    for nova in recalculadas:
        antiga = antigas.get(nova["video_id"])
        if antiga is None:
            faltando.append(nova["video_id"])
            continue
        grupo = "avaliacao" if nova["video_id"] in avaliacao else "treino"
        for coluna in COLUNAS_COMPARADAS:
            a, b = antiga.get(coluna), nova.get(coluna)
            igual = (a == b) if isinstance(a, str) or isinstance(b, str) else _perto(a, b)
            if not igual:
                por_coluna[coluna] = por_coluna.get(coluna, 0) + 1
                alteradas[grupo].add(nova["video_id"])
    n_aval = sum(l["video_id"] in avaliacao for l in recalculadas)
    return {"n_comparadas": len(recalculadas) - len(faltando), "sem_linha_original": sorted(faltando),
            "n_avaliacao": n_aval, "n_treino": len(recalculadas) - n_aval,
            "linhas_alteradas_na_avaliacao": len(alteradas["avaliacao"]),
            "linhas_alteradas_no_treino": len(alteradas["treino"]), "colunas_alteradas": dict(sorted(por_coluna.items())),
            "colunas_originais_seguras_para_esta_dobra": not alteradas["avaliacao"] and not alteradas["treino"]}


# ──── Portões de população do experimento audiovisual: piso, exclusões, duplicatas, sonda de canal e coerê
PISOS = {"n_videos": 600, "n_canais": 14, "n_setores": 2, "videos_por_setor": 50, "canais_por_setor": 3,
         "canais_na_avaliacao_por_janela": 8}
LIMITE_DE_EXCLUSOES = 0.10
LIMIAR_SMD = 0.2
COSSENO_DE_DUPLICATA = 0.98
FOLGA_DA_SONDA = 0.20
MIN_CANAIS_PARA_IC = 8
COLUNA_EMBEDDING_DE_DUPLICATA = f"{col.PREFIXO}emb_media"
SEMENTE = 20261015


class ErroPortao(ValueError):
    """Um portão pré-registrado reprovou: o experimento não pode seguir."""


def verificar_piso(linhas: Sequence[Mapping], janelas: Sequence[Sequence[Mapping]], pisos: Mapping = PISOS) -> dict:
    """Piso da população pós-exclusão e por janela de avaliação; qualquer item abaixo do piso reprova."""
    canais = {l["channel_id"] for l in linhas}
    por_setor: dict[str, dict] = {}
    for l in linhas:
        s = por_setor.setdefault(str(l["setor"]), {"videos": 0, "canais": set()})
        s["videos"] += 1
        s["canais"].add(l["channel_id"])
    medidas = {"n_videos": len(linhas), "n_canais": len(canais), "n_setores": len(por_setor),
               "por_setor": {k: {"videos": v["videos"], "canais": len(v["canais"])} for k, v in sorted(por_setor.items())},
               "canais_na_avaliacao_por_janela": [len({l["channel_id"] for l in j}) for j in janelas]}
    falhas = []
    for item in ("n_videos", "n_canais", "n_setores"):
        if medidas[item] < pisos[item]:
            falhas.append(f"{item}: {medidas[item]} < {pisos[item]}")
    for setor, v in medidas["por_setor"].items():
        if v["videos"] < pisos["videos_por_setor"] or v["canais"] < pisos["canais_por_setor"]:
            falhas.append(f"setor {setor}: {v['videos']} vídeos e {v['canais']} canais "
                          f"(piso {pisos['videos_por_setor']} e {pisos['canais_por_setor']})")
    for k, n in enumerate(medidas["canais_na_avaliacao_por_janela"], start=1):
        if n < pisos["canais_na_avaliacao_por_janela"]:
            falhas.append(f"janela {k}: {n} canais na avaliação < {pisos['canais_na_avaliacao_por_janela']}")
    resultado = {"pisos": dict(pisos), "medidas": medidas, "falhas": falhas, "aprovado": not falhas}
    if falhas:
        raise ErroPortao(f"população abaixo do piso: {falhas}")
    return resultado


def _smd(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    s = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return None if s == 0 else float((np.mean(a) - np.mean(b)) / s)


def comparar_incluidos_excluidos(incluidas: Sequence[Mapping], excluidas: Sequence[Mapping],
                                 ordem_amostra: Mapping[str, int | None]) -> dict:
    """Diferença padronizada (SMD) em y_rel, ref_log, tamanho do histórico, ordem da amostra, canal e setor.

    Só linhas de desenvolvimento. |SMD| > 0,2 vira ressalva escrita; com menos de 2 excluídas não há comparação."""
    saida: dict = {"n_incluidas": len(incluidas), "n_excluidas": len(excluidas), "limiar_smd": LIMIAR_SMD,
                   "smd": {}, "ressalvas": []}
    if len(excluidas) < 2:
        saida["aviso"] = "menos de duas linhas excluídas: comparação não calculada"
        return saida

    def numeros(linhas, chave):
        valores = [chave(l) for l in linhas]
        return np.array([v for v in valores if v is not None and np.isfinite(v)], dtype=float)

    alvos = {"y_rel": lambda l: l.get("y_rel"), "ref_log": lambda l: l.get("ref_log"),
             "hist_n_posts": lambda l: l.get("hist_n_posts"), "ordem_amostra": lambda l: ordem_amostra.get(l["video_id"])}
    for nome, f in alvos.items():
        saida["smd"][nome] = _smd(numeros(incluidas, f), numeros(excluidas, f))
    for nome, campo in (("canal", "channel_id"), ("setor", "setor")):
        piores = (None, 0.0)
        for valor in sorted({str(l[campo]) for l in (*incluidas, *excluidas)}):
            a = np.array([str(l[campo]) == valor for l in incluidas], dtype=float)
            b = np.array([str(l[campo]) == valor for l in excluidas], dtype=float)
            d = _smd(a, b)
            if d is not None and abs(d) > abs(piores[1]):
                piores = (valor, d)
        saida["smd"][nome] = {"maior_em_modulo": piores[1], "categoria": piores[0]}
    for nome, valor in saida["smd"].items():
        v = valor["maior_em_modulo"] if isinstance(valor, dict) else valor
        if v is not None and abs(v) > LIMIAR_SMD:
            saida["ressalvas"].append(f"|SMD| de {nome} = {abs(v):.2f} > {LIMIAR_SMD}: a população incluída difere da excluída")
    return saida


def teto_de_exclusoes(n_entrada: int, n_excluidas: int) -> dict:
    fracao = n_excluidas / n_entrada if n_entrada else 0.0
    acima = fracao > LIMITE_DE_EXCLUSOES
    return {"fracao_excluida": fracao, "limite": LIMITE_DE_EXCLUSOES, "acima_do_teto": acima,
            "ressalva": ("Exclusões acima de 10%: o resultado vale para a população com extração concluída, sujeita a "
                         "viés de sobrevivência, e é declarado como limitação.") if acima else None}


def duplicatas_por_cosseno(linhas: Sequence[Mapping], *, limiar: float = COSSENO_DE_DUPLICATA,
                           coluna: str = COLUNA_EMBEDDING_DE_DUPLICATA) -> dict:
    """Pares com cosseno ≥ limiar na média dos embeddings. Fica o mais antigo de cada grupo; o resto é removido.

    Grupos são componentes conexos dos pares; ordem de publicação e `video_id` desempatam. Linhas sem embedding
    ficam e são contadas. Roda antes de dividir em dobras e sem olhar alvo."""
    com = [l for l in linhas if l.get(coluna) is not None]
    saida = {"limiar": limiar, "coluna": coluna, "n_sem_embedding": len(linhas) - len(com), "pares": [],
             "ids_removidos": []}
    if len(com) < 2:
        return saida
    matriz = np.array([l[coluna] for l in com], dtype=float)
    norma = np.linalg.norm(matriz, axis=1)
    validos = norma > 0
    unit = np.where(validos[:, None], matriz / np.where(validos, norma, 1.0)[:, None], 0.0)
    cos = unit @ unit.T
    ordem = sorted(range(len(com)), key=lambda i: (com[i]["published_at"], com[i]["video_id"]))
    posicao = {i: k for k, i in enumerate(ordem)}
    removidos: set[int] = set()
    for i in ordem:
        if i in removidos:
            continue
        for j in ordem:
            if posicao[j] > posicao[i] and j not in removidos and cos[i, j] >= limiar:
                removidos.add(j)
                saida["pares"].append({"mantido": com[i]["video_id"], "removido": com[j]["video_id"],
                                       "cosseno": float(cos[i, j])})
    saida["ids_removidos"] = sorted(com[j]["video_id"] for j in removidos)
    return saida


def pares_duplicados_entre(grupo_a: Sequence[Mapping], grupo_b: Sequence[Mapping], *,
                           limiar: float = COSSENO_DE_DUPLICATA, coluna: str = COLUNA_EMBEDDING_DE_DUPLICATA) -> dict:
    """Pares (a, b) com cosseno ≥ limiar entre dois grupos (ex.: desenvolvimento × teste). Só conta; nada é removido."""
    a = [l for l in grupo_a if l.get(coluna) is not None]
    b = [l for l in grupo_b if l.get(coluna) is not None]
    saida = {"limiar": limiar, "n_a": len(a), "n_b": len(b), "n_pares": 0, "pares": []}
    if not a or not b:
        return saida

    def unitaria(linhas):
        m = np.array([l[coluna] for l in linhas], dtype=float)
        n = np.linalg.norm(m, axis=1)
        return np.where((n > 0)[:, None], m / np.where(n > 0, n, 1.0)[:, None], 0.0)

    cos = unitaria(a) @ unitaria(b).T
    for i, j in zip(*np.nonzero(cos >= limiar), strict=True):
        saida["pares"].append({"a": a[i]["video_id"], "b": b[j]["video_id"], "cosseno": float(cos[i, j])})
    saida["n_pares"] = len(saida["pares"])
    return saida


def equivalencia_da_duracao(linhas: Sequence[Mapping]) -> dict:
    """`duracao_s` (inteiro) contra `avl_duracao_s` (medida do arquivo): a convenção é o teto da medida."""
    pares = [(l["duracao_s"], l["avl_duracao_s"]) for l in linhas
             if l.get("duracao_s") is not None and l.get("avl_duracao_s") is not None]
    if not pares:
        return {"n": 0}
    iguais = sum(math.ceil(a - 1e-9) == int(d) for d, a in pares)
    diffs = np.array([abs(d - a) for d, a in pares])
    return {"n": len(pares), "fracao_igual_ao_teto": iguais / len(pares), "maior_diferenca_s": float(diffs.max()),
            "diferenca_mediana_s": float(np.median(diffs)),
            "convencao": "duracao_s = teto da duração medida no arquivo (inteiro)"}


def sonda_de_canal(linhas: Sequence[Mapping], *, semente: int = SEMENTE) -> dict:
    """Diagnóstico: quanto os atributos locais identificam o canal em vídeos retidos de canais vistos.

    Classificador fixo (regressão logística, sem ajuste de hiperparâmetros), treinado nas linhas não retidas; retido =
    cada terceiro vídeo de cada canal com ao menos 6 vídeos. Não altera nenhum modelo nem entra em decisão."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src.modelos.preprocessamento import PreprocessadorAudiovisual

    por_canal: dict[str, list] = {}
    for l in sorted(linhas, key=lambda l: (l["published_at"], l["video_id"])):
        por_canal.setdefault(str(l["channel_id"]), []).append(l)
    treino, retido = [], []
    for canal, vs in por_canal.items():
        if len(vs) < 6:
            continue
        for k, l in enumerate(vs):
            (retido if k % 3 == 2 else treino).append(l)
    if len({str(l["channel_id"]) for l in treino}) < 2 or not retido:
        return {"avaliavel": False, "motivo": "canais com ao menos 6 vídeos insuficientes"}
    escalares = [c for c in col.COLUNAS_DE_ATRIBUTO if c not in col.COLUNAS_DE_IMPRESSAO_DIGITAL
                 and c not in col.COLUNAS_VETORIAIS and c not in af.CATEGORICAS_LOCAIS
                 and c != af.COLUNA_DUPLICADA and any(l.get(c) is not None for l in treino)]
    pre = PreprocessadorAudiovisual(escalares, [], {c: af.COMPONENTES_PCA for c in col.COLUNAS_VETORIAIS},
                                    colunas_locais=tuple(escalares)).fit(treino)
    X_treino, X_retido = pre.transform(treino), pre.transform(retido)
    mantidas = [j for j in range(X_treino.shape[1]) if np.isfinite(X_treino[:, j]).any()]
    imp, esc = SimpleImputer(strategy="median"), StandardScaler()
    A = esc.fit_transform(imp.fit_transform(X_treino[:, mantidas]))
    B = esc.transform(imp.transform(X_retido[:, mantidas]))
    y_treino = np.array([str(l["channel_id"]) for l in treino])
    y_retido = np.array([str(l["channel_id"]) for l in retido])
    modelo = LogisticRegression(max_iter=2000, random_state=semente).fit(A, y_treino)
    acerto = float((modelo.predict(B) == y_retido).mean())
    canais = sorted(set(y_treino))
    maioria = float(np.mean(y_retido == max(canais, key=lambda c: int((y_treino == c).sum()))))
    acaso = 1 / len(canais)
    return {"avaliavel": True, "n_treino": len(treino), "n_retidos": len(retido), "n_canais": len(canais),
            "acuracia_nos_retidos": acerto, "acuracia_da_classe_majoritaria": maioria, "acaso_uniforme": acaso,
            "folga_pre_registrada": FOLGA_DA_SONDA,
            "muito_acima_do_acaso": acerto > max(maioria, acaso) + FOLGA_DA_SONDA,
            "leitura": "diagnóstico do atalho de canal; não é critério de decisão nem ajusta modelo"}


def conferir_visoes(desenvolvimento: Sequence[Mapping], completa: Sequence[Mapping],
                    ids_lim: Collection[str] = ()) -> dict:
    """A visão de desenvolvimento é subconjunto da completa, sem teste, intervalo ou LIM, com a mesma elegibilidade."""
    por_id = {l["video_id"]: l for l in completa}
    if len(por_id) != len(completa):
        raise ErroPortao("IDs repetidos na tabela completa")
    ids_dev = [l["video_id"] for l in desenvolvimento]
    if len(set(ids_dev)) != len(ids_dev):
        raise ErroPortao("IDs repetidos na visão de desenvolvimento")
    fora = sorted(set(ids_dev) - set(por_id))
    if fora:
        raise ErroPortao(f"{len(fora)} IDs de desenvolvimento ausentes da tabela completa (ex.: {fora[:3]})")
    proibidos = sorted(l["video_id"] for l in desenvolvimento
                       if str(l.get("bloco", "")).startswith("teste") or l.get("uso_no_contexto") not in ("treino", "avaliacao")
                       or l["video_id"] in set(ids_lim))
    if proibidos:
        raise ErroPortao(f"{len(proibidos)} linhas de teste, intervalo ou LIM na visão de desenvolvimento "
                         f"(ex.: {proibidos[:3]})")
    colunas_estado = [c for c in (*col.COLUNAS_DE_ESTADO, "avl_disponibilidade")]
    divergentes = sorted(l["video_id"] for l in desenvolvimento
                         if any(l.get(c) != por_id[l["video_id"]].get(c) for c in colunas_estado))
    if divergentes:
        raise ErroPortao(f"{len(divergentes)} linhas com elegibilidade diferente nas duas visões (ex.: {divergentes[:3]})")
    return {"n_desenvolvimento": len(ids_dev), "n_completa": len(completa), "subconjunto": True,
            "sem_teste_intervalo_ou_lim": True, "mesma_elegibilidade": True,
            "sha256_ids_desenvolvimento": sha256_ids(sorted(ids_dev))}
