"""Orquestração da avaliação final (protocolo final §3 a §8): braços principal, canais não vistos, sensibilidade e ablação.

O orquestrador não sabe de arquivos: recebe um `fornecedor(espec)` que devolve as linhas do contexto
(`contextos_finais.materializar_contexto` na execução real; dados sintéticos nos testes). Os nomes dos blocos vêm da
configuração, então o MESMO código roda como ensaio com a validação no lugar dos blocos de teste, sem ler rótulo de
teste. Qualquer contexto com bloco de teste exige o portão da consulta única aberto (`protocolo_final`).

Modelos servidos (D-D7): N4 (vídeo, especificação completa), N3 (imagem) e N2 (texto), com q10/q50/q90 no braço
principal; só o q50 nos demais. Ablação: especificações aninhadas E1 a E4, corte único, mesmas linhas.
"""

import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src import familias_atributos as fa
from src import modelo_desenvolvimento as md
from src import modelo_modalidades as mm
from src.avaliacao import baseline_inscritos as bi
from src.avaliacao import bootstrap as bs
from src.avaliacao import contextos_finais as cf
from src.avaliacao import declaracao_faixa as dcl
from src.avaliacao import desenvolvimento as dv
from src.avaliacao import dobras as db
from src.avaliacao import faixas as fx
from src.avaliacao import ganho as gn
from src.avaliacao import relatorio as rl
from src.avaliacao.protocolo_final import ConsultaAberta, PortaoFechado

BLOCOS_ANTERIORES = ("treino", "validacao")
BLOCOS_TESTE = ("teste_1", "teste_2", "teste_3")
MODELOS_SERVIDOS = ("N4", "N3", "N2")
NIVEIS_ABLACAO = ("E1", "E2", "E3", "E4")
FAMILIAS_DA_ABLACAO = {"E1": ("historico",), "E2": ("historico", "legenda"), "E3": ("historico", "legenda", "capa"),
                       "E4": ("historico", "legenda", "capa", "gemini")}
CONTRASTES_ABLACAO = (("E4", "E3"), ("E3", "E2"), ("E2", "E1"))
BASE_DO_ACASO = "media_aritmetica"
# Abaixo disto o PCA e as folhas do modelo não significam nada: a dobra exploratória é pulada e DECLARADA. O braço
# principal e a ablação não pulam: com treino menor que isto a avaliação falha de forma explícita.
MIN_LINHAS_TREINO = 20
Fornecedor = Callable[[cf.EspecContexto], Mapping]  # {"linhas": [...], "manifesto": {"especificacao": {...}}}


@dataclass(frozen=True)
class Configuracao:
    coleta: datetime
    sorteios: Sequence[Mapping]
    blocos_avaliados: tuple[str, ...] = BLOCOS_TESTE
    blocos_anteriores: tuple[str, ...] = BLOCOS_ANTERIORES
    parametros: Mapping = field(default_factory=lambda: dict(mm.CONFIGURACAO_A))
    n_repeticoes: int = bs.N_REPETICOES
    semente: int = bs.SEMENTE
    defasagem: int = 30
    intervalo: int = 30
    defasagem_s1: int = 90
    intervalo_s2: int = 90
    inscritos: Mapping[str, int | None] | None = None  # tabela restrita (baseline 4); None = não avaliado, declarado


# --- especificações dos contextos e dos modelos -----------------------------------------------------------------------


def _permitidos(cfg: Configuracao, k: int) -> tuple[str, ...]:
    return (*cfg.blocos_anteriores, *cfg.blocos_avaliados[:k])


def espec_principal(cfg: Configuracao, k: int) -> cf.EspecContexto:
    bloco = cfg.blocos_avaliados[k - 1]
    return cf.EspecContexto(f"principal_{bloco}", bloco, (bloco,), _permitidos(cfg, k), cfg.defasagem, cfg.intervalo)


def espec_nao_visto(cfg: Configuracao, sorteio: int, k: int, retidos: Sequence[str]) -> cf.EspecContexto:
    bloco = cfg.blocos_avaliados[k - 1]
    return cf.EspecContexto(f"nao_visto_s{sorteio}_{bloco}", bloco, (bloco,), _permitidos(cfg, k), cfg.defasagem,
                            cfg.intervalo, tuple(sorted(retidos)))


def espec_sensibilidade(cfg: Configuracao, rotulo: str, k: int) -> cf.EspecContexto:
    bloco = cfg.blocos_avaliados[k - 1]
    defasagem = cfg.defasagem_s1 if rotulo == "S1" else cfg.defasagem
    intervalo = cfg.intervalo_s2 if rotulo == "S2" else cfg.intervalo
    return cf.EspecContexto(f"sensibilidade_{rotulo}_{bloco}", bloco, (bloco,), _permitidos(cfg, k), defasagem,
                            intervalo)


def espec_ablacao(cfg: Configuracao) -> cf.EspecContexto:
    todos = cfg.blocos_avaliados
    return cf.EspecContexto("ablacao_corte_unico", todos[0], tuple(todos), _permitidos(cfg, len(todos)),
                            cfg.defasagem, cfg.intervalo)


def especs_servidos(base: Mapping) -> dict[str, md.EspecificacaoAtributos]:
    return {"N4": md.EspecificacaoAtributos(**base), "N3": fa.especificacao_da_modalidade(base, "imagem"),
            "N2": fa.especificacao_da_modalidade(base, "texto")}


def especs_ablacao(base: Mapping, hashes_congelados: Mapping[str, str] | None = None
                   ) -> dict[str, md.EspecificacaoAtributos]:
    especs = {nivel: fa.especificacao_ate(base, blocos) for nivel, blocos in FAMILIAS_DA_ABLACAO.items()}
    if hashes_congelados is not None:
        divergentes = sorted(n for n, e in especs.items() if fa.hash_da_especificacao(e) != hashes_congelados.get(n))
        if divergentes:
            raise ValueError(f"especificação diferente da congelada em {divergentes}")
    return especs


# --- configuração × protocolo congelado ---------------------------------------------------------------------------------------


def _projecao_da_configuracao(cfg: Configuracao, ignorados: Collection[str], com_coleta: bool) -> dict:
    projecao = {"intervalo_treino_teste_dias": cfg.intervalo, "defasagem_historico_dias": cfg.defasagem,
                "n_repeticoes": cfg.n_repeticoes, "semente": cfg.semente,
                "n_estimators": cfg.parametros.get("n_estimators"),
                "parametros": {k: v for k, v in cfg.parametros.items() if k not in ignorados},
                "blocos_avaliados": list(cfg.blocos_avaliados), "blocos_anteriores": list(cfg.blocos_anteriores),
                "sementes_dos_sorteios": [s["semente"] for s in cfg.sorteios],
                "S1": {"defasagem_historico_dias": cfg.defasagem_s1, "intervalo_treino_teste_dias": cfg.intervalo},
                "S2": {"defasagem_historico_dias": cfg.defasagem, "intervalo_treino_teste_dias": cfg.intervalo_s2}}
    if com_coleta:
        projecao["coleta"] = cfg.coleta.isoformat()
    return projecao


def _projecao_do_congelado(d: Mapping, ignorados: Collection[str]) -> dict:
    sens = d["sensibilidade_exploratoria"]
    projecao = {"intervalo_treino_teste_dias": d["intervalo_treino_teste_dias"],
                "defasagem_historico_dias": d["defasagem_historico_dias"],
                "n_repeticoes": d["bootstrap"]["n_repeticoes"], "semente": d["bootstrap"]["semente"],
                "n_estimators": d["iteracoes"]["n_estimators"],
                "parametros": {k: v for k, v in d["modelo"]["parametros"].items() if k not in ignorados},
                "blocos_avaliados": list(d["dobras"]["blocos"]), "blocos_anteriores": list(d["blocos_anteriores"]),
                "sementes_dos_sorteios": list(d["canais_nao_vistos"]["sementes"]),
                "S1": dict(sens["S1"]), "S2": dict(sens["S2"])}
    if "coleta_dos_rotulos_utc" in d:
        projecao["coleta"] = d["coleta_dos_rotulos_utc"]
    return projecao


def exigir_configuracao_congelada(cfg: Configuracao, definicoes: Mapping) -> None:
    """A configuração que vai rodar é a do protocolo congelado; qualquer diferença recusa antes de ler dados."""
    if definicoes.get("baseline_inscritos") and cfg.inscritos is None:
        raise ValueError("o protocolo congelado inclui o baseline 4 (inscritos), mas a configuração não traz a tabela")
    ignorados = set(definicoes["modelo"].get("ignorados_na_comparacao", ()))
    congelado = _projecao_do_congelado(definicoes, ignorados)
    atual = _projecao_da_configuracao(cfg, ignorados, "coleta" in congelado)
    diferentes = sorted(k for k in congelado if congelado[k] != atual[k])
    if diferentes:
        raise ValueError(f"a configuração difere do protocolo congelado em {diferentes}")


# --- utilidades -----------------------------------------------------------------------------------------------------------


def _rodar(fornecedor: Fornecedor, espec: cf.EspecContexto, consulta: ConsultaAberta | None) -> tuple[db.Dobra, Mapping]:
    cf.exigir_portao(espec, consulta=consulta)
    dados = fornecedor(espec)
    return db.separar(dados["linhas"]), dados.get("manifesto", {})


def _exigir_treino(dobra: db.Dobra, espec: cf.EspecContexto) -> None:
    """Braços que não podem ser pulados (principal, ablação): treino insuficiente é erro explícito."""
    if len(dobra.treino) < MIN_LINHAS_TREINO:
        raise ValueError(f"treino insuficiente em {espec.nome}: {len(dobra.treino)} linhas (mínimo {MIN_LINHAS_TREINO})")


def _treino_suficiente(dobra: db.Dobra) -> bool:
    return len(dobra.treino) >= MIN_LINHAS_TREINO


def _especificacao_base(manifesto: Mapping) -> Mapping:
    if "especificacao" not in manifesto:
        raise ValueError("o manifesto do dataset não traz a especificação de atributos")
    return manifesto["especificacao"]


def _concatenar(previsoes: Sequence[Mapping]) -> dict:
    return {chave: np.concatenate([np.asarray(p[chave]) for p in previsoes]) for chave in previsoes[0]}


def _celulas(linhas: Sequence[Mapping], bloco: str, sufixo: str = "") -> list[str]:
    return [f"{l['channel_id']}|{bloco}{sufixo}" for l in linhas]


def _secundarias(dobras: Sequence[tuple[Sequence[Mapping], Mapping, Sequence[float]]]) -> dict:
    """F1 macro em 3 faixas (bordas do treino de cada dobra), R² do alvo relativo e frequência de cruzamento dos
    quantis (casos em que a faixa foi corrigida para mínimo e máximo), sobre as linhas finitas das dobras dadas."""
    f1, cruzou = [], []
    for linhas, prev, bordas in dobras:
        real = np.array([np.nan if l["y_rel"] is None else l["y_rel"] for l in linhas], dtype=float)
        previsto = np.asarray(prev["relativo"], dtype=float)
        ok = np.isfinite(real) & np.isfinite(previsto)
        f1.append((real[ok], previsto[ok], bordas))
        cruzou.append(np.asarray(prev["cruzou"], dtype=bool)[ok])
    reais, previstos = np.concatenate([a for a, _, _ in f1]), np.concatenate([b for _, b, _ in f1])
    cruzou = np.concatenate(cruzou)
    return {"f1_macro_3_faixas": fx.f1_macro_empilhado(f1), "r2_alvo_relativo": fx.r2(reais, previstos),
            "acaso_f1": 1 / fx.N_FAIXAS,
            "faixa_cruzamento": {"n": int(cruzou.size), "n_cruzou": int(cruzou.sum()),
                                 "frequencia": float(cruzou.mean()) if cruzou.size else None,
                                 "regra": "q50 central; limites = minimo e maximo das tres saidas"}}


# --- braço principal ------------------------------------------------------------------------------------------------------


def braco_principal(fornecedor: Fornecedor, cfg: Configuracao, consulta: ConsultaAberta | None) -> tuple[list[dict], Mapping]:
    resultados, base = [], None
    for k, bloco in enumerate(cfg.blocos_avaliados, start=1):
        espec = espec_principal(cfg, k)
        dobra, manifesto = _rodar(fornecedor, espec, consulta)
        _exigir_treino(dobra, espec)
        base = base or _especificacao_base(manifesto)
        previsoes = {nome: db.prever_com_faixa(db.treinar_quantis(dobra.treino, spec, cfg.parametros), dobra.avaliacao)
                     for nome, spec in especs_servidos(base).items()}
        resultados.append({"k": k, "bloco": bloco, "dobra": dobra, "bordas": db.bordas_de_faixa(dobra.treino),
                           "previsoes": previsoes, "sha256_dataset": manifesto.get("sha256_dataset"),
                           "reconciliacao": manifesto.get("reconciliacao_anotacoes"),
                           "inscritos": None if cfg.inscritos is None
                           else bi.prever_dobra(dobra.treino, dobra.avaliacao, cfg.inscritos)})
    return resultados, base


def _contexto_da_declaracao(cfg: Configuracao) -> str:
    """Blocos de teste = declaração do teste final; qualquer outro bloco (ensaio) = declaração de desenvolvimento."""
    return "teste_final" if any(b.startswith("teste") for b in cfg.blocos_avaliados) else "desenvolvimento"


def avaliar_principal(resultados: Sequence[Mapping], cfg: Configuracao) -> dict:
    saida = {}
    for nome in MODELOS_SERVIDOS:
        por_dobra = {}
        for r in resultados:
            bloco, _ = rl._bloco(r["dobra"].avaliacao, r["previsoes"][nome], cfg.coleta, n_repeticoes=cfg.n_repeticoes,
                                 semente=cfg.semente, completo=True)
            por_dobra[r["bloco"]] = bloco | _secundarias([(r["dobra"].avaliacao, r["previsoes"][nome], r["bordas"])])
        linhas = [l for r in resultados for l in r["dobra"].avaliacao]
        previsao = _concatenar([r["previsoes"][nome] for r in resultados])
        celulas = [c for r in resultados for c in _celulas(r["dobra"].avaliacao, r["bloco"])]
        empilhado, _ = rl._bloco(linhas, previsao, cfg.coleta, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente,
                                 completo=True, celulas=celulas)
        saida[nome] = {"empilhado": empilhado | _secundarias(
            [(r["dobra"].avaliacao, r["previsoes"][nome], r["bordas"]) for r in resultados])
            | {"declaracao_da_faixa": dcl.declarar(_contexto_da_declaracao(cfg), modelo=nome,
                                                   total=empilhado["cobertura"],
                                                   por_setor=empilhado["cobertura_por_setor"])},
            "por_dobra": por_dobra}
    saida["treino_por_dobra"] = {r["bloco"]: {"n_treino": len(r["dobra"].treino), "n_avaliacao": len(r["dobra"].avaliacao),
                                              "ignoradas": r["dobra"].ignoradas, "bordas_f1": r["bordas"],
                                              "sha256_dataset": r["sha256_dataset"], "reconciliacao": r["reconciliacao"]} for r in resultados}
    saida["baseline_inscritos"] = _baseline_inscritos_principal(resultados, cfg, saida)
    return saida


def _baseline_inscritos_principal(resultados: Sequence[Mapping], cfg: Configuracao, saida: Mapping) -> dict:
    """Baseline 4 (TAPI, D-D1) nas linhas empilhadas do principal; ajuste só no treino de cada dobra."""
    if cfg.inscritos is None:
        return bi.bloco_nao_avaliado("tabela de inscritos não fornecida à avaliação (`inscritos` ausente na "
                                     "configuração)")
    return bi.montar_bloco(
        {r["bloco"]: r["inscritos"][0] for r in resultados}, {r["bloco"]: r["inscritos"][2] for r in resultados},
        [l for r in resultados for l in r["dobra"].avaliacao], np.concatenate([r["inscritos"][1] for r in resultados]),
        {nome: _concatenar([r["previsoes"][nome] for r in resultados])["central_log"] for nome in MODELOS_SERVIDOS},
        celulas=[c for r in resultados for c in _celulas(r["dobra"].avaliacao, r["bloco"])],
        ids_do_principal={nome: saida[nome]["empilhado"]["populacao"]["ids_sha256"] for nome in MODELOS_SERVIDOS},
        n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)


# --- canais não vistos ----------------------------------------------------------------------------------------------------


def _subconjunto_do_principal(resultados: Sequence[Mapping], nome: str, retidos: frozenset[str],
                              blocos: Collection[str]) -> tuple:
    """Linhas do principal nos canais retidos, só dos blocos que o braço de canais não vistos de fato avaliou."""
    linhas, central, celulas = [], [], []
    for r in (r for r in resultados if r["bloco"] in blocos):
        mascara = [l["channel_id"] in retidos for l in r["dobra"].avaliacao]
        linhas += [l for l, m in zip(r["dobra"].avaliacao, mascara, strict=True) if m]
        central += list(np.asarray(r["previsoes"][nome]["central_log"])[mascara])
        celulas += [c for c, m in zip(_celulas(r["dobra"].avaliacao, r["bloco"]), mascara, strict=True) if m]
    return linhas, central, celulas


def braco_nao_vistos(fornecedor: Fornecedor, cfg: Configuracao, base: Mapping, resultados_principal: Sequence[Mapping],
                     consulta: ConsultaAberta | None) -> dict:
    saida = {"sorteios": []}
    agregado = {nome: {"linhas": [], "central": [], "celulas": []} for nome in MODELOS_SERVIDOS}
    inscritos_agregado = _AcumuladoInscritos()
    for d, sorteio in enumerate(cfg.sorteios, start=1):
        retidos = frozenset(sorteio["canais_retidos"])
        acumulado = {nome: {"linhas": [], "central": [], "celulas": []} for nome in MODELOS_SERVIDOS}
        inscritos_sorteio = _AcumuladoInscritos()
        sem_avaliacao, puladas, blocos_avaliados, contextos = [], [], [], {}
        for k, bloco in enumerate(cfg.blocos_avaliados, start=1):
            dobra, manifesto = _rodar(fornecedor, espec_nao_visto(cfg, d, k, retidos), consulta)
            contextos[bloco] = {"n_treino": len(dobra.treino), "n_avaliacao_total": len(dobra.avaliacao),
                                "reconciliacao": manifesto.get("reconciliacao_anotacoes")}
            vazados = sorted({l["channel_id"] for l in dobra.treino if l["channel_id"] in retidos})
            if vazados:
                raise ValueError(f"canal retido no treino do sorteio {d}, {bloco}: {vazados}")
            avaliacao = [l for l in dobra.avaliacao if l["channel_id"] in retidos]
            if not avaliacao:
                sem_avaliacao.append(bloco)
                continue
            if not _treino_suficiente(dobra):
                puladas.append({"bloco": bloco, "n_treino": len(dobra.treino), "motivo": "treino_insuficiente"})
                continue
            blocos_avaliados.append(bloco)
            if cfg.inscritos is not None:
                resultado_inscritos = bi.prever_dobra(dobra.treino, avaliacao, cfg.inscritos)
                for destino in (inscritos_sorteio, inscritos_agregado):
                    destino.acrescentar(f"s{d}|{bloco}", *resultado_inscritos)
            for nome, spec in especs_servidos(base).items():
                central = db.prever_central(db.treinar_q50(dobra.treino, spec, cfg.parametros), avaliacao)
                for destino in (acumulado[nome], agregado[nome]):
                    destino["linhas"] += avaliacao
                    destino["central"] += list(central["central_log"])
                    destino["celulas"] += _celulas(avaliacao, bloco, f"|s{d}")
        saida["sorteios"].append({
            "sorteio": d, "semente": sorteio["semente"], "canais_retidos": sorted(retidos),
            "blocos_sem_avaliacao": sem_avaliacao, "dobras_puladas": puladas, "contextos": contextos,
            "por_modelo": {nome: _avaliar_retidos(acumulado[nome], cfg) for nome in MODELOS_SERVIDOS},
            "baseline_inscritos": _baseline_inscritos_retidos(acumulado, inscritos_sorteio, cfg),
            "principal_nos_mesmos_canais": {
                nome: _avaliar_retidos(dict(zip(("linhas", "central", "celulas"),
                                                _subconjunto_do_principal(resultados_principal, nome, retidos,
                                                                blocos_avaliados))), cfg)
                for nome in MODELOS_SERVIDOS}})
    saida["agregado_dos_sorteios"] = {nome: _avaliar_retidos(agregado[nome], cfg) for nome in MODELOS_SERVIDOS}
    saida["baseline_inscritos_agregado"] = _baseline_inscritos_retidos(agregado, inscritos_agregado, cfg)
    return saida


class _AcumuladoInscritos:
    """Ajuste e previsão do baseline 4 por dobra do braço de canais não vistos, na ordem das linhas acumuladas."""

    def __init__(self) -> None:
        self.previsto: list[np.ndarray] = []
        self.ajustes: dict[str, bi.AjusteInscritos] = {}
        self.cortadas: dict[str, int] = {}

    def acrescentar(self, rotulo: str, ajuste: bi.AjusteInscritos, previsto: np.ndarray, cortadas: int) -> None:
        self.ajustes[rotulo], self.cortadas[rotulo] = ajuste, cortadas
        self.previsto.append(previsto)


def _baseline_inscritos_retidos(acumulado: Mapping[str, Mapping], inscritos: _AcumuladoInscritos,
                                cfg: Configuracao) -> dict:
    if cfg.inscritos is None:
        return bi.bloco_nao_avaliado("tabela de inscritos não fornecida à avaliação (`inscritos` ausente na "
                                     "configuração)")
    linhas = acumulado[MODELOS_SERVIDOS[0]]["linhas"]
    if not linhas:
        return bi.bloco_nao_avaliado("nenhuma linha de canal retido avaliada")
    ids = {}
    for nome in MODELOS_SERVIDOS:
        pop, _ = dv.montar_populacao(linhas, acumulado[nome]["central"], celulas=acumulado[nome]["celulas"])
        ids[nome] = bi.hash_dos_ids(pop.ids)
    return bi.montar_bloco(
        inscritos.ajustes, inscritos.cortadas, linhas, np.concatenate(inscritos.previsto),
        {nome: acumulado[nome]["central"] for nome in MODELOS_SERVIDOS},
        celulas=acumulado[MODELOS_SERVIDOS[0]]["celulas"], ids_do_principal=ids,
        n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)


def _avaliar_retidos(acumulado: Mapping, cfg: Configuracao) -> dict:
    if not acumulado["linhas"]:
        return {"n_videos": 0, "aviso": "nenhuma linha avaliada para estes canais"}
    pop, exclusoes = dv.montar_populacao(acumulado["linhas"], acumulado["central"], celulas=acumulado["celulas"])
    unicos = len({l["video_id"] for l in acumulado["linhas"]})
    return {"exclusoes": exclusoes, "n_videos_unicos": unicos} | dv.avaliar(
        pop, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)


# --- sensibilidade (exploratória) ---------------------------------------------------------------------------------------


def braco_sensibilidade(fornecedor: Fornecedor, cfg: Configuracao, base: Mapping,
                        resultados_principal: Sequence[Mapping], consulta: ConsultaAberta | None) -> dict:
    principal = {l["video_id"]: (l, c, cel) for r in resultados_principal
                 for l, c, cel in zip(r["dobra"].avaliacao, r["previsoes"]["N4"]["central_log"],
                                      _celulas(r["dobra"].avaliacao, r["bloco"]), strict=True)}
    saida = {"natureza": "exploratória; nunca correção; não influencia seleção, limiar nem promoção"}
    spec = especs_servidos(base)["N4"]
    for rotulo in ("S1", "S2"):
        linhas, central, celulas, puladas = [], [], [], []
        for k, bloco in enumerate(cfg.blocos_avaliados, start=1):
            dobra, _ = _rodar(fornecedor, espec_sensibilidade(cfg, rotulo, k), consulta)
            if not _treino_suficiente(dobra):
                puladas.append({"bloco": bloco, "n_treino": len(dobra.treino), "motivo": "treino_insuficiente"})
                continue
            previsto = db.prever_central(db.treinar_q50(dobra.treino, spec, cfg.parametros), dobra.avaliacao)
            linhas += dobra.avaliacao
            central += list(previsto["central_log"])
            celulas += _celulas(dobra.avaliacao, bloco)
        if not linhas:
            saida[rotulo] = {"dobras_puladas": puladas, "aviso": "nenhuma dobra com treino suficiente: braço sem resultado"}
            continue
        pop, exclusoes = dv.montar_populacao(linhas, central, celulas=celulas)
        saida[rotulo] = {"exclusoes": exclusoes, "dobras_puladas": puladas,
                         "proprio": dv.avaliar(pop, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)}
        comuns = [(l, c, cel) for l, c, cel in zip(linhas, central, celulas, strict=True) if l["video_id"] in principal]
        multi, _ = dv.montar_populacao_multi(
            [principal[l["video_id"]][0] for l, _, _ in comuns],
            {"principal": [principal[l["video_id"]][1] for l, _, _ in comuns], rotulo: [c for _, c, _ in comuns]},
            celulas=[cel for _, _, cel in comuns], com_baselines=True)
        saida[rotulo]["contra_o_principal"] = dv.avaliar(multi, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente,
                                                         contrastes=[(rotulo, "principal")])
    return saida


# --- ablação ----------------------------------------------------------------------------------------------------------------


def braco_ablacao(fornecedor: Fornecedor, cfg: Configuracao, base: Mapping, consulta: ConsultaAberta | None,
                  hashes_congelados: Mapping[str, str] | None = None) -> dict:
    espec = espec_ablacao(cfg)
    dobra, manifesto = _rodar(fornecedor, espec, consulta)
    _exigir_treino(dobra, espec)
    previsoes = {nivel: list(db.prever_central(db.treinar_q50(dobra.treino, spec, cfg.parametros),
                                               dobra.avaliacao)["central_log"])
                 for nivel, spec in especs_ablacao(base, hashes_congelados).items()}
    pop, exclusoes = dv.montar_populacao_multi(dobra.avaliacao, previsoes, com_baselines=True)
    contrastes = [*CONTRASTES_ABLACAO, *((nivel, BASE_DO_ACASO) for nivel in NIVEIS_ABLACAO)]
    avaliacao = dv.avaliar(pop, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente, contrastes=contrastes)
    leituras = {}
    for chave, contraste in avaliacao["contrastes"].items():
        ic = contraste["spearman_intra_canal"]["bootstrap"]["ic95"]
        leituras[chave] = gn.leitura_do_ganho(ic) | {
            "menor_efeito_detectavel_aprox": None if ic is None else (ic[1] - ic[0]) / 2}
    elegiveis = avaliacao["spearman_intra_canal"]["E4"]
    return {"n_treino": len(dobra.treino), "n_avaliacao": len(dobra.avaliacao), "exclusoes": exclusoes,
            "mesmas_linhas_para_todos_os_niveis": True, "niveis": list(NIVEIS_ABLACAO),
            "sha256_dataset": manifesto.get("sha256_dataset"),
            "canais_elegiveis_spearman": elegiveis["elegiveis"],
            "canais_excluidos": {"por_tamanho": elegiveis["excluidos_por_tamanho"],
                                 "sem_variacao_no_real": elegiveis["excluidos_sem_variacao_no_real"]},
            "avaliacao": avaliacao, "leitura_do_ganho_spearman_intra_canal": leituras,
            "nao_avalia": "a Camada 1 local do TAPI §6.3 (desvio declarado, não resolvido)"}


# --- execução -------------------------------------------------------------------------------------------------------------------


def gravador_de_bracos(pasta: Path) -> Callable[[str, Mapping], None]:
    """`salvar` que grava cada braço em `<pasta>/<nome>.json`, sem sobrescrever (modo exclusivo)."""
    pasta = Path(pasta)

    def salvar(nome: str, dado: Mapping) -> None:
        pasta.mkdir(parents=True, exist_ok=True)
        with open(pasta / f"{nome}.json", "x", encoding="utf-8") as saida:
            json.dump(rl.sanear(dado), saida, ensure_ascii=False, indent=1)

    return salvar


def _braco(nome: str, funcao: Callable[[], object], salvar: Callable[[str, Mapping], None] | None,
           gravavel: Callable[[object], Mapping] = lambda r: r) -> object:
    """Roda um braço, grava o resultado ao terminar e, se falhar, grava o erro antes de propagar."""
    try:
        resultado = funcao()
    except Exception as erro:
        if salvar:
            salvar(f"{nome}.erro", {"braco": nome, "tipo": type(erro).__name__, "mensagem": str(erro)})
        raise
    if salvar:
        salvar(nome, gravavel(resultado))
    return resultado


def executar(fornecedor: Fornecedor, cfg: Configuracao, *, consulta: ConsultaAberta | None,
             hashes_ablacao: Mapping[str, str] | None = None,
             salvar: Callable[[str, Mapping], None] | None = None) -> dict:
    """Roda os quatro braços e devolve o relatório (JSON-serializável com `relatorio.sanear`).

    Com a consulta ao teste aberta, a configuração tem de ser a congelada e os hashes da ablação são obrigatórios.
    `salvar(nome, dado)` recebe cada braço assim que ele termina (e o erro, se falhar): a consulta é única e nada que
    já foi calculado pode se perder por falha de um braço posterior."""
    if consulta is not None:
        if not isinstance(consulta, ConsultaAberta):
            raise PortaoFechado("a consulta ao teste só vale como a prova devolvida por abrir_consulta_ao_teste")
        exigir_configuracao_congelada(cfg, consulta.definicoes)
        if hashes_ablacao is None:
            raise ValueError("com a consulta ao teste aberta, os hashes congelados da ablação (E1 a E4) são "
                             "obrigatórios")
    resultados, base, principal = _braco(
        "principal", lambda: _principal(fornecedor, cfg, consulta), salvar, gravavel=lambda r: r[2])
    erros: list[Exception] = []

    def independente(nome: str, funcao: Callable[[], Mapping]) -> Mapping:
        """Sensibilidade, canais não vistos e ablação só dependem do principal: a falha de um não derruba os outros."""
        try:
            return _braco(nome, funcao, salvar)
        except Exception as erro:
            erros.append(erro)
            return {"erro": {"tipo": type(erro).__name__, "mensagem": str(erro)}}

    relatorio = {
        "configuracao": {"blocos_avaliados": list(cfg.blocos_avaliados), "blocos_anteriores": list(cfg.blocos_anteriores),
                         "parametros": dict(cfg.parametros), "n_repeticoes": cfg.n_repeticoes, "semente": cfg.semente,
                         "coleta": cfg.coleta.isoformat(), "baseline_inscritos_fornecido": cfg.inscritos is not None},
        "principal": principal,
        "canais_nao_vistos": independente("canais_nao_vistos", lambda: braco_nao_vistos(
            fornecedor, cfg, base, resultados, consulta)),
        "sensibilidade": independente("sensibilidade", lambda: braco_sensibilidade(
            fornecedor, cfg, base, resultados, consulta)),
        "ablacao": independente("ablacao", lambda: braco_ablacao(fornecedor, cfg, base, consulta, hashes_ablacao)),
    }
    if erros:
        raise erros[0]  # todos os braços já foram gravados (ou tiveram o erro gravado); o primeiro erro sobe
    return relatorio


def _principal(fornecedor: Fornecedor, cfg: Configuracao, consulta: ConsultaAberta | None) -> tuple:
    resultados, base = braco_principal(fornecedor, cfg, consulta)
    return resultados, base, avaliar_principal(resultados, cfg)
