"""Experimento audiovisual local: oito configurações aprendidas (M0 a M7) e referências históricas (R0), por dobra."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.avaliacao import (
    experimentos as af,
    experimentos as cs,
    experimentos as dv,
    metricas as bs,
    metricas as fx,
    metricas as od,
    validacao as ad,
    validacao as ah,
)
from src.dados import historico as hs
from src.extracao import pipeline as col
from src.modelos import modalidades as rat, treino as fp


# ──── Experimento audiovisual local: oito configurações aprendidas (M0 a M7) e referências históricas (R0)
N_CONFIGURACOES_APRENDIDAS = len(af.APRENDIDAS)
MAX_AJUSTES = N_CONFIGURACOES_APRENDIDAS * ad.N_JANELAS * len(rat.ALFAS)  # 8 × 2 × 3 = 48
MIN_CANAIS_PARA_IC = 8
TOLERANCIA_DE_EMPATE_MAE = 0.005  # em log; diferenças menores não distinguem configurações elegíveis
DISPONIVEL = "concluida"
COL_DISPONIBILIDADE = "avl_disponibilidade"
BASELINES = tuple(dv.BASELINES)
CONTRASTES_PRIMARIOS = (("M1", "M0"), ("M2", "M0"), ("M3", "M0"))
CONTRASTES_SECUNDARIOS = (("M4", "M0"), ("M5", "M0"), ("M6", "M0"), ("M7", "M0"), ("M5", "M4"), ("M7", "M6"))


class ErroExperimento(ValueError):
    """População, dobra ou resultado incompatível com o protocolo."""


def contrastes_do_protocolo() -> list[tuple[str, str]]:
    """Primários, secundários e cada configuração contra as três referências históricas (descritivos)."""
    contra_referencias = [(c, b) for c in af.APRENDIDAS for b in BASELINES]
    return [*CONTRASTES_PRIMARIOS, *CONTRASTES_SECUNDARIOS, *contra_referencias]


# --- população elegível -------------------------------------------------------------------------------------------


def filtrar_elegiveis(linhas: Sequence[Mapping]) -> tuple[list[dict], dict]:
    """Linhas com extração local concluída e sem estado de recusa; as demais saem de TODAS as configurações."""
    elegiveis, motivos, por_id = [], {}, {}
    for linha in linhas:
        if linha.get(COL_DISPONIBILIDADE) != DISPONIVEL:
            motivo = f"disponibilidade:{linha.get(COL_DISPONIBILIDADE)}"
        elif col.motivos_de_recusa(linha):
            motivo = "estado_de_recusa:" + ",".join(col.motivos_de_recusa(linha))
        else:
            elegiveis.append(dict(linha))
            continue
        motivos[motivo] = motivos.get(motivo, 0) + 1
        por_id[linha["video_id"]] = motivo
    return elegiveis, {"n_entrada": len(linhas), "n_elegiveis": len(elegiveis), "excluidas_por_motivo": motivos,
                       "motivo_por_id": por_id}


# --- dobras prontas -----------------------------------------------------------------------------------------------


@dataclass
class DobraPronta:
    indice: int
    treino: list[dict]
    avaliacao: list[dict]
    corte_de_treino: datetime
    exclusoes: dict
    auditoria: dict
    comparacao_com_originais: dict | None = None
    extras: dict = field(default_factory=dict)


def _tem_alvo(linha: Mapping) -> bool:
    return linha.get("ref_log") is not None and linha.get("y_rel") is not None


def preparar_dobras(estrutura: ad.Estrutura, linhas_elegiveis: Sequence[Mapping], posts: Sequence[hs.Post],
                    canais: Mapping[str, hs.Canal], *, defasagem_dias: int = hs.DEFASAGEM_PADRAO,
                    comparar_originais: bool = True) -> list[DobraPronta]:
    """Recalcula o N1 de cada dobra, audita (falha fechada) e separa treino e avaliação sem referência ou alvo nulos."""
    por_id = {l["video_id"]: l for l in linhas_elegiveis}
    dobras = []
    for janela in estrutura.janelas:
        treino_ids = ad.ids_de_treino(estrutura, janela.indice, linhas_elegiveis)
        ids = [*treino_ids, *janela.ids_avaliacao]
        base = [por_id[v] for v in ids]
        ctx = ah.contexto_da_dobra(f"dobra_{janela.indice}", janela.corte_de_treino)
        novas = ah.recalcular_linhas(posts, canais, base, ctx, defasagem_dias=defasagem_dias)
        auditoria = ah.auditar_dobra(posts, canais, novas, ctx, defasagem_dias=defasagem_dias)
        if not auditoria["aprovado"]:
            raise ErroExperimento(f"auditoria do histórico reprovada na dobra {janela.indice}: "
                                  f"{ {k: len(v) for k, v in auditoria['violacoes'].items() if v} }")
        avaliacao_ids = set(janela.ids_avaliacao)
        treino, avaliacao, sem_treino, sem_avaliacao = [], [], [], []
        for linha in novas:
            eh_avaliacao = linha["video_id"] in avaliacao_ids
            if not _tem_alvo(linha):
                (sem_avaliacao if eh_avaliacao else sem_treino).append(linha["video_id"])
                continue
            (avaliacao if eh_avaliacao else treino).append(
                {**linha, "uso_no_contexto": "avaliacao" if eh_avaliacao else "treino", "dobra": janela.indice})
        comparacao = ah.comparar_com_originais(base, novas, avaliacao_ids) if comparar_originais else None
        dobras.append(DobraPronta(
            janela.indice, treino, avaliacao, janela.corte_de_treino,
            {"treino_sem_referencia_ou_alvo": sorted(sem_treino),
             "avaliacao_sem_referencia_ou_alvo": sorted(sem_avaliacao)}, auditoria, comparacao))
    return dobras


# --- ajuste e previsão --------------------------------------------------------------------------------------------


def _ausencias(linhas: Sequence[Mapping], colunas: Sequence[str]) -> dict[str, int]:
    return {c: sum(l.get(c) is None for l in linhas) for c in colunas}


def ajustar_configuracao(config_id: str, dobra: DobraPronta, total: Mapping, parametros: Mapping,
                         contador: rat.ContadorDeAjustes) -> dict:
    """q10, q50 e q90 da configuração na dobra: previsões da avaliação e diagnóstico do que foi aprendido."""
    espec = af.especificacao_da_configuracao(config_id, total)
    locais = af.colunas_locais_da_especificacao(espec)
    modelos = rat.treinar_quantis(dobra.treino, espec, locais, parametros, contador=contador,
                                  rotulo=f"{config_id}/dobra{dobra.indice}")
    previsao = rat.prever_com_faixa(modelos, dobra.avaliacao)
    pre = modelos["q50"].pipeline.named_steps["pre"]
    diagnostico = {"n_colunas_de_saida": len(pre.nomes_saida_), "colunas_locais_descartadas_por_constancia":
                   list(pre.colunas_descartadas_), "categorias_locais_permitidas": dict(pre.categorias_permitidas_),
                   "ausencias_locais_no_treino": _ausencias(dobra.treino, locais),
                   "ausencias_locais_na_avaliacao": _ausencias(dobra.avaliacao, locais)}
    return {"previsao": previsao, "diagnostico": diagnostico}


def _concatenar(partes: Sequence[Mapping]) -> dict:
    return {k: np.concatenate([np.asarray(p[k]) for p in partes]) for k in partes[0]}


def _contagens_das_dobras(dobras: Sequence[DobraPronta]) -> dict:
    avaliados = [l for d in dobras for l in d.avaliacao]
    treinados = {l["video_id"] for d in dobras for l in d.treino}
    por_dobra = []
    for d in dobras:
        canais_aval = {}
        for l in d.avaliacao:
            canais_aval[l["channel_id"]] = canais_aval.get(l["channel_id"], 0) + 1
        elegiveis = {c: n for c, n in canais_aval.items() if n >= od.MIN_VIDEOS_CANAL}
        por_dobra.append({
            "dobra": d.indice, "n_treino": len(d.treino), "n_avaliacao": len(d.avaliacao),
            "videos_unicos_treino": len({l["video_id"] for l in d.treino}),
            "canais_no_treino": len({l["channel_id"] for l in d.treino}),
            "canais_na_avaliacao": len(canais_aval), "canais_elegiveis_spearman": len(elegiveis),
            "celulas_elegiveis": sorted(f"{c}|d{d.indice}" for c in elegiveis),
            "corte_de_treino_utc": d.corte_de_treino.isoformat(), "exclusoes": d.exclusoes,
            "auditoria_historico": {k: v for k, v in d.auditoria.items() if k != "violacoes"},
            "comparacao_com_colunas_originais": d.comparacao_com_originais})
    return {"videos_unicos_avaliados": len({l["video_id"] for l in avaliados}),
            "videos_unicos_no_treino_de_alguma_dobra": len(treinados),
            "ids_avaliados_sha256": ad.sha256_ids([l["video_id"] for l in avaliados]), "por_dobra": por_dobra}


def _f1_por_configuracao(dobras: Sequence[DobraPronta], previsoes: Mapping[str, Sequence[Mapping]],
                         config_id: str) -> dict:
    partes = []
    reais_todos, previstos_todos = [], []
    for d, prev in zip(dobras, previsoes[config_id], strict=True):
        real = np.array([l["y_rel"] for l in d.avaliacao], dtype=float)
        previsto = np.asarray(prev["relativo"], dtype=float)
        partes.append((real, previsto, fx.tercis_do_treino([l["y_rel"] for l in d.treino])))
        reais_todos.append(real)
        previstos_todos.append(previsto)
    return {"f1_macro_3_faixas": fx.f1_macro_empilhado(partes), "acaso_f1": 1 / fx.N_FAIXAS,
            "r2_alvo_relativo": fx.r2(np.concatenate(reais_todos), np.concatenate(previstos_todos))}


def _faixa_por_configuracao(pop: dv.Populacao, linhas: Sequence[Mapping], prev: Mapping, n_repeticoes: int,
                            semente: int) -> dict:
    posicoes = list(pop.posicoes)
    alinhado = {k: np.asarray(v)[posicoes] for k, v in prev.items()}
    cobertura = dv.cobertura(pop, alinhado["inferior_log"], alinhado["superior_log"], n_repeticoes=n_repeticoes,
                             semente=semente)
    resumo = fp.avaliar_faixa([linhas[i] for i in posicoes], alinhado)
    por_setor = cs.cobertura_por_setor(pop, alinhado["inferior_log"], alinhado["superior_log"],
                                       n_repeticoes=n_repeticoes, semente=semente)
    return {"cobertura": cobertura, "cobertura_por_setor": por_setor, "faixa": resumo}


def _canais_unicos(pop: dv.Populacao) -> list[str]:
    return sorted(set(pop.canais.tolist()))


def _mae(pop: dv.Populacao, metodo: str, mascara: np.ndarray) -> float:
    return float(np.abs(pop.previsto[metodo][mascara] - pop.real_log[mascara]).mean())


def _diagnosticos_adicionais(pop: dv.Populacao, dobras_das_linhas: np.ndarray, concatenadas: Mapping[str, Mapping],
                             configuracoes: Sequence[str], contrastes: Sequence[tuple[str, str]]) -> dict:
    """MAE por dobra e por canal (macro), leave-one-channel-out, pares por dobra e largura × erro; sem novo ajuste."""
    metodos = list(pop.previsto)
    canais = _canais_unicos(pop)
    todas = np.ones(len(pop.ids), dtype=bool)
    por_dobra = {m: {int(k): _mae(pop, m, dobras_das_linhas == k) for k in sorted(set(dobras_das_linhas.tolist()))}
                 for m in metodos}
    por_canal = {m: {c: _mae(pop, m, pop.canais == c) for c in canais} for m in metodos}
    macro = {m: float(np.mean(list(v.values()))) for m, v in por_canal.items()}
    avaliavel = len(canais) >= MIN_CANAIS_PARA_IC
    loco = {}
    for primeiro, segundo in contrastes:
        ponto = _mae(pop, primeiro, todas) - _mae(pop, segundo, todas)
        sem = {c: _mae(pop, primeiro, pop.canais != c) - _mae(pop, segundo, pop.canais != c) for c in canais} \
            if avaliavel else {}
        loco[f"{primeiro}|{segundo}"] = {
            "avaliavel": avaliavel, "n_canais": len(canais), "diferenca_pontual": ponto, "diferencas_sem_cada_canal": sem,
            "minimo": min(sem.values()) if sem else None, "maximo": max(sem.values()) if sem else None,
            "inversao_de_sinal": bool(sem) and any(np.sign(v) != np.sign(ponto) for v in sem.values())}
    rotulos = pop.rotulos_celula()
    pares_por_dobra = {}
    for m in metodos:
        pares_por_dobra[m] = {}
        for k in sorted(set(dobras_das_linhas.tolist())):
            sel = dobras_das_linhas == k
            r = od.acuracia_par_a_par(rotulos[sel], pop.curtidas[sel], pop.previsto[m][sel])
            pares_por_dobra[m][int(k)] = r["distinguiveis"]["acuracia"]
    largura_x_erro = {}
    posicoes = list(pop.posicoes)
    for c in configuracoes:
        largura = (np.asarray(concatenadas[c]["superior_log"]) - np.asarray(concatenadas[c]["inferior_log"]))[posicoes]
        erro = np.abs(pop.previsto[c] - pop.real_log)
        corte = np.quantile(largura, 0.8)
        topo = largura >= corte
        largura_x_erro[c] = {
            "spearman_largura_erro_absoluto": od.correlacao_de_spearman(largura, erro),
            "erro_medio_quintil_mais_largo": float(erro[topo].mean()) if topo.any() else None,
            "erro_medio_demais": float(erro[~topo].mean()) if (~topo).any() else None,
            "leitura": "dispersão relativa da faixa contra o erro; não é confiança calibrada"}
    return {"mae_por_dobra": por_dobra, "mae_por_canal": por_canal, "mae_macro_por_canal": macro,
            "leave_one_channel_out": loco, "acuracia_par_a_par_por_dobra": pares_por_dobra,
            "largura_x_erro": largura_x_erro, "n_canais": len(canais), "ic_avaliavel": avaliavel}


def _sem_ic(avaliacao: dict) -> None:
    """Com menos de 8 canais não há IC: o bootstrap vira `nao_avaliavel` e os marginais somem."""
    nulo = {"ic95": None, "media": None, "mediana": None, "leitura": "nao_avaliavel",
            "motivo": f"menos de {MIN_CANAIS_PARA_IC} canais"}
    for contraste in avaliacao["contrastes"].values():
        for metrica in contraste.values():
            if isinstance(metrica, dict) and "bootstrap" in metrica:
                metrica["bootstrap"] = {**metrica["bootstrap"], **nulo}
    avaliacao["marginais_ic95"] = None


def _exigir_metricas_finitas(avaliacao: Mapping) -> None:
    for metodo, e in avaliacao["erros"].items():
        for chave in ("mae_log", "rmse_log"):
            if not np.isfinite(e[chave]):
                raise rat.ErroDeParada(f"métrica não finita: {chave} de {metodo}")


def avaliar_configuracoes(dobras: Sequence[DobraPronta], total: Mapping, *, configuracoes: Sequence[str],
                          contrastes: Sequence[tuple[str, str]], max_ajustes: int,
                          parametros: Mapping = rat.PARAMETROS, n_repeticoes: int = bs.N_REPETICOES,
                          semente: int = bs.SEMENTE, livro: Path | None = None) -> dict:
    """Ajusta as configurações em todas as dobras (mesmas linhas) e mede erro, ordenação, faixa e contrastes."""
    contador = rat.ContadorDeAjustes(max_ajustes, livro)
    previsoes: dict[str, list[dict]] = {c: [] for c in configuracoes}
    diagnosticos: dict[str, list[dict]] = {c: [] for c in configuracoes}
    for dobra in dobras:
        for config_id in configuracoes:
            resultado = ajustar_configuracao(config_id, dobra, total, parametros, contador)
            previsoes[config_id].append(resultado["previsao"])
            diagnosticos[config_id].append({"dobra": dobra.indice} | resultado["diagnostico"])
    linhas = [l for d in dobras for l in d.avaliacao]
    celulas = [f"{l['channel_id']}|d{d.indice}" for d in dobras for l in d.avaliacao]
    concatenadas = {c: _concatenar(previsoes[c]) for c in configuracoes}
    pop, exclusoes = dv.montar_populacao_multi(
        linhas, {c: concatenadas[c]["central_log"] for c in configuracoes}, celulas=celulas, com_baselines=True)
    if len(pop.ids) != len(linhas):
        raise ErroExperimento(f"as configurações não avaliam as mesmas linhas: {len(pop.ids)} de {len(linhas)} "
                              f"({exclusoes})")
    validos = [c for c in contrastes if c[0] in configuracoes and (c[1] in BASELINES or c[1] in configuracoes)]
    avaliacao = dv.avaliar(pop, n_repeticoes=n_repeticoes, semente=semente, contrastes=validos)
    _exigir_metricas_finitas(avaliacao)
    dobras_das_linhas = np.array([d.indice for d in dobras for _ in d.avaliacao])
    adicionais = _diagnosticos_adicionais(pop, dobras_das_linhas, concatenadas, configuracoes, validos)
    if not adicionais["ic_avaliavel"]:
        _sem_ic(avaliacao)
    por_configuracao = {}
    for c in configuracoes:
        por_configuracao[c] = {
            **_faixa_por_configuracao(pop, linhas, concatenadas[c], n_repeticoes, semente),
            **_f1_por_configuracao(dobras, previsoes, c), "diagnostico_por_dobra": diagnosticos[c],
            "previsoes_cortadas_em_zero": int(np.asarray(concatenadas[c]["cortada_em_zero"]).sum())}
    return {
        "populacao": _contagens_das_dobras(dobras) | {"exclusoes_na_montagem": exclusoes, "n_celulas": avaliacao["n_celulas"]},
        "ajustes": {"feitos": contador.feitos, "maximo": contador.maximo},
        "avaliacao": avaliacao, "por_configuracao": por_configuracao, "diagnosticos_adicionais": adicionais,
        "leituras_separadas": leituras_separadas(avaliacao, validos),
        "rotulo_dos_intervalos": "aproximados: poucos canais, percentil de reamostragem de canais; sem correção de "
                                 "multiplicidade",
        "definicao_das_referencias_r0": {"R0": list(BASELINES), "colunas": dict(af.REFERENCIAS_R0)},
        "parametros": dict(parametros),
    }


def avaliar(dobras: Sequence[DobraPronta], total: Mapping, *, parametros: Mapping = rat.PARAMETROS,
            n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE, max_ajustes: int = MAX_AJUSTES,
            configuracoes: Sequence[str] = af.APRENDIDAS, livro: Path | None = None,
            sonda: Mapping | None = None) -> dict:
    """Experimento de desenvolvimento: todas as configurações, todos os contrastes e a seleção do preditor servido."""
    if len(dobras) != ad.N_JANELAS:
        raise ErroExperimento(f"o protocolo tem {ad.N_JANELAS} dobras; recebi {len(dobras)}")
    nucleo = avaliar_configuracoes(dobras, total, configuracoes=configuracoes, contrastes=contrastes_do_protocolo(),
                                   max_ajustes=max_ajustes, parametros=parametros, n_repeticoes=n_repeticoes,
                                   semente=semente, livro=livro)
    return {"natureza": "exploratória; desenvolvimento; sem teste; sem correção de multiplicidade; não há desfecho "
                        "confirmatório"} | nucleo | {"selecao": selecionar_preditor_servido(nucleo, total, sonda=sonda)}


# --- leitura e seleção --------------------------------------------------------------------------------------------

_ERRO = ("mae_log", "rmse_log")
_ORDENACAO = ("acuracia_par_a_par_distinguiveis", "spearman_intra_canal")
TEXTO_SEM_ORDENACAO = "melhora de erro; ordenação não demonstrada"
TEXTO_COM_ORDENACAO = "melhora de erro e de ordenação demonstradas contra M0 (intervalos aproximados)"


def leituras_separadas(avaliacao: Mapping, contrastes: Sequence[tuple[str, str]]) -> dict:
    """Melhora de erro e de ordenação em blocos separados, com canais, vídeos e pares por contraste."""
    saida = {}
    for primeiro, segundo in contrastes:
        r = avaliacao["contrastes"][f"{primeiro}|{segundo}"]

        def ler(metrica: str, r=r) -> dict:
            b = r[metrica]["bootstrap"]
            return {"diferenca_pontual": r[metrica]["diferenca_pontual"], "ic95": b["ic95"], "leitura": b["leitura"]}

        saida[f"{primeiro}|{segundo}"] = {
            "n_canais": avaliacao["n_canais"], "n_videos": avaliacao["n_videos"],
            "n_pares_distinguiveis": avaliacao["pares"][primeiro]["distinguiveis"]["n"],
            "erro": {m: ler(m) for m in _ERRO}, "ordenacao": {m: ler(m) for m in _ORDENACAO}}
    return saida


def _favoravel(contraste: Mapping | None, metrica: str) -> bool:
    return bool(contraste) and contraste[metrica]["bootstrap"]["leitura"] == "favoravel_ao_modelo"


def _superou_em_erro(nucleo: Mapping, escolhida: str, referencia: str) -> bool:
    """IC inteiro favorável, mesmo sinal nas duas dobras e leave-one-channel-out sem inversão."""
    av, extra = nucleo["avaliacao"], nucleo["diagnosticos_adicionais"]
    if not extra["ic_avaliavel"] or not _favoravel(av["contrastes"].get(f"{escolhida}|{referencia}"), "mae_log"):
        return False
    por_dobra = extra["mae_por_dobra"]
    if not all(por_dobra[escolhida][k] < por_dobra[referencia][k] for k in por_dobra[escolhida]):
        return False
    loco = extra["leave_one_channel_out"][f"{escolhida}|{referencia}"]
    return loco["avaliavel"] and not loco["inversao_de_sinal"]


def _ordenacao_contra_m0(nucleo: Mapping, escolhida: str) -> dict:
    av, extra = nucleo["avaliacao"], nucleo["diagnosticos_adicionais"]
    r = av["contrastes"][f"{escolhida}|M0"]
    por_dobra = extra["acuracia_par_a_par_por_dobra"]
    pares = {k: {"escolhida": por_dobra[escolhida][k], "m0": por_dobra["M0"][k]} for k in por_dobra[escolhida]}
    mesmo_sinal = all((v["escolhida"] or 0) > (v["m0"] or 0) for v in pares.values())
    par, rho = r["acuracia_par_a_par_distinguiveis"], r["spearman_intra_canal"]
    demonstrada = (extra["ic_avaliavel"] and par["bootstrap"]["leitura"] == "favoravel_ao_modelo"
                   and rho["bootstrap"]["leitura"] == "favoravel_ao_modelo" and mesmo_sinal)
    return {"par_a_par_distinguivel": {"diferenca_pontual": par["diferenca_pontual"], "ic95": par["bootstrap"]["ic95"],
                                       "leitura": par["bootstrap"]["leitura"], "por_dobra": pares},
            "spearman_intra_canal": {"diferenca_pontual": rho["diferenca_pontual"], "ic95": rho["bootstrap"]["ic95"],
                                     "leitura": rho["bootstrap"]["leitura"]},
            "mesmo_sinal_nas_duas_dobras": mesmo_sinal, "ordenacao_demonstrada": bool(demonstrada)}


TEXTO_ATALHO_DE_CANAL = ("O ganho de erro sobre M0 pode refletir a identificação do canal pelos atributos locais, e não o "
                         "conteúdo do vídeo.")


def selecionar_preditor_servido(nucleo: Mapping, total: Mapping, *, sonda: Mapping | None = None, elegiveis: Sequence[str] = af.ELEGIVEIS_PARA_SERVIR,
                                empate: float = TOLERANCIA_DE_EMPATE_MAE) -> dict:
    """Regra congelada: menor MAE em log pooled entre as elegíveis; empate (≤ `empate`) vai à de menos famílias de
    extração local e, persistindo, à de menor índice (M1 antes de M2 antes de M3).

    Só desenvolvimento, uma passagem. A escolha sob empate é de engenharia sob incerteza, não estatisticamente
    sustentada. A configuração escolhida é servida mesmo sem superar M0 ou a média histórica; os booleanos abaixo
    obrigam o pacote a trazer o aviso correspondente."""
    av = nucleo["avaliacao"]
    presentes = [c for c in elegiveis if c in av["erros"]]
    if not presentes:
        raise ErroExperimento("nenhuma configuração elegível foi avaliada")
    mae = {c: av["erros"][c]["mae_log"] for c in presentes}
    melhor = min(mae.values())
    colunas = {c: max(d["n_colunas_de_saida"] for d in nucleo["por_configuracao"][c]["diagnostico_por_dobra"])
               for c in presentes}
    familias_locais = {c: sum(f in af.CONFIGURACOES[c].familias for f in (af.LOCAL_COMPACTA, af.LOCAL_EMBEDDINGS))
                       for c in presentes}
    empatadas = [c for c in presentes if mae[c] - melhor <= empate]
    escolhida = min(empatadas, key=lambda c: (familias_locais[c], list(elegiveis).index(c)))
    ordenacao = _ordenacao_contra_m0(nucleo, escolhida)

    def resumo(par: str) -> dict | None:
        r = av["contrastes"].get(par)
        return None if r is None else {"mae_log": r["mae_log"]["bootstrap"]["leitura"],
                                       "diferenca_mae_log": r["mae_log"]["diferenca_pontual"],
                                       "ic95_mae_log": r["mae_log"]["bootstrap"]["ic95"]}

    superou_m0 = _superou_em_erro(nucleo, escolhida, "M0")
    superou_media = _superou_em_erro(nucleo, escolhida, "media_aritmetica")
    return {
        "regra": "menor MAE em log entre as elegíveis; empate dentro da tolerância resolvido pela de menos famílias "
                 "de extração local e depois pela ordem M1, M2, M3; uma única passagem, só desenvolvimento",
        "elegiveis": list(presentes), "tolerancia_de_empate_mae_log": empate, "mae_log": mae,
        "mae_por_dobra": {c: nucleo["diagnosticos_adicionais"]["mae_por_dobra"][c] for c in presentes},
        "mae_macro_por_canal": {c: nucleo["diagnosticos_adicionais"]["mae_macro_por_canal"][c] for c in presentes},
        "colunas_de_saida": colunas, "familias_locais": familias_locais, "empatadas": empatadas,
        "escolhida": escolhida,
        "desempate": {"houve_empate": len(empatadas) > 1,
                      "natureza": "decisão de engenharia sob incerteza; não é sustentada estatisticamente"},
        "contra_m0": resumo(f"{escolhida}|M0"),
        "contra_referencias": {b: resumo(f"{escolhida}|{b}") for b in BASELINES},
        "criterio_de_superou": "IC inteiro favorável, mesmo sinal nas duas dobras e leave-one-channel-out sem inversão",
        "superou_m0_em_erro": superou_m0, "superou_media_historica_em_erro": superou_media,
        "ordenacao_vs_m0": ordenacao, "ordenacao_demonstrada": ordenacao["ordenacao_demonstrada"],
        "texto_obrigatorio": TEXTO_COM_ORDENACAO if ordenacao["ordenacao_demonstrada"] else TEXTO_SEM_ORDENACAO,
        "sem_gemini": af.sem_gemini(af.especificacao_da_configuracao(escolhida, total)),
        "sonda_de_canal": None if sonda is None else dict(sonda),
        "atalho_de_canal_sinalizado": bool(sonda and sonda.get("muito_acima_do_acaso")),
        "aviso_de_atalho_de_canal": TEXTO_ATALHO_DE_CANAL if sonda and sonda.get("muito_acima_do_acaso") else None,
    }
