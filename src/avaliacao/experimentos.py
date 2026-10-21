"""Módulo consolidado. Reúne:
- Modelo × baselines nas mesmas linhas de validação de desenvolvimento, com bootstrap pareado por canal.
- Diagnósticos descritivos por grupo (idade do rótulo, setor, canal, histórico, transcrição).
- Cobertura da faixa por setor, com IC por reamostragem de canais quando estimável (protocolo final §8, §14.9).
- Baseline 4 do TAPI (D-D1): regressão de log(1 + curtidas) sobre log(1 + inscritos atuais), só no relatório.
- Configurações do experimento audiovisual local e mapa de famílias de atributos."""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.avaliacao import metricas as bs, metricas as od
from src.avaliacao.metricas import resumo_erros
from src.dados import curadoria, dataset as ds, historico as hs
from src.extracao import pipeline as col
from src.modelos import treino as fa, treino as md


# ──── Modelo × baselines nas mesmas linhas de validação de desenvolvimento, com bootstrap pareado por cana
BASELINES = {"media_aritmetica": "pred_media_aritmetica", "mediana": "pred_mediana",
             "media_movel_exponencial": "pred_mme_log"}
METODOS = ("modelo", *BASELINES)
DEFINICOES = {
    "mae_rmse_mape": "agregados sobre os vídeos juntos; MAPE adaptado = média de |previsto − real| ÷ max(real, 1) em "
                     "curtidas brutas (razão; 4,41 equivale a 441%)",
    "spearman_global": "todos os vídeos, escala absoluta (ranks médios)",
    "spearman_intra_canal": f"canais com ≥ {od.MIN_VIDEOS_CANAL} vídeos e variação no real; mediana entre canais; "
                            "indefinido não vira zero",
    "contraste_spearman_intra_canal": "só canais em que o modelo e a base têm correlação definida",
    "par_a_par": f"pares do mesmo canal; distinguível = max ≥ {od.FATOR_DISTINGUIVEL} × min nas curtidas observadas "
                 "(zero contra positivo conta; zero contra zero não); empate previsto vale 0,5",
    "bootstrap": f"{bs.N_REPETICOES} reamostragens de canais com reposição, semente {bs.SEMENTE} (escolha registrada), "
                 "mesmos canais para todos os métodos; IC percentil 95%",
}
METRICAS_MENOR_E_MELHOR = {"mae_log": True, "rmse_log": True, "mape_adaptado": True,
                           "acuracia_par_a_par_distinguiveis": False, "acuracia_par_a_par_todos": False,
                           "spearman_global": False}
_CAMPOS_SOMA = ("abs", "sq", "ape", "cred_dist", "n_dist", "cred_todos", "n_todos")


@dataclass(frozen=True)
class Populacao:
    ids: tuple[str, ...]
    canais: np.ndarray
    setores: np.ndarray
    real_log: np.ndarray
    curtidas: np.ndarray
    previsto: dict[str, np.ndarray]
    posicoes: tuple[int, ...] = ()  # índice de cada linha mantida na lista de entrada (para alinhar rótulos de grupo)
    celulas: np.ndarray | None = None  # célula canal × sub-bloco de cada linha; None = a célula é o canal

    def rotulos_celula(self) -> np.ndarray:
        return self.canais if self.celulas is None else self.celulas


def _ausente(valor) -> bool:
    """Ausente = nulo ou não finito (NaN/inf); nenhum ausente é preenchido ou vira zero."""
    return valor is None or not np.isfinite(valor)


def montar_populacao_multi(linhas: Sequence[Mapping], previstos: Mapping[str, Sequence[float]], *,
                           celulas: Sequence[str] | None = None, com_baselines: bool = False
                           ) -> tuple[Populacao, dict]:
    """Linhas com real e a previsão de TODOS os métodos nomeados; as demais saem de todos, com o motivo.

    `com_baselines` exige e acrescenta as três bases do dataset (média aritmética, mediana, média móvel). Sem ele, uma
    base entra como método comum se o chamador a passar em `previstos`.
    """
    nomes = list(previstos)
    for nome, valores in previstos.items():
        if len(valores) != len(linhas):
            raise ValueError(f"previsão {nome!r} com {len(valores)} valores para {len(linhas)} linhas")
    if celulas is not None and len(celulas) != len(linhas):
        raise ValueError("celulas com tamanho diferente das linhas")
    exclusoes = {"sem_baseline": 0, "sem_previsao": 0, "sem_real": 0}
    posicoes = []
    for posicao, linha in enumerate(linhas):
        if _ausente(linha.get("y_log")):
            exclusoes["sem_real"] += 1
        elif any(_ausente(previstos[nome][posicao]) for nome in nomes):
            exclusoes["sem_previsao"] += 1
        elif com_baselines and any(_ausente(linha.get(coluna)) for coluna in BASELINES.values()):
            exclusoes["sem_baseline"] += 1
        else:
            posicoes.append(posicao)
    mantidas = [linhas[p] for p in posicoes]
    real = np.array([l["y_log"] for l in mantidas], dtype=float)
    previsto = {nome: np.array([previstos[nome][p] for p in posicoes], dtype=float) for nome in nomes}
    if com_baselines:
        previsto |= {nome: np.array([l[coluna] for l in mantidas], dtype=float) for nome, coluna in BASELINES.items()}
    populacao = Populacao(
        ids=tuple(l["video_id"] for l in mantidas), canais=np.array([str(l["channel_id"]) for l in mantidas]),
        setores=np.array([str(l["setor"]) for l in mantidas]), real_log=real,
        curtidas=np.array([round(np.expm1(v)) for v in real], dtype=float), previsto=previsto,
        posicoes=tuple(posicoes),
        celulas=None if celulas is None else np.array([str(celulas[p]) for p in posicoes]))
    return populacao, exclusoes


def montar_populacao(linhas: Sequence[Mapping], previsto_modelo_log: Sequence[float], *,
                     celulas: Sequence[str] | None = None) -> tuple[Populacao, dict]:
    """Linhas com real, previsão do modelo e as três bases; as demais saem de todos os métodos, com o motivo."""
    populacao, exclusoes = montar_populacao_multi(linhas, {"modelo": previsto_modelo_log}, celulas=celulas,
                                                  com_baselines=True)
    return populacao, {"sem_baseline": exclusoes["sem_baseline"],
                       "sem_previsao_do_modelo": exclusoes["sem_previsao"], "sem_real": exclusoes["sem_real"]}


@dataclass(frozen=True)
class _PorCanal:
    nomes: list[str]
    n: np.ndarray
    soma: dict[str, dict[str, np.ndarray]]
    rho: dict[str, np.ndarray]  # um valor por CÉLULA (igual a um por canal quando não há células)
    celulas: list[str]
    canal_da_celula: np.ndarray  # índice, em `nomes`, do canal de cada célula


def _por_canal(pop: Populacao) -> _PorCanal:
    grupos = od.agrupar_por_canal(pop.canais)
    nomes = list(grupos)
    indice_canal = {c: k for k, c in enumerate(nomes)}
    rotulos = pop.rotulos_celula()
    grupos_celula = od.agrupar_por_canal(rotulos)
    celulas = list(grupos_celula)
    canal_da_celula = np.zeros(len(celulas), dtype=int)
    for j, celula in enumerate(celulas):
        donos = {str(pop.canais[i]) for i in grupos_celula[celula]}
        if len(donos) != 1:
            raise ValueError(f"a célula {celula!r} reúne vários canais: {sorted(donos)}")
        canal_da_celula[j] = indice_canal[donos.pop()]
    soma: dict[str, dict[str, np.ndarray]] = {}
    rho: dict[str, np.ndarray] = {}
    for metodo, previsto in pop.previsto.items():
        correlacao = od.spearman_por_canal(rotulos, pop.curtidas, previsto)["por_canal"]
        rho[metodo] = np.array([np.nan if correlacao.get(c) is None else correlacao[c] for c in celulas])
        campos = {k: np.zeros(len(nomes)) for k in _CAMPOS_SOMA}
        for k, canal in enumerate(nomes):
            i = grupos[canal]
            erro = previsto[i] - pop.real_log[i]
            campos["abs"][k], campos["sq"][k] = np.abs(erro).sum(), (erro**2).sum()
            campos["ape"][k] = (np.abs(np.expm1(previsto[i]) - pop.curtidas[i]) / np.maximum(pop.curtidas[i], 1)).sum()
        for j, celula in enumerate(celulas):  # pares só dentro da célula; a soma é do canal dono da célula
            i, k = grupos_celula[celula], canal_da_celula[j]
            par = od.contribuicao_de_pares(pop.curtidas[i], previsto[i])
            campos["cred_dist"][k] += par["credito_distinguiveis"]
            campos["n_dist"][k] += par["n_distinguiveis"]
            campos["cred_todos"][k] += par["credito_todos"]
            campos["n_todos"][k] += par["n_todos"]
        soma[metodo] = campos
    return _PorCanal(nomes, np.array([len(grupos[c]) for c in nomes], dtype=float), soma, rho, celulas,
                     canal_da_celula)


def _razao(contagens: np.ndarray, numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    num, den = contagens @ numerador, contagens @ denominador
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _mediana_sorteada(valores: np.ndarray, contagens: np.ndarray) -> float:
    sorteados = np.repeat(valores, contagens)
    sorteados = sorteados[np.isfinite(sorteados)]
    return float(np.median(sorteados)) if len(sorteados) else float("nan")


def _spearman_global(pop: Populacao, metodo: str, linhas: np.ndarray) -> float:
    valor = od.correlacao_de_spearman(pop.curtidas[linhas], pop.previsto[metodo][linhas])
    return float("nan") if valor is None else valor


def _linhas_sorteadas(grupos: Mapping[str, list[int]], nomes: list[str], contagens: np.ndarray) -> np.ndarray:
    partes = [np.tile(grupos[c], int(k)) for c, k in zip(nomes, contagens, strict=True) if k > 0]
    return np.concatenate(partes).astype(int)


def _serie(contagens: np.ndarray, por_canal: _PorCanal, metodo: str, metrica: str) -> np.ndarray:
    s = por_canal.soma[metodo]
    if metrica == "mae_log":
        return _razao(contagens, s["abs"], por_canal.n)
    if metrica == "rmse_log":
        return np.sqrt(_razao(contagens, s["sq"], por_canal.n))
    if metrica == "mape_adaptado":
        return _razao(contagens, s["ape"], por_canal.n)
    if metrica == "acuracia_par_a_par_distinguiveis":
        return _razao(contagens, s["cred_dist"], s["n_dist"])
    if metrica == "acuracia_par_a_par_todos":
        return _razao(contagens, s["cred_todos"], s["n_todos"])
    raise ValueError(f"métrica desconhecida: {metrica}")


def _erros(pop: Populacao, metodo: str) -> dict:
    previsto = pop.previsto[metodo]
    erro = previsto - pop.real_log
    referencia = resumo_erros([int(c) for c in pop.curtidas], [float(p) for p in previsto])
    mae = float(np.abs(erro).mean())
    return {"n": int(len(erro)), "mae_log": mae, "rmse_log": float(np.sqrt((erro**2).mean())),
            "mape_adaptado_razao": referencia["mape_adaptado"],
            "mape_adaptado_percentual": 100.0 * referencia["mape_adaptado"], "n_reais_zero": referencia["n_reais_zero"],
            "vies_medio_log": float(erro.mean()), "diferenca_para_resumo_erros_mae": abs(mae - referencia["mae_log"])}


def _pares_do_contraste(pop: Populacao, contrastes: Sequence[tuple[str, str]] | None
                        ) -> list[tuple[str, str, str]]:
    """(primeiro, segundo, chave de saída). Padrão: `modelo − base`, com o nome da base como chave."""
    if contrastes is None:
        return [("modelo", base, base) for base in BASELINES]
    for primeiro, segundo in contrastes:
        for nome in (primeiro, segundo):
            if nome not in pop.previsto:
                raise ValueError(f"método desconhecido no contraste: {nome!r}; disponíveis: {sorted(pop.previsto)}")
    return [(primeiro, segundo, f"{primeiro}|{segundo}") for primeiro, segundo in contrastes]


def avaliar(pop: Populacao, *, n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE,
            contrastes: Sequence[tuple[str, str]] | None = None) -> dict:
    """Erros, ordenação e contrastes `primeiro − segundo` (ponto e bootstrap pareado por canal) nas mesmas linhas.

    Sem `contrastes`, o padrão é `modelo − base` para as três bases aprovadas (D-D1).
    """
    if not pop.ids:
        raise ValueError("população vazia: nenhuma linha com real e com a previsão de todos os métodos")
    metodos = tuple(pop.previsto)
    pares_contraste = _pares_do_contraste(pop, contrastes)
    grupos = od.agrupar_por_canal(pop.canais)
    por_canal = _por_canal(pop)
    n_canais = len(por_canal.nomes)
    contagens = bs.sorteio_de_canais(n_canais, n_repeticoes=n_repeticoes, semente=semente)
    todos = np.ones((1, n_canais))
    sorteios = [_linhas_sorteadas(grupos, por_canal.nomes, c) for c in contagens]
    tudo = np.arange(len(pop.ids))
    rep_global = {m: np.array([_spearman_global(pop, m, linhas) for linhas in sorteios]) for m in metodos}
    ponto_global = {m: _spearman_global(pop, m, tudo) for m in metodos}
    metricas_somadas = [k for k in METRICAS_MENOR_E_MELHOR if k != "spearman_global"]
    series = {(m, k): _serie(contagens, por_canal, m, k) for m in metodos for k in metricas_somadas}
    pontos = {(m, k): float(_serie(todos, por_canal, m, k)[0]) for m, k in series}
    rotulos = pop.rotulos_celula()
    pares = {m: od.acuracia_par_a_par(rotulos, pop.curtidas, pop.previsto[m]) for m in metodos}
    for resumo in pares.values():
        resumo.pop("por_canal")
    resultado_contrastes = {}
    for primeiro, segundo, chave in pares_contraste:
        resultado = {}
        for metrica, menor in METRICAS_MENOR_E_MELHOR.items():
            if metrica == "spearman_global":
                rep_m, rep_b = rep_global[primeiro], rep_global[segundo]
                ponto_m, ponto_b = ponto_global[primeiro], ponto_global[segundo]
            else:
                rep_m, rep_b = series[(primeiro, metrica)], series[(segundo, metrica)]
                ponto_m, ponto_b = pontos[(primeiro, metrica)], pontos[(segundo, metrica)]
            resultado[metrica] = {"modelo": ponto_m, "base": ponto_b, "diferenca_pontual": ponto_m - ponto_b,
                                  "bootstrap": bs.resumir_diferenca(rep_m, rep_b, menor_e_melhor=menor)}
        resultado["spearman_intra_canal"] = _contraste_spearman_intra(por_canal, primeiro, segundo, contagens)
        resultado_contrastes[chave] = resultado
    marginais = {m: _marginais(m, series, rep_global, por_canal, contagens, metricas_somadas) for m in metodos}
    return {
        "marginais_ic95": marginais,
        "n_videos": len(pop.ids), "n_canais": n_canais, "n_celulas": len(por_canal.celulas),
        "videos_por_canal": {c: len(i) for c, i in grupos.items()},
        "erros": {m: _erros(pop, m) for m in metodos},
        "spearman_global": {m: (None if np.isnan(v) else v) for m, v in ponto_global.items()},
        "spearman_intra_canal": {m: od.spearman_por_canal(rotulos, pop.curtidas, pop.previsto[m]) for m in metodos},
        "pares": pares, "contrastes": resultado_contrastes,
        "bootstrap": {"semente": semente, "n_repeticoes": n_repeticoes, "n_canais_sorteados_por_repeticao": n_canais,
                      "metodo": "percentil 2,5–97,5 de modelo − base; mesmos canais em todos os métodos"},
        "definicoes": DEFINICOES,
    }


def _ic(valores: np.ndarray) -> list[float] | None:
    finitos = np.asarray(valores, dtype=float)
    finitos = finitos[np.isfinite(finitos)]
    return bs.percentis(finitos) if len(finitos) else None


def _marginais(metodo: str, series: dict, rep_global: dict, por_canal: _PorCanal, contagens: np.ndarray,
               metricas_somadas: Sequence[str]) -> dict:
    """IC95% de cada método sozinho (mesmos sorteios de canais), para ler cada número contra o seu próprio acaso."""
    saida = {k: _ic(series[(metodo, k)]) for k in metricas_somadas}
    saida["spearman_global"] = _ic(rep_global[metodo])
    peso = por_canal.canal_da_celula  # cada célula entra com quantas vezes o seu canal foi sorteado
    saida["spearman_intra_canal"] = _ic(np.array([_mediana_sorteada(por_canal.rho[metodo], c[peso]) for c in contagens]))
    return saida


def _contraste_spearman_intra(por_canal: _PorCanal, primeiro: str, segundo: str, contagens: np.ndarray) -> dict:
    comum = np.isfinite(por_canal.rho[primeiro]) & np.isfinite(por_canal.rho[segundo])
    rho_m = np.where(comum, por_canal.rho[primeiro], np.nan)
    rho_b = np.where(comum, por_canal.rho[segundo], np.nan)
    unitaria = np.ones(len(comum), dtype=int)
    ponto_m, ponto_b = _mediana_sorteada(rho_m, unitaria), _mediana_sorteada(rho_b, unitaria)
    sem_populacao = not comum.any()
    peso = por_canal.canal_da_celula
    canais_comuns = sorted({por_canal.nomes[k] for k in por_canal.canal_da_celula[comum]})
    return {
        "n_canais_comuns": len(canais_comuns), "canais_comuns": canais_comuns,
        "n_celulas_comuns": int(comum.sum()), "celulas_comuns": [c for c, ok in zip(por_canal.celulas, comum) if ok],
        "modelo": None if sem_populacao else ponto_m, "base": None if sem_populacao else ponto_b,
        "diferenca_pontual": None if sem_populacao else ponto_m - ponto_b,
        "bootstrap": bs.resumir_diferenca(np.array([_mediana_sorteada(rho_m, c[peso]) for c in contagens]),
                                          np.array([_mediana_sorteada(rho_b, c[peso]) for c in contagens]),
                                          menor_e_melhor=False),
    }


def cobertura(pop: Populacao, inferior: np.ndarray, superior: np.ndarray, *, n_repeticoes: int = bs.N_REPETICOES,
              semente: int = bs.SEMENTE) -> dict:
    """Cobertura da faixa nas linhas da população, com IC por reamostragem de canais (não é Wilson)."""
    dentro = ((pop.real_log >= np.asarray(inferior)) & (pop.real_log <= np.asarray(superior))).astype(float)
    grupos = od.agrupar_por_canal(pop.canais)
    nomes = list(grupos)
    soma = np.array([dentro[grupos[c]].sum() for c in nomes])
    n = np.array([len(grupos[c]) for c in nomes], dtype=float)
    reps = _razao(bs.sorteio_de_canais(len(nomes), n_repeticoes=n_repeticoes, semente=semente), soma, n)
    return {
        "n": int(len(dentro)), "n_dentro": int(dentro.sum()), "cobertura": float(dentro.mean()), "n_canais": len(nomes),
        "ic95_bootstrap_por_canal": bs.percentis(reps[np.isfinite(reps)]), "n_repeticoes": n_repeticoes,
        "metodo_do_intervalo": "bootstrap por canal (percentil 2,5–97,5, reposição); não usa Wilson, que ignora o "
                               "agrupamento por canal",
    }


# ──── Diagnósticos descritivos por grupo (idade do rótulo, setor, canal, histórico, transcrição).
FORA_DAS_FAIXAS = "fora das faixas"


def idade_em_dias(publicado_em: datetime, coleta: datetime) -> int:
    """Dias inteiros entre a publicação e a coleta dos rótulos (curtidas acumuladas lidas na coleta)."""
    if publicado_em.tzinfo is None or coleta.tzinfo is None:
        raise ValueError("a idade do rótulo exige datas com fuso horário (UTC); datas ingênuas mudam com a máquina")
    return (coleta - publicado_em).days


def rotular_faixas(valores: Sequence[float], bordas: Sequence[int], unidade: str = "dias") -> list[str]:
    """Rótulo `AAA–BBB <unidade>` pelo intervalo semiaberto [borda_i, borda_i+1); fora das bordas, `fora das faixas`."""
    rotulos = []
    for valor in valores:
        achado = FORA_DAS_FAIXAS
        for inferior, superior in zip(bordas, bordas[1:], strict=False):
            if inferior <= valor < superior:
                achado = f"{inferior:03d}–{superior - 1:03d} {unidade}"
                break
        rotulos.append(achado)
    return rotulos


def _erros_diagnosticos(pop: Populacao, indices: np.ndarray) -> dict:
    saida = {}
    for metodo in METODOS:
        erro = pop.previsto[metodo][indices] - pop.real_log[indices]
        saida[metodo] = {"mae_log": float(np.abs(erro).mean()), "vies_medio_log": float(erro.mean())}
    return saida


def _ordenacao(pop: Populacao, indices: np.ndarray) -> dict:
    saida = {}
    celulas = pop.rotulos_celula()[indices]  # pares só dentro da célula (o canal, se a população não tem células)
    for metodo in METODOS:
        pares = od.acuracia_par_a_par(celulas, pop.curtidas[indices], pop.previsto[metodo][indices])
        saida[metodo] = {
            "spearman_no_grupo": od.correlacao_de_spearman(pop.curtidas[indices], pop.previsto[metodo][indices]),
            "par_a_par_distinguiveis": pares["distinguiveis"]["acuracia"],
            "n_pares_distinguiveis": pares["distinguiveis"]["n"],
        }
    return saida


def por_grupo(pop: Populacao, rotulos: Sequence[str], *, inferior: np.ndarray | None = None,
              superior: np.ndarray | None = None) -> dict:
    """Erro, viés, ordenação e cobertura (se há faixa) por rótulo, sempre com n e número de canais."""
    rotulos = np.asarray(rotulos)
    if len(rotulos) != len(pop.ids):
        raise ValueError("um rótulo por vídeo da população")
    saida = {}
    for rotulo in sorted(set(rotulos.tolist())):
        indices = np.flatnonzero(rotulos == rotulo)
        grupo = {"n": int(len(indices)), "n_canais": int(len(set(pop.canais[indices]))),
                 "amostra_pequena": bool(len(indices) < od.MIN_VIDEOS_CANAL),
                 "erros": _erros_diagnosticos(pop, indices), "ordenacao": _ordenacao(pop, indices), "cobertura": None}
        if inferior is not None and superior is not None:
            dentro = (pop.real_log[indices] >= np.asarray(inferior)[indices]) & (
                pop.real_log[indices] <= np.asarray(superior)[indices])
            grupo["cobertura"] = {"n": int(len(indices)), "n_dentro": int(dentro.sum()),
                                  "cobertura": float(dentro.mean())}
        saida[rotulo] = grupo
    return saida


# ──── Cobertura da faixa por setor, com IC por reamostragem de canais quando estimável (protocolo final §8
MIN_CANAIS_IC = 3
CANAIS_IC_CONFIAVEL = 8
MIN_LINHAS_IC = 20


def _estado_do_ic(n_canais: int, n: int) -> tuple[str, list[str]]:
    motivos = []
    if n_canais < MIN_CANAIS_IC:
        motivos.append(f"apenas {n_canais} canais no setor (mínimo {MIN_CANAIS_IC} para reamostrar canais)")
    if n < MIN_LINHAS_IC:
        motivos.append(f"apenas {n} linhas no setor (mínimo {MIN_LINHAS_IC})")
    if motivos:
        return "indefinido", motivos
    return ("estimado_aproximado" if n_canais < CANAIS_IC_CONFIAVEL else "estimado"), []


def _ic_por_canal(dentro: np.ndarray, canais: np.ndarray, n_repeticoes: int, semente: int) -> list[float]:
    grupos = od.agrupar_por_canal(canais)
    nomes = list(grupos)
    soma = np.array([dentro[grupos[c]].sum() for c in nomes])
    n = np.array([len(grupos[c]) for c in nomes], dtype=float)
    sorteio = bs.sorteio_de_canais(len(nomes), n_repeticoes=n_repeticoes, semente=semente)
    return bs.percentis((sorteio @ soma) / (sorteio @ n))


def cobertura_por_setor(pop: Populacao, inferior: np.ndarray, superior: np.ndarray, *,
                        n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE) -> dict:
    """Por setor: n, n dentro, cobertura, n de canais, IC95 por canal (ou o motivo de não existir)."""
    dentro = ((pop.real_log >= np.asarray(inferior)) & (pop.real_log <= np.asarray(superior))).astype(float)
    saida = {}
    for setor in sorted(set(pop.setores.tolist())):
        mascara = pop.setores == setor
        canais = pop.canais[mascara]
        n, n_canais = int(mascara.sum()), len(set(canais.tolist()))
        estado, motivos = _estado_do_ic(n_canais, n)
        definido = estado != "indefinido"
        saida[setor] = {
            "n": n, "n_dentro": int(dentro[mascara].sum()), "cobertura": float(dentro[mascara].mean()),
            "n_canais": n_canais, "ic_estado": estado,
            "ic95_bootstrap_por_canal": _ic_por_canal(dentro[mascara], canais, n_repeticoes, semente)
            if definido else None,
            "motivo_ic_indefinido": None if definido else "; ".join(motivos),
            "amostra_insuficiente": not definido,
            "aviso_ic": "IC aproximado: poucos canais no setor; o percentil de poucos grupos tende a ser estreito demais"
            if estado == "estimado_aproximado" else None,
            "n_repeticoes": n_repeticoes,
        }
    return saida


# ──── Baseline 4 do TAPI (D-D1): regressão de log(1 + curtidas) sobre log(1 + inscritos atuais), só no rel
TABELA_CANAIS_RESTRITA = "canais_estatisticas_restritas.parquet"
USO_TREINO = "treino"
MIN_LINHAS_AJUSTE = 20  # mesmo mínimo técnico do treino das dobras (final.MIN_LINHAS_TREINO); não foi calibrado
MIN_CANAIS_AJUSTE = 3  # com dois canais a reta passa exatamente pelas duas médias: não há ajuste a avaliar
ESTADO_OK = "ok"

DEFINICAO = ("regressão linear (mínimos quadrados) de log1p(curtidas do alvo) sobre log1p(inscritos atuais do canal), "
             "ajustada só nas linhas de treino de cada dobra; previsão em log1p, cortada em 0; ausência de inscritos "
             "preservada (sem previsão)")
LIMITACOES = (
    ("Inscritos atuais (lidos na coleta) aplicados a posts de datas anteriores: avaliação retrospectiva; o tamanho do "
    "canal na data do post não é observado, e canais que cresceram ou encolheram desde então ficam desalinhados."),
    ("Há um único valor de inscritos por canal (no máximo 16 valores distintos): a reta é ajustada sobre poucos "
    "pontos e o coeficiente é frágil; é uma referência de contexto, não um previsor validado fora destes canais."),
    ("Os inscritos nunca entram como atributo dos modelos servidos nem de qualquer especificação: só esta referência "
    "os usa, e o ajuste olha apenas o treino de cada dobra."),
    ("Canal sem inscritos fica sem previsão (não vira zero nem média); a comparação com as outras referências vale "
    "só nas linhas em que todas existem, com a cobertura informada."),
)
CONTRASTES = (("modelo", "inscritos"), ("inscritos", "media_aritmetica"), ("inscritos", "mediana"),
              ("inscritos", "media_movel_exponencial"))


@dataclass(frozen=True)
class AjusteInscritos:
    estado: str
    coeficiente: float | None = None
    intercepto: float | None = None
    n_linhas: int = 0
    n_canais: int = 0
    n_linhas_sem_inscritos: int = 0
    canais_do_ajuste: tuple[str, ...] = field(default_factory=tuple)

    def resumo(self) -> dict:
        return {"estado": self.estado, "coeficiente": self.coeficiente, "intercepto": self.intercepto,
                "n_linhas": self.n_linhas, "n_canais": self.n_canais,
                "n_linhas_sem_inscritos": self.n_linhas_sem_inscritos, "canais_do_ajuste": list(self.canais_do_ajuste)}


# --- leitura da tabela restrita -----------------------------------------------------------------------------------


def carregar_inscritos(pasta_base: Path, *, verificar_manifesto: bool = True) -> dict[str, int | None]:
    """`channel_id -> inscritos atuais` da tabela restrita; nulo continua nulo. Com manifesto conferido por padrão."""
    if verificar_manifesto:
        tabela = curadoria.ler_base(Path(pasta_base))[TABELA_CANAIS_RESTRITA]
    else:
        tabela = pq.read_table(Path(pasta_base) / TABELA_CANAIS_RESTRITA)
    return {str(linha["channel_id"]): linha["inscritos"] for linha in tabela.select(["channel_id", "inscritos"])
            .to_pylist()}


def hash_da_tabela(pasta_base: Path) -> str:
    """SHA-256 dos bytes da tabela restrita (para o protocolo congelar a entrada do baseline 4)."""
    return hashlib.sha256((Path(pasta_base) / TABELA_CANAIS_RESTRITA).read_bytes()).hexdigest()


# --- ajuste e previsão ----------------------------------------------------------------------------------------------


def _x(inscritos: int | None) -> float | None:
    return None if inscritos is None or inscritos < 0 else math.log1p(inscritos)


def ajustar(treino: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> AjusteInscritos:
    """Reta de log1p(curtidas) sobre log1p(inscritos) nas linhas de treino com inscritos conhecidos."""
    for linha in treino:
        if linha.get("uso_no_contexto") != USO_TREINO:
            raise ValueError("o baseline 4 só ajusta com linhas de treino: encontrado uso_no_contexto = "
                             f"{linha.get('uso_no_contexto')!r}")
    pares = [(_x(inscritos.get(str(l["channel_id"]))), l["y_log"], str(l["channel_id"])) for l in treino
             if l.get("y_log") is not None and np.isfinite(l["y_log"])]
    com_x = [(x, y, c) for x, y, c in pares if x is not None]
    sem_inscritos = len(pares) - len(com_x)
    canais = tuple(sorted({c for _, _, c in com_x}))
    base = {"n_linhas": len(com_x), "n_canais": len(canais), "n_linhas_sem_inscritos": sem_inscritos,
            "canais_do_ajuste": canais}
    if len(com_x) < MIN_LINHAS_AJUSTE:
        return AjusteInscritos("linhas_insuficientes", **base)
    if len(canais) < MIN_CANAIS_AJUSTE:
        return AjusteInscritos("canais_insuficientes", **base)
    x = np.array([p[0] for p in com_x])
    y = np.array([p[1] for p in com_x])
    variancia = float(((x - x.mean()) ** 2).sum())
    if variancia <= 0.0:
        return AjusteInscritos("sem_variacao_nos_inscritos", **base)
    coeficiente = float(((x - x.mean()) * (y - y.mean())).sum() / variancia)
    return AjusteInscritos(ESTADO_OK, coeficiente, float(y.mean() - coeficiente * x.mean()), **base)


def _bruta(ajuste: AjusteInscritos, linhas: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> np.ndarray:
    if ajuste.estado != ESTADO_OK:
        return np.full(len(linhas), np.nan)
    saida = []
    for linha in linhas:
        x = _x(inscritos.get(str(linha["channel_id"])))
        saida.append(np.nan if x is None else ajuste.intercepto + ajuste.coeficiente * x)
    return np.array(saida, dtype=float)


def prever(ajuste: AjusteInscritos, linhas: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> np.ndarray:
    """Previsão em log1p, cortada em 0; NaN onde o ajuste não existe ou o canal não tem inscritos."""
    bruta = _bruta(ajuste, linhas, inscritos)
    return np.where(np.isnan(bruta), np.nan, np.maximum(bruta, 0.0))


def contar_cortadas(ajuste: AjusteInscritos, linhas: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> int:
    bruta = _bruta(ajuste, linhas, inscritos)
    return int(np.sum(bruta[~np.isnan(bruta)] < 0.0))


# --- bloco de relatório ---------------------------------------------------------------------------------------------


def hash_dos_ids(ids: Sequence[str]) -> str:
    """Mesma fórmula de `relatorio._bloco` para `ids_sha256`: permite conferir população idêntica."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def bloco_nao_avaliado(motivo: str) -> dict:
    return {"estado": "nao_avaliado", "motivo": motivo, "definicao": DEFINICAO, "limitacoes": list(LIMITACOES),
            "desvio": "baseline 4 do TAPI não avaliado nesta população; nenhum valor foi inventado"}


def prever_dobra(treino: Sequence[Mapping], avaliacao: Sequence[Mapping],
                 inscritos: Mapping[str, int | None]) -> tuple[AjusteInscritos, np.ndarray, int]:
    """Ajuste só no treino da dobra e previsão nas linhas de avaliação: (ajuste, previsto, n cortadas em zero)."""
    ajuste = ajustar(treino, inscritos)
    return ajuste, prever(ajuste, avaliacao, inscritos), contar_cortadas(ajuste, avaliacao, inscritos)


def _cobertura(linhas: Sequence[Mapping], previsto: np.ndarray) -> dict:
    com = np.isfinite(previsto)
    sem_canais = sorted({str(l["channel_id"]) for l, ok in zip(linhas, com, strict=True) if not ok})
    return {"n_linhas_avaliacao": len(linhas), "n_com_previsao": int(com.sum()),
            "fracao": float(com.mean()) if len(linhas) else None, "canais_sem_inscritos": sem_canais}


def _avaliar_modelo(linhas: Sequence[Mapping], central: Sequence[float], previsto: np.ndarray,
                    celulas: Sequence[str] | None, n_repeticoes: int, semente: int) -> dict:
    pop, exclusoes = montar_populacao_multi(linhas, {"modelo": central, "inscritos": previsto}, celulas=celulas,
                                               com_baselines=True)
    if not pop.ids:
        return {"estado": "sem_linhas_comuns", "exclusoes": exclusoes}
    return {"estado": "avaliado", "exclusoes": exclusoes,
            "populacao": {"n_avaliadas": len(pop.ids), "n_canais": len(set(pop.canais)),
                          "ids_sha256": hash_dos_ids(pop.ids)},
            "avaliacao": avaliar(pop, n_repeticoes=n_repeticoes, semente=semente, contrastes=CONTRASTES)}


def montar_bloco(ajustes: Mapping[str, AjusteInscritos], cortadas: Mapping[str, int], linhas: Sequence[Mapping],
                 previsto: np.ndarray, centrais: Mapping[str, Sequence[float]], *,
                 celulas: Sequence[str] | None = None, ids_do_principal: Mapping[str, str] | None = None,
                 n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE) -> dict:
    """Bloco do baseline 4: modelo × inscritos × as três referências, nas linhas em que todas existem.

    `ajustes` e `cortadas` são por dobra; `linhas`, `previsto` e `centrais` já estão empilhados na ordem das dobras.
    `ids_do_principal` (modelo -> `ids_sha256` da população principal) permite afirmar se as linhas são as mesmas.
    """
    cobertura = _cobertura(linhas, previsto)
    if cobertura["n_com_previsao"] == 0:
        estados = sorted({a.estado for a in ajustes.values()})
        return bloco_nao_avaliado(f"nenhuma linha com previsão (estados do ajuste: {estados}; canais sem inscritos: "
                                  f"{cobertura['canais_sem_inscritos']})")
    por_modelo = {nome: _avaliar_modelo(linhas, central, previsto, celulas, n_repeticoes, semente)
                  for nome, central in centrais.items()}
    iguais = bool(ids_do_principal) and all(
        por_modelo[nome].get("populacao", {}).get("ids_sha256") == ids_do_principal.get(nome) for nome in por_modelo)
    return {"estado": "avaliado", "definicao": DEFINICAO, "limitacoes": list(LIMITACOES),
            "ajustes_por_dobra": {rotulo: a.resumo() | {"n_cortadas_em_zero_na_avaliacao": cortadas.get(rotulo, 0)}
                                  for rotulo, a in ajustes.items()},
            "cobertura": cobertura, "mesmas_linhas_que_o_principal": iguais, "por_modelo": por_modelo}


# ──── Configurações do experimento audiovisual local e mapa de famílias de atributos.
HISTORICO = "historico"
LOCAL_COMPACTA = "local_compacta"
LOCAL_EMBEDDINGS = "local_embeddings"
LEGENDA = "legenda"
CAPA = "capa"
GEMINI = "gemini"
FAMILIAS = (HISTORICO, LOCAL_COMPACTA, LOCAL_EMBEDDINGS, LEGENDA, CAPA, GEMINI)

CATEGORICAS_LOCAIS = (f"{col.PREFIXO}orientacao", f"{col.PREFIXO}fala_idioma")
COLUNA_DUPLICADA = f"{col.PREFIXO}duracao_s"
MOTIVO_DUPLICADA = "duplica `duracao_s` (família histórico), presente em todas as configurações"
COMPONENTES_PCA = ds.COMPONENTES_PCA

REFERENCIAS_R0 = {"media_aritmetica": "pred_media_aritmetica", "mediana": "pred_mediana",
                  "media_movel_exponencial": "pred_mme_log"}


class ErroConfiguracao(ValueError):
    """Especificação, família ou configuração incoerente com o protocolo."""


@dataclass(frozen=True)
class Configuracao:
    id: str
    descricao: str
    familias: tuple[str, ...]
    aprendida: bool = True

    @property
    def usa_gemini(self) -> bool:
        return GEMINI in self.familias

    @property
    def usa_local(self) -> bool:
        return LOCAL_COMPACTA in self.familias or LOCAL_EMBEDDINGS in self.familias

    @property
    def usa_capa_ou_legenda(self) -> bool:
        return CAPA in self.familias or LEGENDA in self.familias


_LOCAL = (LOCAL_COMPACTA, LOCAL_EMBEDDINGS)
CONFIGURACOES: dict[str, Configuracao] = {c.id: c for c in (
    Configuracao("R0", "referências históricas (média, mediana e média móvel); sem aprendizado", (), aprendida=False),
    Configuracao("M0", "histórico e contexto", (HISTORICO,)),
    Configuracao("M1", "M0 + medidas compactas locais", (HISTORICO, LOCAL_COMPACTA)),
    Configuracao("M2", "M0 + embeddings de quadros locais", (HISTORICO, LOCAL_EMBEDDINGS)),
    Configuracao("M3", "M0 + local completo (compacto e embeddings)", (HISTORICO, *_LOCAL)),
    Configuracao("M4", "histórico + legenda + capa (família existente)", (HISTORICO, LEGENDA, CAPA)),
    Configuracao("M5", "M4 + local completo", (HISTORICO, LEGENDA, CAPA, *_LOCAL)),
    Configuracao("M6", "histórico + legenda + capa + Gemini (família completa)",
                 (HISTORICO, LEGENDA, CAPA, GEMINI)),
    Configuracao("M7", "M6 + local completo", (HISTORICO, LEGENDA, CAPA, GEMINI, *_LOCAL)),
)}
APRENDIDAS = tuple(c for c, cfg in CONFIGURACOES.items() if cfg.aprendida)
# Candidatas a preditor servido: usam atributos locais, não exigem coluna `gem_*` e só dependem de entradas obtidas
# de um arquivo de vídeo enviado (duração, atributos locais, contexto e histórico do canal). Capa e legenda exigem uma
# miniatura e um texto que um arquivo de vídeo não traz; por isso M4 e M5 são avaliadas mas não são servíveis.
ELEGIVEIS_PARA_SERVIR = tuple(c for c in APRENDIDAS if CONFIGURACOES[c].usa_local and not CONFIGURACOES[c].usa_gemini
                              and not CONFIGURACOES[c].usa_capa_ou_legenda)


def familia_da_coluna(coluna: str) -> str:
    """Família de uma coluna preditora; `avl_*` são locais, o resto segue o mapa estabelecido."""
    if coluna.startswith(col.PREFIXO):
        return LOCAL_EMBEDDINGS if coluna in col.COLUNAS_VETORIAIS else LOCAL_COMPACTA
    return fa.familia(coluna)


def colunas_locais_excluidas() -> dict[str, str]:
    """Colunas `avl_*` que não podem ser preditoras, com o motivo."""
    motivos = {c: "impressão digital do arquivo ou da política de amostragem (identifica origem, não conteúdo)"
               for c in col.COLUNAS_DE_IMPRESSAO_DIGITAL}
    motivos |= {c: "estado da etapa de extração (não medido ≠ falha; nunca preditor)" for c in col.COLUNAS_DE_ESTADO}
    motivos[COLUNA_DUPLICADA] = MOTIVO_DUPLICADA
    return motivos


def especificacao_total(base_desenvolvimento: Mapping, local: Mapping) -> dict:
    """União das colunas do desenvolvimento e das locais permitidas, na ordem: desenvolvimento, depois `avl_*`.

    `local` é a especificação candidata do dataset audiovisual (escalares, categóricas, embeddings). Coluna local fora
    de `COLUNAS_PREDITORAS` é erro; as excluídas por regra são removidas e listadas em `excluidas`."""
    excluidas = colunas_locais_excluidas()
    permitidas = set(col.COLUNAS_PREDITORAS)
    declaradas = [*local["escalares"], *local["categoricas"], *local["embeddings"]]
    desconhecidas = sorted(c for c in declaradas if c not in permitidas and c not in excluidas)
    if desconhecidas:
        raise ErroConfiguracao(f"colunas locais fora do contrato do extrator: {desconhecidas}")
    categoricas_vistas = set(local["categoricas"])
    if not categoricas_vistas <= set(CATEGORICAS_LOCAIS):
        raise ErroConfiguracao(f"categóricas locais inesperadas: {sorted(categoricas_vistas - set(CATEGORICAS_LOCAIS))}")
    mantem = lambda c: c not in excluidas  # noqa: E731
    total = {
        "escalares": [*base_desenvolvimento["escalares"], *(c for c in local["escalares"] if mantem(c))],
        "categoricas": [*base_desenvolvimento["categoricas"], *(c for c in local["categoricas"] if mantem(c))],
        "embeddings": {**base_desenvolvimento["embeddings"],
                       **{c: COMPONENTES_PCA for c in local["embeddings"] if mantem(c)}},
    }
    conferir_mapa(total)
    total["excluidas"] = {c: excluidas[c] for c in declaradas if c in excluidas}
    return total


def conferir_mapa(total: Mapping) -> None:
    """Toda coluna de histórico tem de estar na lista aprovada do N1; coluna repetida é erro."""
    todas = [*total["escalares"], *total["categoricas"], *total["embeddings"]]
    if len(set(todas)) != len(todas):
        raise ErroConfiguracao("colunas repetidas na especificação total")
    estranhas = sorted(c for c in todas if familia_da_coluna(c) == HISTORICO and c not in hs.ATRIBUTOS_N1)
    if estranhas:
        raise ErroConfiguracao(f"colunas classificadas como histórico fora do N1 aprovado: {estranhas}")
    proibidas = sorted(c for c in todas if c in ds.PROIBIDAS_COMO_ATRIBUTO)
    if proibidas:
        raise ErroConfiguracao(f"colunas proibidas como atributo: {proibidas}")


def especificacao_da_configuracao(config_id: str, total: Mapping) -> md.EspecificacaoAtributos:
    """Atributos das famílias da configuração, na ordem da especificação total."""
    config = _configuracao_aprendida(config_id)
    pertence = lambda c: familia_da_coluna(c) in config.familias  # noqa: E731
    return md.EspecificacaoAtributos(
        [c for c in total["escalares"] if pertence(c)], [c for c in total["categoricas"] if pertence(c)],
        {c: k for c, k in total["embeddings"].items() if pertence(c)})


def colunas_locais_da_especificacao(espec: md.EspecificacaoAtributos) -> tuple[str, ...]:
    """Colunas `avl_*` da especificação: as que passam por consolidação de categorias e descarte de constantes."""
    return tuple(c for c in (*espec.escalares, *espec.categoricas) if c.startswith(col.PREFIXO))


def _configuracao_aprendida(config_id: str) -> Configuracao:
    if config_id not in CONFIGURACOES or not CONFIGURACOES[config_id].aprendida:
        raise ErroConfiguracao(f"configuração aprendida desconhecida: {config_id!r}; use {list(APRENDIDAS)}")
    return CONFIGURACOES[config_id]


def mapa_de_familias(total: Mapping) -> dict[str, list[str]]:
    """Colunas de cada família, na ordem das famílias do protocolo (mapeamento preservado para auditoria)."""
    todas = [*total["escalares"], *total["categoricas"], *total["embeddings"]]
    return {f: [c for c in todas if familia_da_coluna(c) == f] for f in FAMILIAS}


def resumo_das_configuracoes(total: Mapping) -> dict:
    """Famílias, número de colunas de saída e hash de cada configuração aprendida (entra no protocolo congelado)."""
    saida = {}
    for config_id in APRENDIDAS:
        espec = especificacao_da_configuracao(config_id, total)
        saida[config_id] = {
            "descricao": CONFIGURACOES[config_id].descricao, "familias": list(CONFIGURACOES[config_id].familias),
            "usa_gemini": CONFIGURACOES[config_id].usa_gemini, "usa_local": CONFIGURACOES[config_id].usa_local,
            "elegivel_para_servir": config_id in ELEGIVEIS_PARA_SERVIR,
            "n_escalares": len(espec.escalares), "n_categoricas": len(espec.categoricas),
            "embeddings": dict(espec.embeddings), "n_colunas_de_saida": fa.n_colunas(espec),
            "sha256_especificacao": fa.hash_da_especificacao(espec),
        }
    return saida


def ordem_de_saida(espec: md.EspecificacaoAtributos) -> list[str]:
    """Nomes das colunas de entrada do modelo antes do descarte de constantes: escalares, categóricas, blocos."""
    return [*espec.escalares, *espec.categoricas, *espec.embeddings]


def sem_gemini(espec: md.EspecificacaoAtributos) -> bool:
    return not any(c.startswith("gem_") for c in ordem_de_saida(espec))


def conferir_sem_gemini(espec: md.EspecificacaoAtributos, nomes: Sequence[str] = ()) -> None:
    """Falha se o preditor servível exigir alguma coluna `gem_*` (entrada ou componente de saída)."""
    if not sem_gemini(espec) or any(n.startswith("gem_") for n in nomes):
        raise ErroConfiguracao("o preditor local não pode exigir colunas gem_*")
