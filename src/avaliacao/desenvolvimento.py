"""Modelo × baselines nas mesmas linhas de validação de desenvolvimento, com bootstrap pareado por canal.

Previsões em log (`log1p` de curtidas). Baselines com as fórmulas aprovadas (D-D1), que já vêm no dataset:
média aritmética (`pred_media_aritmetica`), mediana (`pred_mediana`) e média móvel exponencial (`pred_mme_log`).
Uma linha só entra se o real, a previsão do modelo e as três bases existirem: o mesmo conjunto vale para todos os métodos
e as exclusões são contadas por motivo.

MAE, RMSE, MAPE adaptado e acurácia par a par são agregados juntando os vídeos dos canais sorteados (a razão das somas);
o Spearman intra-canal é a mediana entre os canais sorteados (canal sorteado duas vezes conta duas vezes). Nenhum
limiar de promoção é aplicado: o relatório traz a diferença observada e o intervalo.

Células (avaliação final): quando a população empilha dobras com modelos diferentes, `celulas` rotula cada linha com a
sua célula canal × sub-bloco. Pares e Spearman valem dentro da célula; MAE, RMSE e MAPE continuam somados por canal;
o bootstrap continua sorteando CANAIS (cada célula entra com o peso do seu canal). Sem `celulas`, a célula é o canal e
o comportamento é o de sempre.

Métodos e contrastes: por padrão, `modelo` contra as três bases. Para a ablação, `montar_populacao_multi` aceita
qualquer conjunto de previsões nomeadas e `avaliar(..., contrastes=[(a, b), ...])` devolve a diferença `a − b`
com o mesmo bootstrap pareado.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.avaliacao import bootstrap as bs
from src.avaliacao import ordenacao as od
from src.metricas import resumo_erros

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
