"""Módulo consolidado. Reúne:
- Métricas de erro mínimas para conferir previsões em log (DECISOES 9.6).
- Métricas de ordenação (DECISOES 9.6): Spearman global e intra-canal e acurácia par a par intra-canal.
- Bootstrap pareado por canal (DECISOES 9.6: IC 95% com 2.000 reamostragens de canais).
- Faixas por tercis, F1 macro e R² do alvo relativo (D-D3; protocolo final §7).
- Leitura do ganho multimodal pela posição do IC 95% da diferença (DECISOES 9.8, verificação V20)."""

import math
import statistics
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


# ──── Métricas de erro mínimas para conferir previsões em log (DECISOES 9.6).
def _pares(curtidas_reais: Sequence[int | None], previstos_log: Sequence[float | None]) -> list[tuple[int, float]]:
    if len(curtidas_reais) != len(previstos_log):
        raise ValueError("valores reais e previstos com tamanhos diferentes")
    pares = [(r, p) for r, p in zip(curtidas_reais, previstos_log, strict=True) if r is not None and p is not None]
    if not pares:
        raise ValueError("nenhum par com valor real e previsão")
    return pares


def resumo_erros(curtidas_reais: Sequence[int | None], previstos_log: Sequence[float | None]) -> dict:
    """MAE e RMSE em log e MAPE adaptado, com as contagens de pares usados e excluídos."""
    pares = _pares(curtidas_reais, previstos_log)
    erros_log = [previsto - math.log1p(real) for real, previsto in pares]
    sem_previsao = sum(r is not None and p is None for r, p in zip(curtidas_reais, previstos_log, strict=True))
    return {
        "n": len(pares),
        "cobertura": len(pares) / (len(pares) + sem_previsao),
        "n_sem_previsao": sem_previsao,
        "n_sem_valor_real": sum(r is None for r in curtidas_reais),
        "n_reais_zero": sum(real == 0 for real, _ in pares),
        "mae_log": statistics.fmean(abs(e) for e in erros_log),
        "rmse_log": math.sqrt(statistics.fmean(e * e for e in erros_log)),
        "mape_adaptado": statistics.fmean(abs(math.expm1(p) - r) / max(r, 1) for r, p in pares),
    }


# ──── Métricas de ordenação (DECISOES 9.6): Spearman global e intra-canal e acurácia par a par intra-canal
MIN_VIDEOS_CANAL = 5
FATOR_DISTINGUIVEL = 1.2


def _valido(valor) -> bool:
    return valor is not None and bool(np.isfinite(valor))


def _pares_validos(reais: Sequence, previstos: Sequence) -> tuple[np.ndarray, np.ndarray]:
    if len(reais) != len(previstos):
        raise ValueError("valores reais e previstos com tamanhos diferentes")
    pares = [(float(r), float(p)) for r, p in zip(reais, previstos, strict=True) if _valido(r) and _valido(p)]
    if not pares:
        return np.empty(0), np.empty(0)
    return np.array([r for r, _ in pares]), np.array([p for _, p in pares])


def correlacao_de_spearman(reais: Sequence, previstos: Sequence) -> float | None:
    """Spearman com ranks médios; `None` se houver menos de 2 pares ou se um dos lados for constante."""
    r, p = _pares_validos(reais, previstos)
    if len(r) < 2 or np.ptp(r) == 0 or np.ptp(p) == 0:
        return None
    return float(np.corrcoef(rankdata(r), rankdata(p))[0, 1])


def agrupar_por_canal(canais: Sequence) -> dict[str, list[int]]:
    grupos: dict[str, list[int]] = {}
    for i, canal in enumerate(canais):
        grupos.setdefault(str(canal), []).append(i)
    return dict(sorted(grupos.items()))


def spearman_por_canal(canais: Sequence, reais: Sequence, previstos: Sequence, *,
                       minimo: int = MIN_VIDEOS_CANAL) -> dict:
    """Spearman por canal elegível (tamanho e variação do real observados) e mediana entre os definidos."""
    elegiveis, pequenos, constantes, por_canal = [], [], [], {}
    for canal, indices in agrupar_por_canal(canais).items():
        observados = [reais[i] for i in indices if _valido(reais[i])]
        if len(observados) < minimo:
            pequenos.append(canal)
        elif np.ptp(np.array(observados, dtype=float)) == 0:
            constantes.append(canal)
        else:
            elegiveis.append(canal)
            por_canal[canal] = correlacao_de_spearman([reais[i] for i in indices], [previstos[i] for i in indices])
    definidos = [v for v in por_canal.values() if v is not None]
    return {
        "minimo_videos": minimo, "elegiveis": elegiveis, "excluidos_por_tamanho": pequenos,
        "excluidos_sem_variacao_no_real": constantes, "por_canal": por_canal,
        "indefinidos": [c for c, v in por_canal.items() if v is None], "n_definidos": len(definidos),
        "mediana": float(np.median(definidos)) if definidos else None,
    }


def par_distinguivel(a: float, b: float) -> bool:
    menor, maior = min(a, b), max(a, b)
    return maior > menor and maior >= FATOR_DISTINGUIVEL * menor


def contribuicao_de_pares(reais: Sequence, previstos: Sequence) -> dict:
    """Contagens de pares de UM canal (créditos e totais), para somar entre canais ou reamostrar."""
    r, p = _pares_validos(reais, previstos)
    saida = {"credito_distinguiveis": 0.0, "n_distinguiveis": 0, "credito_todos": 0.0, "n_todos": 0,
             "n_empate_previsto_distinguiveis": 0, "n_pares": 0, "n_reais_iguais": 0}
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            saida["n_pares"] += 1
            if r[i] == r[j]:
                saida["n_reais_iguais"] += 1
                continue
            if p[i] == p[j]:
                credito = 0.5
            else:
                credito = 1.0 if (r[i] > r[j]) == (p[i] > p[j]) else 0.0
            saida["credito_todos"] += credito
            saida["n_todos"] += 1
            if par_distinguivel(r[i], r[j]):
                saida["credito_distinguiveis"] += credito
                saida["n_distinguiveis"] += 1
                saida["n_empate_previsto_distinguiveis"] += int(p[i] == p[j])
    return saida


def somar_contribuicoes(contribuicoes: Sequence[Mapping]) -> dict:
    soma = {k: sum(c[k] for c in contribuicoes) for k in
            ("credito_distinguiveis", "n_distinguiveis", "credito_todos", "n_todos",
             "n_empate_previsto_distinguiveis", "n_pares", "n_reais_iguais")}

    def taxa(credito: float, n: int) -> float | None:
        return credito / n if n else None

    return {
        "distinguiveis": {"credito": soma["credito_distinguiveis"], "n": soma["n_distinguiveis"],
                          "acuracia": taxa(soma["credito_distinguiveis"], soma["n_distinguiveis"])},
        "todos": {"credito": soma["credito_todos"], "n": soma["n_todos"],
                  "acuracia": taxa(soma["credito_todos"], soma["n_todos"])},
        "n_empate_previsto_distinguiveis": soma["n_empate_previsto_distinguiveis"],
        "n_pares_mesmo_canal": soma["n_pares"], "n_pares_com_reais_iguais": soma["n_reais_iguais"],
    }


def acuracia_par_a_par(canais: Sequence, reais: Sequence, previstos: Sequence) -> dict:
    """Acurácia par a par intra-canal, em pares distinguíveis e em todos os pares com observações diferentes."""
    por_canal = {c: contribuicao_de_pares([reais[i] for i in idx], [previstos[i] for i in idx])
                 for c, idx in agrupar_por_canal(canais).items()}
    return somar_contribuicoes(list(por_canal.values())) | {"por_canal": por_canal}


# ──── Bootstrap pareado por canal (DECISOES 9.6: IC 95% com 2.000 reamostragens de canais).
SEMENTE = 20261015
N_REPETICOES = 2000
NIVEL = 95.0


def sorteio_de_canais(n_canais: int, *, n_repeticoes: int = N_REPETICOES, semente: int = SEMENTE) -> np.ndarray:
    """Matriz (repetições × canais) com quantas vezes cada canal foi sorteado em cada repetição."""
    rng = np.random.default_rng(semente)
    sorteados = rng.integers(0, n_canais, size=(n_repeticoes, n_canais))
    return np.stack([np.bincount(linha, minlength=n_canais) for linha in sorteados])


def percentis(valores: np.ndarray, nivel: float = NIVEL) -> list[float]:
    cauda = (100.0 - nivel) / 2.0
    inferior, superior = np.percentile(valores, [cauda, 100.0 - cauda])
    return [float(inferior), float(superior)]


def resumir_diferenca(modelo: np.ndarray, base: np.ndarray, *, menor_e_melhor: bool) -> dict:
    """Diferença `modelo − base` por repetição (só onde ambos estão definidos) e sua leitura pela direção da métrica."""
    modelo, base = np.asarray(modelo, dtype=float), np.asarray(base, dtype=float)
    usaveis = np.isfinite(modelo) & np.isfinite(base)
    resumo = {"n_repeticoes": int(len(modelo)), "n_usaveis": int(usaveis.sum()),
              "n_indefinidas": int((~usaveis).sum()),
              "direcao": "menor é melhor" if menor_e_melhor else "maior é melhor", "diferenca": "modelo − base"}
    if not usaveis.any():
        return resumo | {"ic95": None, "media": None, "mediana": None, "leitura": "indefinido"}
    diferencas = (modelo - base)[usaveis]
    inferior, superior = percentis(diferencas)
    favoravel = superior < 0 if menor_e_melhor else inferior > 0
    desfavoravel = inferior > 0 if menor_e_melhor else superior < 0
    leitura = "favoravel_ao_modelo" if favoravel else "desfavoravel_ao_modelo" if desfavoravel else "inconclusivo"
    return resumo | {"ic95": [inferior, superior], "media": float(diferencas.mean()),
                     "mediana": float(np.median(diferencas)), "leitura": leitura}


# ──── Faixas por tercis, F1 macro e R² do alvo relativo (D-D3; protocolo final §7).
N_FAIXAS = 3


def tercis_do_treino(alvo_relativo_do_treino: Sequence[float]) -> list[float]:
    """Os dois tercis (percentis 33,3 e 66,7, interpolação linear) do alvo relativo do treino."""
    valores = np.asarray(alvo_relativo_do_treino, dtype=float)
    return [float(v) for v in np.percentile(valores[np.isfinite(valores)], [100 / 3, 200 / 3])]


def faixas(valores: Sequence[float], bordas: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(bordas, dtype=float), np.asarray(valores, dtype=float), side="right")


def _f1_macro_de_classes(reais: np.ndarray, previstos: np.ndarray) -> float | None:
    f1s = []
    for classe in range(N_FAIXAS):
        tp = int(np.sum((reais == classe) & (previstos == classe)))
        fp = int(np.sum((reais != classe) & (previstos == classe)))
        fn = int(np.sum((reais == classe) & (previstos != classe)))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else None


def f1_macro(reais: Sequence[float], previstos: Sequence[float], bordas: Sequence[float]) -> float | None:
    return _f1_macro_de_classes(faixas(reais, bordas), faixas(previstos, bordas))


def f1_macro_empilhado(dobras: Sequence[tuple[Sequence[float], Sequence[float], Sequence[float]]]) -> float | None:
    """F1 macro sobre as linhas de todas as dobras, cada uma faixada com as bordas do seu próprio treino."""
    reais = np.concatenate([faixas(r, b) for r, _, b in dobras])
    previstos = np.concatenate([faixas(p, b) for _, p, b in dobras])
    return _f1_macro_de_classes(reais, previstos)


def r2(reais: Sequence[float], previstos: Sequence[float]) -> float | None:
    """R² do alvo relativo (pode ser negativo); `None` se o real não varia."""
    r, p = np.asarray(reais, dtype=float), np.asarray(previstos, dtype=float)
    sst = float(np.sum((r - r.mean()) ** 2))
    if sst == 0:
        return None
    return 1.0 - float(np.sum((r - p) ** 2)) / sst


# ──── Leitura do ganho multimodal pela posição do IC 95% da diferença (DECISOES 9.8, verificação V20).
DELTA_RELEVANCIA_PRATICA = 0.02


def leitura_do_ganho(ic95: Sequence[float] | None, *, delta: float = DELTA_RELEVANCIA_PRATICA) -> dict:
    """Sinal, magnitude e leitura combinada de um IC [a, b]; `None` (sem IC) é "indefinido"."""
    if ic95 is None:
        return {"ic95": None, "delta": delta, "sinal": "indefinido", "magnitude": "indefinida",
                "leitura": "indefinido"}
    a, b = float(ic95[0]), float(ic95[1])
    if b < 0:
        sinal = "piora_com_evidencia"
    elif a > 0:
        sinal = "ganho_positivo_com_evidencia"
    else:
        sinal = "sem_evidencia_de_sinal"
    if a >= delta:
        magnitude = "acima_do_limiar"
    elif b < delta:
        magnitude = "abaixo_do_limiar"
    else:
        magnitude = "indeterminada"
    if b < 0:
        leitura = "piora_com_evidencia"
    elif a >= delta:
        leitura = "ganho_positivo_acima_do_limiar"
    elif a > 0:
        leitura = "ganho_positivo_abaixo_do_limiar" if b < delta else "ganho_positivo_magnitude_indeterminada"
    elif b < delta:
        leitura = "sem_evidencia_de_ganho_e_ganho_acima_do_limiar_descartado"
    else:
        leitura = "inconclusivo"
    return {"ic95": [a, b], "delta": delta, "sinal": sinal, "magnitude": magnitude, "leitura": leitura}
