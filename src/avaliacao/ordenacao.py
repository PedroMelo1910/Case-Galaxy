"""Métricas de ordenação (DECISOES 9.6): Spearman global e intra-canal e acurácia par a par intra-canal.

Definições aprovadas, aplicadas sem alteração:
- Spearman intra-canal: por canal com pelo menos `MIN_VIDEOS_CANAL` vídeos observados, agregado pela mediana entre
  canais; ranks médios nos empates. Correlação indefinida (previsão ou real constante) não vira zero: é contada à parte;
- par a par: pares do mesmo canal; empate previsto vale 0,5; "distinguível" = curtidas observadas diferentes com
  `max >= FATOR_DISTINGUIVEL × min` (zero contra positivo é distinguível; zero contra zero não forma par).

Leitura de "todos os pares": todos os pares do mesmo canal com curtidas observadas diferentes. Pares com observações
iguais não têm ordem a acertar e são contados em `n_pares_com_reais_iguais`.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

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
