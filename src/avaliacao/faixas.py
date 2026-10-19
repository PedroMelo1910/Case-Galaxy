"""Faixas por tercis, F1 macro e R² do alvo relativo (D-D3; protocolo final §7).

As bordas são os tercis do alvo relativo no TREINO de cada dobra, aplicados ao real e ao previsto daquela dobra. Uma
faixa é o número de bordas menores ou iguais ao valor (0 = abaixo, 1 = na média, 2 = acima). F1 macro: média dos F1 das
classes que ocorrem no real ou no previsto; classe ausente dos dois lados é indefinida e não entra (nem vira 1). O
acaso fica perto de 1/3. O TAPI fala em quatro faixas ("viral" sem definição operacional); o projeto mantém as três
faixas decididas e declara a diferença.
"""

from collections.abc import Sequence

import numpy as np

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
