"""Métricas de erro mínimas para conferir previsões em log (DECISOES 9.6).

- MAE e RMSE sobre `log(1 + curtidas)`;
- MAPE adaptado: média de |previsto − real| ÷ max(real, 1), em curtidas brutas, com previsto = exp(log) − 1.

Pares sem valor real ou sem previsão ficam fora e são contados, com a cobertura (pares usados ÷ linhas com valor
real) ao lado das métricas; nenhum ausente vira zero. Intervalos de confiança,
métricas de ordenação e relatórios por fatia ficam para o pacote de avaliação.
"""

import math
import statistics
from collections.abc import Sequence


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
