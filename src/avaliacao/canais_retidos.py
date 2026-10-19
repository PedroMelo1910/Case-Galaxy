"""Sorteio dos canais retidos para a avaliação de canais não vistos (D-D2, D-C4; protocolo final §4).

Três sorteios de cerca de 30% dos canais, estratificados por setor × tipo de canal. Alocação por estrato:
arredondamento (meio para cima) de 0,3 × n; estratos com 2 ou mais canais retiram ao menos 1 e nunca todos; um estrato
de um único canal nunca é retirado. Sementes 20261015, 20261016 e 20261017, uma por sorteio. O resultado depende só dos
identificadores e das sementes, não da ordem de entrada.
"""

import math
from collections import Counter
from collections.abc import Mapping

import numpy as np

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
