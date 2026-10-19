"""Diagnósticos descritivos por grupo (idade do rótulo, setor, canal, histórico, transcrição).

Mostram onde o erro, o viés e a cobertura se concentram. Não atribuem causa: um padrão por faixa de idade, por canal ou
entre treino e validação é uma observação, não uma explicação. Viés = previsto − real em log (positivo superestima).
"""

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from src.avaliacao import ordenacao as od
from src.avaliacao.desenvolvimento import METODOS, Populacao

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


def _erros(pop: Populacao, indices: np.ndarray) -> dict:
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
                 "erros": _erros(pop, indices), "ordenacao": _ordenacao(pop, indices), "cobertura": None}
        if inferior is not None and superior is not None:
            dentro = (pop.real_log[indices] >= np.asarray(inferior)[indices]) & (
                pop.real_log[indices] <= np.asarray(superior)[indices])
            grupo["cobertura"] = {"n": int(len(indices)), "n_dentro": int(dentro.sum()),
                                  "cobertura": float(dentro.mean())}
        saida[rotulo] = grupo
    return saida
