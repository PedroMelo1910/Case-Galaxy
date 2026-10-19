"""Cobertura da faixa por setor, com IC por reamostragem de canais quando estimável (protocolo final §8, §14.9).

A unidade sorteada é o canal do setor (vídeos do mesmo canal não são independentes), com a mesma semente e o mesmo
número de repetições do restante da avaliação. O IC só é informado quando há canais e linhas suficientes; senão fica
indefinido, com o motivo escrito: nunca vira zero nem um intervalo degenerado.

Regras técnicas (não calibradas com resultado): menos de `MIN_CANAIS_IC` canais ou de `MIN_LINHAS_IC` linhas no setor =
IC indefinido (amostra insuficiente); entre `MIN_CANAIS_IC` e `CANAIS_IC_CONFIAVEL - 1` canais o IC sai marcado como
aproximado, pois o percentil de poucos grupos tende a ser estreito demais.
"""

import numpy as np

from src.avaliacao import bootstrap as bs
from src.avaliacao import desenvolvimento as dv
from src.avaliacao import ordenacao as od

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


def cobertura_por_setor(pop: dv.Populacao, inferior: np.ndarray, superior: np.ndarray, *,
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
