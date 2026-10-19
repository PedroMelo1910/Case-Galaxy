"""Limiar de largura de baixa confiança (D-E5, D-F2b; protocolo final §8).

O limiar é o percentil 80 das larguras (limite superior − limite inferior da faixa corrigida, em log; igual a q90 − q10
quando não há cruzamento) das previsões do modelo de vídeo nos vídeos da LIM. A LIM
não tem rótulos e não é um conjunto de calibração: só as larguras entram, depois do congelamento do modelo, e nada de
desempenho é ajustado. A regra é validada no modelo de desenvolvimento (erro absoluto dos sinalizados contra o dos não
sinalizados, bootstrap pareado por canal); no modelo servido a validade é presumida e declarada.
"""

from collections.abc import Sequence

import numpy as np

from src.avaliacao import bootstrap as bs

PERCENTIL = 80
LEITURAS = {"favoravel_ao_modelo": "sinalizados_erram_mais_com_evidencia",
            "desfavoravel_ao_modelo": "sinalizados_erram_menos_com_evidencia", "inconclusivo": "inconclusivo",
            "indefinido": "indefinido"}


def larguras(inferior_log: Sequence[float], superior_log: Sequence[float]) -> np.ndarray:
    return np.asarray(superior_log, dtype=float) - np.asarray(inferior_log, dtype=float)


def limiar_de_largura(larguras: Sequence[float], percentil: float = PERCENTIL) -> float:
    valores = np.asarray(larguras, dtype=float)
    if valores.size == 0 or not np.all(np.isfinite(valores)):
        raise ValueError("larguras vazias ou não finitas: o limiar não é definido")
    return float(np.percentile(valores, percentil))


def sinalizar(larguras: Sequence[float], limiar: float) -> np.ndarray:
    """Baixa confiança estatística: largura estritamente acima do limiar."""
    return np.asarray(larguras, dtype=float) > limiar


def validar_regra(larguras: Sequence[float], erros_absolutos: Sequence[float], canais: Sequence[str], *, limiar: float,
                  n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE) -> dict:
    """Erro absoluto médio dos sinalizados e dos não sinalizados, e a diferença com IC por reamostragem de canais."""
    marca = sinalizar(larguras, limiar)
    erro, canais = np.asarray(erros_absolutos, dtype=float), np.asarray([str(c) for c in canais])
    saida = {"limiar": float(limiar), "n_sinalizados": int(marca.sum()), "n_nao_sinalizados": int((~marca).sum())}
    if not marca.any() or marca.all():
        return saida | {"mae_sinalizados": None, "mae_nao_sinalizados": None, "bootstrap": None,
                        "leitura": "sem_sinalizados" if not marca.any() else "todos_sinalizados"}
    nomes = sorted(set(canais))
    por_canal = {c: canais == c for c in nomes}
    soma = {k: np.array([erro[m & sel].sum() for m in por_canal.values()]) for k, sel in
            (("sinal", marca), ("nao", ~marca))}
    n = {k: np.array([(m & sel).sum() for m in por_canal.values()], dtype=float) for k, sel in
         (("sinal", marca), ("nao", ~marca))}
    contagens = bs.sorteio_de_canais(len(nomes), n_repeticoes=n_repeticoes, semente=semente)

    def razao(chave: str) -> np.ndarray:
        num, den = contagens @ soma[chave], contagens @ n[chave]
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    resumo = bs.resumir_diferenca(razao("sinal"), razao("nao"), menor_e_melhor=False)
    return saida | {"mae_sinalizados": float(erro[marca].mean()), "mae_nao_sinalizados": float(erro[~marca].mean()),
                    "bootstrap": resumo, "leitura": LEITURAS[resumo["leitura"]],
                    "nota": "no modelo servido a validade da regra é presumida e declarada"}
