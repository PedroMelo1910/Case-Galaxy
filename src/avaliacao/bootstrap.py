"""Bootstrap pareado por canal (DECISOES 9.6: IC 95% com 2.000 reamostragens de canais).

Vídeos do mesmo canal não são independentes, então a unidade sorteada é o canal, com reposição. O sorteio é gerado
uma vez e reutilizado por todos os métodos comparados (mesmos canais em cada repetição). Semente e número de
repetições são escolhas registradas desta avaliação: a quantidade de repetições vem das decisões, a semente
(20261015, a do projeto) não estava definida e foi fixada aqui.

O intervalo é o percentil 2,5–97,5 das diferenças. Com poucos canais ele é grosseiro: a leitura "inconclusivo" é o
resultado esperado quando o intervalo inclui zero, e nenhuma probabilidade é atribuída ao intervalo.
"""

import numpy as np

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
