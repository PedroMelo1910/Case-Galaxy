"""Leitura do ganho multimodal pela posição do IC 95% da diferença (DECISOES 9.8, verificação V20).

O IC é de `nível de interesse − nível de comparação` (maior é melhor), com limites [a, b]. A leitura combina o sinal
com a magnitude em relação a δ (relevância prática, +0,02: convenção do projeto, não critério do TAPI nem prova):

| posição do IC          | leitura                                                          |
|------------------------|------------------------------------------------------------------|
| b < 0                  | piora com evidência                                              |
| a ≤ 0 ≤ b < δ          | sem evidência de ganho; ganho igual ou acima do limiar descartado |
| a ≤ 0 < δ ≤ b          | inconclusivo                                                     |
| 0 < a e b < δ          | ganho positivo com evidência, abaixo do limiar                   |
| 0 < a < δ ≤ b          | ganho positivo com evidência; magnitude indeterminada            |
| a ≥ δ                  | ganho positivo acima do limiar                                   |
"""

from collections.abc import Sequence

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
