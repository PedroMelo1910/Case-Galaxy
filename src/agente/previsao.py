"""Previsão central (q50), faixa q10–q90 e contribuições do q50 para uma linha de inferência.

Toda a aritmética vem de `faixa_previsao` (já verificada): central = q50, limites = mínimo e máximo das três saídas,
corte em zero e `expm1`. Aqui só se calcula a razão suavizada em relação ao canal sobre os valores exibidos:
`(1 + curtidas previstas) / (1 + curtidas de referência)`, ambos inteiros como mostrados.
"""

import math
from collections.abc import Mapping

from src import faixa_previsao as fp
from src.agente.pacote import Pacote


def faixa_da_linha(pacote: Pacote, linha: Mapping) -> dict:
    """Valores exibidos (central, faixa, razão ao canal) a partir dos três modelos; sem SHAP."""
    linhas = [dict(linha)]
    faixa = fp.prever_faixa(linhas, pacote.q10.prever_relativo(linhas), pacote.q50.prever_relativo(linhas),
                            pacote.q90.prever_relativo(linhas))
    ref_log = float(linha["ref_log"])
    referencia = round(math.expm1(ref_log))
    central = int(faixa["curtidas_central"][0])
    return {
        "curtidas_central": central, "curtidas_inferior": int(faixa["curtidas_inferior"][0]),
        "curtidas_superior": int(faixa["curtidas_superior"][0]),
        "central_log": float(faixa["central_log"][0]), "inferior_log": float(faixa["inferior_log"][0]),
        "superior_log": float(faixa["superior_log"][0]), "q10_log": float(faixa["q10_log"][0]),
        "q50_log": float(faixa["q50_log"][0]), "q90_log": float(faixa["q90_log"][0]),
        "cruzou": bool(faixa["cruzou"][0]), "cortada_em_zero": bool(faixa["cortada_em_zero"][0]),
        "ref_log": ref_log, "curtidas_referencia": referencia,
        "variacao_log_vs_canal": float(faixa["variacao_log_vs_canal"][0]),
        "relative_to_channel": (1 + central) / (1 + referencia),
    }


def prever_linha(pacote: Pacote, linha: Mapping) -> dict:
    """`faixa_da_linha` mais as contribuições SHAP do q50 na escala do modelo."""
    shap = fp.contribuicoes_q50(pacote.q50, [dict(linha)])
    return faixa_da_linha(pacote, linha) | {"shap": {
        "nomes": list(shap["nomes"]), "valores": [float(v) for v in shap["contribuicoes"][0]],
        "base": float(shap["base"]), "saida_bruta_relativa": float(shap["saida_bruta_relativa"][0]),
        "cortada_em_zero": bool(shap["cortada_em_zero"][0]), "transformacoes": list(shap["transformacoes"]),
    }}
