"""Treino e previsão de uma dobra (protocolo final §2 e §3).

Uma dobra recebe as linhas de UM contexto: treina com `uso_no_contexto = treino` e prevê `avaliacao`. Os três quantis
(q10, q50, q90) usam a mesma especificação e os mesmos parâmetros; só o `alpha` muda (D-D7). O central é o q50
reconstruído em log (cortado em 0); a faixa é a corrigida por `faixa_previsao` (q50 mantido; limites = mínimo e máximo
das três saídas, sem rearranjo total dos quantis). Todo ajuste (PCA, códigos de
categoria) acontece dentro de `modelo_desenvolvimento.treinar`, só nas linhas de treino.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src import faixa_previsao as fp
from src import modelo_desenvolvimento as md
from src.avaliacao import faixas as fx

ALFA_CENTRAL = 0.5


@dataclass(frozen=True)
class Dobra:
    treino: list[dict]
    avaliacao: list[dict]
    ignoradas: dict  # usos que não entram (intervalo excluído, canal retido), contados


def separar(linhas: Sequence[Mapping]) -> Dobra:
    usos = Counter(str(l.get("uso_no_contexto")) for l in linhas)
    return Dobra([dict(l) for l in linhas if l.get("uso_no_contexto") == md.USO_TREINO],
                 [dict(l) for l in linhas if l.get("uso_no_contexto") == md.USO_AVALIACAO],
                 {u: n for u, n in sorted(usos.items()) if u not in (md.USO_TREINO, md.USO_AVALIACAO)})


def treinar_q50(treino: Sequence[Mapping], espec: md.EspecificacaoAtributos,
                parametros: Mapping) -> md.ModeloTreinado:
    return md.treinar(treino, espec, parametros=dict(parametros) | {"alpha": ALFA_CENTRAL})


def treinar_quantis(treino: Sequence[Mapping], espec: md.EspecificacaoAtributos,
                    parametros: Mapping) -> dict[str, md.ModeloTreinado]:
    q50 = treinar_q50(treino, espec, parametros)
    return {"q50": q50} | fp.treinar_companheiros(treino, q50)


def prever_central(modelo: md.ModeloTreinado, avaliacao: Sequence[Mapping]) -> dict:
    relativo = np.asarray(modelo.prever_relativo(avaliacao), dtype=float)
    central, cortadas = md.reconstruir_log(avaliacao, relativo)
    return {"relativo": relativo, "central_log": central, "cortadas_em_zero": cortadas}


def prever_com_faixa(modelos: Mapping[str, md.ModeloTreinado], avaliacao: Sequence[Mapping]) -> dict:
    relativos = {q: np.asarray(modelos[q].prever_relativo(avaliacao), dtype=float) for q in ("q10", "q50", "q90")}
    faixa = fp.prever_faixa(avaliacao, relativos["q10"], relativos["q50"], relativos["q90"])
    return faixa | {"relativo": relativos["q50"]}


def bordas_de_faixa(treino: Sequence[Mapping]) -> list[float]:
    """Tercis do alvo relativo do treino da dobra, usados no F1 por faixas (protocolo §7)."""
    return fx.tercis_do_treino([l["y_rel"] for l in treino])
