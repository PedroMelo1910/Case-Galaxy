"""Operações públicas do agente: `prever` e `ordenar`. É a fronteira que o MCP envolverá depois (não implementado).

Ambas recebem o pedido tipado (ou um dict validado pelo mesmo contrato) e devolvem o envelope
`{success, data|error}`. Nenhuma operação levanta exceção para o chamador.
"""

import logging
from collections.abc import Mapping

from pydantic import ValidationError

from src.agente.contratos import (
    MAX_CRIATIVOS_ORDENAR,
    MIN_CRIATIVOS_ORDENAR,
    Codigo,
    ErroAgente,
    PedidoOrdenar,
    PedidoPrever,
    envelope_erro,
    envelope_ok,
)
from src.agente.grafo import construir_grafo, executar_grafo
from src.agente.servicos import Servicos

logger = logging.getLogger(__name__)


def _validar(modelo, dados):
    try:
        return dados if isinstance(dados, modelo) else modelo.model_validate(dados)
    except ValidationError as erro:
        detalhes = [{"campo": ".".join(str(x) for x in e["loc"]), "problema": e["msg"]} for e in erro.errors()]
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, "Entrada inválida.", {"erros": detalhes}) from None


def _interno(falha: Exception) -> ErroAgente:
    logger.exception("falha inesperada na fronteira do agente")
    return ErroAgente(Codigo.ERRO_INTERNO, "Falha interna no agente.", {"tipo": type(falha).__name__})


def prever(pedido: PedidoPrever | Mapping, servicos: Servicos) -> dict:
    try:
        return executar_grafo(servicos, _validar(PedidoPrever, pedido))
    except ErroAgente as erro:
        return envelope_erro(erro)
    except Exception as falha:
        return envelope_erro(_interno(falha))


def _mesmo_canal(pedidos: tuple[PedidoPrever, ...]) -> None:
    ids = {p.channel_id for p in pedidos}
    if None in ids:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, "ordenar exige `channel_id` em todos os criativos.")
    if len(ids) != 1:
        raise ErroAgente(Codigo.CANAIS_DIFERENTES, "Os criativos precisam ser do mesmo canal para serem ordenados.",
                         {"canais": sorted(ids)})


def _mesma_modalidade(pedidos: tuple[PedidoPrever, ...]) -> None:
    """Modelos diferentes não formam uma comparação validada: o ranking vale só dentro de uma modalidade."""
    modalidades = {p.modalidade for p in pedidos}
    if len(modalidades) != 1:
        raise ErroAgente(Codigo.MODALIDADES_DIFERENTES,
                         "Os criativos precisam ser da mesma modalidade para serem ordenados: previsões de modelos "
                         "diferentes não formam uma comparação validada.", {"modalidades": sorted(modalidades)})


def _mesmo_nivel(ordenaveis: list[tuple[int, dict]]) -> None:
    """Dentro de uma modalidade, um vídeo pode ter caído no modelo de imagem (fallback para N3): níveis diferentes são
    modelos diferentes e não formam comparação validada. Recusa, com o que cada item recebeu (sem classificar)."""
    if len({d["model_level"] for _, d in ordenaveis}) <= 1:
        return
    itens = [{"indice": i, "model_level": d["model_level"], "model_version": d["model_version"],
              "fallback": d.get("fallback")} for i, d in ordenaveis]
    raise ErroAgente(Codigo.NIVEIS_DIFERENTES,
                     "Os criativos foram avaliados por modelos de níveis diferentes (por exemplo, um vídeo sem os "
                     "atributos do Gemini caiu no modelo de imagem): não há ranking validado entre eles. Reenvie quando "
                     "todos puderem ser avaliados pelo mesmo modelo.", {"itens": itens})


def ordenar(pedido: PedidoOrdenar | Mapping, servicos: Servicos) -> dict:
    try:
        entrada = _validar(PedidoOrdenar, pedido)
        n = len(entrada.creatives)
        if not MIN_CRIATIVOS_ORDENAR <= n <= MAX_CRIATIVOS_ORDENAR:
            raise ErroAgente(Codigo.ENTRADA_INVALIDA,
                             f"Envie de {MIN_CRIATIVOS_ORDENAR} a {MAX_CRIATIVOS_ORDENAR} criativos.", {"recebidos": n})
        _mesmo_canal(entrada.creatives)
        _mesma_modalidade(entrada.creatives)
        grafo = construir_grafo(servicos)
        resultados = [executar_grafo(servicos, p, grafo) for p in entrada.creatives]
    except ErroAgente as erro:
        return envelope_erro(erro)
    except Exception as falha:
        return envelope_erro(_interno(falha))
    ordenaveis = [(i, r["data"]) for i, r in enumerate(resultados) if r["success"]]
    falhas = [{"indice": i, **r["error"]} for i, r in enumerate(resultados) if not r["success"]]
    if len(ordenaveis) < MIN_CRIATIVOS_ORDENAR:
        return envelope_erro(ErroAgente(Codigo.ENTRADA_INVALIDA,
                                        "Menos de dois criativos puderam ser avaliados; não há ranking.",
                                        {"falhas": falhas}))
    try:
        _mesmo_nivel(ordenaveis)
    except ErroAgente as erro:
        return envelope_erro(erro)
    classificados = sorted(ordenaveis, key=lambda par: (-par[1]["numeric_detail"]["q50_log"], par[0]))
    ranking = [{"posicao": pos, "indice": i, "input_modality": d["input_modality"],
                "model_version": d["model_version"], "model_level": d["model_level"],
                "predicted_metric": d["predicted_metric"], "interval_low": d["interval_low"],
                "interval_high": d["interval_high"], "relative_to_channel": d["relative_to_channel"]}
               for pos, (i, d) in enumerate(classificados, 1)]
    return envelope_ok({
        "criterio": "ordem decrescente do valor central (q50); empates mantêm a ordem de envio; só criativos do "
                    "mesmo canal e da mesma modalidade",
        "ranking": ranking, "nao_avaliados": falhas,
        "resultados": [r["data"] if r["success"] else None for r in resultados],
    })
