"""Operações públicas do agente: `prever` e `ordenar`. É a fronteira que o MCP, o frontend e a CLI chamam.

Ambas recebem o pedido tipado (ou um dict validado pelo mesmo contrato) e devolvem o envelope
`{success, data|error}`. Nenhuma operação levanta exceção para o chamador.
"""

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from src.agente.contratos import (
    MAX_CRIATIVOS_ORDENAR,
    MIN_CRIATIVOS_ORDENAR,
    Codigo,
    ErroAgente,
    PedidoExtrairAudiovisual,
    PedidoOrdenar,
    PedidoPrever,
    envelope_erro,
    envelope_ok,
)
from src.modelos import pacote as ms
from src.agente import atributos as rl
from src.agente.grafo import construir_grafo, executar_grafo
from src.agente.servicos import Servicos
from src.config import RAIZ_PROJETO
from src.extracao.modelos import ErroExtracao
from src.extracao.modelos import ModeloAusente, ModeloCorrompido

logger = logging.getLogger(__name__)

PREDICAO_AUDIOVISUAL_INDISPONIVEL = {
    "disponivel": False, "modelo": None,
    "motivo": "O modelo preditivo que usa estes atributos audiovisuais locais ainda não foi treinado: não há mídia de "
              "treino autorizada mapeada a resultados observados. Esta resposta traz atributos, não previsão."}


def predicao_audiovisual_ativa(raiz: Path = RAIZ_PROJETO) -> dict:
    """Bloco `predicao` da extração: diz, pelo manifesto de pacote ativo, se há preditor local e como obter a previsão.

    Lê só o manifesto (nenhum modelo é carregado). Sem preditor local ativo, ou com manifesto ilegível, vale o texto
    de indisponibilidade."""
    try:
        entrada = ms.versao_ativa(raiz)["pacotes"].get(rl.ROTA_VIDEO_LOCAL)
    except ms.ErroManifesto:
        return dict(PREDICAO_AUDIOVISUAL_INDISPONIVEL)
    if entrada is None:
        return dict(PREDICAO_AUDIOVISUAL_INDISPONIVEL)
    return {"disponivel": True, "modelo": entrada.get("versao_servida"), "rota": rl.ROTA_VIDEO_LOCAL,
            "ferramenta": "prever_criativo",
            "motivo": "Esta ferramenta só extrai atributos. A previsão de curtidas deste arquivo vem do preditor local "
                      "ativo: chame `prever_criativo` com o mesmo arquivo em `creative.arquivo` (`video_route` = "
                      "`auto` ou `local`). A extração já feita fica em cache e não é repetida."}


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


def _compactar_extracao(resultado: Mapping) -> dict:
    """O resultado da extração sem duplicar os vetores de embedding (eles já estão em `colunas`)."""
    etapas = {nome: dict(etapa) for nome, etapa in resultado["etapas"].items()}
    if "embedding" in etapas:
        vetores = [etapas["embedding"].pop(chave, None) for chave in ("emb_media", "emb_gancho")]
        vetor = next((v for v in vetores if v), None)
        if vetor is not None:  # sem vetor (etapa que falhou) não há dimensão: nada de zero fabricado
            etapas["embedding"]["dimensao"] = len(vetor)
    return {**resultado, "etapas": etapas}


def extrair_audiovisual(pedido: PedidoExtrairAudiovisual | Mapping, extrator, *,
                        predicao: Callable[[], dict] | None = None) -> dict:
    """Extrai atributos audiovisuais de um vídeo local (offline). Devolve atributos, nunca uma previsão.

    `extrator.extrair(caminho)` é o `ExtratorAudiovisual`; o envelope é o mesmo do agente. O bloco `predicao` diz se há
    preditor local ativo (`predicao()`, normalmente `predicao_audiovisual_ativa`) e aponta a ferramenta de previsão; sem
    ele, diz que o preditor não está disponível. Esta função não faz fallback para o modelo de imagem.
    """
    try:
        entrada = _validar(PedidoExtrairAudiovisual, pedido)
        resultado = extrator.extrair(entrada.arquivo)
    except ErroAgente as erro:
        return envelope_erro(erro)
    except ErroExtracao as erro:
        return envelope_erro(rl.erro_de_extracao(erro))
    except (ModeloAusente, ModeloCorrompido) as erro:
        return envelope_erro(ErroAgente(Codigo.MODELO_INDISPONIVEL, str(erro), {"modelo": erro.nome}))
    except Exception as falha:
        return envelope_erro(_interno(falha))
    bloco = predicao() if predicao is not None else dict(PREDICAO_AUDIOVISUAL_INDISPONIVEL)
    return envelope_ok({**_compactar_extracao(resultado), "predicao": bloco})


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


def _mesma_rota_pedida(pedidos: tuple[PedidoPrever, ...], servicos: Servicos) -> None:
    """Rotas diferentes (local e Gemini) não formam comparação validada: recusa ANTES de executar qualquer criativo,
    para que um vídeo da rota Gemini não gaste cota num ranking que seria recusado."""
    rotas = [rl.resolver_rota(servicos, p) for p in pedidos]
    if len(set(rotas)) > 1:
        raise ErroAgente(Codigo.ROTAS_DIFERENTES,
                         "Os criativos seguem rotas de previsão diferentes (local e Gemini): modelos diferentes não "
                         "formam uma comparação validada. Use a mesma rota em todos (`video_route`).",
                         {"rotas": [{"indice": i, "rota": r} for i, r in enumerate(rotas)]})


def _mesma_rota_resposta(ordenaveis: list[tuple[int, dict]]) -> None:
    # O fallback para imagem é a rota Gemini em outro nível: quem recusa essa diferença é `_mesmo_nivel`.
    rotas = {rl.ROTA_VIDEO_GEMINI if d["route"]["id"] == rl.ROTA_VIDEO_GEMINI_FALLBACK else d["route"]["id"]
             for _, d in ordenaveis if "route" in d}
    if len(rotas) > 1:
        raise ErroAgente(Codigo.ROTAS_DIFERENTES,
                         "Os criativos foram avaliados por rotas diferentes (por exemplo, um vídeo que caiu no modelo "
                         "de imagem): não há ranking validado entre eles.",
                         {"itens": [{"indice": i, "rota": d["route"]["id"], "model_level": d["model_level"]}
                                    for i, d in ordenaveis]})


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
        _mesma_rota_pedida(entrada.creatives, servicos)
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
        _mesma_rota_resposta(ordenaveis)
        _mesmo_nivel(ordenaveis)
    except ErroAgente as erro:
        return envelope_erro(erro)
    classificados = sorted(ordenaveis, key=lambda par: (-par[1]["numeric_detail"]["q50_log"], par[0]))
    ranking = [{"posicao": pos, "indice": i, "input_modality": d["input_modality"],
                "model_version": d["model_version"], "model_level": d["model_level"],
                "route_id": d.get("route", {}).get("id"), "predicted_metric": d["predicted_metric"], "interval_low": d["interval_low"],
                "interval_high": d["interval_high"], "relative_to_channel": d["relative_to_channel"]}
               for pos, (i, d) in enumerate(classificados, 1)]
    return envelope_ok({
        "criterio": "ordem decrescente do valor central (q50); empates mantêm a ordem de envio; só criativos do "
                    "mesmo canal e da mesma modalidade",
        "ranking": ranking, "nao_avaliados": falhas,
        "resultados": [r["data"] if r["success"] else None for r in resultados],
    })
