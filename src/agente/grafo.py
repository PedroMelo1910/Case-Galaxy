"""Fluxo fixo do agente em LangGraph (ARQUITETURA §7) e a versão direta que reutiliza os mesmos nós.

Ordem: validar_entrada → extrair_atributos → historico_e_comparaveis → prever_modelo → avaliar_confianca → explicar →
montar_saida. Um erro previsto (`ErroAgente`) interrompe o fluxo e vira o envelope de erro; os nós seguintes não
executam. `executar_direto` percorre os mesmos nós em sequência, sem o LangGraph, para provar a equivalência
numérica entre a chamada direta e o grafo.
"""

import itertools
import logging
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src import extracao_legenda as leg
from src.agente import atributos as at
from src.agente import comparaveis as cmp
from src.agente import explicacao as ex
from src.agente import historico_canal as hc
from src.agente import previsao as pv
from src.agente.contratos import Codigo, ErroAgente, PedidoPrever, envelope_erro, envelope_ok
from src.agente.servicos import Servicos
from src.contratos import Setor

logger = logging.getLogger(__name__)
SETORES_SUPORTADOS = ("beleza", "automotivo")
ALVO = ("Curtidas acumuladas observadas, em vídeos com pelo menos 30 dias e até cerca de 1 ano de idade; "
        "não é uma previsão exata aos 30 dias.")
DEFINICAO_REFERENCIA = ("mediana das curtidas dos últimos 20 posts elegíveis do canal publicados até 30 dias antes "
                        "da data planejada")


class Estado(TypedDict, total=False):
    pedido: PedidoPrever
    erro: ErroAgente
    avisos: list[dict]
    extracao: dict
    colunas: dict
    linha: dict
    historico: hc.ResultadoHistorico
    comparaveis: list[dict]
    previsao: dict
    evidencia: dict
    explicacao: dict
    saida: dict


def _aviso(codigo: str, descricao: str, classe: str = "atencao") -> dict:
    return {"classe": classe, "codigo": codigo, "descricao": descricao}


def validar_entrada(s: Servicos, e: Estado) -> dict:
    p = e["pedido"]
    if p.sector not in SETORES_SUPORTADOS:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, f"Setor fora do recorte: {p.sector!r}; use beleza ou automotivo.",
                         {"setores_suportados": list(SETORES_SUPORTADOS)})
    s.pacote.modelo_para(p.modalidade)
    for caminho in (getattr(p.creative, "arquivo", None), p.thumbnail):
        if caminho is not None and not caminho.is_file():
            raise ErroAgente(Codigo.MIDIA_INVALIDA, f"Arquivo não encontrado: {caminho.name}")
    return {"avisos": []}


def _colunas_legenda(extracao: dict) -> dict:
    return {"leg_estado": extracao["estado"], "leg_n_tokens": extracao["n_tokens"],
            "leg_truncado": extracao["truncado"], **{c: extracao[c] for c in leg.CONTAGENS_LEGENDA},
            "leg_emb": extracao["vetor"]}


def extrair_atributos(s: Servicos, e: Estado) -> dict:
    p, avisos = e["pedido"], list(e.get("avisos", []))
    legenda = p.caption
    extr_leg = leg.extrair_legenda(legenda.title if legenda else None, legenda.description if legenda else None,
                                   p.tags, s.codificador_texto)
    if extr_leg["estado"] != "ok":
        avisos.append(_aviso("legenda_incompleta", f"Legenda: {extr_leg['estado']}; o vetor da legenda fica ausente."))
    imagem, origem_capa = at.bytes_da_capa(p, imagens_conferidas=s.imagens_conferidas, baixar=s.baixar_imagem)
    linha_capa = at.extrair_capa(imagem, s.extratores_capa, s.cache_capa)
    if origem_capa == "frame_do_video":
        avisos.append(_aviso("capa_por_frame", "Sem thumbnail: a capa foi o primeiro frame do vídeo, que pode diferir "
                                               "da capa publicada."))
    if linha_capa["capa_estado_imagem"] != "ok" or any(linha_capa[f"{x}_estado"] != "ok" for x in at.ETAPAS_CAPA):
        avisos.append(_aviso("capa_incompleta", "Capa ausente ou com extrator falho; os atributos da capa ficam "
                                                "ausentes, sem valor inventado."))
    url = getattr(p.creative, "url", None)
    anotacao, estado_gemini = s.fonte_anotacao.anotar(url, Setor(p.sector), p.duration_s)
    if anotacao is None:
        avisos.append(_aviso("atributos_gemini_ausentes",
                             f"Sem anotação do Gemini ({estado_gemini}); o modelo completo recebe esses atributos "
                             "como ausentes, não há modelo reduzido neste pacote."))
    video_id = at.video_id_da_url(url) or "criativo_planejado"
    gem = at.colunas_gemini(anotacao, p.sector, s.codificador_transcricao, s.pasta_cache_transcricao, video_id)
    if p.duration_s is None:
        avisos.append(_aviso("duracao_ausente", "Duração não informada; o atributo fica ausente."))
    extracao = {"legenda": extr_leg["estado"], "capa_origem": origem_capa, "capa": linha_capa["capa_estado_imagem"],
                "gemini": estado_gemini, "transcricao": gem.get("gem_transc_status"),
                "embedding_transcricao": gem.get(at.ds.COLUNA_ESTADO_EMBEDDING)}
    return {"avisos": avisos, "extracao": extracao, "colunas": {**_colunas_legenda(extr_leg), **linha_capa, **gem}}


def historico_e_comparaveis(s: Servicos, e: Estado) -> dict:
    p = e["pedido"]
    resultado = hc.obter_historico(p, s.base, consultor=s.consultor_historico, agora=s.relogio())
    linha = {**resultado.linha, **e["colunas"]}
    linha["tipo_canal"] = resultado.tipo_canal
    url = getattr(p.creative, "url", None)
    comparaveis = cmp.selecionar(s.linhas_desenvolvimento, linha, channel_id=p.channel_id, setor=p.sector,
                                 planejada=p.planned_datetime, excluir_video_id=at.video_id_da_url(url))
    return {"linha": linha, "historico": resultado, "comparaveis": comparaveis,
            "avisos": [*e.get("avisos", []), *resultado.avisos]}


def prever_modelo(s: Servicos, e: Estado) -> dict:
    espec = s.pacote.q50.especificacao
    exigidas = [*espec.escalares, *espec.categoricas, *espec.embeddings, "ref_log"]
    faltam = [c for c in exigidas if c not in e["linha"]]
    if faltam:
        raise ErroAgente(Codigo.ERRO_INTERNO, "A linha de inferência não tem todas as colunas do modelo.",
                         {"faltam": faltam[:10]})
    return {"previsao": pv.prever_linha(s.pacote, e["linha"])}


def avaliar_confianca(s: Servicos, e: Estado) -> dict:
    avisos, pred = list(e.get("avisos", [])), e["previsao"]
    avisos.append(_aviso("modelo_provisorio", "Modelo provisório de desenvolvimento: configuração escolhida em uma "
                                              "busca única, sem avaliação final nem teste."))
    avisos.append(_aviso("faixa_sem_cobertura_suficiente", s.pacote.declaracao_faixa["aviso"]))
    avisos.append(_aviso("sem_sinal_de_confianca_da_lim", "Não há limiar estatístico de confiança medido na amostra "
                                                           "LIM; nenhum sinal de confiança é afirmado.", "info"))
    if pred["cortada_em_zero"]:
        avisos.append(_aviso("previsao_cortada_em_zero", "A saída bruta do modelo ficou abaixo de zero e foi cortada "
                                                          "em 0 curtidas."))
    if pred["cruzou"]:
        avisos.append(_aviso("quantis_cruzados", "q10, q50 e q90 se cruzaram; a faixa usa mínimo e máximo das três "
                                                  "saídas e mantém o q50 como central.", "info"))
    return {"avisos": avisos}


def explicar(s: Servicos, e: Estado) -> dict:
    codigos = [a["codigo"] for a in e["avisos"]]
    evidencia = ex.construir_evidencia(e["previsao"], e["linha"], codigos)
    return {"evidencia": evidencia, "explicacao": ex.explicar(evidencia, s.subagente)}


def montar_saida(s: Servicos, e: Estado) -> dict:
    p, pred, h, expl = e["pedido"], e["previsao"], e["historico"], e["explicacao"]
    evidencia = e["evidencia"]
    saida = {
        "input_modality": p.modalidade, "model_level": "completo (desenvolvimento, provisório)",
        "predicted_metric": pred["curtidas_central"], "interval_low": pred["curtidas_inferior"],
        "interval_high": pred["curtidas_superior"],
        "interval_nominal_level": s.pacote.declaracao_faixa["nivel_nominal"],
        "coverage_estimate": s.pacote.declaracao_faixa,
        "relative_to_channel": pred["relative_to_channel"], "reference_likes": pred["curtidas_referencia"],
        "reference_source": "canal" if h.linha["estado_referencia"] == "canal" else "setor",
        "reference_definition": DEFINICAO_REFERENCIA, "target_definition": ALVO,
        "history": h.resumo(), "retrospective": p.planned_datetime <= s.base.coletado_em,
        "warnings": e["avisos"], "top_drivers": evidencia["drivers"], "family_contributions": evidencia["familias"],
        "explanation": expl["texto"], "explanation_source": expl["fonte"],
        "explanation_fallback_reason": expl["motivo_fallback"], "comparables": e["comparaveis"],
        "extraction": e["extracao"], "model_version": s.pacote.versao, "bundle_id": s.pacote.identificacao,
        "numeric_detail": {k: pred[k] for k in (
            "central_log", "inferior_log", "superior_log", "q10_log", "q50_log", "q90_log", "ref_log",
            "variacao_log_vs_canal", "cruzou", "cortada_em_zero")},
    }
    return {"saida": saida}


NOS: tuple[tuple[str, Callable[[Servicos, Estado], dict]], ...] = (
    ("validar_entrada", validar_entrada), ("extrair_atributos", extrair_atributos),
    ("historico_e_comparaveis", historico_e_comparaveis), ("prever_modelo", prever_modelo),
    ("avaliar_confianca", avaliar_confianca), ("explicar", explicar), ("montar_saida", montar_saida),
)


def _protegido(nome: str, funcao: Callable[[Servicos, Estado], dict], servicos: Servicos):
    def no(estado: Estado) -> dict:
        if estado.get("erro") is not None:
            return {}
        try:
            return funcao(servicos, estado)
        except ErroAgente as erro:
            return {"erro": erro}
        except Exception as falha:  # noqa: BLE001 - nunca vaza rastro nem chave: vira ERRO_INTERNO tipado
            logger.exception("falha na etapa %s do agente", nome)
            return {"erro": ErroAgente(Codigo.ERRO_INTERNO, "Falha interna no fluxo do agente.",
                                       {"etapa": nome, "tipo": type(falha).__name__})}
    return no


def construir_grafo(servicos: Servicos):
    construtor = StateGraph(Estado)
    for nome, funcao in NOS:
        construtor.add_node(nome, _protegido(nome, funcao, servicos))
    construtor.add_edge(START, NOS[0][0])
    for (a, _), (b, _) in itertools.pairwise(NOS):
        construtor.add_edge(a, b)
    construtor.add_edge(NOS[-1][0], END)
    return construtor.compile()


def _envelope(estado: Estado) -> dict:
    erro = estado.get("erro")
    return envelope_erro(erro) if erro is not None else envelope_ok(estado["saida"])


def executar_grafo(servicos: Servicos, pedido: PedidoPrever, grafo=None) -> dict:
    grafo = grafo or construir_grafo(servicos)
    return _envelope(grafo.invoke({"pedido": pedido}))


def executar_direto(servicos: Servicos, pedido: PedidoPrever) -> dict:
    estado: Estado = {"pedido": pedido}
    for nome, funcao in NOS:
        estado.update(_protegido(nome, funcao, servicos)(estado))
    return _envelope(estado)
