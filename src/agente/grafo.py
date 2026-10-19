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
from src import familias_atributos as fa
from src.agente import atributos as at
from src.agente import avisos_gemini as ag
from src.agente import comparaveis as cmp
from src.agente import entrada_texto as et
from src.agente import explicacao as ex
from src.agente import historico_canal as hc
from src.agente import previsao as pv
from src.agente.contratos import (
    Codigo,
    ErroAgente,
    PedidoPrever,
    envelope_erro,
    envelope_ok,
    id_do_proprio_video,
)
from src.agente.servicos import Servicos
from src.agente.video_local import MAX_DURACAO_S, SondaVideo, sondar_video
from src.avaliacao import driver_lim as dl
from src.contratos import Setor

logger = logging.getLogger(__name__)
SETORES_SUPORTADOS = ("beleza", "automotivo")
ALVO = ("Curtidas acumuladas observadas, em vídeos com pelo menos 30 dias e até cerca de 1 ano de idade; "
        "não é uma previsão exata aos 30 dias.")
DEFINICAO_REFERENCIA = ("mediana das curtidas dos últimos 20 posts elegíveis do canal publicados até 30 dias antes "
                        "da data planejada")


TEMPO_MATERIAL_PUBLICADO = ("A data planejada já passou: o material já foi publicado. Esta saída é uma demonstração "
                            "operacional com material existente, não é previsão feita antes da publicação.")
TEMPO_MATERIAL_FUTURO = "A data planejada ainda não ocorreu: previsão anterior à publicação."
INTERPRETACAO_DA_ENTRADA = {"texto": "texto interpretado como legenda planejada",
                            "imagem": "imagem interpretada como capa do criativo planejado"}
MODALIDADE_DO_FALLBACK = "imagem"  # D-F5: Gemini falhou → N3 (histórico, legenda e capa) com evidência insuficiente
TOLERANCIA_DURACAO_INFORMADA_S = 1


def material_ja_publicado(s: Servicos, p: PedidoPrever) -> bool:
    """A data planejada já ocorreu segundo o relógio do serviço; não depende da data do snapshot do histórico."""
    return p.planned_datetime <= s.relogio()


class Estado(TypedDict, total=False):
    pedido: PedidoPrever
    erro: ErroAgente
    avisos: list[dict]
    extracao: dict
    colunas: dict
    linha: dict
    historico: hc.ResultadoHistorico
    comparaveis: list[dict]
    sonda: SondaVideo
    fallback: dict
    previsao: dict
    confianca: dict
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
    s.pacote_para(p.modalidade)
    et.legenda_planejada(p)  # texto divergente da legenda, ou em branco, é erro de entrada antes de qualquer extração
    for caminho in (getattr(p.creative, "arquivo", None), p.thumbnail):
        if caminho is not None and not caminho.is_file():
            raise ErroAgente(Codigo.MIDIA_INVALIDA, f"Arquivo não encontrado: {caminho.name}")
    saida: dict = {"avisos": _avisos_de_entradas_ignoradas(p)}
    if p.modalidade == "video":
        saida |= _validar_video(p, saida["avisos"])
    return saida


def _validar_video(p: PedidoPrever, avisos: list[dict]) -> dict:
    """Regra do Short (até 180 s, vertical ou quadrado) ANTES de qualquer rede. Arquivo local: a sonda manda na duração."""
    if p.duration_s is not None and p.duration_s > MAX_DURACAO_S:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA,
                         f"A duração informada passa de {MAX_DURACAO_S} s; o agente aceita Shorts de até {MAX_DURACAO_S} s.",
                         {"duration_s": p.duration_s, "limite_s": MAX_DURACAO_S})
    arquivo = getattr(p.creative, "arquivo", None)
    if arquivo is None:
        return {}
    sonda = sondar_video(arquivo)
    duracao = max(1, round(sonda.duracao_s))
    if p.duration_s is not None and abs(p.duration_s - duracao) > TOLERANCIA_DURACAO_INFORMADA_S:
        avisos.append(_aviso("duracao_do_arquivo_prevalece",
                             f"A duração informada ({p.duration_s} s) difere da medida no arquivo ({duracao} s); vale a "
                             "do arquivo.", "info"))
    return {"sonda": sonda, "pedido": p.model_copy(update={"duration_s": duracao})}


def _avisos_de_entradas_ignoradas(p: PedidoPrever) -> list[dict]:
    """Entradas que os modelos de imagem e de texto não usam ficam avisadas, nunca ignoradas em silêncio."""
    if p.modalidade == "video":
        return []
    avisos = []
    if p.duration_s is not None:
        avisos.append(_aviso("duracao_ignorada", "A duração informada não é usada pelos modelos de imagem e de texto.",
                             "info"))
    if p.modalidade == "texto" and p.thumbnail is not None:
        avisos.append(_aviso("thumbnail_ignorada", "O modelo de texto não usa capa: a thumbnail informada foi ignorada.",
                             "info"))
    if p.modalidade == "imagem" and p.thumbnail is not None:
        avisos.append(_aviso("thumbnail_ignorada", "A imagem do criativo é a capa planejada: a thumbnail informada "
                                                   "foi ignorada.", "info"))
    return avisos


def _colunas_legenda(extracao: dict) -> dict:
    return {"leg_estado": extracao["estado"], "leg_n_tokens": extracao["n_tokens"],
            "leg_truncado": extracao["truncado"], **{c: extracao[c] for c in leg.CONTAGENS_LEGENDA},
            "leg_emb": extracao["vetor"]}


def extrair_atributos(s: Servicos, e: Estado) -> dict:
    if e["pedido"].modalidade == "video":
        return _extrair_video(s, e)
    return _extrair_reduzida(s, e)


def _extrair_reduzida(s: Servicos, e: Estado) -> dict:
    """Imagem e texto: só a legenda planejada (e a capa, na imagem). Sem Gemini, sem transcrição e sem duração."""
    p, avisos = e["pedido"], list(e.get("avisos", []))
    legenda = et.legenda_planejada(p)
    extr_leg = leg.extrair_legenda(legenda.title if legenda else None, legenda.description if legenda else None,
                                   p.tags, s.codificador_texto)
    if extr_leg["estado"] != "ok":
        if p.modalidade == "texto":  # o texto é a única entrada: sem ele a previsão seria só histórico, sob o nome do texto
            raise ErroAgente(
                Codigo.ENTRADA_INVALIDA if extr_leg["estado"] == "texto_vazio" else Codigo.MODELO_INDISPONIVEL,
                f"A legenda planejada não pôde ser processada ({extr_leg['estado']}).",
                {"estado": extr_leg["estado"], "erro": extr_leg.get("erro")})
        avisos.append(_aviso("legenda_incompleta", f"Legenda: {extr_leg['estado']}; o vetor da legenda fica ausente."))
    colunas, origem_capa, estado_capa = _colunas_legenda(extr_leg), "nao_aplicavel", "nao_aplicavel"
    if p.modalidade == "imagem":
        # A imagem do criativo é a capa planejada; uma `thumbnail` enviada junto é ignorada (avisada na validação).
        imagem, origem_capa = at.bytes_da_capa(p.model_copy(update={"thumbnail": None}),
                                               imagens_conferidas=s.imagens_conferidas, baixar=s.baixar_imagem)
        linha_capa = at.extrair_capa(imagem, s.extratores_capa, s.cache_capa)
        estado_capa, colunas = linha_capa["capa_estado_imagem"], colunas | linha_capa
        if estado_capa in ("imagem_corrompida", "sem_imagem"):
            raise ErroAgente(Codigo.MIDIA_INVALIDA, "A imagem do criativo não pôde ser lida ou está corrompida.",
                             {"estado": estado_capa})
        if estado_capa != "ok" or any(linha_capa[f"{x}_estado"] != "ok" for x in at.ETAPAS_CAPA):
            avisos.append(_aviso("capa_incompleta", "Capa ausente ou com extrator falho; os atributos da capa ficam "
                                                    "ausentes, sem valor inventado."))
    extracao = {"legenda": extr_leg["estado"], "capa_origem": origem_capa, "capa": estado_capa,
                "gemini": "nao_aplicavel", "transcricao": None, "embedding_transcricao": None}
    return {"avisos": avisos, "extracao": extracao, "colunas": colunas}


def _anotar_video(s: Servicos, e: Estado) -> tuple[dict | None, str, list[dict]]:
    """(anotação, estado, avisos): cache primeiro; depois URL pública ou Files API, conforme a origem do vídeo."""
    p = e["pedido"]
    arquivo = getattr(p.creative, "arquivo", None)
    if arquivo is None:
        anotacao, estado = s.fonte_anotacao.anotar(getattr(p.creative, "url", None), Setor(p.sector), p.duration_s)
        return anotacao, estado, []
    if s.fonte_arquivo is None:
        return None, "upload_nao_configurado", []
    resultado = s.fonte_arquivo.anotar(e["sonda"], Setor(p.sector))
    return resultado.anotacao, resultado.estado, list(resultado.avisos)


def _fallback_para_imagem(s: Servicos, estado_gemini: str, linha_capa: dict) -> dict:
    """D-F5 / ARQUITETURA §12: sem os atributos do Gemini, o vídeo é servido pelo modelo de imagem (N3).

    O modelo de imagem só vale com a capa: sem ela (ou sem pacote de imagem) a previsão seria só histórico e legenda
    sob o nome de um modelo de imagem, então é erro explícito, nunca previsão silenciosa.
    """
    cota = estado_gemini == "falha_cota_esgotada"
    try:
        s.pacote_para(MODALIDADE_DO_FALLBACK)
    except ErroAgente as erro:
        raise ErroAgente(erro.codigo, "Sem os atributos do Gemini e sem modelo de imagem configurado: não há modelo "
                         "aplicável a este vídeo.", {"gemini": estado_gemini}) from None
    if linha_capa["capa_estado_imagem"] != "ok":
        raise ErroAgente(
            Codigo.COTA_ESGOTADA if cota else Codigo.MIDIA_INVALIDA,
            "Sem os atributos do Gemini e sem a capa do vídeo: não há entrada visual para nenhum modelo. Envie "
            "`thumbnail` ou habilite o modo online do servidor.",
            {"gemini": estado_gemini, "capa": linha_capa["capa_estado_imagem"]})
    return {"de": "video", "para": MODALIDADE_DO_FALLBACK, "motivo": estado_gemini}


def _avisos_do_fallback(estado_gemini: str) -> list[dict]:
    causa = ag.aviso_do_estado(estado_gemini)
    principal = _aviso("fallback_video_para_imagem",
                       f"Sem anotação do Gemini ({estado_gemini}): a previsão veio do modelo de imagem (histórico, "
                       "legenda e capa), não do modelo completo de vídeo. Evidência insuficiente.")
    return [principal, *([causa] if causa else [])]


def _avisos_de_transcricao(gem: dict) -> list[dict]:
    avisos = []
    if gem.get("gem_transc_status") == "incerta":
        avisos.append(_aviso("transcricao_incerta", "Transcrição incerta: os atributos de transcrição ficam ausentes, "
                                                    "sem tratar como \"sem fala\"."))
    if gem.get(at.ds.COLUNA_ESTADO_EMBEDDING) == "falha_extrator":
        avisos.append(_aviso("transcricao_embedding_falhou", "O codificador da transcrição falhou: o embedding da "
                                                             "transcrição fica ausente."))
    return avisos


def _extrair_video(s: Servicos, e: Estado) -> dict:
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
    anotacao, estado_gemini, avisos_fonte = _anotar_video(s, e)
    avisos.extend(avisos_fonte)
    extracao = {"legenda": extr_leg["estado"], "capa_origem": origem_capa, "capa": linha_capa["capa_estado_imagem"],
                "gemini": estado_gemini}
    if anotacao is None:
        fallback = _fallback_para_imagem(s, estado_gemini, linha_capa)
        avisos.extend(_avisos_do_fallback(estado_gemini))
        extracao |= {"transcricao": None, "embedding_transcricao": None}
        return {"avisos": avisos, "extracao": extracao, "fallback": fallback,
                "colunas": {**_colunas_legenda(extr_leg), **linha_capa}}
    url = getattr(p.creative, "url", None)
    video_id = at.video_id_da_url(url) or "criativo_planejado"
    gem = at.colunas_gemini(anotacao, p.sector, s.codificador_transcricao, s.pasta_cache_transcricao, video_id)
    avisos.extend(_avisos_de_transcricao(gem))
    if p.duration_s is None:
        avisos.append(_aviso("duracao_ausente", "Duração não informada; o atributo fica ausente."))
    extracao |= {"transcricao": gem.get("gem_transc_status"), "embedding_transcricao": gem.get(at.ds.COLUNA_ESTADO_EMBEDDING)}
    return {"avisos": avisos, "extracao": extracao, "colunas": {**_colunas_legenda(extr_leg), **linha_capa, **gem}}


def historico_e_comparaveis(s: Servicos, e: Estado) -> dict:
    p = e["pedido"]
    resultado = hc.obter_historico(p, s.base, consultor=s.consultor_historico, agora=s.relogio())
    linha = {**resultado.linha, **e["colunas"]}
    linha["tipo_canal"] = resultado.tipo_canal
    comparaveis = cmp.selecionar(s.colecao_de_referencia, linha, channel_id=p.channel_id, setor=p.sector,
                                 planejada=p.planned_datetime, excluir_video_id=id_do_proprio_video(p),
                                 origem=s.origem_da_colecao)
    return {"linha": linha, "historico": resultado, "comparaveis": comparaveis,
            "avisos": [*e.get("avisos", []), *resultado.avisos]}


def pacote_efetivo(s: Servicos, e: Estado):
    """Pacote que de fato prevê: o da modalidade, ou o de imagem quando o vídeo caiu no fallback para N3."""
    return s.pacote_para(MODALIDADE_DO_FALLBACK if e.get("fallback") else e["pedido"].modalidade)


def prever_modelo(s: Servicos, e: Estado) -> dict:
    pacote = pacote_efetivo(s, e)
    espec = pacote.q50.especificacao
    exigidas = [*espec.escalares, *espec.categoricas, *espec.embeddings, "ref_log"]
    faltam = [c for c in exigidas if c not in e["linha"]]
    if faltam:
        raise ErroAgente(Codigo.ERRO_INTERNO, "A linha de inferência não tem todas as colunas do modelo.",
                         {"faltam": faltam[:10]})
    return {"previsao": pv.prever_linha(pacote, e["linha"])}


def confianca_estatistica(pacote, pred: dict) -> dict:
    """Regra de largura (D-F2 b): só com limiar vinculado a ESTE pacote (vídeo final); senão, não se aplica."""
    if pacote.limiar is None:
        return {"aplicavel": False, "motivo": "sem limiar de largura vinculado a este pacote (imagem, texto, fallback "
                                              "ou pacote sem amostra LIM)"}
    largura = float(pred["superior_log"] - pred["inferior_log"])
    sinalizada = bool(dl.sinalizar_com(pacote.limiar, pacote.identidade_sha256, [largura])[0])
    return {"aplicavel": True, "regra": "largura da faixa (log) acima do percentil 80 das larguras da LIM neste pacote",
            "limiar_log": pacote.limiar.valor, "largura_log": largura, "sinalizada": sinalizada,
            "n_lim": pacote.limiar.n, "pacote_vinculado": pacote.identidade_sha256,
            "validade": "presumida e declarada no pacote servido; não é garantia de cobertura nem de desempenho",
            "diagnostico_de_desenvolvimento": dict(pacote.detalhe_do_limiar or {})}


def _texto_do_limiar(c: dict) -> str:
    base = (f"Regra de baixa confiança por largura: limiar {c['limiar_log']:.4f} em log (percentil 80 das larguras da "
            f"faixa deste pacote em {c['n_lim']} vídeos da LIM, sem rótulos). A validade da regra neste pacote é "
            "presumida e declarada.")
    d = c["diagnostico_de_desenvolvimento"]
    if d:
        base += (f" No pacote de desenvolvimento o diagnóstico foi {d['leitura']} (erro médio {d['mae_sinalizados']:.3f} "
                 f"nos sinalizados contra {d['mae_nao_sinalizados']:.3f}; IC 95% da diferença "
                 f"[{d['ic95_da_diferenca'][0]:.3f}; {d['ic95_da_diferenca'][1]:.3f}]).")
    return base + " Não garante cobertura nem desempenho."


def _avisos_do_modelo(pacote) -> list[dict]:
    if pacote.eh_final:
        avisos = [_aviso("modelo_refit_final", "Modelo retreinado com todos os vídeos anotados da amostra (inclusive os "
                         "de teste), configuração A fixa, sem busca nem calibração. A avaliação final descreve o "
                         "procedimento, não este artefato.")]
    elif pacote.modalidade == "video":
        avisos = [_aviso("modelo_provisorio", "Modelo provisório de desenvolvimento: configuração escolhida em uma "
                         "busca única, sem avaliação final nem teste.")]
    else:
        avisos = [_aviso("modelo_provisorio", "Modelo reduzido provisório de desenvolvimento: configuração A fixa, "
                         "herdada do modelo de vídeo, sem busca, calibração nem avaliação final.")]
    if pacote.modalidade != "video":
        avisos.append(_aviso("evidencia_limitada_sem_video",
                             "Evidência limitada: sem informação do vídeo (conteúdo, transcrição e atributos do "
                             f"Gemini), a previsão usa só {fa.MODALIDADES[pacote.modalidade].descricao}."))
    return avisos


def avaliar_confianca(s: Servicos, e: Estado) -> dict:
    avisos, pred = list(e.get("avisos", [])), e["previsao"]
    pacote = pacote_efetivo(s, e)
    avisos.extend(_avisos_do_modelo(pacote))
    avisos.append(_aviso("faixa_sem_cobertura_suficiente", pacote.declaracao_faixa["aviso"]))
    confianca = confianca_estatistica(pacote, pred)
    if not confianca["aplicavel"]:
        avisos.append(_aviso("sem_sinal_de_confianca_da_lim", "Não há limiar estatístico de confiança medido na amostra "
                                                               "LIM; nenhum sinal de confiança é afirmado.", "info"))
    else:
        avisos.append(_aviso("limiar_de_largura_presumido", _texto_do_limiar(confianca), "info"))
        if confianca["sinalizada"]:
            avisos.append(_aviso("baixa_confianca_estatistica",
                                 f"Baixa confiança estatística: a faixa ({confianca['largura_log']:.3f} em log) é mais "
                                 f"larga que o limiar ({confianca['limiar_log']:.3f}). Indica incerteza maior, não prova "
                                 "erro maior; a validade da regra neste pacote é presumida."))
    if material_ja_publicado(s, e["pedido"]):
        avisos.append(_aviso("material_ja_publicado", TEMPO_MATERIAL_PUBLICADO))
    if pred["cortada_em_zero"]:
        avisos.append(_aviso("previsao_cortada_em_zero", "A saída bruta do modelo ficou abaixo de zero e foi cortada "
                                                          "em 0 curtidas."))
    if pred["cruzou"]:
        avisos.append(_aviso("quantis_cruzados", "q10, q50 e q90 se cruzaram; a faixa usa mínimo e máximo das três "
                                                  "saídas e mantém o q50 como central.", "info"))
    return {"avisos": avisos, "confianca": confianca}


def explicar(s: Servicos, e: Estado) -> dict:
    codigos = [a["codigo"] for a in e["avisos"]]
    evidencia = ex.construir_evidencia(e["previsao"], e["linha"], codigos)
    explicacao = ex.explicar(evidencia, s.subagente)
    return {"evidencia": evidencia, "explicacao": explicacao,
            "avisos": _avisos_da_explicacao(e["avisos"], explicacao, configurado=s.subagente is not None)}


def _avisos_da_explicacao(avisos: list[dict], explicacao: dict, *, configurado: bool) -> list[dict]:
    """Subagente configurado que cai no texto fixo fica explícito nos avisos (sem subagente nada muda: é o padrão)."""
    if not configurado or explicacao["fonte"] != "template":
        return avisos
    novos = [_aviso("explicacao_por_modelo_fixo",
                    f"A explicação por LLM não foi usada ({explicacao['motivo_fallback']}); vale o texto de modelo fixo.",
                    "info")]
    causa = ag.aviso_do_motivo_de_explicacao(explicacao["motivo_fallback"])
    if causa is not None:
        novos.append(causa)
    existentes = {a["codigo"] for a in avisos}
    return [*avisos, *(a for a in novos if a["codigo"] not in existentes)]


def montar_saida(s: Servicos, e: Estado) -> dict:
    p, pred, h, expl = e["pedido"], e["previsao"], e["historico"], e["explicacao"]
    evidencia, pacote = e["evidencia"], pacote_efetivo(s, e)
    saida = {
        "input_modality": p.modalidade, "model_level": pacote.nivel,
        "predicted_metric": pred["curtidas_central"], "interval_low": pred["curtidas_inferior"],
        "interval_high": pred["curtidas_superior"],
        "interval_nominal_level": pacote.declaracao_faixa["nivel_nominal"],
        "coverage_estimate": pacote.declaracao_faixa,
        "relative_to_channel": pred["relative_to_channel"], "reference_likes": pred["curtidas_referencia"],
        "reference_source": "canal" if h.linha["estado_referencia"] == "canal" else "setor",
        "reference_definition": DEFINICAO_REFERENCIA, "target_definition": ALVO,
        "history": h.resumo(), "retrospective": material_ja_publicado(s, p),
        "prediction_timing": TEMPO_MATERIAL_PUBLICADO if material_ja_publicado(s, p) else TEMPO_MATERIAL_FUTURO,
        "warnings": e["avisos"], "top_drivers": evidencia["drivers"], "family_contributions": evidencia["familias"],
        "explanation": expl["texto"], "explanation_source": expl["fonte"],
        "explanation_fallback_reason": expl["motivo_fallback"], "comparables": e["comparaveis"],
        "extraction": e["extracao"], "model_version": pacote.versao, "bundle_id": pacote.identificacao,
        "statistical_confidence": e["confianca"],
        "numeric_detail": {k: pred[k] for k in (
            "central_log", "inferior_log", "superior_log", "q10_log", "q50_log", "q90_log", "ref_log",
            "variacao_log_vs_canal", "cruzou", "cortada_em_zero")},
    }
    if p.modalidade in INTERPRETACAO_DA_ENTRADA:
        saida["input_interpretation"] = INTERPRETACAO_DA_ENTRADA[p.modalidade]
    if e.get("fallback"):
        saida |= _campos_do_fallback(e["fallback"])
    elif e["confianca"].get("sinalizada"):
        saida |= _campos_da_baixa_confianca(e["confianca"])
    return {"saida": saida}


def _campos_da_baixa_confianca(c: dict) -> dict:
    """Largura acima do limiar vinculado: classe `regra_estatistica` (D-F2 b), com a ressalva de validade presumida."""
    motivo = (f"baixa confiança estatística: largura {c['largura_log']:.3f} acima do limiar {c['limiar_log']:.3f} "
              "(validade presumida e declarada neste pacote)")
    return {"confidence_flag": "baixa",
            "confidence_reasons": [{"classe": "regra_estatistica", "codigo": "largura_acima_do_limiar",
                                    "descricao": motivo}]}


def _campos_do_fallback(fallback: dict) -> dict:
    """Só no fallback: o que mudou de modelo e por quê, com a baixa confiança da ARQUITETURA §9.3 (D-F2)."""
    motivo = f"sem os atributos do Gemini ({fallback['motivo']}): previsão pelo modelo de imagem, sem o vídeo"
    return {"fallback": fallback, "confidence_flag": "baixa",
            "confidence_reasons": [{"classe": "evidencia_insuficiente", "codigo": "sem_vlm", "descricao": motivo}]}


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
        except Exception as falha:
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
