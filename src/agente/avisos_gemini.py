"""Avisos tipados para as falhas do Gemini (cota, taxa, teto da execução, resposta inválida, envio de arquivo).

O estado da fonte de anotação (`falha_...`) e o motivo do subagente de explicação viram o mesmo código de aviso, para que
o envelope diga POR QUE o caminho degradou, sem rastro de pilha nem chave. Nada aqui consulta rede.
"""

import re

_POR_ESTADO = {
    "falha_cota_esgotada": (
        "cota_gemini_esgotada", "A cota diária do Gemini está esgotada; nenhuma nova tentativa foi feita."),
    "falha_LimiteDeTaxa": (
        "limite_de_taxa_gemini",
        "O Gemini recusou por limite de requisições por minuto; esta chamada não foi repetida."),
    "falha_LimiteDaExecucao": (
        "limite_da_execucao_gemini",
        "O teto de requisições ou de segundos de vídeo desta execução foi atingido; nada foi enviado."),
}
_ENVIO_DE_ARQUIVO = {"falha_ArquivoComFalha", "falha_ArquivoNaoFicouAtivo", "falha_ArquivoNaoEncontrado",
                     "falha_duracao_do_arquivo", "falha_arquivo_alterado", "falha_tamanho_divergente"}
_INDISPONIVEL = {"falha_ChaveInvalida", "falha_AcessoNegado", "falha_ModeloIndisponivel", "falha_RegistroIndisponivel",
                 "falha_FalhaTemporaria", "falha_RequisicaoRecusada", "falha_ErroGemini"}
_MOTIVO_COTA = re.compile(r"CotaDiariaEsgotada")


def aviso_do_estado(estado: str | None) -> dict | None:
    """Aviso (`{classe, codigo, descricao}`) da causa, ou None quando o estado não é uma falha com causa tipada."""
    if not estado:
        return None
    if estado in _POR_ESTADO:
        codigo, descricao = _POR_ESTADO[estado]
    elif estado.startswith(("falha_invalida_", "falha_resposta_")):
        codigo, descricao = "extracao_gemini_invalida", "O Gemini não devolveu uma anotação válida para este vídeo."
    elif estado in _ENVIO_DE_ARQUIVO:
        codigo, descricao = "envio_de_arquivo_ao_gemini_falhou", "O envio do vídeo local ao Gemini não foi concluído."
    elif estado in _INDISPONIVEL:
        codigo, descricao = "gemini_indisponivel", "O Gemini ficou indisponível para esta chamada."
    else:
        return None
    return {"classe": "atencao", "codigo": codigo, "descricao": descricao}


def aviso_do_motivo_de_explicacao(motivo: str | None) -> dict | None:
    """Aviso quando o subagente falhou por cota diária; os demais motivos ficam só em `explanation_fallback_reason`."""
    if motivo and _MOTIVO_COTA.search(motivo):
        return aviso_do_estado("falha_cota_esgotada")
    return None
