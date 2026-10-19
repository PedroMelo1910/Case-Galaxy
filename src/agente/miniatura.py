"""Baixa a miniatura pública de um vídeo do YouTube (`i.ytimg.com`), a única capa possível para uma URL nunca vista.

Só imagem, nunca vídeo ou áudio (D-B2). Só `https` e só o host `i.ytimg.com`, também em redirecionamentos; o ID é
validado antes de entrar na URL; o corpo é lido com limite de tamanho. Tenta `maxresdefault` (o que a API chama de
`maxres`) e cai em `hqdefault`. Falha devolve None, e o grafo trata a capa ausente de forma explícita.
"""

import logging
import re
from collections.abc import Callable

import httpx

from src.agente.atributos import MAX_BYTES_IMAGEM

logger = logging.getLogger(__name__)
HOST = "i.ytimg.com"
NOMES = ("maxresdefault", "hqdefault")
TEMPO_LIMITE_S = 15.0
_ID_VIDEO = re.compile(r"[A-Za-z0-9_-]{11}")


def _so_o_host_da_miniatura(requisicao: httpx.Request) -> None:
    if requisicao.url.scheme != "https" or requisicao.url.host != HOST:
        raise httpx.TransportError("host fora de i.ytimg.com recusado")


def criar_baixador(transporte: httpx.BaseTransport | None = None, *,
                   max_bytes: int = MAX_BYTES_IMAGEM) -> Callable[[str], bytes | None]:
    cliente = httpx.Client(transport=transporte, timeout=TEMPO_LIMITE_S, follow_redirects=True,
                           event_hooks={"request": [_so_o_host_da_miniatura]})

    def baixar(video_id: str) -> bytes | None:
        if not _ID_VIDEO.fullmatch(video_id or ""):
            return None
        for nome in NOMES:
            try:
                with cliente.stream("GET", f"https://{HOST}/vi/{video_id}/{nome}.jpg") as resposta:
                    if resposta.status_code != 200:
                        continue
                    dados = b""
                    for bloco in resposta.iter_bytes():
                        dados += bloco
                        if len(dados) > max_bytes:
                            dados = b""
                            break
                    if dados:
                        return dados
            except httpx.HTTPError as erro:
                logger.warning("miniatura não baixada (%s)", type(erro).__name__)
        return None

    return baixar
