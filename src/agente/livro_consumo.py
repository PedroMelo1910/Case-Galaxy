"""Livro de consumo das chamadas externas do agente, no formato dos scripts de verificação ao vivo.

Uma linha JSON por evento: `quando_utc`, `tipo` (`gemini_video`, `gemini_texto`, `gemini_arquivo`, `youtube`), `fase`
(`envio` ANTES de a requisição sair, `resultado` depois) e os campos do evento. É o mesmo formato de
`VERIFICACOES/agente_ao_vivo/consumo.jsonl`, lido por `saldo_video.py` e `orcamento_anotacao.py`, para que a cota
gasta pelo MCP continue conciliável com a da anotação do dataset. Nunca grava chave, URL com chave nem conteúdo do pedido.
"""

import json
import os
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx

_CAMPOS_PERMITIDOS = ("tipo", "fase", "operacao", "duracao_video_s", "modelo", "sha256_corpo", "resultado", "uso",
                      "unidades", "recurso")


def _agora_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class LivroConsumo:
    """Acrescenta eventos ao arquivo, um por vez (trava), gravando e forçando ao disco antes de devolver."""

    def __init__(self, arquivo: Path, *, agora: Callable[[], str] = _agora_utc) -> None:
        self.arquivo = Path(arquivo)
        self._agora = agora
        self._trava = threading.Lock()

    def registrar(self, evento: Mapping) -> None:
        linha = {"quando_utc": self._agora(), **{k: evento[k] for k in _CAMPOS_PERMITIDOS if k in evento}}
        texto = json.dumps(linha, ensure_ascii=False) + "\n"
        with self._trava, self.arquivo.open("a", encoding="utf-8") as saida:
            saida.write(texto)
            saida.flush()
            os.fsync(saida.fileno())

    def registrar_gemini(self, evento: Mapping) -> None:
        self.registrar(evento)

    def registrar_youtube(self, recurso: str) -> None:
        self.registrar({"tipo": "youtube", "fase": "envio", "unidades": 1, "recurso": recurso})


class TransporteYouTubeComLivro(httpx.BaseTransport):
    """Anota cada requisição à YouTube Data API (1 unidade) ANTES de enviá-la. Sem livro gravável, nada sai."""

    def __init__(self, interno: httpx.BaseTransport, livro: LivroConsumo) -> None:
        self._interno, self._livro = interno, livro

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            self._livro.registrar_youtube(request.url.path)  # só o caminho: nunca a consulta, onde poderia haver chave
        except OSError:
            raise httpx.TransportError("o consumo não pôde ser registrado") from None
        return self._interno.handle_request(request)

    def close(self) -> None:
        self._interno.close()
