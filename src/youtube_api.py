"""Cliente mínimo da YouTube Data API v3: leitura pública com chave de API.

A chave vai no cabeçalho `x-goog-api-key`, nunca na URL, para não aparecer em
logs de requisição nem em mensagens de erro.
"""

import time
from collections.abc import Callable, Iterator, Sequence

import httpx
from pydantic import SecretStr

from src.config import Configuracao

URL_BASE = "https://www.googleapis.com/youtube/v3"
CABECALHO_CHAVE = "x-goog-api-key"
MAX_IDS_POR_CHAMADA = 50
CUSTO_POR_CHAMADA = 1  # channels.list, playlistItems.list e videos.list custam 1 unidade
ALTURA_MAXIMA_PLAYER = 1280  # com maxHeight, a API informa embedWidth/embedHeight quando conhece a proporção
TENTATIVAS = 3
TEMPO_LIMITE_S = 30.0
PARTES_CANAL = "snippet,contentDetails,statistics"
PARTES_PLAYLIST = "contentDetails"
PARTES_VIDEO = "snippet,contentDetails,statistics,status,player"

STATUS_TEMPORARIOS = frozenset({429, 500, 502, 503, 504})
MOTIVOS_TEMPORARIOS = frozenset({"rateLimitExceeded", "userRateLimitExceeded", "backendError"})
MOTIVOS_COTA = frozenset({"quotaExceeded", "dailyLimitExceeded"})
MOTIVOS_CHAVE = frozenset({"keyInvalid", "keyExpired", "API_KEY_INVALID", "API_KEY_EXPIRED"})
MOTIVOS_API_DESATIVADA = frozenset({"accessNotConfigured", "SERVICE_DISABLED"})


class ErroYouTube(Exception):
    """Falha da API com código estável. A mensagem nunca contém a chave."""

    codigo = "ERRO_YOUTUBE"

    def __init__(self, mensagem: str, *, status: int | None = None, motivos: frozenset[str] = frozenset()):
        super().__init__(mensagem)
        self.status = status
        self.motivos = motivos


class CotaEsgotada(ErroYouTube):
    codigo = "COTA_ESGOTADA"


class ChaveInvalida(ErroYouTube):
    codigo = "CHAVE_INVALIDA"


class ApiDesativada(ErroYouTube):
    codigo = "API_DESATIVADA"


class RecursoNaoEncontrado(ErroYouTube):
    codigo = "NAO_ENCONTRADO"


class AcessoNegado(ErroYouTube):
    codigo = "ACESSO_NEGADO"


class FalhaTemporaria(ErroYouTube):
    codigo = "FALHA_TEMPORARIA"


class RespostaInvalida(ErroYouTube):
    codigo = "RESPOSTA_INVALIDA"


class LimiteDeUnidades(ErroYouTube):
    """A próxima chamada ultrapassaria o teto de unidades definido para esta execução."""

    codigo = "LIMITE_DE_UNIDADES"


def _motivos(corpo: object) -> frozenset[str]:
    erro = corpo.get("error") if isinstance(corpo, dict) else None
    if not isinstance(erro, dict):
        return frozenset()
    itens = [*erro.get("errors", []), *erro.get("details", [])]
    return frozenset(i["reason"] for i in itens if isinstance(i, dict) and isinstance(i.get("reason"), str))


def _mensagem_api(corpo: object) -> str:
    erro = corpo.get("error") if isinstance(corpo, dict) else None
    return str(erro.get("message", "")) if isinstance(erro, dict) else ""


def _classe_do_erro(status: int, motivos: frozenset[str]) -> type[ErroYouTube]:
    if motivos & MOTIVOS_COTA:
        return CotaEsgotada
    if motivos & MOTIVOS_CHAVE:
        return ChaveInvalida
    if motivos & MOTIVOS_API_DESATIVADA:
        return ApiDesativada
    if status in STATUS_TEMPORARIOS or motivos & MOTIVOS_TEMPORARIOS:
        return FalhaTemporaria
    if status == 404:
        return RecursoNaoEncontrado
    if status in (401, 403):
        return AcessoNegado
    return ErroYouTube


def _lotes(ids: Sequence[str], tamanho: int) -> Iterator[list[str]]:
    for inicio in range(0, len(ids), tamanho):
        yield list(ids[inicio : inicio + tamanho])


class ClienteYouTube:
    """Chamadas `channels.list`, `playlistItems.list` e `videos.list`, com contagem de cota."""

    def __init__(
        self,
        chave: SecretStr,
        *,
        transporte: httpx.BaseTransport | None = None,
        espera: Callable[[float], None] = time.sleep,
        limite_unidades: int | None = None,
    ):
        if not chave.get_secret_value().strip():
            raise ValueError("Chave da YouTube Data API vazia.")
        self._chave = chave
        self._espera = espera
        self._http = httpx.Client(base_url=URL_BASE, timeout=TEMPO_LIMITE_S, transport=transporte)
        self.unidades_usadas = 0
        self.limite_unidades = limite_unidades

    @classmethod
    def a_partir_da_configuracao(
        cls,
        configuracao: Configuracao | None = None,
        *,
        transporte: httpx.BaseTransport | None = None,
        espera: Callable[[float], None] = time.sleep,
        limite_unidades: int | None = None,
    ) -> "ClienteYouTube":
        """Cliente com a chave YOUTUBE_API_KEY do .env; falha antes de qualquer chamada se ela faltar."""
        chave = (configuracao or Configuracao()).chave_youtube()
        return cls(chave, transporte=transporte, espera=espera, limite_unidades=limite_unidades)

    def __enter__(self) -> "ClienteYouTube":
        return self

    def __exit__(self, *_: object) -> None:
        self.fechar()

    def fechar(self) -> None:
        self._http.close()

    def canais(self, ids: Sequence[str]) -> list[dict]:
        itens: list[dict] = []
        for lote in _lotes(ids, MAX_IDS_POR_CHAMADA):
            itens.extend(self._itens("channels", {"part": PARTES_CANAL, "id": ",".join(lote)}))
        return itens

    def canal_por_handle(self, handle: str) -> dict | None:
        """Canal identificado pelo handle público (ex.: @marca); None se o handle não existir. Custa 1 unidade."""
        itens = self._itens("channels", {"part": "snippet", "forHandle": handle})
        return itens[0] if itens else None

    def pagina_uploads(self, playlist_id: str, token: str | None = None) -> dict:
        parametros: dict[str, str | int] = {
            "part": PARTES_PLAYLIST,
            "playlistId": playlist_id,
            "maxResults": MAX_IDS_POR_CHAMADA,
        }
        if token:
            parametros["pageToken"] = token
        return self._get("playlistItems", parametros)

    def videos(self, ids: Sequence[str]) -> list[dict]:
        itens: list[dict] = []
        for lote in _lotes(ids, MAX_IDS_POR_CHAMADA):
            parametros = {"part": PARTES_VIDEO, "id": ",".join(lote), "maxHeight": ALTURA_MAXIMA_PLAYER}
            itens.extend(self._itens("videos", parametros))
        return itens

    def _itens(self, recurso: str, parametros: dict) -> list[dict]:
        itens = self._get(recurso, parametros).get("items", [])
        if not isinstance(itens, list):
            raise RespostaInvalida(f"{recurso}: campo 'items' inválido")
        return itens

    def _get(self, recurso: str, parametros: dict) -> dict:
        for tentativa in range(1, TENTATIVAS + 1):
            resultado = self._tentar(recurso, parametros)
            if isinstance(resultado, dict):
                return resultado
            if not isinstance(resultado, FalhaTemporaria) or tentativa == TENTATIVAS:
                raise resultado
            self._espera(2.0**tentativa)
        raise AssertionError("inalcançável")

    def _tentar(self, recurso: str, parametros: dict) -> dict | ErroYouTube:
        if self.limite_unidades is not None and self.unidades_usadas + CUSTO_POR_CHAMADA > self.limite_unidades:
            return LimiteDeUnidades(
                f"{recurso}: chamada recusada; {self.unidades_usadas} de {self.limite_unidades} unidades já usadas"
            )
        cabecalhos = {CABECALHO_CHAVE: self._chave.get_secret_value()}
        try:
            resposta = self._http.get(f"/{recurso}", params=parametros, headers=cabecalhos)
        except httpx.TransportError as erro:
            return FalhaTemporaria(f"{recurso}: falha de rede ({type(erro).__name__})")
        self.unidades_usadas += CUSTO_POR_CHAMADA
        try:
            corpo = resposta.json()
        except ValueError:
            corpo = None
        if resposta.status_code == 200:
            if not isinstance(corpo, dict):
                return RespostaInvalida(f"{recurso}: resposta 200 sem JSON válido")
            return corpo
        return self._erro(recurso, resposta.status_code, corpo)

    def _erro(self, recurso: str, status: int, corpo: object) -> ErroYouTube:
        motivos = _motivos(corpo)
        texto = f"{recurso}: HTTP {status} {sorted(motivos)} {_mensagem_api(corpo)}".strip()
        texto = texto.replace(self._chave.get_secret_value(), "***")
        return _classe_do_erro(status, motivos)(texto, status=status, motivos=motivos)
