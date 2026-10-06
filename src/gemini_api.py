"""Cliente mínimo do `generateContent` da Gemini API (endpoint estável v1beta), com tetos por execução.

Cada chamada de `gerar` é UMA requisição. Antes de enviar, o cliente recusa se a execução já atingiu o
teto de requisições ou de segundos de vídeo, e respeita o intervalo mínimo entre requisições. Toda
requisição enviada é contada, inclusive as que falham na rede ou voltam inválidas. A chave vai no
cabeçalho `x-goog-api-key` e é removida de qualquer mensagem de erro.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

import httpx
from pydantic import SecretStr

from src.config import Configuracao

URL_BASE = "https://generativelanguage.googleapis.com/v1beta"
CABECALHO_CHAVE = "x-goog-api-key"
TEMPO_LIMITE_S = 240.0
_TIPO_COTA = "type.googleapis.com/google.rpc.QuotaFailure"
_TIPO_ESPERA = "type.googleapis.com/google.rpc.RetryInfo"
_CAMPOS_DE_USO = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")


class ErroGemini(Exception):
    """Falha de uma requisição. `fatal`: a execução inteira deve parar; senão, só o vídeo é afetado."""

    codigo = "ERRO_GEMINI"
    fatal = False

    def __init__(self, mensagem: str, *, status: int | None = None):
        super().__init__(mensagem)
        self.status = status


class ChaveInvalida(ErroGemini):
    codigo, fatal = "CHAVE_INVALIDA", True


class ModeloIndisponivel(ErroGemini):
    codigo, fatal = "MODELO_INDISPONIVEL", True


class AcessoNegado(ErroGemini):
    codigo, fatal = "ACESSO_NEGADO", True


class CotaDiariaEsgotada(ErroGemini):
    codigo, fatal = "COTA_DIARIA_ESGOTADA", True


class LimiteDaExecucao(ErroGemini):
    """Teto de requisições ou de segundos de vídeo desta execução: recusado antes de enviar."""

    codigo, fatal = "LIMITE_DA_EXECUCAO", True


class RequisicaoRecusada(ErroGemini):
    """HTTP 4xx específico do pedido (por exemplo, vídeo inacessível); não adianta repetir."""

    codigo = "REQUISICAO_RECUSADA"


class FalhaTemporaria(ErroGemini):
    codigo = "FALHA_TEMPORARIA"


class LimiteDeTaxa(FalhaTemporaria):
    """429 de limite por minuto: pode ser repetido depois da espera sugerida."""

    codigo = "LIMITE_DE_TAXA"

    def __init__(
        self, mensagem: str, *, status: int | None, espera_sugerida_s: float | None, quota_ids: tuple[str, ...]
    ):
        super().__init__(mensagem, status=status)
        self.espera_sugerida_s = espera_sugerida_s
        self.quota_ids = quota_ids


@dataclass(frozen=True)
class Limites:
    """Tetos explícitos da execução. O intervalo mínimo deriva do RPM conferido no AI Studio."""

    max_requisicoes: int
    max_segundos_video: int
    intervalo_minimo_s: float

    def __post_init__(self) -> None:
        if self.max_requisicoes < 1 or self.max_segundos_video < 1:
            raise ValueError("max_requisicoes e max_segundos_video devem ser inteiros maiores ou iguais a 1")
        if self.intervalo_minimo_s < 0:
            raise ValueError("intervalo_minimo_s não pode ser negativo")


@dataclass(frozen=True)
class RespostaGemini:
    """Resposta HTTP 200. `situacao` é "ok" só quando há texto e a geração terminou normalmente."""

    situacao: str
    texto: str | None
    duracao_s: float
    uso: dict = field(default_factory=dict)
    versao_modelo: str | None = None
    id_resposta: str | None = None


def _detalhes(corpo: object) -> list[dict]:
    erro = corpo.get("error") if isinstance(corpo, dict) else None
    detalhes = erro.get("details", []) if isinstance(erro, dict) else []
    return [d for d in detalhes if isinstance(d, dict)] if isinstance(detalhes, list) else []


def _mensagem(corpo: object) -> str:
    erro = corpo.get("error") if isinstance(corpo, dict) else None
    return str(erro.get("message", "")) if isinstance(erro, dict) else ""


def _quota_ids(detalhes: list[dict]) -> tuple[str, ...]:
    ids = []
    for detalhe in detalhes:
        if detalhe.get("@type") == _TIPO_COTA:
            ids += [
                str(v.get("quotaId")) for v in detalhe.get("violations", []) if isinstance(v, dict) and v.get("quotaId")
            ]
    return tuple(ids)


def _espera_sugerida(detalhes: list[dict]) -> float | None:
    for detalhe in detalhes:
        if detalhe.get("@type") == _TIPO_ESPERA:
            achado = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detalhe.get("retryDelay", "")))
            if achado:
                return float(achado[1])
    return None


def _uso(corpo: dict) -> dict:
    metadados = corpo.get("usageMetadata") if isinstance(corpo.get("usageMetadata"), dict) else {}
    uso = {nome: metadados[nome] for nome in _CAMPOS_DE_USO if isinstance(metadados.get(nome), int)}
    detalhes = metadados.get("promptTokensDetails")
    if isinstance(detalhes, list):
        uso["por_modalidade"] = {
            str(d.get("modality")): d.get("tokenCount", 0)
            for d in detalhes
            if isinstance(d, dict) and d.get("modality")
        }
    return uso


def _situacao_e_texto(corpo: dict) -> tuple[str, str | None]:
    if isinstance(corpo.get("promptFeedback"), dict) and corpo["promptFeedback"].get("blockReason"):
        return "bloqueada", None
    candidatos = corpo.get("candidates")
    if not isinstance(candidatos, list) or not candidatos or not isinstance(candidatos[0], dict):
        return "sem_candidato", None
    candidato = candidatos[0]
    conteudo = candidato.get("content") if isinstance(candidato.get("content"), dict) else {}
    partes = conteudo.get("parts") if isinstance(conteudo.get("parts"), list) else []
    textos = [
        p["text"] for p in partes if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")
    ]
    texto = "".join(textos) or None
    fim = candidato.get("finishReason")
    if fim == "MAX_TOKENS":
        return "truncada", texto
    if fim not in (None, "STOP"):
        return f"finalizacao_{fim}", texto
    return ("ok", texto) if texto is not None else ("sem_texto", None)


class ClienteGemini:
    """Uma requisição por chamada de `gerar`, com contagem, tetos e intervalo mínimo."""

    def __init__(
        self,
        chave: SecretStr,
        modelo: str,
        limites: Limites,
        *,
        transporte: httpx.BaseTransport | None = None,
        relogio: Callable[[], float] = time.monotonic,
        espera: Callable[[float], None] = time.sleep,
    ):
        if not chave.get_secret_value().strip():
            raise ValueError("Chave da Gemini API vazia.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.\-]*", modelo):
            raise ValueError(f"identificador de modelo inválido: {modelo!r}")
        self._chave = chave
        self.modelo = modelo
        self.limites = limites
        self._relogio = relogio
        self._espera = espera
        self._http = httpx.Client(base_url=URL_BASE, timeout=TEMPO_LIMITE_S, transport=transporte)
        self._ultimo_envio: float | None = None
        self.requisicoes_enviadas = 0
        self.segundos_video_enviados = 0

    @classmethod
    def a_partir_da_configuracao(
        cls,
        modelo: str,
        limites: Limites,
        *,
        configuracao: Configuracao | None = None,
        transporte: httpx.BaseTransport | None = None,
    ) -> "ClienteGemini":
        """Cliente com a GEMINI_API_KEY do .env; falha antes de qualquer chamada se ela faltar."""
        chave = (configuracao or Configuracao()).chave_gemini()
        return cls(chave, modelo, limites, transporte=transporte)

    def __repr__(self) -> str:
        return f"ClienteGemini(modelo={self.modelo!r}, requisicoes_enviadas={self.requisicoes_enviadas})"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.fechar()

    def fechar(self) -> None:
        self._http.close()

    def _verificar_tetos(self, duracao_video_s: int) -> None:
        if self.requisicoes_enviadas + 1 > self.limites.max_requisicoes:
            raise LimiteDaExecucao(f"teto de {self.limites.max_requisicoes} requisições atingido")
        if self.segundos_video_enviados + duracao_video_s > self.limites.max_segundos_video:
            raise LimiteDaExecucao(f"teto de {self.limites.max_segundos_video} segundos de vídeo atingido")

    def _respeitar_intervalo(self) -> None:
        if self._ultimo_envio is None:
            return
        restante = self.limites.intervalo_minimo_s - (self._relogio() - self._ultimo_envio)
        if restante > 0:
            self._espera(restante)

    def gerar(self, corpo: dict, *, duracao_video_s: int) -> RespostaGemini:
        self._verificar_tetos(duracao_video_s)
        self._respeitar_intervalo()
        self._ultimo_envio = inicio = self._relogio()
        self.requisicoes_enviadas += 1
        self.segundos_video_enviados += duracao_video_s
        try:
            resposta = self._http.post(
                f"/models/{self.modelo}:generateContent",
                json=corpo,
                headers={CABECALHO_CHAVE: self._chave.get_secret_value()},
            )
        except httpx.RequestError as erro:
            raise FalhaTemporaria(f"falha de rede ({type(erro).__name__})") from None
        duracao = self._relogio() - inicio
        try:
            dados = resposta.json()
        except ValueError:
            dados = None
        if resposta.status_code != 200:
            raise self._erro(resposta.status_code, dados)
        if not isinstance(dados, dict):
            return RespostaGemini(situacao="corpo_invalido", texto=None, duracao_s=duracao)
        situacao, texto = _situacao_e_texto(dados)
        return RespostaGemini(
            situacao=situacao,
            texto=texto,
            duracao_s=duracao,
            uso=_uso(dados),
            versao_modelo=dados.get("modelVersion") if isinstance(dados.get("modelVersion"), str) else None,
            id_resposta=dados.get("responseId") if isinstance(dados.get("responseId"), str) else None,
        )

    def _erro(self, status: int, corpo: object) -> ErroGemini:
        detalhes = _detalhes(corpo)
        mensagem = _mensagem(corpo)
        texto = f"HTTP {status}: {mensagem}".replace(self._chave.get_secret_value(), "***").strip()
        motivos = {str(d.get("reason")) for d in detalhes if d.get("reason")}
        if "API_KEY_INVALID" in motivos or "api key not valid" in mensagem.lower():
            return ChaveInvalida(texto, status=status)
        if status == 429:
            ids = _quota_ids(detalhes)
            if any("PerDay" in i for i in ids):
                return CotaDiariaEsgotada(texto, status=status)
            return LimiteDeTaxa(texto, status=status, espera_sugerida_s=_espera_sugerida(detalhes), quota_ids=ids)
        if status == 404:
            return ModeloIndisponivel(texto, status=status)
        if status in (401, 403):
            return AcessoNegado(texto, status=status)
        if status >= 500:
            return FalhaTemporaria(texto, status=status)
        return RequisicaoRecusada(texto, status=status)
