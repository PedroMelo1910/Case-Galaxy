"""Módulo consolidado. Reúne:
- Cliente mínimo do `generateContent` e da Files API da Gemini API (endpoint estável v1beta), com tetos por execução.
- Avisos tipados para as falhas do Gemini (cota, taxa, teto da execução, resposta inválida, envio de arquivo)."""

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from src.config import Configuracao


# ──── Cliente mínimo do `generateContent` e da Files API da Gemini API (endpoint estável v1beta), com teto
URL_BASE = "https://generativelanguage.googleapis.com/v1beta"
HOST_API = "generativelanguage.googleapis.com"
URL_ENVIO_ARQUIVOS = f"https://{HOST_API}/upload/v1beta/files"
CABECALHO_CHAVE = "x-goog-api-key"
TEMPO_LIMITE_S = 240.0
TEMPO_LIMITE_ENVIO_S = 900.0
BLOCO_ENVIO_BYTES = 1_048_576
BLOQUEIO_COTA_PADRAO_S = 1800.0
_NOME_ARQUIVO = re.compile(r"files/[a-z0-9][a-z0-9-]{0,39}")
_DURACAO_VIDEO = re.compile(r"(\d+(?:\.\d+)?)s")
logger = logging.getLogger(__name__)
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


class ArquivoNaoEncontrado(ErroGemini):
    """404 de um arquivo da Files API (inexistente ou já expirado); não significa modelo indisponível."""

    codigo = "ARQUIVO_NAO_ENCONTRADO"


class ArquivoComFalha(ErroGemini):
    """O Gemini marcou o arquivo como FAILED ao processá-lo."""

    codigo = "ARQUIVO_COM_FALHA"


class ArquivoNaoFicouAtivo(ErroGemini):
    """O arquivo continuou em PROCESSING depois do número máximo de consultas."""

    codigo = "ARQUIVO_NAO_FICOU_ATIVO"


class RegistroIndisponivel(ErroGemini):
    """O evento de envio não pôde ser gravado no livro de consumo: a requisição não sai."""

    codigo, fatal = "REGISTRO_INDISPONIVEL", True


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
class ArquivoGemini:
    """Arquivo da Files API. `sha256_enviado` é o hash dos bytes realmente enviados (só no envio)."""

    nome: str
    uri: str
    mime: str
    estado: str
    tamanho_bytes: int | None = None
    duracao_s: float | None = None
    sha256_enviado: str | None = None


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
    """Uma requisição por chamada de `gerar`, com contagem, tetos e intervalo mínimo; mais a Files API."""

    def __init__(
        self,
        chave: SecretStr,
        modelo: str,
        limites: Limites,
        *,
        transporte: httpx.BaseTransport | None = None,
        relogio: Callable[[], float] = time.monotonic,
        espera: Callable[[float], None] = time.sleep,
        registro: Callable[[dict], None] | None = None,
        bloqueio_cota_s: float = BLOQUEIO_COTA_PADRAO_S,
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
        self._registro = registro
        self._bloqueio_cota_s = bloqueio_cota_s
        self._bloqueado_ate: float | None = None
        self._http = httpx.Client(base_url=URL_BASE, timeout=TEMPO_LIMITE_S, transport=transporte)
        self._ultimo_envio: float | None = None
        self.requisicoes_enviadas = 0
        self.segundos_video_enviados = 0
        self.operacoes: dict[str, int] = {}

    @classmethod
    def a_partir_da_configuracao(
        cls,
        modelo: str,
        limites: Limites,
        *,
        configuracao: Configuracao | None = None,
        transporte: httpx.BaseTransport | None = None,
        registro: Callable[[dict], None] | None = None,
    ) -> "ClienteGemini":
        """Cliente com a GEMINI_API_KEY do .env; falha antes de qualquer chamada se ela faltar."""
        chave = (configuracao or Configuracao()).chave_gemini()
        return cls(chave, modelo, limites, transporte=transporte, registro=registro)

    def __repr__(self) -> str:
        return f"ClienteGemini(modelo={self.modelo!r}, requisicoes_enviadas={self.requisicoes_enviadas})"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.fechar()

    def fechar(self) -> None:
        self._http.close()

    # --- tetos, bloqueio e registro ---------------------------------------------------------------------------------

    def _verificar_tetos(self, duracao_video_s: int, requisicoes: int = 1) -> None:
        if self.requisicoes_enviadas + requisicoes > self.limites.max_requisicoes:
            raise LimiteDaExecucao(f"teto de {self.limites.max_requisicoes} requisições atingido")
        if self.segundos_video_enviados + duracao_video_s > self.limites.max_segundos_video:
            raise LimiteDaExecucao(f"teto de {self.limites.max_segundos_video} segundos de vídeo atingido")

    def verificar_orcamento(self, duracao_video_s: int, requisicoes: int) -> None:
        """Levanta `LimiteDaExecucao` ou `CotaDiariaEsgotada` se o fluxo planejado já não cabe; não envia nada."""
        self._verificar_bloqueio()
        self._verificar_tetos(duracao_video_s, requisicoes)

    def _verificar_bloqueio(self) -> None:
        if self._bloqueado_ate is not None and self._relogio() < self._bloqueado_ate:
            raise CotaDiariaEsgotada("cota diária do Gemini esgotada: nenhuma nova requisição foi enviada")

    def _respeitar_intervalo(self) -> None:
        if self._ultimo_envio is None:
            return
        restante = self.limites.intervalo_minimo_s - (self._relogio() - self._ultimo_envio)
        if restante > 0:
            self._espera(restante)

    def _tipo_de_registro(self, operacao: str, duracao_video_s: int) -> str:
        if operacao != "gerar":
            return "gemini_arquivo"
        return "gemini_video" if duracao_video_s > 0 else "gemini_texto"

    def _registrar_envio(self, operacao: str, duracao_video_s: int, sha256_corpo: str | None) -> None:
        if self._registro is None:
            return
        evento = {"fase": "envio", "operacao": operacao, "tipo": self._tipo_de_registro(operacao, duracao_video_s),
                  "duracao_video_s": duracao_video_s, "modelo": self.modelo}
        if sha256_corpo is not None:
            evento["sha256_corpo"] = sha256_corpo
        try:
            self._registro(evento)
        except Exception as falha:  # noqa: BLE001 - sem livro, nada sai: o consumo precisa ser reconciliável
            raise RegistroIndisponivel(f"o consumo não pôde ser registrado ({type(falha).__name__})") from None

    def _registrar_resultado(self, operacao: str, duracao_video_s: int, resultado: str, uso: dict | None = None) -> None:
        if self._registro is None:
            return
        evento = {"fase": "resultado", "operacao": operacao, "tipo": self._tipo_de_registro(operacao, duracao_video_s),
                  "resultado": resultado}
        if uso:
            evento["uso"] = uso
        try:
            self._registro(evento)
        except Exception as falha:  # noqa: BLE001 - a requisição já saiu; só o resultado ficou sem registro
            logger.warning("resultado não registrado no livro de consumo (%s)", type(falha).__name__)

    def _requisitar(self, operacao: str, metodo: str, url: str, *, duracao_video_s: int = 0,
                    sha256_corpo: str | None = None, verificar: bool = True, intervalo: bool = False,
                    **opcoes: object) -> tuple[httpx.Response, float]:
        """Uma requisição HTTP: tetos, registro ANTES de enviar, contagem e tradução de falha de rede."""
        if verificar:
            self._verificar_bloqueio()
            self._verificar_tetos(duracao_video_s)
        if intervalo:
            self._respeitar_intervalo()
        self._registrar_envio(operacao, duracao_video_s, sha256_corpo)
        self.requisicoes_enviadas += 1
        self.operacoes[operacao] = self.operacoes.get(operacao, 0) + 1
        self.segundos_video_enviados += duracao_video_s
        inicio = self._relogio()
        if intervalo:
            self._ultimo_envio = inicio
        try:
            resposta = self._http.request(metodo, url, **opcoes)
        except httpx.RequestError as erro:
            self._registrar_resultado(operacao, duracao_video_s, "falha_de_rede")
            raise FalhaTemporaria(f"falha de rede ({type(erro).__name__})") from None
        return resposta, self._relogio() - inicio

    @staticmethod
    def _json(resposta: httpx.Response) -> object:
        try:
            return resposta.json()
        except ValueError:
            return None

    def _levantar(self, operacao: str, duracao_video_s: int, resposta: httpx.Response, dados: object,
                  *, arquivo: bool = False) -> None:
        """Erro HTTP já traduzido e registrado; cota diária ativa o bloqueio das próximas chamadas."""
        erro = self._erro(resposta.status_code, dados)
        if arquivo and isinstance(erro, ModeloIndisponivel):
            erro = ArquivoNaoEncontrado(str(erro), status=resposta.status_code)
        if isinstance(erro, CotaDiariaEsgotada):
            self._bloqueado_ate = self._relogio() + self._bloqueio_cota_s
        self._registrar_resultado(operacao, duracao_video_s, erro.codigo)
        raise erro

    def _cabecalho_chave(self) -> dict[str, str]:
        return {CABECALHO_CHAVE: self._chave.get_secret_value()}

    # --- generateContent --------------------------------------------------------------------------------------------

    def gerar(self, corpo: dict, *, duracao_video_s: int) -> RespostaGemini:
        hash_corpo = hashlib.sha256(json.dumps(corpo, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        resposta, duracao = self._requisitar(
            "gerar", "POST", f"/models/{self.modelo}:generateContent", duracao_video_s=duracao_video_s,
            sha256_corpo=hash_corpo, intervalo=True, json=corpo, headers=self._cabecalho_chave())
        dados = self._json(resposta)
        if resposta.status_code != 200:
            self._levantar("gerar", duracao_video_s, resposta, dados)
        if not isinstance(dados, dict):
            self._registrar_resultado("gerar", duracao_video_s, "corpo_invalido")
            return RespostaGemini(situacao="corpo_invalido", texto=None, duracao_s=duracao)
        situacao, texto = _situacao_e_texto(dados)
        uso = _uso(dados)
        self._registrar_resultado("gerar", duracao_video_s, situacao, uso)
        return RespostaGemini(
            situacao=situacao,
            texto=texto,
            duracao_s=duracao,
            uso=uso,
            versao_modelo=dados.get("modelVersion") if isinstance(dados.get("modelVersion"), str) else None,
            id_resposta=dados.get("responseId") if isinstance(dados.get("responseId"), str) else None,
        )

    # --- Files API --------------------------------------------------------------------------------------------------

    @staticmethod
    def _validar_nome(nome: str) -> str:
        if not isinstance(nome, str) or not _NOME_ARQUIVO.fullmatch(nome):
            raise ValueError("nome de arquivo da Files API inválido")
        return nome

    @staticmethod
    def _arquivo(dados: object, sha256_enviado: str | None = None) -> ArquivoGemini:
        corpo = dados.get("file", dados) if isinstance(dados, dict) else {}
        if not isinstance(corpo, dict) or not isinstance(corpo.get("name"), str) \
                or not isinstance(corpo.get("uri"), str):
            raise RequisicaoRecusada("resposta da Files API sem nome ou URI do arquivo")
        if urlparse(corpo["uri"]).hostname != HOST_API or not _NOME_ARQUIVO.fullmatch(corpo["name"]):
            raise RequisicaoRecusada("resposta da Files API com URI ou nome inesperado")
        duracao = None
        metadados = corpo.get("videoMetadata")
        if isinstance(metadados, dict):
            achado = _DURACAO_VIDEO.fullmatch(str(metadados.get("videoDuration", "")))
            duracao = float(achado[1]) if achado else None
        tamanho = corpo.get("sizeBytes")
        return ArquivoGemini(
            nome=corpo["name"], uri=corpo["uri"], mime=str(corpo.get("mimeType", "")),
            estado=str(corpo.get("state", "STATE_UNSPECIFIED")),
            tamanho_bytes=int(tamanho) if isinstance(tamanho, (int, str)) and str(tamanho).isdigit() else None,
            duracao_s=duracao, sha256_enviado=sha256_enviado)

    def enviar_arquivo(self, caminho: Path, mime: str, *, nome_exibicao: str) -> ArquivoGemini:
        """Envio retomável em duas requisições (iniciar e enviar/finalizar). A chave só vai na primeira.

        Os bytes seguem em blocos (o arquivo nunca é lido inteiro na memória) e o SHA-256 do que saiu é devolvido.
        """
        caminho = Path(caminho)
        tamanho = caminho.stat().st_size
        self._verificar_bloqueio()
        self._verificar_tetos(0, requisicoes=2)
        inicio, _ = self._requisitar(
            "arquivo_iniciar", "POST", URL_ENVIO_ARQUIVOS, verificar=False,
            headers={**self._cabecalho_chave(), "X-Goog-Upload-Protocol": "resumable",
                     "X-Goog-Upload-Command": "start", "X-Goog-Upload-Header-Content-Length": str(tamanho),
                     "X-Goog-Upload-Header-Content-Type": mime, "Content-Type": "application/json"},
            json={"file": {"display_name": nome_exibicao}})
        if inicio.status_code != 200:
            self._levantar("arquivo_iniciar", 0, inicio, self._json(inicio), arquivo=True)
        self._registrar_resultado("arquivo_iniciar", 0, "ok")
        url_envio = inicio.headers.get("x-goog-upload-url", "")
        partes = urlparse(url_envio)
        if partes.scheme != "https" or partes.hostname != HOST_API:  # os bytes nunca saem para outro host
            raise RequisicaoRecusada("a Files API devolveu uma URL de envio inesperada")
        hash_enviado = hashlib.sha256()

        def blocos():
            with caminho.open("rb") as origem:
                while bloco := origem.read(BLOCO_ENVIO_BYTES):
                    hash_enviado.update(bloco)
                    yield bloco

        try:
            resposta, _ = self._requisitar(
                "arquivo_enviar", "POST", url_envio, verificar=False, content=blocos(),
                headers={"Content-Length": str(tamanho), "X-Goog-Upload-Offset": "0",
                         "X-Goog-Upload-Command": "upload, finalize"},
                timeout=httpx.Timeout(TEMPO_LIMITE_ENVIO_S))
        except OSError:
            self._registrar_resultado("arquivo_enviar", 0, "arquivo_local_ilegivel")
            raise RequisicaoRecusada("o arquivo local não pôde ser lido durante o envio") from None
        dados = self._json(resposta)
        if resposta.status_code != 200:
            self._levantar("arquivo_enviar", 0, resposta, dados, arquivo=True)
        self._registrar_resultado("arquivo_enviar", 0, "ok")
        return self._arquivo(dados, hash_enviado.hexdigest())

    def consultar_arquivo(self, nome: str) -> ArquivoGemini:
        nome = self._validar_nome(nome)
        resposta, _ = self._requisitar("arquivo_estado", "GET", f"/{nome}", headers=self._cabecalho_chave())
        dados = self._json(resposta)
        if resposta.status_code != 200:
            self._levantar("arquivo_estado", 0, resposta, dados, arquivo=True)
        self._registrar_resultado("arquivo_estado", 0, "ok")
        return self._arquivo(dados)

    def aguardar_ativo(self, nome: str, *, intervalo_s: float = 5.0, max_consultas: int = 12) -> ArquivoGemini:
        """Consulta o estado até ACTIVE; FAILED ou esgotar `max_consultas` é erro (nunca espera sem limite)."""
        for numero in range(1, max_consultas + 1):
            arquivo = self.consultar_arquivo(nome)
            if arquivo.estado == "ACTIVE":
                return arquivo
            if arquivo.estado == "FAILED":
                raise ArquivoComFalha("o Gemini não conseguiu processar o arquivo enviado")
            if numero < max_consultas:
                self._espera(intervalo_s)
        raise ArquivoNaoFicouAtivo(f"o arquivo não ficou ativo em {max_consultas} consultas")

    def apagar_arquivo(self, nome: str) -> None:
        """Exclui o arquivo. Ignora os tetos e o bloqueio por cota; é contada e registrada como as demais."""
        nome = self._validar_nome(nome)
        resposta, _ = self._requisitar("arquivo_apagar", "DELETE", f"/{nome}", verificar=False,
                                       headers=self._cabecalho_chave())
        if resposta.status_code != 200:
            self._levantar("arquivo_apagar", 0, resposta, self._json(resposta), arquivo=True)
        self._registrar_resultado("arquivo_apagar", 0, "ok")

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


# ──── Avisos tipados para as falhas do Gemini (cota, taxa, teto da execução, resposta inválida, envio de a
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
