"""Extração de um vídeo pelo Gemini: pedido, política de tentativas e estados do resultado.

O pedido leva só a URL pública do vídeo e o setor do canal (para escolher os campos setoriais),
nunca estatísticas de desempenho. Uma anotação só é válida depois da validação do conteúdo
(esquema e coerência), não apenas com HTTP 200. Quando nenhuma resposta válida é obtida, o
resultado é `falha_extracao`: os atributos ficam ausentes e a falha nunca equivale a "sem fala".
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from src import gemini_api as ga
from src import rubrica as rb
from src.contratos import IdVideo, ItemExtracao, Setor

MAX_CARACTERES_RESPOSTA = 20_000


@dataclass(frozen=True)
class ConfiguracaoExtracao:
    """Tudo o que muda a resposta do modelo. Qualquer alteração muda a chave de compatibilidade do cache."""

    modelo: str
    temperatura: float
    max_tokens_saida: int = 16_384
    processamento_midia: str = "STATIC"
    tipo_midia: str = "video/mp4"  # a API exige o tipo quando o modo de processamento é declarado
    versao_rubrica: str = rb.VERSAO_RUBRICA
    versao_prompt: str = rb.VERSAO_PROMPT

    def parametros_geracao(self) -> dict:
        return {
            "responseMimeType": "application/json",
            "temperature": self.temperatura,
            "maxOutputTokens": self.max_tokens_saida,
        }

    def descricao(self) -> dict:
        return asdict(self) | {"ferramentas": [], "pedido_e_esquema_sha256": self._hash_pedidos()}

    def _hash_pedidos(self) -> str:
        conteudo = {str(s): [rb.texto_do_pedido(s), rb.esquema_json(s)] for s in rb.SETORES_SUPORTADOS}
        return hashlib.sha256(json.dumps(conteudo, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def chave_compatibilidade(self) -> str:
        return hashlib.sha256(json.dumps(self.descricao(), sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Politica:
    """Limites por vídeo: requisições no total e novas tentativas depois de resposta inválida."""

    max_tentativas_por_video: int
    max_novas_tentativas_invalida: int
    espera_inicial_s: float = 5.0
    espera_maxima_s: float = 60.0

    def __post_init__(self) -> None:
        if self.max_tentativas_por_video < 1 or self.max_novas_tentativas_invalida < 0:
            raise ValueError("política inválida: pelo menos 1 tentativa e novas tentativas não negativas")
        if not 0 <= self.espera_inicial_s <= self.espera_maxima_s:
            raise ValueError("esperas inválidas: 0 <= espera_inicial_s <= espera_maxima_s")


POLITICA_ANOTACAO = Politica(max_tentativas_por_video=3, max_novas_tentativas_invalida=1)
POLITICA_REPETICAO = Politica(max_tentativas_por_video=2, max_novas_tentativas_invalida=0)


class Tentativa(BaseModel):
    """Uma requisição enviada, com o que voltou. Nunca contém a chave."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fase: Literal["anotacao", "repeticao"]
    numero: int = Field(ge=1)
    instante_utc: AwareDatetime
    resultado: str
    detalhe: str | None = None
    http_status: int | None = None
    duracao_s: float | None = None
    uso: dict = Field(default_factory=dict)
    versao_modelo: str | None = None
    texto_resposta: str | None = None


class ResultadoAnotacao(BaseModel):
    """Estado final da extração de um vídeo. `falha_extracao` deixa anotação e transcrição ausentes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: IdVideo
    setor: Setor
    estado: Literal["ok", "falha_extracao"]
    transcricao_status: Literal["ok", "sem_fala", "incerta", "falha_extracao"]
    anotacao: dict | None
    motivo_falha: str | None
    tentativas: tuple[Tentativa, ...]

    @property
    def tem_fala(self) -> bool | None:
        """Derivado do status: ok/incerta → sim; sem_fala → não; falha → desconhecido."""
        return {"ok": True, "incerta": True, "sem_fala": False}.get(self.transcricao_status)

    @model_validator(mode="after")
    def _estado_coerente(self) -> "ResultadoAnotacao":
        if self.estado == "ok":
            if self.anotacao is None or self.motivo_falha is not None:
                raise ValueError("estado ok exige anotação e nenhum motivo de falha")
            if self.anotacao.get("transcricao_status") != self.transcricao_status:
                raise ValueError("transcricao_status difere do registrado na anotação")
        elif self.anotacao is not None or self.transcricao_status != "falha_extracao" or not self.motivo_falha:
            raise ValueError("falha_extracao exige anotação ausente, status falha_extracao e motivo")
        return self


class ExecucaoInterrompida(RuntimeError):
    """Erro que vale para a execução inteira (cota diária, chave, modelo, teto). Guarda as tentativas do vídeo."""

    def __init__(self, codigo: str, mensagem: str, tentativas: tuple[Tentativa, ...]):
        super().__init__(f"{codigo}: {mensagem}")
        self.codigo = codigo
        self.tentativas = tentativas


def parte_de_video(uri: str, mime: str, config: ConfiguracaoExtracao) -> dict:
    """Parte `fileData` do pedido: URL pública ou URI da Files API, com o mesmo `mediaProcessing` da configuração."""
    return {"fileData": {"fileUri": uri, "mimeType": mime}, "mediaProcessing": config.processamento_midia}


def corpo_com_video(parte_video: dict, setor: Setor, config: ConfiguracaoExtracao) -> dict:
    """Pedido único para as duas rotas (URL e arquivo enviado): só muda a parte do vídeo, nunca rubrica, esquema,
    temperatura nem limite de saída, para que as anotações das duas rotas sejam comparáveis (verificação V24)."""
    return {
        "contents": [{"role": "user", "parts": [parte_video, {"text": rb.texto_do_pedido(setor)}]}],
        "generationConfig": config.parametros_geracao() | {"responseJsonSchema": rb.esquema_json(setor)},
    }


def corpo_para_url(url: str, setor: Setor, config: ConfiguracaoExtracao) -> dict:
    return corpo_com_video(parte_de_video(url, config.tipo_midia, config), setor, config)


def corpo_para_arquivo(uri: str, mime: str, setor: Setor, config: ConfiguracaoExtracao) -> dict:
    """Pedido de um vídeo enviado à Files API: o tipo MIME é o do arquivo real, o resto é igual ao da rota por URL."""
    return corpo_com_video(parte_de_video(uri, mime, config), setor, config)


def corpo_da_requisicao(item: ItemExtracao, config: ConfiguracaoExtracao) -> dict:
    """Pedido do generateContent: vídeo pela URL, seguido do texto; sem ferramentas nem metadados do vídeo."""
    if not isinstance(item, ItemExtracao):
        raise TypeError("o pedido ao Gemini só aceita ItemExtracao (contrato sem estatísticas de desempenho)")
    return corpo_para_url(item.url_video, item.setor, config)


def _espera(erro: ga.ErroGemini, ordem: int, politica: Politica) -> float:
    sugerida = erro.espera_sugerida_s if isinstance(erro, ga.LimiteDeTaxa) else None
    base = sugerida if sugerida is not None else politica.espera_inicial_s * 2 ** (ordem - 1)
    return min(base, politica.espera_maxima_s)


def _avaliar(resposta: ga.RespostaGemini, setor: Setor) -> tuple[str, dict | None, str | None]:
    if resposta.situacao != "ok" or resposta.texto is None:
        return f"resposta_{resposta.situacao}", None, None
    try:
        anotacao = rb.validar_resposta(resposta.texto, setor)
    except rb.RespostaInvalida as erro:
        return f"invalida_{erro.categoria}", None, erro.motivo
    return "valida", anotacao.model_dump(mode="json"), None


def _falha(item: ItemExtracao, motivo: str, tentativas: list[Tentativa]) -> ResultadoAnotacao:
    return ResultadoAnotacao(
        video_id=item.video_id,
        setor=item.setor,
        estado="falha_extracao",
        transcricao_status="falha_extracao",
        anotacao=None,
        motivo_falha=motivo,
        tentativas=tuple(tentativas),
    )


def anotar(
    item: ItemExtracao,
    cliente: ga.ClienteGemini,
    config: ConfiguracaoExtracao,
    *,
    politica: Politica,
    fase: Literal["anotacao", "repeticao"] = "anotacao",
    espera: Callable[[float], None] = time.sleep,
    agora: Callable[[], datetime] = lambda: datetime.now(UTC),
    ao_tentar: Callable[[Tentativa], None] | None = None,
) -> ResultadoAnotacao:
    """Anota um vídeo respeitando a política. Erros fatais viram `ExecucaoInterrompida`.

    `ao_tentar` recebe cada tentativa assim que ela termina, para que o registro de requisições
    não perca nenhuma, mesmo se a execução for interrompida no meio do vídeo.
    """
    if cliente.modelo != config.modelo:
        raise ValueError("o cliente e a configuração usam modelos diferentes")
    corpo = corpo_da_requisicao(item, config)
    tentativas: list[Tentativa] = []

    def registrar(tentativa: Tentativa) -> None:
        tentativas.append(tentativa)
        if ao_tentar is not None:
            ao_tentar(tentativa)

    invalidas = temporarias = 0
    for numero in range(1, politica.max_tentativas_por_video + 1):
        registro = {"fase": fase, "numero": numero, "instante_utc": agora()}
        try:
            resposta = cliente.gerar(corpo, duracao_video_s=item.duracao_s)
        except ga.ErroGemini as erro:
            if isinstance(erro, ga.LimiteDaExecucao):
                raise ExecucaoInterrompida(erro.codigo, str(erro), tuple(tentativas)) from None
            registrar(
                Tentativa(**registro, resultado=f"erro_{erro.codigo}", detalhe=str(erro), http_status=erro.status)
            )
            if erro.fatal:
                raise ExecucaoInterrompida(erro.codigo, str(erro), tuple(tentativas)) from None
            if isinstance(erro, ga.RequisicaoRecusada):
                return _falha(item, "requisicao_recusada", tentativas)
            temporarias += 1
            if numero < politica.max_tentativas_por_video:
                espera(_espera(erro, temporarias, politica))
            continue
        resultado, anotacao, detalhe = _avaliar(resposta, item.setor)
        registrar(
            Tentativa(
                **registro,
                resultado=resultado,
                detalhe=detalhe,
                http_status=200,
                duracao_s=round(resposta.duracao_s, 3),
                uso=resposta.uso,
                versao_modelo=resposta.versao_modelo,
                texto_resposta=resposta.texto[:MAX_CARACTERES_RESPOSTA] if resposta.texto else None,
            )
        )
        if anotacao is not None:
            return ResultadoAnotacao(
                video_id=item.video_id,
                setor=item.setor,
                estado="ok",
                transcricao_status=anotacao["transcricao_status"],
                anotacao=anotacao,
                motivo_falha=None,
                tentativas=tuple(tentativas),
            )
        invalidas += 1
        if invalidas > politica.max_novas_tentativas_invalida:
            return _falha(item, "resposta_invalida", tentativas)
    return _falha(item, "tentativas_esgotadas", tentativas)
