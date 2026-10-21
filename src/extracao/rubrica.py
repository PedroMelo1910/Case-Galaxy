"""Módulo consolidado. Reúne:
- Rubrica v1 de atributos observáveis e contrato de transcrição da extração pelo Gemini.
- Extração de um vídeo pelo Gemini: pedido, política de tentativas e estados do resultado.
- Cache da extração pelo Gemini: um JSON por vídeo, com a configuração que o produziu.
- Anotação de um vídeo LOCAL pela Files API do Gemini, com o mesmo modelo, rubrica, esquema e temperatura da rota por URL."""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    create_model,
    Field,
    model_validator,
    StringConstraints,
    ValidationError,
)

from src.dados.coleta import IdVideo, ItemExtracao, Setor
from src.extracao.video_local import MAX_DURACAO_S, SondaVideo
from src.integracoes import gemini as ga


# ──── Rubrica v1 de atributos observáveis e contrato de transcrição da extração pelo Gemini.
VERSAO_RUBRICA = "r1"
VERSAO_PROMPT = "p1"
SETORES_SUPORTADOS = (Setor.BELEZA, Setor.AUTOMOTIVO)
OPCOES_SIM_NAO = ("sim", "nao", "indeterminado")
OPCOES_TOM = ("ausente", "calmo", "animado", "instrucional", "indeterminado")
STATUS_DO_MODELO = ("ok", "sem_fala", "incerta")
PADRAO_IDIOMA = r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$"
MAX_ITENS_LISTA = 5
LIMITES_TEXTO = {"curto": 80, "medio": 400, "transcricao": 8000}

SimNao = Literal["sim", "nao", "indeterminado"]
Nota = Annotated[int, Field(ge=0, le=10)] | None
Idioma = Annotated[str, StringConstraints(pattern=PADRAO_IDIOMA)] | None
ItemCurto = Annotated[str, StringConstraints(min_length=1, max_length=LIMITES_TEXTO["curto"])]


class RespostaInvalida(ValueError):
    """Resposta do modelo que não serve como anotação. `categoria`: sintaxe, esquema ou coerencia."""

    def __init__(self, categoria: str, motivo: str):
        super().__init__(f"{categoria}: {motivo}")
        self.categoria = categoria
        self.motivo = motivo


@dataclass(frozen=True)
class Campo:
    nome: str
    tipo: str  # sim_nao, nota, texto_curto, texto_medio, transcricao, lista, tom, status, idioma
    descricao: str


CAMPOS_COMUNS = (
    Campo(
        "hook_primeiros_3s",
        "texto_medio",
        "Descrição objetiva, em até duas frases, do que se vê e ouve nos primeiros 3 segundos.",
    ),
    Campo(
        "forca_hook_0a10",
        "nota",
        "Quanto os primeiros 3 segundos prendem a atenção por elementos observáveis (movimento, rosto, pergunta, contraste, texto).",
    ),
    Campo("tem_rosto", "sim_nao", "Algum rosto humano aparece de forma reconhecível em algum momento."),
    Campo(
        "close_up_produto",
        "sim_nao",
        "O produto ou objeto principal aparece em plano fechado, ocupando boa parte do quadro.",
    ),
    Campo(
        "before_after_ou_transformacao", "sim_nao", "Mostra comparação de antes e depois ou uma transformação visível."
    ),
    Campo("try_on_ou_demonstracao", "sim_nao", "Alguém usa, aplica, veste ou demonstra o produto em funcionamento."),
    Campo(
        "texto_na_tela",
        "sim_nao",
        "Há texto legível sobreposto ou escrito na tela (legendas, títulos, preços), além de logotipos.",
    ),
    Campo("densidade_texto_0a10", "nota", "Quantidade de texto na tela ao longo do vídeo; 0 = nenhum texto."),
    Campo(
        "ritmo_cortes_0a10",
        "nota",
        "Frequência de cortes de cena; 0 = plano único sem cortes, 10 = cortes muito frequentes.",
    ),
    Campo("energia_musica_0a10", "nota", "Energia da música de fundo; 0 = não há música."),
    Campo("tom_narracao", "tom", "Tom predominante da voz falada; ausente = não há fala."),
    Campo(
        "brilho_saturacao_0a10",
        "nota",
        "Brilho e saturação das cores; 0 = escuro e sem cor, 10 = muito claro e saturado.",
    ),
    Campo("cor_dominante", "texto_curto", "Cor predominante na imagem, em uma ou duas palavras."),
    Campo(
        "estetica_geral_0a10", "nota", "Qualidade visual da produção: nitidez, iluminação, enquadramento e acabamento."
    ),
    Campo(
        "clareza_beneficio_produto_0a10",
        "nota",
        "Clareza com que o vídeo comunica um benefício do produto; null se não houver produto identificável.",
    ),
    Campo("elementos_video_top5", "lista", "Até 5 elementos visuais mais marcantes, em palavras curtas."),
    Campo(
        "elementos_audio_top5",
        "lista",
        "Até 5 elementos sonoros mais marcantes (voz, música, efeitos), em palavras curtas.",
    ),
    Campo("transcricao_status", "status", "ok, sem_fala ou incerta, conforme as regras de transcrição do pedido."),
    Campo("transcricao_texto", "transcricao", "Fala transcrita literalmente, no idioma original; vazio se sem_fala."),
    Campo(
        "transcricao_idioma",
        "idioma",
        "Código BCP 47 do idioma falado predominante (ex.: pt-BR); null se sem_fala ou desconhecido.",
    ),
)

CAMPOS_POR_SETOR = {
    Setor.BELEZA: (
        Campo("swatch", "sim_nao", "Mostra amostra de cor ou textura do produto aplicada na pele ou numa superfície."),
        Campo(
            "aplicacao_na_pele", "sim_nao", "Mostra o produto sendo aplicado na pele, no rosto, no cabelo ou nas unhas."
        ),
    ),
    Setor.AUTOMOTIVO: (
        Campo("close_do_carro", "sim_nao", "Mostra o veículo ou partes dele em plano fechado."),
        Campo("som_do_motor", "sim_nao", "O som do motor ou do escapamento é audível."),
        Campo("pov_direcao", "sim_nao", "Câmera no ponto de vista de quem dirige, com o veículo em movimento."),
        Campo("interior", "sim_nao", "Mostra o interior do veículo (painel, bancos, cabine)."),
    ),
}

CAMPOS_SIM_NAO_COMUNS = tuple(c.nome for c in CAMPOS_COMUNS if c.tipo == "sim_nao")

_TIPO_PYTHON: dict[str, Any] = {
    "sim_nao": SimNao,
    "nota": Nota,
    "texto_curto": Annotated[str, StringConstraints(max_length=LIMITES_TEXTO["curto"])],
    "texto_medio": Annotated[str, StringConstraints(max_length=LIMITES_TEXTO["medio"])],
    "transcricao": Annotated[str, StringConstraints(max_length=LIMITES_TEXTO["transcricao"])],
    "lista": Annotated[tuple[ItemCurto, ...], Field(max_length=MAX_ITENS_LISTA)],
    "tom": Literal["ausente", "calmo", "animado", "instrucional", "indeterminado"],
    "status": Literal["ok", "sem_fala", "incerta"],
    "idioma": Idioma,
}

_TIPO_JSON: dict[str, dict] = {
    "sim_nao": {"type": "string", "enum": list(OPCOES_SIM_NAO)},
    "nota": {"type": ["integer", "null"], "minimum": 0, "maximum": 10},
    "texto_curto": {"type": "string"},
    "texto_medio": {"type": "string"},
    "transcricao": {"type": "string"},
    "lista": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_ITENS_LISTA},
    "tom": {"type": "string", "enum": list(OPCOES_TOM)},
    "status": {"type": "string", "enum": list(STATUS_DO_MODELO)},
    "idioma": {"type": ["string", "null"]},
}


class _AnotacaoBase(BaseModel):
    """Regras de coerência da transcrição, comuns aos setores."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    @model_validator(mode="after")
    def _transcricao_coerente(self) -> "_AnotacaoBase":
        status, texto, idioma = self.transcricao_status, self.transcricao_texto, self.transcricao_idioma  # type: ignore[attr-defined]
        if status == "ok" and (not texto.strip() or idioma is None):
            raise ValueError("status ok exige texto transcrito e idioma")
        if status == "sem_fala" and (texto.strip() or idioma is not None):
            raise ValueError("status sem_fala exige texto vazio e idioma nulo")
        return self


def _verificar_setor(setor: Setor) -> None:
    if setor not in SETORES_SUPORTADOS:
        raise ValueError(f"setor sem rubrica nesta versão: {setor}")


def campos_do_setor(setor: Setor) -> tuple[Campo, ...]:
    _verificar_setor(setor)
    return CAMPOS_COMUNS + CAMPOS_POR_SETOR[setor]


def campos_sim_nao(setor: Setor) -> tuple[str, ...]:
    return tuple(c.nome for c in campos_do_setor(setor) if c.tipo == "sim_nao")


def _criar_modelo(setor: Setor) -> type[_AnotacaoBase]:
    definicoes = {c.nome: (_TIPO_PYTHON[c.tipo], Field(description=c.descricao)) for c in campos_do_setor(setor)}
    return create_model(f"Anotacao{str(setor).capitalize()}", __base__=_AnotacaoBase, **definicoes)


_MODELOS = {setor: _criar_modelo(setor) for setor in SETORES_SUPORTADOS}


def modelo_do_setor(setor: Setor) -> type[_AnotacaoBase]:
    _verificar_setor(setor)
    return _MODELOS[setor]


def esquema_json(setor: Setor) -> dict:
    """Esquema no subconjunto de JSON Schema aceito pelo `responseJsonSchema` do Gemini."""
    campos = campos_do_setor(setor)
    return {
        "type": "object",
        "properties": {c.nome: _TIPO_JSON[c.tipo] | {"description": c.descricao} for c in campos},
        "required": [c.nome for c in campos],
        "additionalProperties": False,
    }


_NOME_SETOR = {Setor.BELEZA: "beleza", Setor.AUTOMOTIVO: "automotivo"}

_PEDIDO = """Analise este YouTube Short do setor {setor} e preencha o JSON pedido.

Regras:
1. Use somente o que se vê e se ouve no próprio vídeo. Ignore título, descrição, canal e qualquer informação externa.
2. Não estime desempenho, alcance ou popularidade do vídeo; descreva apenas atributos observáveis.
3. Campos sim/nao: responda "indeterminado" quando não for possível observar com segurança. Nunca use "nao" para indicar dúvida.
4. Notas de 0 a 10: use null quando não for possível avaliar. O valor 0 significa ausência observada, nunca dúvida.
5. Transcrição da fala:
   - Fala são palavras faladas por pessoas ou por voz sintética (narração, locução, diálogo, voz de fundo inteligível), em qualquer idioma. Letra de música cantada e texto escrito na tela não são fala.
   - Transcreva literalmente, no idioma original, sem traduzir, resumir ou corrigir.
   - "ok": há fala e ela foi transcrita por inteiro; informe o idioma (código BCP 47, como pt-BR).
   - "sem_fala": não há fala; deixe o texto vazio e o idioma null.
   - "incerta": há fala, mas parte dela é ininteligível ou você não tem segurança; transcreva o que entender, marque os trechos não compreendidos com [inaudível] e informe o idioma se souber, senão null."""


def texto_do_pedido(setor: Setor) -> str:
    _verificar_setor(setor)
    return _PEDIDO.format(setor=_NOME_SETOR[setor])


def _motivo_de_validacao(erro: ValidationError) -> tuple[str, str]:
    erros = erro.errors(include_url=False, include_input=False)
    categoria = "coerencia" if all(not e["loc"] for e in erros) else "esquema"
    partes = [f"{'.'.join(str(p) for p in e['loc']) or 'resposta'}: {e['msg']}" for e in erros[:5]]
    return categoria, "; ".join(partes)


def validar_resposta(texto: str, setor: Setor) -> _AnotacaoBase:
    """Anotação validada (esquema + coerência) a partir do texto devolvido pelo modelo."""
    try:
        dados = json.loads(texto)
    except (TypeError, ValueError) as erro:
        raise RespostaInvalida("sintaxe", f"texto não é JSON ({type(erro).__name__})") from erro
    if not isinstance(dados, dict):
        raise RespostaInvalida("sintaxe", "JSON não é um objeto")
    try:
        return modelo_do_setor(setor).model_validate_json(texto)
    except ValidationError as erro:
        raise RespostaInvalida(*_motivo_de_validacao(erro)) from erro


# ──── Extração de um vídeo pelo Gemini: pedido, política de tentativas e estados do resultado.
MAX_CARACTERES_RESPOSTA = 20_000


@dataclass(frozen=True)
class ConfiguracaoExtracao:
    """Tudo o que muda a resposta do modelo. Qualquer alteração muda a chave de compatibilidade do cache."""

    modelo: str
    temperatura: float
    max_tokens_saida: int = 16_384
    processamento_midia: str = "STATIC"
    tipo_midia: str = "video/mp4"  # a API exige o tipo quando o modo de processamento é declarado
    versao_rubrica: str = VERSAO_RUBRICA
    versao_prompt: str = VERSAO_PROMPT

    def parametros_geracao(self) -> dict:
        return {
            "responseMimeType": "application/json",
            "temperature": self.temperatura,
            "maxOutputTokens": self.max_tokens_saida,
        }

    def descricao(self) -> dict:
        return asdict(self) | {"ferramentas": [], "pedido_e_esquema_sha256": self._hash_pedidos()}

    def _hash_pedidos(self) -> str:
        conteudo = {str(s): [texto_do_pedido(s), esquema_json(s)] for s in SETORES_SUPORTADOS}
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
        "contents": [{"role": "user", "parts": [parte_video, {"text": texto_do_pedido(setor)}]}],
        "generationConfig": config.parametros_geracao() | {"responseJsonSchema": esquema_json(setor)},
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
        anotacao = validar_resposta(resposta.texto, setor)
    except RespostaInvalida as erro:
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


# ──── Cache da extração pelo Gemini: um JSON por vídeo, com a configuração que o produziu.
FORMATO = 1
FORMATO_UPLOAD = 1
RETENCAO_UPLOAD_DIAS = 30
ROTA_UPLOAD = "upload_files_api"
_PADRAO_VIDEO = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PADRAO_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMATO_INSTANTE = "%Y-%m-%dT%H:%M:%SZ"


class CacheIncompativel(RuntimeError):
    """Arquivo produzido por outra configuração (modelo, pedido, esquema ou parâmetros)."""


class CacheCorrompido(RuntimeError):
    """Arquivo ilegível, de formato desconhecido ou com anotação que não passa na validação."""


class CacheAnotacoes:
    def __init__(self, raiz: Path, config: ConfiguracaoExtracao):
        self.raiz = Path(raiz)
        self.config = config
        self.chave = config.chave_compatibilidade()

    def caminho(self, video_id: str) -> Path:
        if not _PADRAO_VIDEO.fullmatch(video_id):
            raise ValueError(f"video_id inválido: {video_id!r}")
        return self.raiz / f"{video_id}.json"

    def ler(self, video_id: str) -> ResultadoAnotacao | None:
        caminho = self.caminho(video_id)
        if not caminho.is_file():
            return None
        try:
            conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError) as erro:
            raise CacheCorrompido(f"{video_id}: arquivo de cache ilegível ({type(erro).__name__})") from erro
        if not isinstance(conteudo, dict) or conteudo.get("formato") != FORMATO:
            raise CacheCorrompido(f"{video_id}: formato de cache desconhecido")
        if conteudo.get("chave_compatibilidade") != self.chave:
            raise CacheIncompativel(
                f"{video_id}: o cache foi gerado com outra configuração; use outra pasta ou nova versão, sem reaproveitar"
            )
        return self._resultado(video_id, conteudo.get("resultado"))

    def _resultado(self, video_id: str, dados: object) -> ResultadoAnotacao:
        try:
            resultado = ResultadoAnotacao.model_validate(dados)
            if resultado.anotacao is not None:
                modelo_do_setor(resultado.setor).model_validate_json(json.dumps(resultado.anotacao))
        except (ValidationError, ValueError) as erro:
            raise CacheCorrompido(f"{video_id}: conteúdo do cache não passa na validação") from erro
        if resultado.video_id != video_id:
            raise CacheCorrompido(f"{video_id}: o arquivo contém outro vídeo")
        return resultado

    def gravar(self, item: ItemExtracao, resultado: ResultadoAnotacao, *, agora: datetime | None = None) -> Path:
        if resultado.video_id != item.video_id:
            raise ValueError("o resultado e o item têm video_id diferentes")
        instante = (agora or datetime.now(UTC)).astimezone(UTC)
        conteudo = {
            "formato": FORMATO,
            "chave_compatibilidade": self.chave,
            "configuracao": self.config.descricao(),
            "item": item.model_dump(mode="json"),
            "resultado": resultado.model_dump(mode="json"),
            "gravado_em_utc": instante.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return gravar_json_atomico(self.caminho(item.video_id), conteudo)


class CacheUploads:
    """Anotações de vídeos enviados pelo usuário, chaveadas pelo SHA-256 do ARQUIVO (nunca por um `video_id` falso).

    Guarda só o que o cache de URL guarda (a anotação validada e a configuração que a produziu) mais o hash do arquivo,
    o setor e a data de expiração. Nenhum nome, caminho ou byte do vídeo. Retenção: `RETENCAO_UPLOAD_DIAS` dias a partir
    da gravação (mesma regra de 30 dias dos dados derivados do YouTube, por prudência); depois disso a entrada é
    apagada na leitura. Só anotações válidas são guardadas. Chave de compatibilidade própria da rota, derivada da
    configuração de extração, para que uma mudança de modelo, rubrica, pedido ou parâmetros invalide a entrada.
    """

    def __init__(self, raiz: Path, config: ConfiguracaoExtracao, *, agora: Callable[[], datetime] | None = None):
        self.raiz = Path(raiz)
        self.config = config
        self._agora = agora or (lambda: datetime.now(UTC))
        self.chave = hashlib.sha256(json.dumps({"base": config.chave_compatibilidade(), "rota": ROTA_UPLOAD},
                                               sort_keys=True).encode("utf-8")).hexdigest()

    def caminho(self, sha256: str) -> Path:
        if not _PADRAO_SHA256.fullmatch(sha256):
            raise ValueError("sha256 inválido")
        return self.raiz / f"{sha256}.json"

    def ler(self, sha256: str, setor: str) -> dict | None:
        """Anotação guardada para o arquivo, ou None (ausente, expirada ou de outro setor)."""
        caminho = self.caminho(sha256)
        if not caminho.is_file():
            return None
        try:
            conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError) as erro:
            raise CacheCorrompido(f"upload {sha256[:12]}: arquivo de cache ilegível ({type(erro).__name__})") from erro
        if not isinstance(conteudo, dict) or conteudo.get("formato") != FORMATO_UPLOAD:
            raise CacheCorrompido(f"upload {sha256[:12]}: formato de cache desconhecido")
        if conteudo.get("chave_compatibilidade") != self.chave:
            raise CacheIncompativel(f"upload {sha256[:12]}: o cache foi gerado com outra configuração")
        try:
            expira = datetime.strptime(conteudo["expira_em_utc"], _FORMATO_INSTANTE).replace(tzinfo=UTC)
        except (KeyError, ValueError, TypeError) as erro:
            raise CacheCorrompido(f"upload {sha256[:12]}: data de expiração ilegível") from erro
        if expira <= self._agora():
            caminho.unlink(missing_ok=True)
            return None
        if conteudo.get("sha256_arquivo") != sha256 or conteudo.get("setor") != setor:
            return None
        try:
            modelo_do_setor(setor).model_validate_json(json.dumps(conteudo.get("anotacao")))
        except (ValidationError, ValueError) as erro:
            raise CacheCorrompido(f"upload {sha256[:12]}: anotação não passa na validação") from erro
        return dict(conteudo["anotacao"])

    def gravar(self, sha256: str, setor: str, anotacao: dict) -> Path:
        agora = self._agora().astimezone(UTC)
        conteudo = {
            "formato": FORMATO_UPLOAD, "chave_compatibilidade": self.chave, "rota": ROTA_UPLOAD,
            "configuracao": self.config.descricao(), "origem": "arquivo_enviado_pelo_usuario",
            "sha256_arquivo": sha256, "setor": setor, "anotacao": anotacao,
            "gravado_em_utc": agora.strftime(_FORMATO_INSTANTE),
            "expira_em_utc": (agora + timedelta(days=RETENCAO_UPLOAD_DIAS)).strftime(_FORMATO_INSTANTE),
        }
        return gravar_json_atomico(self.caminho(sha256), conteudo)


def gravar_json_atomico(destino: Path, conteudo: object) -> Path:
    """Grava num temporário da mesma pasta e renomeia; em caso de falha, remove o temporário."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    dados = json.dumps(conteudo, ensure_ascii=False, indent=1).encode("utf-8")
    descritor, temporario = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".parcial", dir=destino.parent)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(dados)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, destino)
    except BaseException:
        Path(temporario).unlink(missing_ok=True)
        raise
    return destino


# ──── Anotação de um vídeo LOCAL pela Files API do Gemini, com o mesmo modelo, rubrica, esquema e temperat
logger = logging.getLogger(__name__)
REQUISICOES_MINIMAS = 4  # iniciar, enviar, uma consulta de estado e a geração (a exclusão ignora os tetos)
TOLERANCIA_DURACAO_S = 1.0


@dataclass(frozen=True)
class ResultadoUpload:
    """`estado`: ok | cache | indisponivel | falha_...; `anotacao` só existe em ok e cache."""

    anotacao: dict | None
    estado: str
    avisos: list[dict] = field(default_factory=list)


def _aviso_nao_apagado() -> dict:
    return {"classe": "atencao", "codigo": "arquivo_gemini_nao_apagado",
            "descricao": "O arquivo temporário enviado ao Gemini não pôde ser apagado agora; ele expira sozinho em 48 horas."}


@dataclass
class FonteAnotacaoArquivo:
    """Anotação de arquivo local. Sem `cliente` só lê o cache; sem nenhum dos dois, o estado é `indisponivel`."""

    config: ConfiguracaoExtracao
    cache: CacheUploads | None = None
    cliente: ga.ClienteGemini | None = None
    intervalo_estado_s: float = 5.0
    max_consultas_estado: int = 12

    def anotar(self, sonda: SondaVideo, setor: Setor) -> ResultadoUpload:
        guardada = self._do_cache(sonda, setor)
        if guardada is not None:
            return ResultadoUpload(guardada, "cache")
        if self.cliente is None:
            return ResultadoUpload(None, "indisponivel")
        try:
            self.cliente.verificar_orcamento(sonda.duracao_inteira_s, REQUISICOES_MINIMAS)
        except ga.ErroGemini as erro:
            return ResultadoUpload(None, _estado_de_erro(erro))
        return self._enviar_e_anotar(sonda, setor)

    def _do_cache(self, sonda: SondaVideo, setor: Setor) -> dict | None:
        if self.cache is None:
            return None
        try:
            return self.cache.ler(sonda.sha256, str(setor))
        except (CacheIncompativel, CacheCorrompido) as erro:
            logger.warning("cache de upload ignorado (%s)", type(erro).__name__)
            return None

    def _enviar_e_anotar(self, sonda: SondaVideo, setor: Setor) -> ResultadoUpload:
        try:
            enviado = self.cliente.enviar_arquivo(sonda.caminho, sonda.mime,
                                                  nome_exibicao=f"casegalaxy-{sonda.sha256[:12]}")
        except ga.ErroGemini as erro:  # sem nome devolvido não há o que apagar
            return ResultadoUpload(None, _estado_de_erro(erro))
        avisos: list[dict] = []
        try:
            resultado = self._anotar_arquivo_enviado(sonda, setor, enviado)
        except ga.ErroGemini as erro:
            resultado = ResultadoUpload(None, _estado_de_erro(erro))
        finally:
            self._apagar(enviado.nome, avisos)
        return replace(resultado, avisos=avisos)

    def _anotar_arquivo_enviado(self, sonda: SondaVideo, setor: Setor, enviado: ga.ArquivoGemini) -> ResultadoUpload:
        divergencia = _divergencia(sonda, enviado)
        if divergencia is not None:
            return ResultadoUpload(None, divergencia)
        ativo = self.cliente.aguardar_ativo(enviado.nome, intervalo_s=self.intervalo_estado_s,
                                            max_consultas=self.max_consultas_estado)
        divergencia = _divergencia_de_duracao(ativo)
        if divergencia is not None:
            return ResultadoUpload(None, divergencia)
        anotacao, estado = self._gerar(sonda, setor, ativo.uri)
        if anotacao is not None and self.cache is not None:
            self._guardar(sonda, setor, anotacao)
        return ResultadoUpload(anotacao, estado)

    def _guardar(self, sonda: SondaVideo, setor: Setor, anotacao: dict) -> None:
        """A anotação já foi paga em cota: falha de disco no cache não pode jogá-la fora."""
        try:
            self.cache.gravar(sonda.sha256, str(setor), anotacao)
        except OSError:
            logger.warning("o cache de uploads não pôde ser gravado; a anotação segue sem cache")

    def _gerar(self, sonda: SondaVideo, setor: Setor, uri: str) -> tuple[dict | None, str]:
        corpo = corpo_para_arquivo(uri, sonda.mime, setor, self.config)
        resposta = self.cliente.gerar(corpo, duracao_video_s=sonda.duracao_inteira_s)
        if resposta.situacao != "ok" or resposta.texto is None:
            return None, f"falha_resposta_{resposta.situacao}"
        try:
            return validar_resposta(resposta.texto, setor).model_dump(mode="json"), "ok"
        except RespostaInvalida as erro:
            return None, f"falha_invalida_{erro.categoria}"

    def _apagar(self, nome: str, avisos: list[dict]) -> None:
        try:
            self.cliente.apagar_arquivo(nome)
        except Exception as falha:
            logger.warning("arquivo temporário do Gemini não apagado (%s)", type(falha).__name__)
            avisos.append(_aviso_nao_apagado())


def _estado_de_erro(erro: ga.ErroGemini) -> str:
    return "falha_cota_esgotada" if isinstance(erro, ga.CotaDiariaEsgotada) else f"falha_{type(erro).__name__}"


def _divergencia(sonda: SondaVideo, enviado: ga.ArquivoGemini) -> str | None:
    if enviado.sha256_enviado != sonda.sha256:
        return "falha_arquivo_alterado"
    if enviado.tamanho_bytes is not None and enviado.tamanho_bytes != sonda.tamanho_bytes:
        return "falha_tamanho_divergente"
    return None


def _divergencia_de_duracao(ativo: ga.ArquivoGemini) -> str | None:
    if ativo.duracao_s is not None and ativo.duracao_s > MAX_DURACAO_S + TOLERANCIA_DURACAO_S:
        return "falha_duracao_do_arquivo"
    return None
