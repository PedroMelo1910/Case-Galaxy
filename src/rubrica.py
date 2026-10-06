"""Rubrica v1 de atributos observáveis e contrato de transcrição da extração pelo Gemini.

Uma tabela de campos é a fonte única do esquema JSON enviado ao Gemini e do modelo Pydantic que
valida a resposta. Desconhecido nunca vira "nao" ou 0: campos sim/não têm `indeterminado` e notas
aceitam `null`. O status `falha_extracao` não é oferecido ao modelo: só o código local o atribui,
quando a chamada falha ou a resposta não passa na validação.
"""

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, create_model, model_validator

from src.contratos import Setor

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
