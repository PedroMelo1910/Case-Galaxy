"""Módulo consolidado. Reúne:
- Mensagens compreensíveis para os erros do envelope `{success: false, error}` e para falhas da própria interface.
- Canais da base curada do produto (a mesma base de histórico que o agente usa para canais conhecidos).
- Leitura do histórico de posts do canal informado pelo usuário (texto colado ou arquivo CSV).
- Do formulário ao pedido do agente: validação com mensagens claras e montagem do `dict` de `prever`."""

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import RAIZ_PROJETO
from src.interface_web.textos import exibivel


# ──── Mensagens compreensíveis para os erros do envelope `{success: false, error}` e para falhas da própri
COMANDO_VERIFICAR = "uv run casegalaxy-demo --verificar-ambiente"
COMANDO_BAIXAR = "uv run casegalaxy-demo --baixar-codificadores"
COMANDO_PREPARAR_AV = "uv run casegalaxy-preparar-av"

ROTULOS_DE_CAMPO = {
    "creative": "Vídeo", "creative.arquivo": "Vídeo", "creative.tipo": "Vídeo", "caption": "Legenda",
    "caption.title": "Título", "caption.description": "Descrição", "planned_datetime": "Data e hora planejadas",
    "sector": "Setor", "channel_id": "Canal", "channel_stats": "Histórico enviado",
    "channel_stats.video_id": "Histórico enviado", "channel_type": "Tipo do canal", "own_video_id": "ID do próprio vídeo",
    "video": "Vídeo", "setor": "Setor", "data": "Data e hora planejadas", "canal": "Canal", "tipo_canal": "Tipo do canal",
    "historico": "Histórico enviado", "titulo": "Título", "descricao": "Legenda ou contexto",
    "duration_s": "Duração", "video_route": "Rota do vídeo", "rota_video": "Rota do vídeo", "consultar_youtube": "Histórico ao vivo", "thumbnail": "Capa", "tags": "Tags",
}


@dataclass(frozen=True)
class MensagemErro:
    titulo: str
    texto: str
    itens: tuple[str, ...] = field(default_factory=tuple)
    dicas: tuple[str, ...] = field(default_factory=tuple)


_TITULOS = {
    "ENTRADA_INVALIDA": "Alguma informação precisa de ajuste",
    "MIDIA_INVALIDA": "Não foi possível usar este vídeo",
    "HISTORICO_INDISPONIVEL": "Sem histórico do canal",
    "MODELO_INDISPONIVEL": "Modelo ou pesos indisponíveis neste computador",
    "COTA_ESGOTADA": "Cota de consulta esgotada",
    "ERRO_INTERNO": "A análise não pôde ser concluída",
}
TITULO_GENERICO = "A análise não pôde ser concluída"
_DICAS = {
    "MODELO_INDISPONIVEL": (
        f"Para ver o que falta, rode no terminal: {COMANDO_VERIFICAR}",
        f"Para obter os pesos dos codificadores (precisa de internet): {COMANDO_BAIXAR}",
        f"Para os pesos da extração do vídeo: {COMANDO_PREPARAR_AV}",
        "Sem o preditor local ativo, escolha a rota Gemini (exige o modo online) ou use 'Reprodução de resposta salva', "
        "que não precisa de modelos."),
    "MIDIA_INVALIDA": ("Use MP4, MOV ou WebM com até 100 MB e 180 s, em formato vertical ou quadrado.",),
    "HISTORICO_INDISPONIVEL": ("Escolha um canal da base ou envie o histórico de posts do canal.",),
    "COTA_ESGOTADA": ("Tente de novo mais tarde ou use o histórico enviado por você.",),
    "ERRO_INTERNO": ("Tente de novo. Se persistir, rode o verificador do ambiente descrito no manual de uso.",),
}


def _itens_de_detalhes(detalhes: dict) -> tuple[str, ...]:
    itens = []
    for erro in detalhes.get("erros", ()) or ():
        campo = str(erro.get("campo", ""))
        rotulo = ROTULOS_DE_CAMPO.get(campo, ROTULOS_DE_CAMPO.get(campo.split(".")[0], campo or "Entrada"))
        itens.append(f"{rotulo}: {exibivel(erro.get('problema', ''))}")
    return tuple(itens)


def mensagem_do_envelope(envelope: dict) -> MensagemErro:
    """Mensagem para um envelope de erro (ou para um envelope malformado)."""
    erro = envelope.get("error") if isinstance(envelope, dict) else None
    if not isinstance(erro, dict):
        return MensagemErro(_TITULOS["ERRO_INTERNO"], "A resposta do agente veio em formato inesperado.",
                            dicas=_DICAS["ERRO_INTERNO"])
    codigo = str(erro.get("code", "ERRO_INTERNO"))
    detalhes = erro.get("details") if isinstance(erro.get("details"), dict) else {}
    return MensagemErro(titulo=_TITULOS.get(codigo, TITULO_GENERICO),
                        texto=exibivel(erro.get("message") or "Falha sem descrição."),
                        itens=_itens_de_detalhes(detalhes), dicas=_DICAS.get(codigo, ()))


# ──── Canais da base curada do produto (a mesma base de histórico que o agente usa para canais conhecidos)
@dataclass(frozen=True)
class CanalBase:
    channel_id: str
    titulo: str
    setor: str
    tipo_canal: str

    @property
    def rotulo(self) -> str:
        return f"{self.titulo} ({self.tipo_canal})"


def listar_canais_da_base(raiz: Path = RAIZ_PROJETO) -> tuple[CanalBase, ...]:
    """Canais da base curada do manifesto de pacote ativo, ordenados por setor e título.

    Lê só a tabela de canais (título, setor e tipo registrados), depois de conferir o manifesto da base.
    Levanta exceção se a base não existir; o chamador mostra a lista como indisponível.
    """
    from src.dados import curadoria
    from src.modelos import pacote as ms

    versao = ms.versao_ativa(Path(raiz))
    if versao.get("contexto_execucao"):
        from src.dados.execucao import carregar
        _, base = carregar(Path(raiz), versao)
        return tuple(CanalBase(c.channel_id, c.channel_id, c.setor, c.tipo_canal)
                     for c in sorted(base.canais.values(), key=lambda c: (c.setor, c.channel_id)))
    tabelas = curadoria.ler_base(Path(raiz) / versao["base_historica"]["pasta"])
    linhas = tabelas["canais.parquet"].select(["channel_id", "titulo", "setor", "tipo_canal"]).to_pylist()
    canais = [CanalBase(l["channel_id"], l["titulo"] or l["channel_id"], l["setor"], l["tipo_canal"]) for l in linhas]
    return tuple(sorted(canais, key=lambda c: (c.setor, c.titulo.lower())))


# ──── Leitura do histórico de posts do canal informado pelo usuário (texto colado ou arquivo CSV).
MAX_POSTS = 200
COLUNAS_OBRIGATORIAS = ("published_at", "likes", "collected_at")
MODELO_CSV = "published_at,likes,collected_at,video_id\n2026-08-01T10:00:00Z,1200,2026-10-18T21:00:00Z,AAAAAAAAAAA"
_ID_VIDEO = re.compile(r"^[A-Za-z0-9_-]{11}$")
_MAX_MENSAGENS = 8


def _instante(texto: str) -> datetime:
    valor = datetime.fromisoformat(texto.strip().replace("Z", "+00:00"))
    return valor if valor.tzinfo else valor.replace(tzinfo=UTC)


def _separador(cabecalho: str) -> str:
    return max((";", "\t", ","), key=cabecalho.count)


def interpretar_historico(texto: str) -> tuple[list[dict], list[str]]:
    """(posts no formato do pedido, mensagens de erro). Com erro, a lista de posts vem vazia."""
    linhas = [l for l in texto.replace("\r\n", "\n").split("\n") if l.strip()]
    if not linhas:
        return [], ["O histórico está vazio."]
    leitor = csv.DictReader(io.StringIO("\n".join(linhas)), delimiter=_separador(linhas[0]))
    campos = [(c or "").strip().lower() for c in leitor.fieldnames or ()]
    faltam = [c for c in COLUNAS_OBRIGATORIAS if c not in campos]
    if faltam:
        return [], [f"Faltam colunas no cabeçalho: {', '.join(faltam)}. Use: {MODELO_CSV.splitlines()[0]}"]
    leitor.fieldnames = campos
    posts, erros = [], []
    for numero, linha in enumerate(leitor, start=2):
        if len(erros) >= _MAX_MENSAGENS:
            erros.append("Há mais linhas com problema; corrija as primeiras e envie de novo.")
            break
        try:
            posts.append(_post(linha))
        except ValueError as problema:
            erros.append(f"Linha {numero}: {problema}")
    ids = [p["video_id"] for p in posts if "video_id" in p]
    if not erros and len(ids) != len(set(ids)):
        erros.append("Há video_id repetido no histórico.")
    if not erros and len(posts) > MAX_POSTS:
        erros.append(f"O histórico tem {len(posts)} posts; o limite é {MAX_POSTS}.")
    if not erros and not posts:
        erros.append("O histórico não tem nenhum post.")
    return ([], erros) if erros else (posts, [])


def _post(linha: dict) -> dict:
    try:
        publicado, coletado = _instante(linha["published_at"] or ""), _instante(linha["collected_at"] or "")
    except (ValueError, AttributeError):
        raise ValueError("data inválida; use o formato 2026-08-01T10:00:00Z.") from None
    if coletado < publicado:
        raise ValueError("collected_at é anterior a published_at.")
    bruto = (linha.get("likes") or "").strip()
    if bruto and not bruto.isdigit():
        raise ValueError("likes deve ser um número inteiro sem separador de milhar (ou ficar vazio).")
    identificador = (linha.get("video_id") or "").strip()
    if identificador and not _ID_VIDEO.fullmatch(identificador):
        raise ValueError("video_id deve ter 11 caracteres (letras, números, - ou _).")
    post = {"published_at": publicado.isoformat(), "likes": int(bruto) if bruto else None,
            "collected_at": coletado.isoformat()}
    if identificador:
        post["video_id"] = identificador
    return post


# ──── Do formulário ao pedido do agente: validação com mensagens claras e montagem do `dict` de `prever`.
MODO_BASE = "base"
MODO_HISTORICO = "historico"
MODO_ONLINE = "online"
TIPO_NAO_INFORMADO = "nao_informar"
ROTA_AUTO, ROTA_LOCAL, ROTA_GEMINI = "auto", "local", "gemini"
ROTAS_DE_VIDEO = {
    ROTA_AUTO: "Automática: preditor local quando o vídeo é um arquivo",
    ROTA_LOCAL: "Preditor local (offline)",
    ROTA_GEMINI: "Gemini (online, opcional)",
}
SETORES = ("beleza", "automotivo")
EXTENSOES_VIDEO = ("mp4", "mov", "webm", "avi", "mpeg", "mpg", "flv", "wmv", "3gp")
MAX_BYTES_VIDEO = 100_000_000
MAX_CARACTERES = 10_000
FUSO = ZoneInfo("America/Sao_Paulo")
_TIPOS_VALIDOS = ("marca", "criador", TIPO_NAO_INFORMADO)


@dataclass(frozen=True)
class EntradaFormulario:
    arquivo_nome: str | None = None
    arquivo_bytes: int | None = None
    titulo: str = ""
    descricao: str = ""
    setor: str | None = None
    data: date | None = None
    hora: time | None = None
    modo_canal: str | None = None
    canal_base_id: str | None = None
    historico_texto: str = ""
    historico_conflito: bool = False  # texto colado e arquivo enviados ao mesmo tempo
    canal_online_id: str = ""
    tipo_canal: str | None = None
    own_video_id: str = ""
    rota_video: str = ROTA_AUTO
    online_disponivel: bool = False


@dataclass(frozen=True)
class ErroCampo:
    campo: str
    mensagem: str


def _extensao(nome: str) -> str:
    return nome.rsplit(".", 1)[-1].lower() if "." in nome else ""


def _erros_do_video(e: EntradaFormulario) -> list[ErroCampo]:
    if not e.arquivo_nome:
        return [ErroCampo("video", "Envie o vídeo que será analisado.")]
    erros = []
    if _extensao(e.arquivo_nome) not in EXTENSOES_VIDEO:
        erros.append(ErroCampo("video", "Formato não reconhecido. Use " + ", ".join(x.upper() for x in EXTENSOES_VIDEO[:3])
                               + " ou outro formato de vídeo comum."))
    if (e.arquivo_bytes or 0) > MAX_BYTES_VIDEO:
        erros.append(ErroCampo("video", "O vídeo passa de 100 MB. Envie um arquivo menor."))
    if e.arquivo_bytes == 0:
        erros.append(ErroCampo("video", "O arquivo está vazio."))
    return erros


def _erros_de_texto(e: EntradaFormulario) -> list[ErroCampo]:
    campos = (("titulo", "O título", e.titulo), ("descricao", "A legenda ou descrição", e.descricao))
    return [ErroCampo(c, f"{n} passa de {MAX_CARACTERES} caracteres.") for c, n, t in campos if len(t) > MAX_CARACTERES]


def _erros_de_canal(e: EntradaFormulario, canais: dict[str, CanalBase]) -> list[ErroCampo]:
    if e.modo_canal not in (MODO_BASE, MODO_HISTORICO, MODO_ONLINE):
        return [ErroCampo("canal", "Escolha como informar o contexto do canal.")]
    if e.modo_canal == MODO_BASE:
        canal = canais.get(e.canal_base_id or "")
        if canal is None:
            return [ErroCampo("canal", "Escolha um canal da base.")]
        if e.setor and canal.setor != e.setor:
            return [ErroCampo("canal", f"O canal escolhido é do setor {canal.setor}, diferente do setor selecionado.")]
        return []
    erros = []
    if e.tipo_canal not in _TIPOS_VALIDOS:
        erros.append(ErroCampo("tipo_canal", "Informe se o canal é marca ou criador (ou escolha 'não informar')."))
    if e.modo_canal == MODO_HISTORICO:
        erros += _erros_do_historico(e)
    else:
        erros += _erros_do_canal_online(e)
    return erros


def _erros_do_historico(e: EntradaFormulario) -> list[ErroCampo]:
    if e.historico_conflito:
        return [ErroCampo("historico", "Use só uma forma de informar o histórico: texto colado ou arquivo CSV.")]
    if not e.historico_texto.strip():
        return [ErroCampo("historico", "Cole ou envie o histórico de posts do canal (published_at, likes, collected_at).")]
    _, mensagens = interpretar_historico(e.historico_texto)
    return [ErroCampo("historico", m) for m in mensagens]


def _erros_do_canal_online(e: EntradaFormulario) -> list[ErroCampo]:
    if not e.online_disponivel:
        return [ErroCampo("canal", "A consulta ao vivo está desligada neste computador (modo online não habilitado).")]
    identificador = e.canal_online_id.strip()
    if not (identificador.startswith("UC") and len(identificador) == 24):
        return [ErroCampo("canal", "O ID do canal do YouTube começa com UC e tem 24 caracteres.")]
    return []


def _erros_da_rota(e: EntradaFormulario) -> list[ErroCampo]:
    if e.rota_video not in ROTAS_DE_VIDEO:
        return [ErroCampo("rota_video", "Escolha uma rota de vídeo válida.")]
    if e.rota_video == ROTA_GEMINI and not e.online_disponivel:
        return [ErroCampo("rota_video", "A rota Gemini exige o modo online, que está desligado neste computador.")]
    return []


def validar(e: EntradaFormulario, canais: dict[str, CanalBase]) -> list[ErroCampo]:
    """Todos os problemas do formulário, na ordem em que a tela os apresenta."""
    erros = _erros_do_video(e) + _erros_de_texto(e)
    if e.setor not in SETORES:
        erros.append(ErroCampo("setor", "Escolha o setor do vídeo."))
    if e.data is None or e.hora is None:
        erros.append(ErroCampo("data", "Informe a data e a hora planejadas para a publicação."))
    erros += _erros_de_canal(e, canais)
    erros += _erros_da_rota(e)
    identificador = e.own_video_id.strip()
    if identificador and not (len(identificador) == 11 and identificador.replace("-", "").replace("_", "").isalnum()):
        erros.append(ErroCampo("own_video_id", "O ID do vídeo no YouTube tem 11 caracteres."))
    return erros


def instante_planejado(data: date, hora: time) -> str:
    """Data e hora de Brasília em ISO 8601 com fuso UTC."""
    return datetime.combine(data, hora, tzinfo=FUSO).astimezone(UTC).isoformat().replace("+00:00", "Z")


def montar_pedido(e: EntradaFormulario, canais: dict[str, CanalBase], caminho: str | None) -> dict:
    """Pedido de `prever`. Com `caminho=None` o criativo fica sem arquivo (serve só para a chave do cache).

    Pressupõe `validar(e, canais) == []`.
    """
    criativo: dict = {"tipo": "video"}
    if caminho is not None:
        criativo["arquivo"] = caminho
    pedido: dict = {"creative": criativo, "sector": e.setor,
                    "planned_datetime": instante_planejado(e.data, e.hora)}
    if e.rota_video != ROTA_AUTO:  # sem o campo vale a rota automática do agente
        pedido["video_route"] = e.rota_video
    if e.titulo.strip() or e.descricao.strip():
        pedido["caption"] = {"title": e.titulo.strip() or None, "description": e.descricao.strip() or None}
    if e.own_video_id.strip():
        pedido["own_video_id"] = e.own_video_id.strip()
    if e.modo_canal == MODO_BASE:
        pedido["channel_id"] = e.canal_base_id
        return pedido
    if e.tipo_canal in ("marca", "criador"):
        pedido["channel_type"] = e.tipo_canal
    if e.modo_canal == MODO_HISTORICO:
        pedido["channel_stats"], _ = interpretar_historico(e.historico_texto)
    else:
        pedido["channel_id"] = e.canal_online_id.strip()
        pedido["consultar_youtube"] = True
    return pedido


def pedido_de_resposta_salva(url: str) -> dict:
    """Pedido mínimo do adaptador de respostas salvas: a resposta é escolhida pela URL do vídeo."""
    return {"creative": {"tipo": "video", "url": url}}
