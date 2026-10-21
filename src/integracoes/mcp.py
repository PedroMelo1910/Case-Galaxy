"""Módulo consolidado. Reúne:
- Fronteira do MCP: raízes de arquivo, caminhos, tamanhos e limites de entrada, antes de chamar o agente.
- Livro de consumo das chamadas externas do agente, no formato dos scripts de verificação ao vivo.
- Modo ONLINE do servidor MCP: opt-in por ambiente, com tetos por execução e livro de consumo obrigatório.
- Servidor MCP fino (stdio) sobre `prever` e `ordenar` do agente."""

import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.tools import Tool
from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel

from src.agente import api
from src.agente.contratos import (
    Codigo,
    envelope_erro,
    ErroAgente,
    MAX_BYTES_IMAGEM,
    MAX_CRIATIVOS_ORDENAR,
    MIN_CRIATIVOS_ORDENAR,
    PedidoExtrairAudiovisual,
    PedidoOrdenar,
    PedidoPrever,
)
from src.agente.servicos import MODELO_GEMINI, Servicos
from src.config import Configuracao
from src.dados.historico import criar_consultor
from src.extracao.modelos import (
    AmbienteIncompleto,
    CODIGO_SAIDA_AMBIENTE_INCOMPLETO,
    exigir_ambiente,
    MAX_BYTES as MAX_BYTES_EXTRACAO_LOCAL,
)
from src.extracao.video_local import MAX_BYTES_VIDEO
from src.integracoes import gemini as ga, youtube as mn, youtube as yt
from src.proveniencia import agora_utc as _agora_utc


# ──── Fronteira do MCP: raízes de arquivo, caminhos, tamanhos e limites de entrada, antes de chamar o agen
VARIAVEL_RAIZES = "CASEGALAXY_MCP_RAIZES"
MAX_CARACTERES_TEXTO = 10_000
MAX_POSTS_HISTORICO = 200
MAX_TAGS = 100  # acrescentado após a revisão: as tags entram no texto codificado e não podem burlar o limite de texto
MAX_CARACTERES_ID = 100  # `channel_id` de canal do YouTube tem 24 caracteres


class ErroConfiguracao(ValueError):
    """Configuração do servidor inválida (por exemplo, uma raiz permitida que não existe)."""


@dataclass(frozen=True)
class ConfiguracaoFronteira:
    raizes: tuple[Path, ...] = ()
    max_bytes_imagem: int = MAX_BYTES_IMAGEM
    max_bytes_video: int = MAX_BYTES_VIDEO
    max_bytes_extracao_local: int = MAX_BYTES_EXTRACAO_LOCAL  # a extração local não envia nada: o limite é de recurso
    max_caracteres_texto: int = MAX_CARACTERES_TEXTO
    max_posts_historico: int = MAX_POSTS_HISTORICO
    max_tags: int = MAX_TAGS
    max_caracteres_id: int = MAX_CARACTERES_ID

    @classmethod
    def do_ambiente(cls, ambiente: Mapping[str, str]) -> "ConfiguracaoFronteira":
        """Raízes de `CASEGALAXY_MCP_RAIZES` (separadas por `os.pathsep`); ausente ou vazia = nenhuma raiz."""
        partes = [p.strip() for p in (ambiente.get(VARIAVEL_RAIZES) or "").split(os.pathsep) if p.strip()]
        raizes = []
        for parte in partes:
            try:
                raiz = Path(parte).resolve(strict=True)
            except (OSError, ValueError, RuntimeError) as erro:
                raise ErroConfiguracao(f"{VARIAVEL_RAIZES}: a raiz {parte!r} não existe ou não pode ser lida") from erro
            if not raiz.is_dir():
                raise ErroConfiguracao(f"{VARIAVEL_RAIZES}: a raiz {parte!r} não é um diretório")
            raizes.append(raiz)
        return cls(raizes=tuple(raizes))


def _erro_de_arquivo(mensagem: str, **detalhes: object) -> ErroAgente:
    return ErroAgente(Codigo.MIDIA_INVALIDA, mensagem, detalhes)


def _arquivo_permitido(caminho: Path, campo: str, limite_bytes: int, config: ConfiguracaoFronteira) -> Path:
    """Caminho resolvido de um arquivo regular dentro das raízes e dentro do limite de tamanho."""
    if not config.raizes:
        raise _erro_de_arquivo(f"O acesso a arquivos não está configurado neste servidor: defina as raízes "
                               f"permitidas em {VARIAVEL_RAIZES}.", campo=campo)
    try:
        resolvido = Path(caminho).resolve(strict=False)
    except (OSError, ValueError, RuntimeError):  # byte nulo, laço de links simbólicos, nome ilegível
        raise _erro_de_arquivo("Caminho de arquivo inválido.", campo=campo) from None
    if not any(resolvido.is_relative_to(raiz) for raiz in config.raizes):
        raise _erro_de_arquivo("O arquivo está fora das raízes permitidas.", campo=campo)
    try:
        if not resolvido.exists():
            raise _erro_de_arquivo("Arquivo não encontrado.", campo=campo)
        if not resolvido.is_file():
            raise _erro_de_arquivo("O caminho informado não é um arquivo.", campo=campo)
        tamanho = resolvido.stat().st_size
    except OSError:  # permissão negada, arquivo removido entre a checagem e o uso, volume indisponível
        raise _erro_de_arquivo("O arquivo não pôde ser lido.", campo=campo) from None
    if tamanho > limite_bytes:
        raise _erro_de_arquivo(f"Arquivo maior que o limite de {limite_bytes} bytes.", campo=campo,
                               limite_bytes=limite_bytes)
    return resolvido


def _limitar_texto(valor: str | None, campo: str, config: ConfiguracaoFronteira) -> None:
    if valor is not None and len(valor) > config.max_caracteres_texto:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, f"Texto maior que o limite de {config.max_caracteres_texto} "
                         "caracteres.", {"campo": campo, "limite_caracteres": config.max_caracteres_texto})


def _limitar_identificacao_e_tags(pedido: PedidoPrever, config: ConfiguracaoFronteira) -> None:
    """`channel_id` e `tags` não são ilimitados: as tags entram no texto codificado, junto com título e descrição."""
    if pedido.channel_id is not None and len(pedido.channel_id) > config.max_caracteres_id:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, f"channel_id maior que o limite de {config.max_caracteres_id} "
                         "caracteres.", {"campo": "channel_id", "limite_caracteres": config.max_caracteres_id})
    if len(pedido.tags) > config.max_tags:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, f"Mais de {config.max_tags} tags.",
                         {"campo": "tags", "limite_tags": config.max_tags, "recebidas": len(pedido.tags)})
    if sum(len(tag) for tag in pedido.tags) > config.max_caracteres_texto:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, f"Tags somam mais que o limite de {config.max_caracteres_texto} "
                         "caracteres.", {"campo": "tags", "limite_caracteres": config.max_caracteres_texto})


def _limitar_entradas(pedido: PedidoPrever, config: ConfiguracaoFronteira) -> None:
    _limitar_identificacao_e_tags(pedido, config)
    criativo = pedido.creative
    if criativo.tipo == "texto":
        _limitar_texto(criativo.texto, "creative.texto", config)
    if pedido.caption is not None:
        _limitar_texto(pedido.caption.title, "caption.title", config)
        _limitar_texto(pedido.caption.description, "caption.description", config)
    posts = pedido.channel_stats
    if posts is not None and len(posts) > config.max_posts_historico:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, f"Histórico fornecido maior que o limite de "
                         f"{config.max_posts_historico} posts.",
                         {"campo": "channel_stats", "limite_posts": config.max_posts_historico,
                          "recebidos": len(posts)})


def verificar_pedido(dados: Mapping | PedidoPrever, config: ConfiguracaoFronteira) -> PedidoPrever:
    """Pedido validado pelo contrato do agente, dentro dos limites, com os caminhos de arquivo já resolvidos.

    Levanta `ErroAgente` (`ENTRADA_INVALIDA` ou `MIDIA_INVALIDA`); o chamador o converte no envelope de erro.
    """
    pedido = api._validar(PedidoPrever, dados)
    _limitar_entradas(pedido, config)
    mudancas: dict = {}
    criativo = pedido.creative
    if criativo.tipo in ("imagem", "video") and getattr(criativo, "arquivo", None) is not None:
        limite = config.max_bytes_imagem if criativo.tipo == "imagem" else config.max_bytes_video
        resolvido = _arquivo_permitido(criativo.arquivo, "creative.arquivo", limite, config)
        mudancas["creative"] = criativo.model_copy(update={"arquivo": resolvido})
    if pedido.thumbnail is not None:
        mudancas["thumbnail"] = _arquivo_permitido(pedido.thumbnail, "thumbnail", config.max_bytes_imagem, config)
    return pedido.model_copy(update=mudancas) if mudancas else pedido


def verificar_pedido_extracao(dados: Mapping | PedidoExtrairAudiovisual,
                              config: ConfiguracaoFronteira) -> PedidoExtrairAudiovisual:
    """Pedido de extração audiovisual: contrato, raízes permitidas, caminho resolvido e limite de tamanho."""
    pedido = api._validar(PedidoExtrairAudiovisual, dados)
    resolvido = _arquivo_permitido(pedido.arquivo, "arquivo", config.max_bytes_extracao_local, config)
    return pedido.model_copy(update={"arquivo": resolvido})


def verificar_pedido_ordenar(dados: Mapping | PedidoOrdenar, config: ConfiguracaoFronteira) -> PedidoOrdenar:
    """Como `verificar_pedido` para cada criativo; o erro informa o `indice`. Contagem inválida fica para o agente."""
    entrada = api._validar(PedidoOrdenar, dados)
    if not MIN_CRIATIVOS_ORDENAR <= len(entrada.creatives) <= MAX_CRIATIVOS_ORDENAR:
        return entrada
    verificados = []
    for indice, criativo in enumerate(entrada.creatives):
        try:
            verificados.append(verificar_pedido(criativo, config))
        except ErroAgente as erro:
            raise ErroAgente(erro.codigo, erro.mensagem, {**erro.detalhes, "indice": indice}) from None
    return entrada.model_copy(update={"creatives": tuple(verificados)})


# ──── Livro de consumo das chamadas externas do agente, no formato dos scripts de verificação ao vivo.
_CAMPOS_PERMITIDOS = ("tipo", "fase", "operacao", "duracao_video_s", "modelo", "sha256_corpo", "resultado", "uso",
                      "unidades", "recurso")


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


# ──── Modo ONLINE do servidor MCP: opt-in por ambiente, com tetos por execução e livro de consumo obrigató
logger = logging.getLogger(__name__)
VARIAVEL_ONLINE = "CASEGALAXY_MCP_ONLINE"
VARIAVEL_LIVRO = "CASEGALAXY_MCP_LIVRO"
VARIAVEL_MAX_REQUISICOES = "CASEGALAXY_MCP_MAX_REQUISICOES"
VARIAVEL_MAX_SEGUNDOS = "CASEGALAXY_MCP_MAX_SEGUNDOS_VIDEO"
VARIAVEL_MAX_UNIDADES_YOUTUBE = "CASEGALAXY_MCP_MAX_UNIDADES_YOUTUBE"
VARIAVEL_INTERVALO = "CASEGALAXY_MCP_INTERVALO_S"
PADRAO_MAX_REQUISICOES = 30
PADRAO_MAX_SEGUNDOS = 360
PADRAO_MAX_UNIDADES_YOUTUBE = 20
PADRAO_INTERVALO_S = 5.0
LIMITE_SUPERIOR = 10_000
MAX_INTERVALO_S = 600.0


@dataclass(frozen=True)
class ConfiguracaoOnline:
    livro: Path
    max_requisicoes_gemini: int = PADRAO_MAX_REQUISICOES
    max_segundos_video: int = PADRAO_MAX_SEGUNDOS
    max_unidades_youtube: int = PADRAO_MAX_UNIDADES_YOUTUBE
    intervalo_minimo_s: float = PADRAO_INTERVALO_S

    @classmethod
    def do_ambiente(cls, ambiente: Mapping[str, str]) -> "ConfiguracaoOnline | None":
        """None = offline (variável ausente, vazia ou `0`). `1` liga; qualquer outro valor é erro, nunca online por engano."""
        valor = (ambiente.get(VARIAVEL_ONLINE) or "").strip()
        if valor in ("", "0"):
            return None
        if valor != "1":
            raise ErroConfiguracao(f"{VARIAVEL_ONLINE}: use 1 para ligar o modo online ou 0/vazio para desligar")
        return cls(
            livro=_caminho_do_livro(ambiente.get(VARIAVEL_LIVRO)),
            max_requisicoes_gemini=_inteiro(ambiente, VARIAVEL_MAX_REQUISICOES, PADRAO_MAX_REQUISICOES),
            max_segundos_video=_inteiro(ambiente, VARIAVEL_MAX_SEGUNDOS, PADRAO_MAX_SEGUNDOS),
            max_unidades_youtube=_inteiro(ambiente, VARIAVEL_MAX_UNIDADES_YOUTUBE, PADRAO_MAX_UNIDADES_YOUTUBE),
            intervalo_minimo_s=_intervalo(ambiente.get(VARIAVEL_INTERVALO)))


def _caminho_do_livro(bruto: str | None) -> Path:
    if not (bruto or "").strip():
        raise ErroConfiguracao(f"{VARIAVEL_LIVRO}: o modo online exige o arquivo do livro de consumo (para a cota "
                               "continuar conciliável com a anotação do dataset)")
    caminho = Path(bruto.strip()).resolve()
    if not caminho.parent.is_dir() or caminho.is_dir():
        raise ErroConfiguracao(f"{VARIAVEL_LIVRO}: a pasta do livro de consumo não existe ou o caminho é uma pasta")
    return caminho


def _inteiro(ambiente: Mapping[str, str], variavel: str, padrao: int) -> int:
    bruto = (ambiente.get(variavel) or "").strip()
    if not bruto:
        return padrao
    if not bruto.isdigit() or not 1 <= int(bruto) <= LIMITE_SUPERIOR:
        raise ErroConfiguracao(f"{variavel}: informe um inteiro entre 1 e {LIMITE_SUPERIOR}")
    return int(bruto)


def _intervalo(bruto: str | None) -> float:
    texto = (bruto or "").strip()
    if not texto:
        return PADRAO_INTERVALO_S
    try:
        valor = float(texto)
    except ValueError:
        raise ErroConfiguracao(f"{VARIAVEL_INTERVALO}: informe um número em segundos") from None
    if not 0 <= valor <= MAX_INTERVALO_S:
        raise ErroConfiguracao(f"{VARIAVEL_INTERVALO}: informe entre 0 e {MAX_INTERVALO_S:.0f} segundos")
    return valor


def carregar_configuracao() -> Configuracao:
    """Ponto único de leitura das chaves (`.env` ou ambiente); os testes o substituem."""
    return Configuracao()


def exigir_chaves(configuracao: Configuracao) -> None:
    """O modo online precisa das duas chaves; falta de qualquer uma é erro de partida que nomeia só a variável."""
    for nome, chave in (("GEMINI_API_KEY", configuracao.gemini_api_key), ("YOUTUBE_API_KEY", configuracao.youtube_api_key)):
        if chave is None:
            raise ErroConfiguracao(f"{nome} não está definida no .env nem no ambiente (exigida pelo modo online)")


def criar_servicos_online(online: ConfiguracaoOnline, *, configuracao: Configuracao | None = None,
                          transporte_gemini: httpx.BaseTransport | None = None,
                          transporte_youtube: httpx.BaseTransport | None = None,
                          transporte_miniatura: httpx.BaseTransport | None = None,
                          fabrica: Callable[..., Servicos] = Servicos.padrao) -> Servicos:
    """Serviços reais com os clientes online. Os transportes são injetáveis: os testes nunca tocam a rede."""
    configuracao = configuracao or carregar_configuracao()
    exigir_chaves(configuracao)
    livro = LivroConsumo(online.livro)
    gemini = ga.ClienteGemini(
        configuracao.chave_gemini(), MODELO_GEMINI,
        ga.Limites(max_requisicoes=online.max_requisicoes_gemini, max_segundos_video=online.max_segundos_video,
                   intervalo_minimo_s=online.intervalo_minimo_s),
        transporte=transporte_gemini, registro=livro.registrar_gemini)
    youtube = yt.ClienteYouTube(
        configuracao.chave_youtube(), limite_unidades=online.max_unidades_youtube,
        transporte=TransporteYouTubeComLivro(transporte_youtube or httpx.HTTPTransport(), livro))
    logger.info("modo online: %d requisições, %d s de vídeo e %d unidades do YouTube por execução; livro em %s",
                online.max_requisicoes_gemini, online.max_segundos_video, online.max_unidades_youtube,
                online.livro.name)
    return fabrica(cliente_gemini=gemini, consultor_historico=criar_consultor(youtube), explicar_com_gemini=True,
                   baixar_imagem=mn.criar_baixador(transporte_miniatura))


# ──── Servidor MCP fino (stdio) sobre `prever` e `ordenar` do agente.
NOME_DO_SERVIDOR = "casegalaxy-agente"
_DESCRICAO_PREVER_BASE = (
    "Prevê as curtidas de um criativo planejado para um canal de beleza ou automotivo: vídeo (URL pública do YouTube "
    "ou arquivo de até 100 MB, 180 s, vertical ou quadrado), imagem (interpretada como a capa planejada) ou texto "
    "(interpretado como a legenda planejada). Devolve a previsão central (q50), a faixa (limites = mínimo e máximo das "
    "saídas q10, q50 e q90, com q50 sempre dentro dela) com a cobertura observada de cada modelo, a referência do canal, os principais fatores, os avisos e a versão do modelo. O resultado é o "
    "envelope {success, data | error} do agente. Toda resposta traz `route`, a rota que a produziu (`video_local`, "
    "`video_gemini`, `video_gemini_fallback_imagem`, `imagem` ou `texto`). Vídeo em arquivo: `video_route` escolhe a "
    "rota (`auto`, `local` ou `gemini`); com preditor local ativo, `auto` usa a rota local (offline, sem Gemini, com "
    "atributos audiovisuais extraídos no equipamento e cache por conteúdo); vídeo por URL pública usa a rota Gemini. Arquivos só dentro das raízes permitidas pelo servidor. Histórico "
    "fornecido: informe `video_id` de cada post e, para imagem, texto ou vídeo em arquivo que já existe no canal, "
    "`own_video_id`, para excluir o próprio criativo da referência; sem identidade o aviso "
    "`identidade_do_proprio_video_ausente` declara a limitação. ")
DESCRICAO_PREVER_OFFLINE = _DESCRICAO_PREVER_BASE + (
    "Modo offline: sem Gemini nem YouTube. Na rota local o vídeo em arquivo é previsto sem Gemini e sem rede (pesos "
    "preparados antes com `casegalaxy-preparar-av`; sem eles o erro é explícito, sem volta ao modelo de imagem). Na rota "
    "Gemini, um vídeo sem anotação guardada (cache) é avaliado pelo modelo de imagem (N3), com aviso "
    "`fallback_video_para_imagem` e baixa confiança, e exige capa (`thumbnail`). A consulta ao YouTube "
    "(`consultar_youtube`) não está habilitada e a explicação é o texto de modelo fixo, rotulado como determinístico.")
DESCRICAO_ORDENAR = (
    "Ordena de 2 a 5 criativos do mesmo canal, da mesma modalidade, da mesma rota e do mesmo nível de modelo pelo "
    "valor central previsto (q50). Criativos de modalidades, rotas (local e Gemini) ou níveis diferentes (por "
    "exemplo, um vídeo que caiu no modelo de imagem) são recusados: modelos diferentes não formam uma comparação "
    "validada. Cada item do ranking informa a "
    "modalidade, o nível e a versão do modelo. Devolve o envelope {success, data | error}.")
DESCRICAO_EXTRAIR = (
    "Extrai atributos audiovisuais de um arquivo de vídeo local, 100% offline e sem Gemini nem YouTube: sonda do arquivo, "
    "5 a 8 quadros reais, embeddings SigLIP2, rosto, texto na tela (OCR), objetos (YOLOX, classes COCO), cor, brilho, "
    "contraste e saturação, movimento (fluxo óptico), cortes de cena, energia do áudio e fala (faster-whisper). Cada "
    "etapa traz o próprio estado; falha, ausência de áudio e dados insuficientes nunca viram zero. Não devolve previsão: "
    "a resposta traz `predicao`, que diz se há preditor local ativo e aponta `prever_criativo` (com o mesmo arquivo) "
    "para obter a previsão; a extração fica em cache e não é repetida. Sem preditor local ativo, `predicao.disponivel` "
    "é falso. Esta ferramenta é sempre offline e nunca consome cota do Gemini ou do YouTube, "
    "mesmo com o servidor em modo ONLINE. Arquivos só dentro das raízes permitidas pelo servidor.")
INSTRUCOES_OFFLINE = (
    "Agente preditivo de curtidas do Case Galaxies (modelos retreinados com todos os vídeos anotados). A cobertura "
    "informada vem do procedimento de avaliação, não foi medida nos modelos servidos e não é garantida. "
    "Modo offline: nenhuma consulta ao Gemini ou ao YouTube é feita por este servidor. Toda previsão traz a "
    "cobertura observada da faixa e os avisos do modelo usado; leia-os antes de usar o número.")


def descricao_prever_online(online: "on.ConfiguracaoOnline") -> str:
    return _DESCRICAO_PREVER_BASE + (
        f"Modo ONLINE (opt-in): vídeo público novo e arquivo de vídeo local são anotados pelo Gemini (arquivo local: "
        f"enviado à Files API e apagado depois do uso; se a exclusão falhar ele expira em 48 h) e a explicação é "
        f"redigida pelo Gemini (texto fixo se falhar). `consultar_youtube: true` lê só os uploads do canal, nunca as "
        f"estatísticas do vídeo consultado. Cada chamada consome a cota do projeto; tetos por execução do servidor: "
        f"{online.max_requisicoes_gemini} requisições e {online.max_segundos_video} s de vídeo no Gemini, "
        f"{online.max_unidades_youtube} unidades do YouTube. Cota esgotada ou falha do Gemini nunca é repetida na "
        f"mesma chamada: o vídeo cai no modelo de imagem (N3) com aviso.")


def instrucoes_online(online: "on.ConfiguracaoOnline") -> str:
    return (
        "Agente preditivo de curtidas do Case Galaxies (modelos retreinados com todos os vídeos anotados). A cobertura "
        "informada vem do procedimento de avaliação, não foi medida nos modelos servidos e não é garantida. "
        "Modo ONLINE (opt-in): este servidor pode chamar o Gemini e o YouTube, e cada chamada consome a cota do "
        f"projeto (tetos por execução: {online.max_requisicoes_gemini} requisições e {online.max_segundos_video} s de "
        f"vídeo no Gemini, {online.max_unidades_youtube} unidades do YouTube; consumo registrado em livro). Toda "
        "previsão traz a cobertura observada da faixa e os avisos do modelo usado; leia-os antes de usar o número.")


# Compatibilidade: nomes antigos usados por testes e documentação.
DESCRICAO_PREVER = DESCRICAO_PREVER_OFFLINE
INSTRUCOES = INSTRUCOES_OFFLINE


def _json_estrito(valor: object) -> str:
    """Datas vão em ISO 8601; qualquer outro tipo fora do JSON levanta erro em vez de virar texto em silêncio."""
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    raise TypeError(f"valor fora do JSON no envelope do agente: {type(valor).__name__}")


def como_resultado(envelope: Mapping) -> CallToolResult:
    """Envelope do agente como resultado MCP: estruturado e em texto JSON; `is_error` quando `success` é falso."""
    conteudo = json.loads(json.dumps(envelope, ensure_ascii=False, default=_json_estrito, allow_nan=False))
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(conteudo, ensure_ascii=False))],
                          structured_content=conteudo, is_error=not conteudo["success"])


def _resultado_seguro(envelope: Mapping) -> CallToolResult:
    try:
        return como_resultado(envelope)
    except (ValueError, TypeError):  # número não finito ou valor não serializável: nunca emitir JSON inválido
        logger.exception("o envelope do agente não pôde ser serializado")
        return como_resultado(envelope_erro(ErroAgente(
            Codigo.ERRO_INTERNO, "A resposta do agente não pôde ser serializada.", {})))


class _SobDemanda:
    """Serviços de UM pedido, criados só quando algum atributo é lido: entrada recusada não carrega modelos.

    A falha da criação fica anotada em `falha` e é repetida a cada leitura do mesmo pedido, sem nova tentativa: um
    ranking com três criativos tenta criar os serviços uma vez, não três. O pedido seguinte tenta de novo.
    """

    def __init__(self, obter: Callable[[], Servicos]) -> None:
        self._obter, self._servicos, self.falha = obter, None, None

    def __getattr__(self, nome: str) -> Any:
        if nome.startswith("_"):  # introspecção (LangGraph, copy, pickle) não cria serviços; evita recursão sem `__init__`
            raise AttributeError(nome)
        if self.falha is not None:
            raise self.falha
        if self._servicos is None:
            try:
                self._servicos = self._obter()
            except Exception as falha:
                self.falha = falha
                raise
        return getattr(self._servicos, nome)


class _Aplicacao:
    def __init__(self, fabrica: Callable[[], Servicos], config: ConfiguracaoFronteira) -> None:
        self.config, self._fabrica, self._servicos = config, fabrica, None
        self._trava = threading.RLock()

    def _obter(self) -> Servicos:
        with self._trava:
            if self._servicos is None:  # uma falha não fica gravada: a próxima chamada tenta de novo
                self._servicos = self._fabrica()
            return self._servicos

    def executar(self, verificar: Callable, operacao: Callable, dados: Mapping) -> dict:
        """Fronteira, depois a função do agente; um pedido por vez (modelos e extratores locais não são concorrentes)."""
        try:
            pedido = verificar(dados, self.config)
        except ErroAgente as erro:
            return envelope_erro(erro)
        except Exception as falha:
            logger.exception("falha inesperada na fronteira do MCP")
            return envelope_erro(ErroAgente(Codigo.ERRO_INTERNO, "Falha interna na fronteira do servidor.",
                                            {"tipo": type(falha).__name__}))
        servicos = _SobDemanda(self._obter)
        with self._trava:
            resultado = operacao(pedido, servicos)
        if isinstance(servicos.falha, AmbienteIncompleto):  # falta biblioteca ou peso: a mensagem já diz o que fazer
            logger.error("ambiente incompleto: %s", servicos.falha)  # no stderr, com os caminhos completos
            return envelope_erro(ErroAgente(Codigo.MODELO_INDISPONIVEL, servicos.falha.mensagem_sem_caminhos(),
                                            servicos.falha.detalhes()))  # ao cliente, sem caminhos absolutos
        if servicos.falha is not None:  # a pane dos serviços não pode aparecer como erro da entrada do usuário
            logger.error("serviços do agente indisponíveis", exc_info=servicos.falha)
            return envelope_erro(ErroAgente(Codigo.ERRO_INTERNO, "Serviços do agente indisponíveis.",
                                            {"tipo": type(servicos.falha).__name__}))
        return resultado


def _esquema_do_argumento(modelo: type[BaseModel]) -> dict:
    """Esquema anunciado: um argumento `pedido` com o esquema do contrato (`$defs` sobe para a raiz)."""
    esquema = modelo.model_json_schema()
    definicoes = esquema.pop("$defs", {})
    parametros: dict[str, Any] = {"type": "object", "properties": {"pedido": esquema}, "required": ["pedido"]}
    if definicoes:
        parametros["$defs"] = definicoes
    return parametros


def _ferramenta(funcao: Callable, modelo: type[BaseModel], descricao: str) -> Tool:
    ferramenta = Tool.from_function(funcao, description=descricao)
    ferramenta.parameters = _esquema_do_argumento(modelo)
    return ferramenta


class _ExtratorSobDemanda:
    """Extrator audiovisual criado só na primeira extração: entrada recusada ou outra ferramenta não carregam nada."""

    def __init__(self, fabrica: Callable[[], Any] | None) -> None:
        self._fabrica, self._extrator = fabrica or self._padrao, None

    @staticmethod
    def _padrao() -> Any:
        from src.extracao.pipeline import ExtratorAudiovisual

        return ExtratorAudiovisual()

    def extrair(self, caminho: Any) -> dict:
        if self._extrator is None:
            self._extrator = self._fabrica()
        return self._extrator.extrair(caminho)


def criar_servidor(fabrica_servicos: Callable[[], Servicos], config: ConfiguracaoFronteira,
                   online: ConfiguracaoOnline | None = None,
                   fabrica_extrator: Callable[[], Any] | None = None,
                   predicao_audiovisual: Callable[[], dict] | None = None) -> MCPServer:
    """Servidor MCP com `prever_criativo`, `ordenar_criativos` e `extrair_atributos_audiovisual`.

    `fabrica_servicos` roda na primeira necessidade. `online` só descreve o modo nas ferramentas e nas instruções; quem
    liga os clientes é a fábrica de serviços. `fabrica_extrator` (padrão: `ExtratorAudiovisual()`) cria o extrator local
    na primeira extração; a extração é sempre offline e não usa os serviços de previsão. `predicao_audiovisual` diz, na
    resposta da extração, se há preditor local ativo (`api.predicao_audiovisual_ativa`); sem ele, vale o texto de
    indisponibilidade.
    """
    app = _Aplicacao(fabrica_servicos, config)
    extrator = _ExtratorSobDemanda(fabrica_extrator)

    def prever_criativo(pedido: dict[str, Any]) -> CallToolResult:
        return _resultado_seguro(app.executar(verificar_pedido, lambda p, s: api.prever(p, s), pedido))

    def ordenar_criativos(pedido: dict[str, Any]) -> CallToolResult:
        return _resultado_seguro(app.executar(verificar_pedido_ordenar, lambda p, s: api.ordenar(p, s), pedido))

    def extrair_atributos_audiovisual(pedido: dict[str, Any]) -> CallToolResult:
        return _resultado_seguro(app.executar(verificar_pedido_extracao,
                                              lambda p, s: api.extrair_audiovisual(p, extrator, predicao=predicao_audiovisual), pedido))

    descricao_prever = DESCRICAO_PREVER_OFFLINE if online is None else descricao_prever_online(online)
    descricao_ordenar = DESCRICAO_ORDENAR if online is None else DESCRICAO_ORDENAR + " Modo ONLINE: cada criativo pode consumir cota do Gemini e do YouTube."
    ferramentas = [_ferramenta(prever_criativo, PedidoPrever, descricao_prever),
                   _ferramenta(ordenar_criativos, PedidoOrdenar, descricao_ordenar),
                   _ferramenta(extrair_atributos_audiovisual, PedidoExtrairAudiovisual, DESCRICAO_EXTRAIR)]
    instrucoes = INSTRUCOES_OFFLINE if online is None else instrucoes_online(online)
    return MCPServer(NOME_DO_SERVIDOR, instructions=instrucoes, tools=ferramentas, log_level="WARNING")


def silenciar_registros_de_rede() -> None:
    """O httpx registra a URL de cada requisição em INFO (inclusive a de envio de arquivo ao Gemini): só avisos."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> int:
    """Serve por stdio até o cliente encerrar. Retorno 2: configuração inválida (mensagem no stderr, sem chaves). Retorno 3:
    faltam bibliotecas do sistema ou pesos dos codificadores (a mensagem no stderr lista cada item e o comando)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    silenciar_registros_de_rede()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # os pesos dos modelos locais nunca são baixados, nem no modo online
    try:
        config = ConfiguracaoFronteira.do_ambiente(os.environ)
        online = ConfiguracaoOnline.do_ambiente(os.environ)
        if online is not None:
            exigir_chaves(carregar_configuracao())
    except ErroConfiguracao as erro:
        print(f"Configuração inválida do servidor MCP: {erro}", file=sys.stderr)
        return 2
    try:
        exigir_ambiente()  # falha na partida, com a lista do que falta, em vez de erro a cada chamada
    except AmbienteIncompleto as erro:
        print(erro, file=sys.stderr)
        return CODIGO_SAIDA_AMBIENTE_INCOMPLETO
    fabrica = Servicos.padrao if online is None else (lambda: criar_servicos_online(online))
    criar_servidor(fabrica, config, online=online, predicao_audiovisual=api.predicao_audiovisual_ativa).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
