"""Servidor MCP fino (stdio) sobre `prever` e `ordenar` do agente.

Não há outro pipeline: cada ferramenta valida a fronteira (`fronteira_mcp`) e chama a mesma função pública do agente
(`api.prever` e `api.ordenar`), devolvendo o mesmo envelope `{success, data | error}`: em `structured_content` (para
máquinas) e como texto JSON (para clientes sem suporte a conteúdo estruturado). `is_error` é verdadeiro quando
`success` é falso; o código do erro continua no envelope.

Offline por padrão: os serviços são `Servicos.padrao()` sem cliente do Gemini nem do YouTube, criados só na primeira
necessidade (uma entrada recusada não carrega modelos). Um vídeo cuja anotação não está guardada (cache da amostra ou de
uploads) é avaliado pelo modelo de imagem (N3), com aviso e baixa confiança, se houver capa; sem rede nenhuma.

Modo online (opt-in, `CASEGALAXY_MCP_ONLINE=1`, ver `online.py`): anotação de vídeo público novo e de arquivo local pela
Files API do Gemini, explicação por LLM e histórico atual do canal no YouTube, com tetos por execução e livro de consumo.
As descrições das ferramentas e as instruções do servidor dizem o modo ativo.

Protocolo e diagnóstico ficam separados: o stdout é só do protocolo; o log vai para o stderr.

Uso: `python -m src.agente.servidor_mcp`, com `CASEGALAXY_MCP_RAIZES` listando as raízes de arquivo permitidas.
"""

import json
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping
from datetime import date, datetime
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.tools import Tool
from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel

from src.agente import api
from src.agente import fronteira_mcp as fm
from src.agente import online as on
from src.agente.contratos import (
    Codigo,
    ErroAgente,
    PedidoOrdenar,
    PedidoPrever,
    envelope_erro,
)
from src.agente.servicos import Servicos

logger = logging.getLogger(__name__)

NOME_DO_SERVIDOR = "casegalaxy-agente"
_DESCRICAO_PREVER_BASE = (
    "Prevê as curtidas de um criativo planejado para um canal de beleza ou automotivo: vídeo (URL pública do YouTube "
    "ou arquivo de até 100 MB, 180 s, vertical ou quadrado), imagem (interpretada como a capa planejada) ou texto "
    "(interpretado como a legenda planejada). Devolve a previsão central (q50), a faixa (limites = mínimo e máximo das "
    "saídas q10, q50 e q90, com q50 sempre dentro dela) com a cobertura observada de cada modelo, a referência do canal, os principais fatores, os avisos e a versão do modelo. O resultado é o "
    "envelope {success, data | error} do agente. Arquivos só dentro das raízes permitidas pelo servidor. Histórico "
    "fornecido: informe `video_id` de cada post e, para imagem, texto ou vídeo em arquivo que já existe no canal, "
    "`own_video_id`, para excluir o próprio criativo da referência; sem identidade o aviso "
    "`identidade_do_proprio_video_ausente` declara a limitação. ")
DESCRICAO_PREVER_OFFLINE = _DESCRICAO_PREVER_BASE + (
    "Modo offline: sem Gemini nem YouTube. Um vídeo sem anotação guardada (cache) é avaliado pelo modelo de imagem "
    "(N3), com aviso `fallback_video_para_imagem` e baixa confiança, e exige capa (`thumbnail`); a consulta ao YouTube "
    "(`consultar_youtube`) não está habilitada e a explicação é o texto de modelo fixo.")
DESCRICAO_ORDENAR = (
    "Ordena de 2 a 5 criativos do mesmo canal, da mesma modalidade e do mesmo nível de modelo pelo valor central "
    "previsto (q50). Criativos de modalidades ou níveis diferentes (por exemplo, um vídeo que caiu no modelo de "
    "imagem) são recusados: modelos diferentes não formam uma comparação validada. Cada item do ranking informa a "
    "modalidade, o nível e a versão do modelo. Devolve o envelope {success, data | error}.")
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
    def __init__(self, fabrica: Callable[[], Servicos], config: fm.ConfiguracaoFronteira) -> None:
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


def criar_servidor(fabrica_servicos: Callable[[], Servicos], config: fm.ConfiguracaoFronteira,
                   online: on.ConfiguracaoOnline | None = None) -> MCPServer:
    """Servidor MCP com `prever_criativo` e `ordenar_criativos`; `fabrica_servicos` roda na primeira necessidade.

    `online` só descreve o modo nas ferramentas e nas instruções; quem liga os clientes é a fábrica de serviços.
    """
    app = _Aplicacao(fabrica_servicos, config)

    def prever_criativo(pedido: dict[str, Any]) -> CallToolResult:
        return _resultado_seguro(app.executar(fm.verificar_pedido, lambda p, s: api.prever(p, s), pedido))

    def ordenar_criativos(pedido: dict[str, Any]) -> CallToolResult:
        return _resultado_seguro(app.executar(fm.verificar_pedido_ordenar, lambda p, s: api.ordenar(p, s), pedido))

    descricao_prever = DESCRICAO_PREVER_OFFLINE if online is None else descricao_prever_online(online)
    descricao_ordenar = DESCRICAO_ORDENAR if online is None else DESCRICAO_ORDENAR + " Modo ONLINE: cada criativo pode consumir cota do Gemini e do YouTube."
    ferramentas = [_ferramenta(prever_criativo, PedidoPrever, descricao_prever),
                   _ferramenta(ordenar_criativos, PedidoOrdenar, descricao_ordenar)]
    instrucoes = INSTRUCOES_OFFLINE if online is None else instrucoes_online(online)
    return MCPServer(NOME_DO_SERVIDOR, instructions=instrucoes, tools=ferramentas, log_level="WARNING")


def silenciar_registros_de_rede() -> None:
    """O httpx registra a URL de cada requisição em INFO (inclusive a de envio de arquivo ao Gemini): só avisos."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> int:
    """Serve por stdio até o cliente encerrar. Retorno 2: configuração inválida (mensagem no stderr, sem chaves)."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    silenciar_registros_de_rede()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # os pesos dos modelos locais nunca são baixados, nem no modo online
    try:
        config = fm.ConfiguracaoFronteira.do_ambiente(os.environ)
        online = on.ConfiguracaoOnline.do_ambiente(os.environ)
        if online is not None:
            on.exigir_chaves(on.carregar_configuracao())
    except fm.ErroConfiguracao as erro:
        print(f"Configuração inválida do servidor MCP: {erro}", file=sys.stderr)
        return 2
    fabrica = Servicos.padrao if online is None else (lambda: on.criar_servicos_online(online))
    criar_servidor(fabrica, config, online=online).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
