"""Modo ONLINE do servidor MCP: opt-in por ambiente, com tetos por execução e livro de consumo obrigatório.

Sem `CASEGALAXY_MCP_ONLINE=1` nada aqui é usado e o servidor continua offline, como antes. Com ele, os serviços do agente
ganham: cliente do Gemini (anotação de vídeo público novo e de arquivo local pela Files API, e explicação), consultor de
histórico do YouTube (só o canal; nunca as estatísticas do vídeo consultado) e baixador de miniatura. As chaves vêm do
`.env`/ambiente pela `Configuracao` existente e nunca aparecem em mensagem, log ou livro.

Variáveis: `CASEGALAXY_MCP_ONLINE` (`1` liga; vazio ou `0` desliga; qualquer outro valor é erro),
`CASEGALAXY_MCP_LIVRO` (obrigatória: arquivo do livro de consumo), e opcionais `CASEGALAXY_MCP_MAX_REQUISICOES` (30),
`CASEGALAXY_MCP_MAX_SEGUNDOS_VIDEO` (360), `CASEGALAXY_MCP_MAX_UNIDADES_YOUTUBE` (20) e `CASEGALAXY_MCP_INTERVALO_S` (5).
Os tetos valem por EXECUÇÃO do servidor (processo). Não substituem o orçamento diário: esse é conferido pelos scripts de
verificação, lendo o mesmo livro.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from src import gemini_api as ga
from src import youtube_api as yt
from src.agente import livro_consumo as lc
from src.agente import miniatura as mn
from src.agente.fronteira_mcp import ErroConfiguracao
from src.agente.servicos import MODELO_GEMINI, Servicos
from src.agente.youtube_historico import criar_consultor
from src.config import Configuracao

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
    livro = lc.LivroConsumo(online.livro)
    gemini = ga.ClienteGemini(
        configuracao.chave_gemini(), MODELO_GEMINI,
        ga.Limites(max_requisicoes=online.max_requisicoes_gemini, max_segundos_video=online.max_segundos_video,
                   intervalo_minimo_s=online.intervalo_minimo_s),
        transporte=transporte_gemini, registro=livro.registrar_gemini)
    youtube = yt.ClienteYouTube(
        configuracao.chave_youtube(), limite_unidades=online.max_unidades_youtube,
        transporte=lc.TransporteYouTubeComLivro(transporte_youtube or httpx.HTTPTransport(), livro))
    logger.info("modo online: %d requisições, %d s de vídeo e %d unidades do YouTube por execução; livro em %s",
                online.max_requisicoes_gemini, online.max_segundos_video, online.max_unidades_youtube,
                online.livro.name)
    return fabrica(cliente_gemini=gemini, consultor_historico=criar_consultor(youtube), explicar_com_gemini=True,
                   baixar_imagem=mn.criar_baixador(transporte_miniatura))

