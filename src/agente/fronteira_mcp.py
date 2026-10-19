"""Fronteira do MCP: raízes de arquivo, caminhos, tamanhos e limites de entrada, antes de chamar o agente.

Sem o SDK do MCP: é lógica pura sobre o contrato do agente, testável sozinha. O que o cliente envia é validado pelo
mesmo contrato (`PedidoPrever` e `PedidoOrdenar`, com o mesmo erro `ENTRADA_INVALIDA`) e depois pelos limites abaixo.

- Acesso a arquivo só dentro de raízes configuradas em `CASEGALAXY_MCP_RAIZES`; sem configuração, qualquer arquivo é
  recusado. O caminho é resolvido (`..` e links simbólicos seguidos) e só então comparado com as raízes; a verificação
  de pertencer às raízes vem antes da de existência, para o erro não revelar quais arquivos existem fora delas.
- Tamanho máximo: imagem e thumbnail 20 MB; vídeo em arquivo 100 MB. Texto e legenda: 10.000 caracteres. Histórico
  fornecido: 200 posts. O vídeo local é conferido de novo pela sonda (`video_local`): contêiner, 180 s e vertical ou quadrado, antes de qualquer rede. Acrescentados
  após a revisão: as tags somam, no máximo, o mesmo limite de caracteres do texto (e são até 100), porque entram no
  texto codificado; `channel_id` tem até 100 caracteres.
- Mensagens e detalhes de erro nunca trazem caminho absoluto, raízes nem variáveis de ambiente.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.agente import api
from src.agente.atributos import MAX_BYTES_IMAGEM
from src.agente.contratos import (
    MAX_CRIATIVOS_ORDENAR,
    MIN_CRIATIVOS_ORDENAR,
    Codigo,
    ErroAgente,
    PedidoOrdenar,
    PedidoPrever,
)
from src.agente.video_local import MAX_BYTES_VIDEO

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
