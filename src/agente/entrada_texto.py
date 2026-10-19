"""Texto do criativo como legenda planejada (ARQUITETURA §7): mapeamento explícito, equivalência e conflito.

Interpretação documentada: um criativo de texto é a legenda planejada. `creative.texto` é o texto principal e é
mapeado para a representação de legenda (título e descrição) usada no treino, com a mesma composição e a mesma
normalização da extração (`leg.normalizar_texto` e `leg.compor_legenda`):

- sem `caption` (ou com `caption` em branco), o texto inteiro vira a descrição e o título fica ausente;
- com `caption`, ela é aceita se o texto composto (título e descrição) for equivalente ao texto, sem duplicar nada;
- se divergirem, é um erro de entrada: nenhuma das duas é escolhida e elas não são concatenadas.

Para vídeo e imagem a legenda continua sendo a informada em `caption`.
"""

from src import extracao_legenda as leg
from src.agente.contratos import Codigo, ErroAgente, Legenda, PedidoPrever


def _composta(legenda: Legenda) -> str:
    return leg.compor_legenda(legenda.title, legenda.description, None)


def legenda_planejada(pedido: PedidoPrever) -> Legenda | None:
    """Legenda que as extrações usam; para texto, vem de `creative.texto` (com `caption` só se equivalente)."""
    criativo = pedido.creative
    if criativo.tipo != "texto":
        return pedido.caption
    texto = leg.normalizar_texto(criativo.texto)
    if not texto:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, "O texto do criativo está sem conteúdo.",
                         {"campos": ["creative.texto"]})
    informada = pedido.caption
    if informada is None or not _composta(informada):
        return Legenda(title=None, description=criativo.texto)
    if _composta(informada) == texto:
        return informada
    raise ErroAgente(Codigo.ENTRADA_INVALIDA,
                     "O texto do criativo e a legenda (`caption`) divergem; envie só um deles ou textos equivalentes.",
                     {"campos": ["creative.texto", "caption"]})
