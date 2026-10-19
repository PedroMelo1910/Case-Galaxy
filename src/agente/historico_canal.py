"""Histórico do canal para um pedido: origem (snapshot, fornecido ou consulta atual) e atributos N1.

Reutiliza `historico.calcular`, a mesma função que gerou as linhas de treino, com um alvo hipotético publicado em
`planned_datetime` e sem curtidas. Assim valem as regras aprovadas: 20 últimos posts elegíveis publicados até
p − 30 dias, fallback do setor com menos de 5 posts e nenhum dado do próprio vídeo, da LIM ou posterior a p.

Três origens, sempre informadas com data:
- `historico_do_snapshot`: base curada coletada em `data_referencia` (não é uma consulta atual);
- `historico_fornecido`: posts enviados no pedido, com `collected_at`;
- `consulta_atual`: YouTube Data API consultada agora, só metadados e curtidas de posts anteriores.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src import historico as hist
from src.agente.contratos import (
    ORIGEM_CONSULTA_ATUAL,
    ORIGEM_FORNECIDO,
    ORIGEM_SNAPSHOT,
    Codigo,
    ErroAgente,
    PedidoPrever,
    PostFornecido,
    id_do_proprio_video,
)

ID_ALVO_PLANEJADO = "__planejado__"

Consultor = Callable[..., Sequence[PostFornecido]]
"""`(channel_id, publicados_ate)`: posts até o limite com curtidas e posts recentes só para a cadência. Se o atributo
`aceita_exclusao_do_alvo` for verdadeiro, recebe também `excluir_video_id=`: o vídeo-alvo não entra na leitura."""


@dataclass(frozen=True)
class BaseHistorica:
    """Posts e canais do snapshot, o contexto de treino permitido e a data em que o snapshot foi coletado."""

    posts: Sequence[hist.Post]
    canais: Mapping[str, hist.Canal]
    contexto: hist.ContextoTreino
    coletado_em: datetime


@dataclass(frozen=True)
class ResultadoHistorico:
    linha: dict
    origem: str
    data_referencia: datetime
    canal_conhecido: bool
    tipo_canal: str | None
    avisos: tuple[dict, ...] = field(default_factory=tuple)

    def resumo(self) -> dict:
        return {
            "origem": self.origem, "data_referencia": self.data_referencia.isoformat(),
            "estado_referencia": self.linha["estado_referencia"], "n_posts": self.linha["hist_n_posts"],
            "limite_historico_utc": self.linha["limite_historico_utc"].isoformat(),
            "primeiro_post_utc": _iso(self.linha["hist_primeiro_utc"]),
            "ultimo_post_utc": _iso(self.linha["hist_ultimo_utc"]), "canal_conhecido": self.canal_conhecido,
            "tipo_canal": self.tipo_canal,
        }


def _iso(instante: datetime | None) -> str | None:
    return instante.isoformat() if instante else None


def _aviso(codigo: str, descricao: str, classe: str = "atencao") -> dict:
    return {"classe": classe, "codigo": codigo, "descricao": descricao}


def _posts_externos(estatisticas: Sequence[PostFornecido], channel_id: str,
                    excluir_video_id: str | None = None) -> list[hist.Post]:
    """Posts do histórico fornecido ou consultado. O próprio criativo (id da URL ou `own_video_id`) sai antes dos
    cálculos, qualquer que seja a data planejada; posts sem id mantêm `externo:i` e só a data os protege."""
    posts = []
    for i, p in enumerate(estatisticas):
        id_real = getattr(p, "video_id", None)
        if id_real is not None and id_real == excluir_video_id:
            continue
        posts.append(hist.Post(video_id=id_real or f"externo:{i}", channel_id=channel_id, published_at=p.published_at,
                               curtidas=p.likes, papel="so_historico"))
    return posts


def _recusar_ids_de_outros_canais(externos: Sequence[hist.Post], restantes: Sequence[hist.Post]) -> None:
    """Um `video_id` fornecido que já pertence a outro canal da base é entrada inválida, não erro interno."""
    existentes = {p.video_id for p in restantes}
    colisoes = sorted(p.video_id for p in externos if p.video_id in existentes)
    if colisoes:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA,
                         "`video_id` do histórico fornecido pertence a outro canal da base.",
                         {"campo": "channel_stats.video_id", "ids": colisoes[:5]})


def _escolher_fonte(pedido: PedidoPrever, base: BaseHistorica, consultor: Consultor | None,
                    canal_conhecido: bool, agora: datetime) -> tuple[str, list[PostFornecido] | None, datetime]:
    if pedido.channel_stats:  # lista vazia não apaga o histórico do snapshot
        coletas = [p.collected_at for p in pedido.channel_stats]
        return ORIGEM_FORNECIDO, list(pedido.channel_stats), max(coletas, default=agora)
    if pedido.consultar_youtube:
        if pedido.channel_id is None:
            raise ErroAgente(Codigo.HISTORICO_INDISPONIVEL, "Consulta ao YouTube pedida sem `channel_id`.")
        if consultor is None:
            raise ErroAgente(Codigo.HISTORICO_INDISPONIVEL,
                             "A consulta ao YouTube não está habilitada neste servidor (modo offline): envie "
                             "`channel_stats` ou use um servidor com o modo online.", {"channel_id": pedido.channel_id})
        limite = pedido.planned_datetime - timedelta(days=hist.DEFASAGEM_PADRAO)
        return ORIGEM_CONSULTA_ATUAL, list(_consultar(consultor, pedido, limite)), agora
    if canal_conhecido:
        return ORIGEM_SNAPSHOT, None, base.coletado_em
    raise ErroAgente(
        Codigo.HISTORICO_INDISPONIVEL,
        "Canal fora da base e sem histórico: envie `channel_stats` ou peça a consulta ao YouTube. "
        "Não é um histórico curto: não há fonte de histórico.", {"channel_id": pedido.channel_id})


def _consultar(consultor: Consultor, pedido: PedidoPrever, limite: datetime) -> Sequence[PostFornecido]:
    """Consulta o canal; um consultor que sabe excluir o vídeo-alvo da leitura de estatísticas recebe o id dele.

    O pedido ao YouTube nunca deve pedir as curtidas, visualizações ou comentários do vídeo consultado (ARQUITETURA §7).
    """
    alvo = id_do_proprio_video(pedido)
    if alvo is not None and getattr(consultor, "aceita_exclusao_do_alvo", False):
        return consultor(pedido.channel_id, limite, excluir_video_id=alvo)
    return consultor(pedido.channel_id, limite)


def _id_alvo(pedido: PedidoPrever) -> str:
    return id_do_proprio_video(pedido) or ID_ALVO_PLANEJADO


def obter_historico(pedido: PedidoPrever, base: BaseHistorica, *, consultor: Consultor | None = None,
                    agora: datetime | None = None) -> ResultadoHistorico:
    agora = agora or datetime.now(UTC)
    canal_id = pedido.channel_id or "__canal_informado__"
    registrado = base.canais.get(canal_id)
    if registrado is not None and registrado.setor != pedido.sector:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, "O setor informado difere do setor registrado do canal.",
                         {"informado": pedido.sector, "registrado": registrado.setor})
    avisos: list[dict] = []
    tipo = registrado.tipo_canal if registrado is not None else pedido.channel_type
    if registrado is not None and pedido.channel_type and pedido.channel_type != registrado.tipo_canal:
        avisos.append(_aviso("tipo_canal_divergente", "tipo_canal informado difere do registrado; vale o registrado."))
    if registrado is None and tipo is None:
        avisos.append(_aviso("tipo_canal_ausente",
                             "Canal novo sem tipo_canal (marca ou criador): o atributo fica ausente, sem palpite."))
    origem, externos, data_ref = _escolher_fonte(pedido, base, consultor, registrado is not None, agora)
    posts = list(base.posts)
    if externos is not None:  # histórico fornecido ou consultado substitui o do snapshot para este canal
        restantes = [p for p in posts if p.channel_id != canal_id]
        novos = _posts_externos(externos, canal_id, id_do_proprio_video(pedido))
        _recusar_ids_de_outros_canais(novos, restantes)
        posts = restantes + novos
    canais = dict(base.canais)
    canais[canal_id] = hist.Canal(channel_id=canal_id, setor=pedido.sector, tipo_canal=tipo)
    alvo = hist.Post(video_id=_id_alvo(pedido), channel_id=canal_id, published_at=pedido.planned_datetime,
                     curtidas=None, papel="alvo", duracao_s=pedido.duration_s)
    linha = hist.calcular(posts, canais, [alvo], base.contexto)[0]
    avisos.extend(_avisos_de_origem(origem, data_ref, pedido.planned_datetime, linha))
    avisos.extend(_avisos_de_identidade(pedido, origem, externos, data_ref))
    return ResultadoHistorico(linha, origem, data_ref, registrado is not None, tipo, tuple(avisos))


def _avisos_de_identidade(pedido: PedidoPrever, origem: str, externos: Sequence[PostFornecido] | None,
                          data_ref: datetime) -> list[dict]:
    """Declara a limitação quando o agente não consegue garantir a exclusão do próprio criativo.

    Só importa se o histórico foi lido depois da data planejada (demonstração retrospectiva): um criativo ainda não
    publicado não pode estar nele. Sem identidade, ou com posts sem `video_id`, só a defasagem de 30 dias protege.
    """
    if pedido.planned_datetime > data_ref:
        return []
    if id_do_proprio_video(pedido) is None:
        return [_aviso(
            "identidade_do_proprio_video_ausente",
            "Sem URL do YouTube nem `own_video_id`, o agente não sabe qual post do histórico é o próprio criativo: "
            "se ele já foi publicado, só a defasagem de 30 dias o mantém fora da referência (e a cadência pode "
            "contá-lo). Informe `own_video_id` e o `video_id` de cada post para excluí-lo antes dos cálculos.")]
    sem_id = sum(1 for p in externos or () if getattr(p, "video_id", None) is None) \
        if origem == ORIGEM_FORNECIDO else 0
    if sem_id:
        return [_aviso(
            "historico_fornecido_sem_ids",
            f"{sem_id} post{'s' if sem_id != 1 else ''} do histórico fornecido sem `video_id`: não é possível "
            "garantir que o próprio criativo esteja fora; para esses posts só a defasagem de 30 dias protege.")]
    return []


def _avisos_de_origem(origem: str, data_ref: datetime, planejada: datetime, linha: dict) -> list[dict]:
    if linha["estado_referencia"] == "indisponivel":
        raise ErroAgente(Codigo.HISTORICO_INDISPONIVEL,
                         "Sem referência: nem o canal nem o setor têm posts elegíveis até a data planejada.")
    avisos = []
    if linha["hist_n_posts"] < hist.MIN_POSTS:
        avisos.append(_aviso("historico_curto",
                             f"Histórico curto: {linha['hist_n_posts']} posts elegíveis (mínimo {hist.MIN_POSTS}); "
                             "a referência vem do setor."))
    if linha["estado_referencia"] == "setor" and linha["limite_setor_utc"] is not None \
            and linha["limite_setor_utc"] < linha["limite_historico_utc"]:
        avisos.append(_aviso("setor_limitado_ao_corte_do_treino",
                             f"A referência do setor usa só posts até {linha['limite_setor_utc']:%Y-%m-%d}, o corte "
                             "do treino do modelo; posts mais novos do setor não entram.", "info"))
    if origem == ORIGEM_SNAPSHOT:
        atraso = (planejada - data_ref).days
        texto = (f"Histórico do snapshot coletado em {data_ref:%Y-%m-%d}, não uma consulta atual."
                 + (f" A data planejada é {atraso} dias posterior; posts recentes não aparecem." if atraso > 0 else ""))
        avisos.append(_aviso("historico_do_snapshot", texto, "info" if atraso <= 0 else "atencao"))
    elif origem == ORIGEM_FORNECIDO:
        avisos.append(_aviso("historico_fornecido", f"Histórico enviado no pedido, lido em até {data_ref:%Y-%m-%d}.",
                             "info"))
    else:
        avisos.append(_aviso("consulta_atual", f"Histórico consultado ao YouTube em {data_ref:%Y-%m-%d %H:%MZ}.",
                             "info"))
    return avisos
