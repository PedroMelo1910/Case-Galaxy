"""Adaptador de histórico atual do canal pela YouTube Data API (consulta de metadados, sem baixar mídia).

Percorre os uploads do mais novo para o mais antigo (`playlistItems.list`, 50 por página) até reunir `MAX_ELEGIVEIS`
publicações com data até o limite (p − 30 dias) ou esgotar `max_paginas`; depois lê as estatísticas dos IDs reunidos
(`videos.list`, 50 por chamada). Custo máximo: 1 + max_paginas + ceil(50 · max_paginas / 50) unidades.

Só entram vídeos públicos de até 180 s. As publicações até o limite levam curtidas; as mais recentes entram com
`likes = None` e servem só à cadência. Inscritos, visualizações e comentários não são usados. Se o limite de páginas
cortar a busca antes de 20 posts elegíveis, o histórico fica curto e isso aparece nos avisos do fluxo.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from src import youtube_api as yt
from src.agente.contratos import Codigo, ErroAgente, PostConsultado, PostFornecido
from src.coleta import duracao_em_segundos
from src.contratos import DURACAO_MAXIMA_SHORT_S
from src.historico import MAX_POSTS

MAX_PAGINAS_PADRAO = 3


def _instante(texto: str) -> datetime:
    return datetime.fromisoformat(texto).astimezone(UTC)


def criar_consultor(cliente: yt.ClienteYouTube, *, max_paginas: int = MAX_PAGINAS_PADRAO,
                    agora: Callable[[], datetime] = lambda: datetime.now(UTC)
                    ) -> Callable[[str, datetime], Sequence[PostFornecido]]:
    def listar_ids(uploads: str, limite: datetime) -> list[str]:
        ids, elegiveis, token = [], 0, None
        for _ in range(max_paginas):
            pagina = cliente.pagina_uploads(uploads, token)
            for item in pagina.get("items", []):
                ids.append(item["contentDetails"]["videoId"])
                publicado = item["contentDetails"].get("videoPublishedAt")
                elegiveis += bool(publicado) and _instante(publicado) <= limite
            token = pagina.get("nextPageToken")
            if elegiveis >= MAX_POSTS or not token:
                break
        return ids

    def consultar(channel_id: str, limite: datetime) -> list[PostFornecido]:
        try:
            canais = cliente.canais([channel_id])
            if not canais:
                raise ErroAgente(Codigo.HISTORICO_INDISPONIVEL, "Canal não encontrado no YouTube.",
                                 {"channel_id": channel_id})
            ids = listar_ids(canais[0]["contentDetails"]["relatedPlaylists"]["uploads"], limite)
            recursos = cliente.videos(ids) if ids else []
        except yt.CotaEsgotada as erro:
            raise ErroAgente(Codigo.COTA_ESGOTADA, "Cota da YouTube Data API esgotada.") from erro
        except (yt.ErroYouTube, KeyError) as erro:
            raise ErroAgente(Codigo.HISTORICO_INDISPONIVEL, "Falha ao consultar o histórico no YouTube.",
                             {"tipo": type(erro).__name__}) from erro
        coletado = agora()
        posts = []
        for r in recursos:
            duracao = duracao_em_segundos(r.get("contentDetails", {}).get("duration"))
            if r.get("status", {}).get("privacyStatus") != "public" or not duracao or duracao > DURACAO_MAXIMA_SHORT_S:
                continue
            try:
                publicado = _instante(r["snippet"]["publishedAt"])
            except (KeyError, ValueError, TypeError):
                continue
            curtidas = r.get("statistics", {}).get("likeCount")
            likes = int(curtidas) if curtidas is not None and publicado <= limite else None
            posts.append(PostConsultado(video_id=r.get("id"), published_at=publicado, likes=likes, collected_at=coletado))
        return posts

    return consultar
