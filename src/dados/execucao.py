"""Contexto privado de execução: histórico autorizado, sem redistribuir o snapshot de pesquisa.

O arquivo opcional CASEGALAXY_CONTEXTO (ou data/privado/contexto.json) contém canais e posts.
Sem arquivo, o pedido fornece channel_stats ou usa o consultor YouTube já configurado.
"""

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from src.agente.contratos import PostFornecido
from src.dados.historico import BaseHistorica, Canal, ContextoTreino, Post


def carregar(raiz: Path, versao: dict):
    contexto = versao["contexto_execucao"]
    corte = datetime.fromisoformat(contexto["treino_ate_utc"].replace("Z", "+00:00"))
    arquivo = Path(os.environ.get("CASEGALAXY_CONTEXTO", "data/privado/contexto.json"))
    if not arquivo.is_absolute():
        arquivo = Path(raiz) / arquivo
    if os.environ.get("CASEGALAXY_CONTEXTO") and not arquivo.is_file():
        raise ValueError("arquivo CASEGALAXY_CONTEXTO não encontrado")
    bruto = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.is_file() else {}
    if set(bruto) - {"canais", "posts"}:
        raise ValueError("contexto privado aceita somente canais e posts")
    canais = {}
    for c in bruto.get("canais", []):
        canal = Canal(**c)
        if canal.setor not in ("beleza", "automotivo") or canal.tipo_canal not in ("marca", "criador"):
            raise ValueError("setor ou tipo inválido no contexto privado")
        if canal.channel_id in canais:
            raise ValueError("canal duplicado no contexto privado")
        canais[canal.channel_id] = canal
    posts, linhas, coletas, ids = [], [], [], set()
    for registro in bruto.get("posts", []):
        dados = dict(registro)
        canal_id = dados.pop("channel_id")
        if canal_id not in canais:
            raise ValueError("post sem canal declarado no contexto privado")
        p = PostFornecido.model_validate(dados)
        if not p.video_id or p.video_id in ids:
            raise ValueError("contexto privado exige video_id único por post")
        ids.add(p.video_id)
        posts.append(Post(p.video_id, canal_id, p.published_at, p.likes, "so_historico"))
        coletas.append(p.collected_at)
        linhas.append({"video_id": p.video_id, "channel_id": canal_id, "published_at": p.published_at,
                       "setor": canais[canal_id].setor, "uso_no_contexto": "treino",
                       "y_log": math.log1p(p.likes) if p.likes is not None else None})
    base = BaseHistorica(posts, canais, ContextoTreino("final", corte), max(coletas, default=datetime.now(UTC)))
    return linhas, base


def comparaveis_do_pedido(pedido):
    """Posts enviados pelo usuário; sem embeddings, a seleção existente usa recência.

    Não fabrica IDs ou links para posts que não vieram identificados.
    """
    return [{"video_id": p.video_id, "channel_id": pedido.channel_id or "__canal_informado__",
             "published_at": p.published_at, "setor": pedido.sector, "uso_no_contexto": "treino",
             "y_log": math.log1p(p.likes)} for p in (pedido.channel_stats or ())
            if p.video_id and p.likes is not None]
