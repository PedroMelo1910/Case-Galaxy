"""Comparáveis reais do conjunto de desenvolvimento (treino e validação), nunca de teste final nem da LIM.

Regras: exclui o próprio criativo; só vídeos publicados até `planned_datetime − 30 dias` (idade mínima de maturidade
na data da consulta); mesmo setor; até 3 do mesmo canal e o restante (até 5 no total) de outros canais do setor.
Sem canal na base, os 5 vêm do setor. A similaridade é o cosseno médio dos embeddings de legenda e capa disponíveis;
sem embedding, o desempate é a recência. Nenhum exemplo é sintético.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import numpy as np

from src.contratos import url_do_video

USOS_PERMITIDOS = ("treino", "avaliacao")
DIAS_MATURIDADE = 30
N_DO_CANAL = 3
N_TOTAL = 5
BLOCOS_EMBEDDING = ("leg_emb", "capa_emb")


def _cosseno(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None:
        return None
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape or not (np.isfinite(va).all() and np.isfinite(vb).all()):
        return None
    norma = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb) / norma if norma > 0 else None


def similaridade(consulta: Mapping, candidato: Mapping) -> float | None:
    valores = [s for b in BLOCOS_EMBEDDING if (s := _cosseno(consulta.get(b), candidato.get(b))) is not None]
    return float(np.mean(valores)) if valores else None


def _item(linha: Mapping, sim: float | None, mesmo_canal: bool) -> dict:
    return {
        "video_id": linha["video_id"], "url": url_do_video(linha["video_id"]),
        "channel_id": linha["channel_id"], "published_at": linha["published_at"].isoformat(),
        "curtidas": round(math.expm1(linha["y_log"])), "similaridade": sim, "mesmo_canal": mesmo_canal,
        "origem": "desenvolvimento_real", "uso_no_contexto": linha["uso_no_contexto"],
    }


def _ordenar(par: tuple[Mapping, float | None]) -> tuple:
    linha, sim = par
    return (-(sim if sim is not None else -2.0), -linha["published_at"].timestamp(), linha["video_id"])


def selecionar(linhas: Sequence[Mapping], consulta: Mapping, *, channel_id: str | None, setor: str,
               planejada: datetime, excluir_video_id: str | None) -> list[dict]:
    """Até 5 comparáveis reais; `consulta` leva `leg_emb` e `capa_emb` do criativo (podem faltar)."""
    limite = planejada - timedelta(days=DIAS_MATURIDADE)
    elegiveis = [
        l for l in linhas
        if l["uso_no_contexto"] in USOS_PERMITIDOS and l["video_id"] != excluir_video_id and l["setor"] == setor
        and l["published_at"] <= limite and l.get("y_log") is not None
    ]
    pontuadas = [(l, similaridade(consulta, l)) for l in elegiveis]
    do_canal = sorted((p for p in pontuadas if channel_id and p[0]["channel_id"] == channel_id), key=_ordenar)
    outros = sorted((p for p in pontuadas if not (channel_id and p[0]["channel_id"] == channel_id)), key=_ordenar)
    escolhidos = [(*p, True) for p in do_canal[:N_DO_CANAL]]
    escolhidos += [(*p, False) for p in outros[:N_TOTAL - len(escolhidos)]]
    return [_item(l, s, m) for l, s, m in escolhidos]
