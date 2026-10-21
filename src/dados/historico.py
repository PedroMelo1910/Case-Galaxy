"""Módulo consolidado. Reúne:
- Histórico do canal, atributos N1 e previsões de referência (C5; D-C3, D-C4, D-D1, D-D8).
- Histórico do canal para um pedido: origem (snapshot, fornecido ou consulta atual) e atributos N1.
- Adaptador de histórico atual do canal pela YouTube Data API (consulta de metadados, sem baixar mídia)."""

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyarrow as pa
import pyarrow.parquet as pq

from src.avaliacao import metricas
from src.agente.contratos import (
    Codigo,
    ErroAgente,
    id_do_proprio_video,
    ORIGEM_CONSULTA_ATUAL,
    ORIGEM_FORNECIDO,
    ORIGEM_SNAPSHOT,
    PedidoPrever,
    PostConsultado,
    PostFornecido,
)
from src.config import RAIZ_PROJETO
from src.dados import curadoria, particoes, snapshot
from src.dados.coleta import duracao_em_segundos, DURACAO_MAXIMA_SHORT_S, Papel
from src.integracoes import youtube as yt
from src.proveniencia import sha256_bytes as _sha256


# ──── Histórico do canal, atributos N1 e previsões de referência (C5; D-C3, D-C4, D-D1, D-D8).
VERSAO = "1"
DEFASAGEM_PADRAO = 30
DEFASAGEM_MAXIMA = 3650
MAX_POSTS = 20
MIN_POSTS = 5
MEIA_VIDA_POSTS = 5
JANELA_CADENCIA_DIAS = 90
FUSO_HORARIO = "America/Sao_Paulo"
PAPEIS_HISTORICO = frozenset({Papel.ALVO.value, Papel.SO_HISTORICO.value})
CONTEXTOS_MATERIALIZAVEIS = ("validacao",)
DIRETORIO_PADRAO = RAIZ_PROJETO / "data" / "features"
ARQUIVO_TABELA = "n1.parquet"
ARQUIVO_MANIFESTO = "manifesto.json"
ARQUIVO_VERIFICACAO = "verificacao_desenvolvimento.json"

COLUNAS_IDENTIFICACAO = ("video_id", "channel_id", "published_at", "bloco", "uso_no_contexto")
ATRIBUTOS_N1 = (
    "hist_ref_log_mediana", "hist_log_media", "hist_mme_log", "hist_n_posts", "hist_ref_setor",
    "dias_desde_post_anterior", "posts_90d", "duracao_s", "dia_semana_local", "hora_local", "setor", "tipo_canal",
)
COLUNAS_ALVO = ("y_log", "y_rel")
COLUNAS_REFERENCIA = ("ref_log", "pred_media_aritmetica", "pred_mediana", "pred_mme_log")
COLUNAS_AUDITORIA = ("estado_referencia", "previsoes_de", "limite_historico_utc", "hist_primeiro_utc",
                     "hist_ultimo_utc", "limite_setor_utc", "n_posts_setor")
PROIBIDAS = frozenset({
    "video_id", "channel_id", "published_at", "curtidas", "visualizacoes", "comentarios", "inscritos",
    "total_videos", "topic_details", "topicDetails", "mes", *COLUNAS_ALVO, *COLUNAS_REFERENCIA,
})

REGRAS = {
    "populacao_historica": "posts com papel alvo ou so_historico de canais elegíveis da base curada, inclusive alvos "
    "anteriores fora da amostra de anotação; nunca a LIM, o próprio alvo, posts posteriores ou curtidas ausentes "
    "(que não viram zero); zero genuíno permanece",
    "selecao": "os 20 últimos na ordem total (published_at, video_id) entre os publicados até p − D, inclusive",
    "referencia": "ref_log = log1p(mediana das curtidas brutas); y_rel = log1p(curtidas do alvo) − ref_log; "
    "mediana de n par = média dos dois centrais",
    "previsoes": {
        "pred_media_aritmetica": "log1p(média aritmética das curtidas brutas): baseline do TAPI",
        "pred_mediana": "log1p(mediana das curtidas brutas) = ref_log",
        "pred_mme_log": "média de log1p(curtidas) com peso 0,5^(idade/5); idade 0 = o mais recente na ordem acima",
    },
    "minimo_posts": MIN_POSTS,
    "fallback_setor": "com hist_n_posts < 5: log1p(mediana das curtidas brutas de todos os posts elegíveis do setor "
    "(alvo e só histórico; sem LIM nem o próprio alvo) de canais permitidos no contexto, publicados até "
    "min(treino_ate_utc do contexto, p − D)). A mesma mediana preenche ref_log, as três previsões e os três "
    "atributos históricos: não são três estimadores. hist_n_posts mantém o número real do canal; "
    "hist_ref_setor = 1. Sem posts no setor: estado indisponível, sem zero e sem outro setor. Agrupada por post: "
    "canais com mais posts pesam mais (Pedro, 14/10)",
    "cadencia": "publicações elegíveis do canal (alvo ou só histórico, com ou sem curtidas visíveis; sem LIM) "
    "estritamente anteriores a p, sem a defasagem D; dias_desde_post_anterior em dias decimais, ausente sem "
    "publicação anterior; posts_90d conta p − 90 dias ≤ publicação < p",
    "contexto": "dia_semana_local (0 = segunda) e hora_local (0–23) no fuso declarado; duracao_s; setor e tipo_canal "
    "da lista curada",
    "fuso_horario": FUSO_HORARIO,
    "fora_dos_atributos": "mês, ID do canal, inscritos, visualizações, comentários, totais do canal, topicDetails e "
    "estatísticas do próprio vídeo; identificadores ficam só nas colunas de identificação",
}
LIMITACOES = [
    "Curtidas do histórico e dos alvos são as observadas na coleta, não reconstruídas na data simulada: a defasagem "
    "de D dias é aproximação, não correção (D-C3, D-C6). Avaliação retrospectiva aproximada.",
    "Os controles testados (próprio alvo, posts posteriores, LIM, corte do treino, canais excluídos) não provam a "
    "ausência de todo vazamento temporal.",
    "Cadência e lista de posts sujeitas a viés de sobrevivência: vídeos apagados antes da coleta não aparecem.",
    "Fallback agrupado por post: canais com mais posts influenciam mais a mediana do setor.",
    "A defasagem de 90 dias é aceita, mas a sensibilidade não foi executada; nela, o corte do treino continua o da "
    "partição congelada até decisão do pacote da sensibilidade.",
]
NAO_INCLUIDO = [
    "contextos de teste (consulta única, sob protocolo registrado)",
    "regressão sobre inscritos (baseline 4, só no relatório)",
    "sorteios de canais não vistos: o contexto já aceita canais excluídos, mas nenhum sorteio foi feito",
    "modelos, intervalos de confiança, ablação e sensibilidade 30 × 90",
]


class ErroHistorico(ValueError):
    """Entrada, contexto ou coluna incompatível com as regras do histórico."""


@dataclass(frozen=True)
class Post:
    video_id: str
    channel_id: str
    published_at: datetime
    curtidas: int | None
    papel: str
    duracao_s: int | None = None

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            raise ErroHistorico(f"{self.video_id}: published_at sem fuso")


@dataclass(frozen=True)
class Canal:
    channel_id: str
    setor: str
    tipo_canal: str


@dataclass(frozen=True)
class ContextoTreino:
    """O que o treino de uma avaliação pode usar: publicações até `treino_ate_utc`, fora dos canais excluídos."""

    nome: str
    treino_ate_utc: datetime
    canais_excluidos: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.treino_ate_utc.tzinfo is None:
            raise ErroHistorico(f"contexto {self.nome}: treino_ate_utc sem fuso")


def _validar_defasagem(defasagem_dias: object) -> int:
    if (not isinstance(defasagem_dias, int) or isinstance(defasagem_dias, bool)
            or not 1 <= defasagem_dias <= DEFASAGEM_MAXIMA):
        raise ErroHistorico(f"defasagem inválida: {defasagem_dias!r}; use um inteiro de 1 a {DEFASAGEM_MAXIMA} dias")
    return defasagem_dias


def _fuso() -> ZoneInfo:
    try:
        return ZoneInfo(FUSO_HORARIO)
    except ZoneInfoNotFoundError as erro:
        raise ErroHistorico(f"base de fusos sem {FUSO_HORARIO}; nenhum fuso alternativo é usado") from erro


def _ordem(post: Post) -> tuple[datetime, str]:
    return post.published_at, post.video_id


def historico_do_alvo(posts: Iterable[Post], alvo: Post, *, defasagem_dias: int) -> list[Post]:
    """Os 20 últimos posts elegíveis do canal publicados até p − D, em ordem cronológica."""
    limite = alvo.published_at - timedelta(days=_validar_defasagem(defasagem_dias))
    elegiveis = [
        p for p in posts
        if p.channel_id == alvo.channel_id and p.papel in PAPEIS_HISTORICO and p.curtidas is not None
        and p.video_id != alvo.video_id and p.published_at <= limite
    ]
    return sorted(elegiveis, key=_ordem)[-MAX_POSTS:]


def media_movel_exponencial_log(curtidas_cronologicas: Sequence[int], meia_vida: int = MEIA_VIDA_POSTS) -> float:
    """Média ponderada de log1p(curtidas); o mais recente pesa 1 e o de idade k pesa 0,5^(k/meia_vida)."""
    if not curtidas_cronologicas:
        raise ErroHistorico("média móvel sem posts")
    ultimo = len(curtidas_cronologicas) - 1
    pesos = [0.5 ** ((ultimo - i) / meia_vida) for i in range(len(curtidas_cronologicas))]
    return sum(w * math.log1p(c) for w, c in zip(pesos, curtidas_cronologicas, strict=True)) / sum(pesos)


def referencia_setor(
    posts: Iterable[Post], canais: Mapping[str, Canal], *, setor: str, limite: datetime, excluir_video: str,
    canais_excluidos: frozenset[str],
) -> tuple[float | None, int]:
    """log1p(mediana agrupada das curtidas do setor até o limite) e o número de posts; (None, 0) sem posts."""
    curtidas = [
        p.curtidas for p in posts
        if p.channel_id in canais and canais[p.channel_id].setor == setor and p.channel_id not in canais_excluidos
        and p.papel in PAPEIS_HISTORICO and p.curtidas is not None and p.video_id != excluir_video
        and p.published_at <= limite
    ]
    return (math.log1p(statistics.median(curtidas)), len(curtidas)) if curtidas else (None, 0)


def cadencia(posts_do_canal: Iterable[Post], alvo: Post) -> tuple[float | None, int]:
    """Dias desde a publicação anterior (ausente se não houver) e publicações nos 90 dias antes de p."""
    anteriores = [p.published_at for p in posts_do_canal
                  if p.papel in PAPEIS_HISTORICO and p.video_id != alvo.video_id and p.published_at < alvo.published_at]
    if not anteriores:
        return None, 0
    inicio_janela = alvo.published_at - timedelta(days=JANELA_CADENCIA_DIAS)
    dias = (alvo.published_at - max(anteriores)).total_seconds() / 86400
    return dias, sum(instante >= inicio_janela for instante in anteriores)


def _referencias(historico: Sequence[Post], valor_setor: float | None) -> dict:
    if len(historico) >= MIN_POSTS:
        curtidas = [p.curtidas for p in historico]
        mediana = math.log1p(statistics.median(curtidas))
        return {"ref_log": mediana, "pred_media_aritmetica": math.log1p(statistics.fmean(curtidas)),
                "pred_mediana": mediana, "pred_mme_log": media_movel_exponencial_log(curtidas),
                "estado_referencia": "canal", "previsoes_de": "canal"}
    estado = "setor" if valor_setor is not None else "indisponivel"
    return {**dict.fromkeys(COLUNAS_REFERENCIA, valor_setor), "estado_referencia": estado,
            "previsoes_de": "setor_mesma_mediana" if valor_setor is not None else "indisponivel"}


def _linha(alvo: Post, canal: Canal, todos: Sequence[Post], do_canal: Sequence[Post], canais: Mapping[str, Canal],
           contexto: ContextoTreino, defasagem_dias: int) -> dict:
    historico = historico_do_alvo(do_canal, alvo, defasagem_dias=defasagem_dias)
    limite_historico = alvo.published_at - timedelta(days=defasagem_dias)
    valor_setor = limite_setor = n_setor = None
    if len(historico) < MIN_POSTS:
        limite_setor = min(contexto.treino_ate_utc, limite_historico)
        valor_setor, n_setor = referencia_setor(todos, canais, setor=canal.setor, limite=limite_setor,
                                                excluir_video=alvo.video_id, canais_excluidos=contexto.canais_excluidos)
    refs = _referencias(historico, valor_setor)
    dias, posts_90d = cadencia(do_canal, alvo)
    local = alvo.published_at.astimezone(_fuso())
    y_log = math.log1p(alvo.curtidas) if alvo.curtidas is not None else None
    return {
        "video_id": alvo.video_id, "channel_id": alvo.channel_id, "published_at": alvo.published_at,
        "hist_ref_log_mediana": refs["pred_mediana"], "hist_log_media": refs["pred_media_aritmetica"],
        "hist_mme_log": refs["pred_mme_log"], "hist_n_posts": len(historico),
        "hist_ref_setor": int(len(historico) < MIN_POSTS), "dias_desde_post_anterior": dias, "posts_90d": posts_90d,
        "duracao_s": alvo.duracao_s, "dia_semana_local": local.weekday(), "hora_local": local.hour,
        "setor": canal.setor, "tipo_canal": canal.tipo_canal,
        "y_log": y_log, "y_rel": y_log - refs["ref_log"] if y_log is not None and refs["ref_log"] is not None else None,
        **refs,
        "limite_historico_utc": limite_historico,
        "hist_primeiro_utc": historico[0].published_at if historico else None,
        "hist_ultimo_utc": historico[-1].published_at if historico else None,
        "limite_setor_utc": limite_setor, "n_posts_setor": n_setor,
    }


def calcular(posts: Sequence[Post], canais: Mapping[str, Canal], alvos: Sequence[Post], contexto: ContextoTreino,
             *, defasagem_dias: int = DEFASAGEM_PADRAO) -> list[dict]:
    """Uma linha por alvo, em ordem (published_at, video_id), com atributos, alvo, referências e auditoria."""
    _validar_defasagem(defasagem_dias)
    repetidos = [v for v, n in Counter(p.video_id for p in posts).items() if n > 1]
    if repetidos:
        raise ErroHistorico(f"vídeos repetidos na base: {sorted(repetidos)[:5]}")
    elegiveis = [p for p in posts if p.channel_id in canais]
    por_canal: dict[str, list[Post]] = {}
    for post in elegiveis:
        por_canal.setdefault(post.channel_id, []).append(post)
    linhas = []
    for alvo in sorted(alvos, key=_ordem):
        if alvo.channel_id not in canais:
            raise ErroHistorico(f"{alvo.video_id}: canal fora da lista de canais elegíveis")
        linhas.append(_linha(alvo, canais[alvo.channel_id], elegiveis, por_canal.get(alvo.channel_id, []), canais,
                             contexto, defasagem_dias))
    return linhas


def conferir_atributos(colunas: Sequence[str]) -> None:
    """Recusa coluna proibida, repetida ou fora da lista aprovada da família N1 (D-D8)."""
    invalidas = [c for c in colunas if c in PROIBIDAS or c not in ATRIBUTOS_N1]
    if invalidas or len(set(colunas)) != len(colunas):
        raise ErroHistorico(f"colunas não permitidas como atributo: {invalidas or 'repetidas'}")


def matriz_atributos(linhas: Iterable[Mapping]) -> list[dict]:
    """Só as colunas aprovadas de N1, na ordem fixa; identificadores, alvo e referências ficam de fora."""
    conferir_atributos(ATRIBUTOS_N1)
    return [{c: linha[c] for c in ATRIBUTOS_N1} for linha in linhas]


# --- leitura da base e materialização ---------------------------------------------------------------------------

DATA_UTC = pa.timestamp("us", tz="UTC")
_PAPEIS_COLUNA = {
    **dict.fromkeys(COLUNAS_IDENTIFICACAO, "identificacao"), **dict.fromkeys(ATRIBUTOS_N1, "atributo"),
    **dict.fromkeys(COLUNAS_ALVO, "alvo"), **dict.fromkeys(COLUNAS_REFERENCIA, "referencia"),
    **dict.fromkeys(COLUNAS_AUDITORIA, "auditoria"),
}
_TIPOS = {
    "video_id": pa.string(), "channel_id": pa.string(), "published_at": DATA_UTC, "bloco": pa.string(),
    "uso_no_contexto": pa.string(), "hist_n_posts": pa.int32(), "hist_ref_setor": pa.int8(), "posts_90d": pa.int32(),
    "duracao_s": pa.int32(), "dia_semana_local": pa.int8(), "hora_local": pa.int8(), "setor": pa.string(),
    "tipo_canal": pa.string(), "estado_referencia": pa.string(), "previsoes_de": pa.string(),
    "limite_historico_utc": DATA_UTC, "hist_primeiro_utc": DATA_UTC, "hist_ultimo_utc": DATA_UTC,
    "limite_setor_utc": DATA_UTC, "n_posts_setor": pa.int32(),
}
ESQUEMA = pa.schema(
    [pa.field(nome, _TIPOS.get(nome, pa.float64()), metadata={"papel_coluna": papel})
     for nome, papel in _PAPEIS_COLUNA.items()],
    metadata={"tabela": "historico_n1", "versao_historico": VERSAO},
)


def _iso(instante: datetime | None) -> str | None:
    return instante.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if instante else None


def carregar_base(pasta_base: Path) -> tuple[list[Post], dict[str, Canal]]:
    """Posts e canais da base curada, só com as colunas usadas; tabelas restritas nunca são usadas aqui."""
    tabelas = curadoria.ler_base(Path(pasta_base))
    colunas = ["video_id", "channel_id", "published_at", "curtidas", "papel", "duracao_s"]
    posts = [Post(**v) for v in tabelas["videos.parquet"].select(colunas).to_pylist()]
    canais = {c["channel_id"]: Canal(**c)
              for c in tabelas["canais.parquet"].select(["channel_id", "setor", "tipo_canal"]).to_pylist()}
    return posts, canais


def contexto_da_particao(particao: Mapping, bloco: str, canais_excluidos: Iterable[str] = ()) -> ContextoTreino:
    """Contexto de treino de um bloco avaliado, com o corte lido do manifesto congelado (sem recortar de novo)."""
    resumo = {b["bloco"]: b for b in particao["blocos"]}
    if bloco not in resumo or not resumo[bloco]["treino_ate_utc"]:
        raise ErroHistorico(f"bloco sem corte de treino na partição: {bloco!r}")
    corte = datetime.strptime(resumo[bloco]["treino_ate_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return ContextoTreino(bloco, corte, frozenset(canais_excluidos))


def uso_no_contexto(bloco: str, channel_id: str, publicado: datetime, contexto: ContextoTreino) -> str:
    """avaliacao (bloco avaliado), canal_retido, treino (até o corte) ou intervalo_excluido (depois do corte)."""
    if bloco == contexto.nome:
        return "avaliacao"
    if channel_id in contexto.canais_excluidos:
        return "canal_retido"
    return "treino" if publicado <= contexto.treino_ate_utc else "intervalo_excluido"


def _contagens(linhas: Sequence[dict], posts: Sequence[Post], canais: Mapping[str, Canal],
               contexto: ContextoTreino) -> dict:
    fallback: dict[str, dict[str, dict[str, int]]] = {}
    for linha in linhas:
        if linha["estado_referencia"] != "canal":
            por_uso = fallback.setdefault(linha["setor"], {}).setdefault(linha["uso_no_contexto"], {})
            por_uso[linha["estado_referencia"]] = por_uso.get(linha["estado_referencia"], 0) + 1
    no_corte: dict[str, Counter] = {}
    for p in posts:
        if (p.channel_id in canais and p.channel_id not in contexto.canais_excluidos and p.papel in PAPEIS_HISTORICO
                and p.curtidas is not None and p.published_at <= contexto.treino_ate_utc):
            no_corte.setdefault(canais[p.channel_id].setor, Counter())[p.channel_id] += 1
    return {
        "linhas_por_uso": dict(sorted(Counter(linha["uso_no_contexto"] for linha in linhas).items())),
        "estado_referencia": dict(sorted(Counter(linha["estado_referencia"] for linha in linhas).items())),
        "fallback": fallback,
        "alvos_com_zero_curtidas": sum(linha["y_log"] == 0 for linha in linhas),
        "posts_do_setor_ate_o_corte": {
            setor: {"posts": sum(c.values()), "fracao_do_maior_canal": round(max(c.values()) / sum(c.values()), 4),
                    "por_canal": dict(c.most_common())}
            for setor, c in sorted(no_corte.items())
        },
    }


def _verificacao(linhas: Sequence[dict], curtidas: Mapping[str, int | None], contexto: ContextoTreino,
                 defasagem_dias: int) -> dict:
    avaliadas = [linha for linha in linhas if linha["uso_no_contexto"] == "avaliacao"]
    reais = [curtidas[linha["video_id"]] for linha in avaliadas]
    if not any(r is not None and linha["ref_log"] is not None for r, linha in zip(reais, avaliadas, strict=True)):
        raise ErroHistorico(f"bloco {contexto.nome} sem alvo com curtidas e referência: nada a conferir")
    return {
        "contexto": contexto.nome, "defasagem_dias": defasagem_dias, "blocos_usados": [contexto.nome],
        "aviso": "conferência das previsões de referência no desenvolvimento; sem intervalo de confiança; não é o "
        "relatório do experimento nem usa o teste",
        "linhas": len(avaliadas),
        "estado_referencia": dict(sorted(Counter(linha["estado_referencia"] for linha in avaliadas).items())),
        "metricas": {c: metricas.resumo_erros(reais, [linha[c] for linha in avaliadas])
                     for c in COLUNAS_REFERENCIA[1:]},
    }


def _manifesto(pasta_base: Path, arquivo_particao: Path, particao: Mapping, contexto: ContextoTreino,
               defasagem_dias: int, contagens: dict, arquivos: Mapping[str, bytes]) -> dict:
    prov = json.loads((Path(pasta_base) / curadoria.PROVENIENCIA).read_text(encoding="utf-8"))
    return {
        "versao_historico": VERSAO,
        "tipo": "atributos_n1_e_previsoes_de_referencia",
        "gerado_em_utc": _iso(datetime.now(UTC)),
        "base": {**particao["base"], "pasta": f"{Path(pasta_base).parent.name}/{Path(pasta_base).name}"},
        "particao": {"arquivo": f"{Path(arquivo_particao).parent.name}/{Path(arquivo_particao).name}",
                     "versao": particao["versao_particao"],
                     "sha256_arquivo": _sha256(Path(arquivo_particao).read_bytes()),
                     "sha256_atribuicao": particao["sha256_atribuicao"]},
        "contexto": {"nome": contexto.nome, "treino_ate_utc": _iso(contexto.treino_ate_utc),
                     "canais_excluidos": sorted(contexto.canais_excluidos),
                     "blocos_materializados": ["treino", contexto.nome],
                     "uso_no_contexto": "treino: publicado até treino_ate_utc; intervalo_excluido: do bloco de treino, "
                     "mas depois do corte; canal_retido: canal excluído do treino; avaliacao: bloco avaliado",
                     "para_quem_consome": "treinar só com uso_no_contexto = treino e avaliar só com avaliacao; "
                     "linhas com ref_log ou y_rel nulos (referência indisponível ou curtidas ausentes) ficam fora "
                     "das métricas e são contadas, nunca preenchidas com zero"},
        "defasagem_dias": defasagem_dias,
        "regras": REGRAS,
        "colunas": {papel: [c for c, p in _PAPEIS_COLUNA.items() if p == papel]
                    for papel in ("identificacao", "atributo", "alvo", "referencia", "auditoria")},
        "tabela": {"arquivo": ARQUIVO_TABELA, "linhas": contagens["linhas"],
                   "sha256": _sha256(arquivos[ARQUIVO_TABELA])},
        "verificacao": {"arquivo": ARQUIVO_VERIFICACAO, "sha256": _sha256(arquivos[ARQUIVO_VERIFICACAO])},
        "contagens": {k: v for k, v in contagens.items() if k != "linhas"},
        "codigo": snapshot.estado_do_codigo(),
        "retencao": {"apagar_ate_utc": prov["retencao"]["apagar_ate_utc"],
                     "herdada_de": f"snapshot={prov['origem']['snapshot_id']}"},
        "limitacoes": LIMITACOES,
        "nao_incluido": NAO_INCLUIDO,
    }


def _gravar_pasta(final: Path, arquivos: Mapping[str, bytes]) -> Path:
    if final.exists():
        raise FileExistsError(f"saída já existe e não será sobrescrita: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporaria = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".parcial", dir=final.parent))
    try:
        for nome, dados in arquivos.items():
            with open(temporaria / nome, "wb") as arquivo:
                arquivo.write(dados)
                arquivo.flush()
                os.fsync(arquivo.fileno())
        if final.exists():
            raise FileExistsError(f"saída já existe e não será sobrescrita: {final}")
        temporaria.rename(final)
    except BaseException:
        shutil.rmtree(temporaria, ignore_errors=True)
        raise
    return final


def materializar(pasta_base: Path, arquivo_particao: Path, *, contexto: str = "validacao",
                 defasagem_dias: int = DEFASAGEM_PADRAO, destino_raiz: Path = DIRETORIO_PADRAO) -> Path:
    """Grava atributos N1 e referências dos alvos de treino e do bloco avaliado; nunca sobrescreve."""
    if contexto not in CONTEXTOS_MATERIALIZAVEIS:
        raise ErroHistorico(f"contexto {contexto!r} não é materializado neste pacote; só {CONTEXTOS_MATERIALIZAVEIS}")
    _validar_defasagem(defasagem_dias)
    _fuso()
    particao = particoes.ler_particao(arquivo_particao, pasta_base=pasta_base)
    final = (Path(destino_raiz) / f"snapshot={particao['base']['snapshot_id']}" / f"historico=v{VERSAO}"
             / f"defasagem={defasagem_dias}" / f"contexto={contexto}")
    if final.exists():
        raise FileExistsError(f"saída já existe e não será sobrescrita: {final}")
    ctx = contexto_da_particao(particao, contexto)
    posts, canais = carregar_base(pasta_base)
    atribuicao = particao["atribuicao"]
    alvos = [p for p in posts if p.papel == Papel.ALVO.value and atribuicao.get(p.video_id) in ("treino", contexto)]
    linhas = [
        linha | {"bloco": atribuicao[linha["video_id"]],
                 "uso_no_contexto": uso_no_contexto(atribuicao[linha["video_id"]], linha["channel_id"],
                                                    linha["published_at"], ctx)}
        for linha in calcular(posts, canais, alvos, ctx, defasagem_dias=defasagem_dias)
    ]
    tabela = pa.Table.from_pylist([{c: linha[c] for c in ESQUEMA.names} for linha in linhas], schema=ESQUEMA)
    saida = pa.BufferOutputStream()
    pq.write_table(tabela, saida)
    verificacao = _verificacao(linhas, {p.video_id: p.curtidas for p in alvos}, ctx, defasagem_dias)
    arquivos = {
        ARQUIVO_TABELA: saida.getvalue().to_pybytes(),
        ARQUIVO_VERIFICACAO: json.dumps(verificacao, ensure_ascii=False, indent=1).encode("utf-8"),
    }
    contagens = {"linhas": len(linhas), **_contagens(linhas, posts, canais, ctx)}
    manifesto = _manifesto(pasta_base, arquivo_particao, particao, ctx, defasagem_dias, contagens, arquivos)
    arquivos[ARQUIVO_MANIFESTO] = json.dumps(manifesto, ensure_ascii=False, indent=1).encode("utf-8")
    return _gravar_pasta(final, arquivos)


def main(argv: Sequence[str] | None = None) -> int:
    """0 = gravado; 2 = entrada recusada; 3 = saída já existe; 4 = falha ao gravar."""
    parser = argparse.ArgumentParser(prog="python -m src.dados.historico", description="Atributos N1 e referências.")
    parser.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n> da base curada")
    parser.add_argument("--particao", type=Path, required=True, help="particao.json congelado")
    parser.add_argument("--contexto", default="validacao", choices=CONTEXTOS_MATERIALIZAVEIS)
    parser.add_argument("--defasagem", type=int, default=DEFASAGEM_PADRAO, help="D em dias (30; 90 na sensibilidade)")
    parser.add_argument("--destino", type=Path, default=DIRETORIO_PADRAO, help="raiz de data/features")
    args = parser.parse_args(argv)
    try:
        pasta = materializar(args.base, args.particao, contexto=args.contexto, defasagem_dias=args.defasagem,
                             destino_raiz=args.destino)
    except FileExistsError as erro:
        print(f"{erro}.", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError, KeyError) as erro:
        print(f"Histórico recusado ({type(erro).__name__}): {erro}. Nada foi gravado.", file=sys.stderr)
        return 2
    except OSError as erro:
        print(f"Falha ao gravar o histórico ({type(erro).__name__}: {erro}).", file=sys.stderr)
        return 4
    manifesto = json.loads((pasta / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))
    print(json.dumps({"pasta": str(pasta), "contagens": manifesto["contagens"]}, ensure_ascii=False, indent=1))
    return 0


# ──── Histórico do canal para um pedido: origem (snapshot, fornecido ou consulta atual) e atributos N1.
ID_ALVO_PLANEJADO = "__planejado__"

Consultor = Callable[..., Sequence[PostFornecido]]
"""`(channel_id, publicados_ate)`: posts até o limite com curtidas e posts recentes só para a cadência. Se o atributo
`aceita_exclusao_do_alvo` for verdadeiro, recebe também `excluir_video_id=`: o vídeo-alvo não entra na leitura."""


@dataclass(frozen=True)
class BaseHistorica:
    """Posts e canais do snapshot, o contexto de treino permitido e a data em que o snapshot foi coletado."""

    posts: Sequence[Post]
    canais: Mapping[str, Canal]
    contexto: ContextoTreino
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
            "primeiro_post_utc": _iso_historico_canal(self.linha["hist_primeiro_utc"]),
            "ultimo_post_utc": _iso_historico_canal(self.linha["hist_ultimo_utc"]), "canal_conhecido": self.canal_conhecido,
            "tipo_canal": self.tipo_canal,
        }


def _iso_historico_canal(instante: datetime | None) -> str | None:
    return instante.isoformat() if instante else None


def _aviso(codigo: str, descricao: str, classe: str = "atencao") -> dict:
    return {"classe": classe, "codigo": codigo, "descricao": descricao}


def _posts_externos(estatisticas: Sequence[PostFornecido], channel_id: str,
                    excluir_video_id: str | None = None) -> list[Post]:
    """Posts do histórico fornecido ou consultado. O próprio criativo (id da URL ou `own_video_id`) sai antes dos
    cálculos, qualquer que seja a data planejada; posts sem id mantêm `externo:i` e só a data os protege."""
    posts = []
    for i, p in enumerate(estatisticas):
        id_real = getattr(p, "video_id", None)
        if id_real is not None and id_real == excluir_video_id:
            continue
        posts.append(Post(video_id=id_real or f"externo:{i}", channel_id=channel_id, published_at=p.published_at,
                               curtidas=p.likes, papel="so_historico"))
    return posts


def _recusar_ids_de_outros_canais(externos: Sequence[Post], restantes: Sequence[Post]) -> None:
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
        limite = pedido.planned_datetime - timedelta(days=DEFASAGEM_PADRAO)
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
    canais[canal_id] = Canal(channel_id=canal_id, setor=pedido.sector, tipo_canal=tipo)
    alvo = Post(video_id=_id_alvo(pedido), channel_id=canal_id, published_at=pedido.planned_datetime,
                     curtidas=None, papel="alvo", duracao_s=pedido.duration_s)
    linha = calcular(posts, canais, [alvo], base.contexto)[0]
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
    if linha["hist_n_posts"] < MIN_POSTS:
        avisos.append(_aviso("historico_curto",
                             f"Histórico curto: {linha['hist_n_posts']} posts elegíveis (mínimo {MIN_POSTS}); "
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


# ──── Adaptador de histórico atual do canal pela YouTube Data API (consulta de metadados, sem baixar mídia
MAX_PAGINAS_PADRAO = 3


def _instante(texto: str) -> datetime:
    return datetime.fromisoformat(texto).astimezone(UTC)


def criar_consultor(cliente: yt.ClienteYouTube, *, max_paginas: int = MAX_PAGINAS_PADRAO,
                    agora: Callable[[], datetime] = lambda: datetime.now(UTC)
                    ) -> Callable[..., Sequence[PostFornecido]]:
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

    def consultar(channel_id: str, limite: datetime, excluir_video_id: str | None = None) -> list[PostFornecido]:
        try:
            canais = cliente.canais([channel_id])
            if not canais:
                raise ErroAgente(Codigo.HISTORICO_INDISPONIVEL, "Canal não encontrado no YouTube.",
                                 {"channel_id": channel_id})
            ids = [i for i in listar_ids(canais[0]["contentDetails"]["relatedPlaylists"]["uploads"], limite)
                   if i != excluir_video_id]
            recursos = cliente.videos(ids) if ids else []
        except yt.CotaEsgotada as erro:
            raise ErroAgente(Codigo.COTA_ESGOTADA, "Cota da YouTube Data API esgotada.",
                             {"origem": "youtube_data_api"}) from erro
        except yt.LimiteDeUnidades as erro:
            raise ErroAgente(Codigo.COTA_ESGOTADA, "O teto de unidades da YouTube Data API desta execução foi atingido.",
                             {"origem": "limite_da_execucao"}) from erro
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

    consultar.aceita_exclusao_do_alvo = True
    return consultar


if __name__ == "__main__":
    sys.exit(main())
