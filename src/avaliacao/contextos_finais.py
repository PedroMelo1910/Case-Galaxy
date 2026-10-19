"""Contextos da avaliação final: N1 e dataset de cada dobra, braço e sorteio (protocolo final §3 a §6).

Um contexto fixa: o corte de treino (primeira publicação do bloco-âncora menos o intervalo), a defasagem do histórico,
os canais excluídos do ajuste, os blocos avaliados e os blocos com alvos materializados. As linhas dos blocos
anteriores publicadas até o corte viram TREINO (janela expansiva); as demais entram no intervalo excluído; o bloco
avaliado é `avaliacao` mesmo para canais retidos (que nunca são treino).

Reaproveita, sem reescrever, o cálculo do histórico (`historico.calcular`), o manifesto do N1 e a montagem do dataset
(`dataset_desenvolvimento.montar`) do desenvolvimento, para que o contexto final seja construído pelo mesmo código.
Qualquer contexto que materialize bloco de teste lê rótulos de teste e só roda com o portão da consulta aberto.
"""

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src import dataset_desenvolvimento as ds
from src import historico as hs
from src import particoes
from src.avaliacao.protocolo_final import ConsultaAberta, PortaoFechado
from src.contratos import Papel

FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"
PREFIXO_TESTE = "teste"


@dataclass(frozen=True)
class EspecContexto:
    nome: str
    bloco_corte: str  # bloco cuja primeira publicação, menos o intervalo, define o corte de treino
    avaliados: tuple[str, ...]
    permitidos: tuple[str, ...]  # blocos com alvos materializados (treino, validação, testes anteriores, avaliados)
    defasagem_dias: int
    intervalo_dias: int
    canais_excluidos: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fora = [b for b in self.avaliados if b not in self.permitidos]
        if fora:
            raise ValueError(f"avaliados fora dos permitidos: {fora}")

    @property
    def pasta(self) -> str:
        return re.sub(r"[^A-Za-z0-9_=.-]", "_", self.nome)

    @property
    def usa_teste(self) -> bool:
        return any(b.startswith(PREFIXO_TESTE) for b in self.permitidos)


def exigir_portao(espec: EspecContexto, *, consulta: ConsultaAberta | None) -> None:
    """Contexto com bloco de teste só existe com a PROVA do portão da consulta única (`ConsultaAberta`).

    Uma declaração do chamador (um dicionário com `aberta`) não vale: a prova só nasce de `abrir_consulta_ao_teste`
    e o marcador criado antes da leitura ainda precisa existir e ser do mesmo protocolo."""
    if not espec.usa_teste:
        return
    if not isinstance(consulta, ConsultaAberta):
        raise PortaoFechado(f"o contexto {espec.nome!r} lê rótulos de teste e o portão da consulta ao teste não está "
                            "aberto")
    consulta.verificar()
    consulta.marcar_leitura()  # fase 2: a partir daqui rótulos de teste podem ser lidos


def corte_de_treino(particao: Mapping, bloco: str, intervalo_dias: int) -> datetime:
    resumo = {b["bloco"]: b for b in particao["blocos"]}
    if bloco not in resumo or not resumo[bloco].get("primeira_publicacao_utc"):
        raise ValueError(f"bloco sem primeira publicação na partição: {bloco!r}")
    primeira = datetime.strptime(resumo[bloco]["primeira_publicacao_utc"], FORMATO_UTC).replace(tzinfo=UTC)
    return primeira - timedelta(days=intervalo_dias)


def contexto_de(espec: EspecContexto, particao: Mapping) -> hs.ContextoTreino:
    return hs.ContextoTreino(espec.nome, corte_de_treino(particao, espec.bloco_corte, espec.intervalo_dias),
                             frozenset(espec.canais_excluidos))


def rotular_uso(bloco: str, canal: str, publicado: datetime, *, avaliados: Collection[str],
                ctx: hs.ContextoTreino) -> str:
    """avaliacao (bloco avaliado, qualquer canal), canal_retido, treino (até o corte) ou intervalo_excluido."""
    if bloco in avaliados:
        return "avaliacao"
    if canal in ctx.canais_excluidos:
        return "canal_retido"
    return "treino" if publicado <= ctx.treino_ate_utc else "intervalo_excluido"


def n1_do_contexto(posts: Sequence[hs.Post], canais: Mapping[str, hs.Canal], atribuicao: Mapping[str, str],
                   ctx: hs.ContextoTreino, *, avaliados: Collection[str], permitidos: Collection[str],
                   defasagem_dias: int, ids_amostra: Collection[str] | None = None) -> list[dict]:
    """Linhas do N1 dos alvos dos blocos permitidos (e da amostra, se dada), com `bloco` e `uso_no_contexto`."""
    alvos = [p for p in posts if p.papel == Papel.ALVO.value and atribuicao.get(p.video_id) in permitidos
             and (ids_amostra is None or p.video_id in ids_amostra)]
    return [linha | {"bloco": atribuicao[linha["video_id"]],
                     "uso_no_contexto": rotular_uso(atribuicao[linha["video_id"]], linha["channel_id"],
                                                    linha["published_at"], avaliados=avaliados, ctx=ctx)}
            for linha in hs.calcular(posts, canais, alvos, ctx, defasagem_dias=defasagem_dias)]


def _descricao(espec: EspecContexto, ctx: hs.ContextoTreino) -> dict:
    """Tudo que distingue um contexto de outro com o mesmo nome, incluindo o corte vindo da partição."""
    return {"nome": espec.nome, "bloco_corte": espec.bloco_corte, "avaliados": list(espec.avaliados),
            "permitidos": list(espec.permitidos), "intervalo_dias": espec.intervalo_dias,
            "defasagem_dias": espec.defasagem_dias, "canais_excluidos": sorted(espec.canais_excluidos),
            "treino_ate_utc": ctx.treino_ate_utc.strftime(FORMATO_UTC)}


def _escrever_n1(linhas: Sequence[dict], pasta: Path, *, pasta_base: Path, arquivo_particao: Path, particao: Mapping,
                 ctx: hs.ContextoTreino, espec: EspecContexto, posts: Sequence[hs.Post],
                 canais: Mapping[str, hs.Canal]) -> Path:
    descricao = _descricao(espec, ctx)
    if (pasta / hs.ARQUIVO_MANIFESTO).is_file() and (pasta / hs.ARQUIVO_TABELA).is_file():
        gravado = json.loads((pasta / hs.ARQUIVO_MANIFESTO).read_text(encoding="utf-8")).get("contexto_final")
        if gravado != descricao:
            raise ValueError(f"a pasta do contexto {espec.nome!r} guarda outro contexto (corte, intervalo, defasagem "
                             "ou canais diferentes): não será reaproveitada")
        return pasta  # mesmo contexto já gravado: idempotente
    tabela = pa.Table.from_pylist([{c: l[c] for c in hs.ESQUEMA.names} for l in linhas], schema=hs.ESQUEMA)
    saida = pa.BufferOutputStream()
    pq.write_table(tabela, saida)
    verificacao = {"contexto": ctx.nome, "defasagem_dias": espec.defasagem_dias,
                   "aviso": "contexto da avaliação final; sem métricas de referência calculadas aqui",
                   "linhas_avaliadas": sum(l["uso_no_contexto"] == "avaliacao" for l in linhas)}
    arquivos = {hs.ARQUIVO_TABELA: saida.getvalue().to_pybytes(),
                hs.ARQUIVO_VERIFICACAO: json.dumps(verificacao, ensure_ascii=False, indent=1).encode("utf-8")}
    contagens = {"linhas": len(linhas), **hs._contagens(linhas, posts, canais, ctx)}
    manifesto = hs._manifesto(pasta_base, arquivo_particao, particao, ctx, espec.defasagem_dias, contagens, arquivos)
    manifesto["contexto_final"] = descricao
    arquivos[hs.ARQUIVO_MANIFESTO] = json.dumps(manifesto, ensure_ascii=False, indent=1).encode("utf-8")
    return hs._gravar_pasta(pasta, arquivos)


def materializar_contexto(espec: EspecContexto, *, pasta_base: Path, arquivo_particao: Path, destino: Path,
                          pasta_legenda_capa: Path, arquivo_amostra: Path,
                          carregar_anotacoes: Callable[[Sequence[str]], Mapping[str, Mapping]], codificador,
                          cache_embeddings: Path, consulta: ConsultaAberta | None = None) -> dict:
    """N1 do contexto → dataset (mesmo `montar` do desenvolvimento) → linhas. Idempotente por contexto."""
    exigir_portao(espec, consulta=consulta)
    particao = particoes.ler_particao(arquivo_particao, pasta_base=pasta_base)
    posts, canais = hs.carregar_base(pasta_base)
    amostra = json.loads(Path(arquivo_amostra).read_text(encoding="utf-8"))
    ids = {x["video_id"] for x in amostra["lista"]}
    ctx = contexto_de(espec, particao)
    linhas = n1_do_contexto(posts, canais, particao["atribuicao"], ctx, avaliados=espec.avaliados,
                            permitidos=espec.permitidos, defasagem_dias=espec.defasagem_dias, ids_amostra=ids)
    pasta_n1 = _escrever_n1(linhas, Path(destino) / "n1" / espec.pasta, pasta_base=pasta_base,
                            arquivo_particao=arquivo_particao, particao=particao, ctx=ctx, espec=espec, posts=posts,
                            canais=canais)
    pasta_dataset = ds.montar(
        tabela_n1=pasta_n1 / hs.ARQUIVO_TABELA, manifesto_n1=pasta_n1 / hs.ARQUIVO_MANIFESTO,
        pasta_legenda_capa=pasta_legenda_capa, arquivo_amostra=arquivo_amostra, carregar_anotacoes=carregar_anotacoes,
        codificador=codificador, cache_embeddings=cache_embeddings, destino_raiz=Path(destino) / "dataset" / espec.pasta)
    dataset, manifesto = ds.carregar_dataset(pasta_dataset)
    return {"espec": espec, "pasta_n1": pasta_n1, "pasta_dataset": pasta_dataset, "linhas": dataset,
            "manifesto": manifesto}
