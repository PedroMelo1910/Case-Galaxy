"""Módulo consolidado. Reúne:
- Hash das entradas que definem as features do teste (protocolo final §10 d; revisão ML, C1).
- Contextos da avaliação final: N1 e dataset de cada dobra, braço e sorteio (protocolo final §3 a §6).
- Fornecedor de contextos da avaliação final sobre os dados reais (protocolo final §3 a §6).
- Orquestração da avaliação final (protocolo final §3 a §8): braços principal, canais não vistos, sensibilidade e ablação.
- Driver da avaliação final: recalcula do disco todos os hashes, congela, abre o portão e executa (revisão ML, C1)."""

import argparse
import hashlib
import importlib.metadata as metadados
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.avaliacao import (
    experimentos as bi,
    experimentos as dv,
    limiar as dcl,
    metricas as bs,
    metricas as fx,
    metricas as gn,
    protocolos as pf,
    relatorio as rl,
    validacao as cr,
    validacao as db,
)
from src.avaliacao.protocolos import ConsultaAberta, PortaoFechado
from src.dados import dataset as ds, historico as hs, particoes
from src.dados.coleta import Papel
from src.extracao import texto as leg
from src.modelos import modalidades as mm, treino as fa, treino as md
from src.proveniencia import sha256_bytes as _sha


# ──── Hash das entradas que definem as features do teste (protocolo final §10 d; revisão ML, C1).
def hash_do_arquivo(arquivo: Path) -> str:
    resumo = hashlib.sha256()
    with open(arquivo, "rb") as entrada:
        for bloco in iter(lambda: entrada.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def hash_da_pasta(pasta: Path) -> str:
    """SHA-256 do conjunto (caminho relativo, hash do conteúdo) de todos os arquivos da pasta, em ordem."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise FileNotFoundError(f"pasta de entrada ausente: {pasta}")
    linhas = [f"{arquivo.relative_to(pasta).as_posix()}\t{hash_do_arquivo(arquivo)}"
              for arquivo in sorted(p for p in pasta.rglob("*") if p.is_file())]
    return _sha("\n".join(linhas).encode("utf-8"))


def hash_das_anotacoes(pasta_cache: Path, ids: Collection[str]) -> dict:
    """Hash dos arquivos `<id>.json` dos IDs pedidos; os ausentes ficam listados (e mudam o hash)."""
    pasta_cache = Path(pasta_cache)
    linhas, ausentes = [], []
    for video_id in sorted(set(ids)):
        arquivo = pasta_cache / f"{video_id}.json"
        if arquivo.is_file():
            linhas.append(f"{video_id}\t{hash_do_arquivo(arquivo)}")
        else:
            ausentes.append(video_id)
            linhas.append(f"{video_id}\tAUSENTE")
    return {"sha256": _sha("\n".join(linhas).encode("utf-8")), "n": len(linhas) - len(ausentes), "ausentes": ausentes}


def hashes_das_entradas(*, caches_de_anotacao: Sequence[tuple[Path, Collection[str]]], pasta_legenda_capa: Path,
                        cache_embeddings: Path, tabela_inscritos: Path, uv_lock: Path) -> Mapping[str, str]:
    """As entradas do protocolo, na forma esperada por `abrir_consulta_ao_teste`."""
    partes = [hash_das_anotacoes(pasta, ids)["sha256"] for pasta, ids in caches_de_anotacao]
    return {"anotacoes_teste_e_lim": _sha("\n".join(partes).encode("utf-8")),
            "legenda_capa": hash_da_pasta(pasta_legenda_capa), "embeddings": hash_da_pasta(cache_embeddings),
            "inscritos": hash_do_arquivo(tabela_inscritos), "uv_lock": hash_do_arquivo(uv_lock)}


# ──── Contextos da avaliação final: N1 e dataset de cada dobra, braço e sorteio (protocolo final §3 a §6).
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


# ──── Fornecedor de contextos da avaliação final sobre os dados reais (protocolo final §3 a §6).
@dataclass(frozen=True)
class FornecedorReal:
    pasta_base: Path
    arquivo_particao: Path
    destino: Path
    pasta_legenda_capa: Path
    arquivo_amostra: Path
    carregar_anotacoes: Callable[[Sequence[str]], Mapping[str, Mapping]]
    codificador: object
    cache_embeddings: Path
    consulta: ConsultaAberta | None = None

    def __call__(self, espec: EspecContexto) -> Mapping:
        exigir_portao(espec, consulta=self.consulta)
        return materializar_contexto(
            espec, pasta_base=self.pasta_base, arquivo_particao=self.arquivo_particao, destino=self.destino,
            pasta_legenda_capa=self.pasta_legenda_capa, arquivo_amostra=self.arquivo_amostra,
            carregar_anotacoes=self.carregar_anotacoes, codificador=self.codificador,
            cache_embeddings=self.cache_embeddings, consulta=self.consulta)


# ──── Orquestração da avaliação final (protocolo final §3 a §8): braços principal, canais não vistos, sens
BLOCOS_ANTERIORES = ("treino", "validacao")
BLOCOS_TESTE = ("teste_1", "teste_2", "teste_3")
MODELOS_SERVIDOS = ("N4", "N3", "N2")
NIVEIS_ABLACAO = ("E1", "E2", "E3", "E4")
FAMILIAS_DA_ABLACAO = {"E1": ("historico",), "E2": ("historico", "legenda"), "E3": ("historico", "legenda", "capa"),
                       "E4": ("historico", "legenda", "capa", "gemini")}
CONTRASTES_ABLACAO = (("E4", "E3"), ("E3", "E2"), ("E2", "E1"))
BASE_DO_ACASO = "media_aritmetica"
# Abaixo disto o PCA e as folhas do modelo não significam nada: a dobra exploratória é pulada e DECLARADA. O braço
# principal e a ablação não pulam: com treino menor que isto a avaliação falha de forma explícita.
MIN_LINHAS_TREINO = 20
Fornecedor = Callable[[EspecContexto], Mapping]  # {"linhas": [...], "manifesto": {"especificacao": {...}}}


@dataclass(frozen=True)
class Configuracao:
    coleta: datetime
    sorteios: Sequence[Mapping]
    blocos_avaliados: tuple[str, ...] = BLOCOS_TESTE
    blocos_anteriores: tuple[str, ...] = BLOCOS_ANTERIORES
    parametros: Mapping = field(default_factory=lambda: dict(mm.CONFIGURACAO_A))
    n_repeticoes: int = bs.N_REPETICOES
    semente: int = bs.SEMENTE
    defasagem: int = 30
    intervalo: int = 30
    defasagem_s1: int = 90
    intervalo_s2: int = 90
    inscritos: Mapping[str, int | None] | None = None  # tabela restrita (baseline 4); None = não avaliado, declarado


# --- especificações dos contextos e dos modelos -----------------------------------------------------------------------


def _permitidos(cfg: Configuracao, k: int) -> tuple[str, ...]:
    return (*cfg.blocos_anteriores, *cfg.blocos_avaliados[:k])


def espec_principal(cfg: Configuracao, k: int) -> EspecContexto:
    bloco = cfg.blocos_avaliados[k - 1]
    return EspecContexto(f"principal_{bloco}", bloco, (bloco,), _permitidos(cfg, k), cfg.defasagem, cfg.intervalo)


def espec_nao_visto(cfg: Configuracao, sorteio: int, k: int, retidos: Sequence[str]) -> EspecContexto:
    bloco = cfg.blocos_avaliados[k - 1]
    return EspecContexto(f"nao_visto_s{sorteio}_{bloco}", bloco, (bloco,), _permitidos(cfg, k), cfg.defasagem,
                            cfg.intervalo, tuple(sorted(retidos)))


def espec_sensibilidade(cfg: Configuracao, rotulo: str, k: int) -> EspecContexto:
    bloco = cfg.blocos_avaliados[k - 1]
    defasagem = cfg.defasagem_s1 if rotulo == "S1" else cfg.defasagem
    intervalo = cfg.intervalo_s2 if rotulo == "S2" else cfg.intervalo
    return EspecContexto(f"sensibilidade_{rotulo}_{bloco}", bloco, (bloco,), _permitidos(cfg, k), defasagem,
                            intervalo)


def espec_ablacao(cfg: Configuracao) -> EspecContexto:
    todos = cfg.blocos_avaliados
    return EspecContexto("ablacao_corte_unico", todos[0], tuple(todos), _permitidos(cfg, len(todos)),
                            cfg.defasagem, cfg.intervalo)


def especs_servidos(base: Mapping) -> dict[str, md.EspecificacaoAtributos]:
    return {"N4": md.EspecificacaoAtributos(**base), "N3": fa.especificacao_da_modalidade(base, "imagem"),
            "N2": fa.especificacao_da_modalidade(base, "texto")}


def especs_ablacao(base: Mapping, hashes_congelados: Mapping[str, str] | None = None
                   ) -> dict[str, md.EspecificacaoAtributos]:
    especs = {nivel: fa.especificacao_ate(base, blocos) for nivel, blocos in FAMILIAS_DA_ABLACAO.items()}
    if hashes_congelados is not None:
        divergentes = sorted(n for n, e in especs.items() if fa.hash_da_especificacao(e) != hashes_congelados.get(n))
        if divergentes:
            raise ValueError(f"especificação diferente da congelada em {divergentes}")
    return especs


# --- configuração × protocolo congelado ---------------------------------------------------------------------------------------


def _projecao_da_configuracao(cfg: Configuracao, ignorados: Collection[str], com_coleta: bool) -> dict:
    projecao = {"intervalo_treino_teste_dias": cfg.intervalo, "defasagem_historico_dias": cfg.defasagem,
                "n_repeticoes": cfg.n_repeticoes, "semente": cfg.semente,
                "n_estimators": cfg.parametros.get("n_estimators"),
                "parametros": {k: v for k, v in cfg.parametros.items() if k not in ignorados},
                "blocos_avaliados": list(cfg.blocos_avaliados), "blocos_anteriores": list(cfg.blocos_anteriores),
                "sementes_dos_sorteios": [s["semente"] for s in cfg.sorteios],
                "S1": {"defasagem_historico_dias": cfg.defasagem_s1, "intervalo_treino_teste_dias": cfg.intervalo},
                "S2": {"defasagem_historico_dias": cfg.defasagem, "intervalo_treino_teste_dias": cfg.intervalo_s2}}
    if com_coleta:
        projecao["coleta"] = cfg.coleta.isoformat()
    return projecao


def _projecao_do_congelado(d: Mapping, ignorados: Collection[str]) -> dict:
    sens = d["sensibilidade_exploratoria"]
    projecao = {"intervalo_treino_teste_dias": d["intervalo_treino_teste_dias"],
                "defasagem_historico_dias": d["defasagem_historico_dias"],
                "n_repeticoes": d["bootstrap"]["n_repeticoes"], "semente": d["bootstrap"]["semente"],
                "n_estimators": d["iteracoes"]["n_estimators"],
                "parametros": {k: v for k, v in d["modelo"]["parametros"].items() if k not in ignorados},
                "blocos_avaliados": list(d["dobras"]["blocos"]), "blocos_anteriores": list(d["blocos_anteriores"]),
                "sementes_dos_sorteios": list(d["canais_nao_vistos"]["sementes"]),
                "S1": dict(sens["S1"]), "S2": dict(sens["S2"])}
    if "coleta_dos_rotulos_utc" in d:
        projecao["coleta"] = d["coleta_dos_rotulos_utc"]
    return projecao


def exigir_configuracao_congelada(cfg: Configuracao, definicoes: Mapping) -> None:
    """A configuração que vai rodar é a do protocolo congelado; qualquer diferença recusa antes de ler dados."""
    if definicoes.get("baseline_inscritos") and cfg.inscritos is None:
        raise ValueError("o protocolo congelado inclui o baseline 4 (inscritos), mas a configuração não traz a tabela")
    ignorados = set(definicoes["modelo"].get("ignorados_na_comparacao", ()))
    congelado = _projecao_do_congelado(definicoes, ignorados)
    atual = _projecao_da_configuracao(cfg, ignorados, "coleta" in congelado)
    diferentes = sorted(k for k in congelado if congelado[k] != atual[k])
    if diferentes:
        raise ValueError(f"a configuração difere do protocolo congelado em {diferentes}")


# --- utilidades -----------------------------------------------------------------------------------------------------------


def _rodar(fornecedor: Fornecedor, espec: EspecContexto, consulta: ConsultaAberta | None) -> tuple[db.Dobra, Mapping]:
    exigir_portao(espec, consulta=consulta)
    dados = fornecedor(espec)
    return db.separar(dados["linhas"]), dados.get("manifesto", {})


def _exigir_treino(dobra: db.Dobra, espec: EspecContexto) -> None:
    """Braços que não podem ser pulados (principal, ablação): treino insuficiente é erro explícito."""
    if len(dobra.treino) < MIN_LINHAS_TREINO:
        raise ValueError(f"treino insuficiente em {espec.nome}: {len(dobra.treino)} linhas (mínimo {MIN_LINHAS_TREINO})")


def _treino_suficiente(dobra: db.Dobra) -> bool:
    return len(dobra.treino) >= MIN_LINHAS_TREINO


def _especificacao_base(manifesto: Mapping) -> Mapping:
    if "especificacao" not in manifesto:
        raise ValueError("o manifesto do dataset não traz a especificação de atributos")
    return manifesto["especificacao"]


def _concatenar(previsoes: Sequence[Mapping]) -> dict:
    return {chave: np.concatenate([np.asarray(p[chave]) for p in previsoes]) for chave in previsoes[0]}


def _celulas(linhas: Sequence[Mapping], bloco: str, sufixo: str = "") -> list[str]:
    return [f"{l['channel_id']}|{bloco}{sufixo}" for l in linhas]


def _secundarias(dobras: Sequence[tuple[Sequence[Mapping], Mapping, Sequence[float]]]) -> dict:
    """F1 macro em 3 faixas (bordas do treino de cada dobra), R² do alvo relativo e frequência de cruzamento dos
    quantis (casos em que a faixa foi corrigida para mínimo e máximo), sobre as linhas finitas das dobras dadas."""
    f1, cruzou = [], []
    for linhas, prev, bordas in dobras:
        real = np.array([np.nan if l["y_rel"] is None else l["y_rel"] for l in linhas], dtype=float)
        previsto = np.asarray(prev["relativo"], dtype=float)
        ok = np.isfinite(real) & np.isfinite(previsto)
        f1.append((real[ok], previsto[ok], bordas))
        cruzou.append(np.asarray(prev["cruzou"], dtype=bool)[ok])
    reais, previstos = np.concatenate([a for a, _, _ in f1]), np.concatenate([b for _, b, _ in f1])
    cruzou = np.concatenate(cruzou)
    return {"f1_macro_3_faixas": fx.f1_macro_empilhado(f1), "r2_alvo_relativo": fx.r2(reais, previstos),
            "acaso_f1": 1 / fx.N_FAIXAS,
            "faixa_cruzamento": {"n": int(cruzou.size), "n_cruzou": int(cruzou.sum()),
                                 "frequencia": float(cruzou.mean()) if cruzou.size else None,
                                 "regra": "q50 central; limites = minimo e maximo das tres saidas"}}


# --- braço principal ------------------------------------------------------------------------------------------------------


def braco_principal(fornecedor: Fornecedor, cfg: Configuracao, consulta: ConsultaAberta | None) -> tuple[list[dict], Mapping]:
    resultados, base = [], None
    for k, bloco in enumerate(cfg.blocos_avaliados, start=1):
        espec = espec_principal(cfg, k)
        dobra, manifesto = _rodar(fornecedor, espec, consulta)
        _exigir_treino(dobra, espec)
        base = base or _especificacao_base(manifesto)
        previsoes = {nome: db.prever_com_faixa(db.treinar_quantis(dobra.treino, spec, cfg.parametros), dobra.avaliacao)
                     for nome, spec in especs_servidos(base).items()}
        resultados.append({"k": k, "bloco": bloco, "dobra": dobra, "bordas": db.bordas_de_faixa(dobra.treino),
                           "previsoes": previsoes, "sha256_dataset": manifesto.get("sha256_dataset"),
                           "reconciliacao": manifesto.get("reconciliacao_anotacoes"),
                           "inscritos": None if cfg.inscritos is None
                           else bi.prever_dobra(dobra.treino, dobra.avaliacao, cfg.inscritos)})
    return resultados, base


def _contexto_da_declaracao(cfg: Configuracao) -> str:
    """Blocos de teste = declaração do teste final; qualquer outro bloco (ensaio) = declaração de desenvolvimento."""
    return "teste_final" if any(b.startswith("teste") for b in cfg.blocos_avaliados) else "desenvolvimento"


def avaliar_principal(resultados: Sequence[Mapping], cfg: Configuracao) -> dict:
    saida = {}
    for nome in MODELOS_SERVIDOS:
        por_dobra = {}
        for r in resultados:
            bloco, _ = rl._bloco(r["dobra"].avaliacao, r["previsoes"][nome], cfg.coleta, n_repeticoes=cfg.n_repeticoes,
                                 semente=cfg.semente, completo=True)
            por_dobra[r["bloco"]] = bloco | _secundarias([(r["dobra"].avaliacao, r["previsoes"][nome], r["bordas"])])
        linhas = [l for r in resultados for l in r["dobra"].avaliacao]
        previsao = _concatenar([r["previsoes"][nome] for r in resultados])
        celulas = [c for r in resultados for c in _celulas(r["dobra"].avaliacao, r["bloco"])]
        empilhado, _ = rl._bloco(linhas, previsao, cfg.coleta, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente,
                                 completo=True, celulas=celulas)
        saida[nome] = {"empilhado": empilhado | _secundarias(
            [(r["dobra"].avaliacao, r["previsoes"][nome], r["bordas"]) for r in resultados])
            | {"declaracao_da_faixa": dcl.declarar(_contexto_da_declaracao(cfg), modelo=nome,
                                                   total=empilhado["cobertura"],
                                                   por_setor=empilhado["cobertura_por_setor"])},
            "por_dobra": por_dobra}
    saida["treino_por_dobra"] = {r["bloco"]: {"n_treino": len(r["dobra"].treino), "n_avaliacao": len(r["dobra"].avaliacao),
                                              "ignoradas": r["dobra"].ignoradas, "bordas_f1": r["bordas"],
                                              "sha256_dataset": r["sha256_dataset"], "reconciliacao": r["reconciliacao"]} for r in resultados}
    saida["baseline_inscritos"] = _baseline_inscritos_principal(resultados, cfg, saida)
    return saida


def _baseline_inscritos_principal(resultados: Sequence[Mapping], cfg: Configuracao, saida: Mapping) -> dict:
    """Baseline 4 (TAPI, D-D1) nas linhas empilhadas do principal; ajuste só no treino de cada dobra."""
    if cfg.inscritos is None:
        return bi.bloco_nao_avaliado("tabela de inscritos não fornecida à avaliação (`inscritos` ausente na "
                                     "configuração)")
    return bi.montar_bloco(
        {r["bloco"]: r["inscritos"][0] for r in resultados}, {r["bloco"]: r["inscritos"][2] for r in resultados},
        [l for r in resultados for l in r["dobra"].avaliacao], np.concatenate([r["inscritos"][1] for r in resultados]),
        {nome: _concatenar([r["previsoes"][nome] for r in resultados])["central_log"] for nome in MODELOS_SERVIDOS},
        celulas=[c for r in resultados for c in _celulas(r["dobra"].avaliacao, r["bloco"])],
        ids_do_principal={nome: saida[nome]["empilhado"]["populacao"]["ids_sha256"] for nome in MODELOS_SERVIDOS},
        n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)


# --- canais não vistos ----------------------------------------------------------------------------------------------------


def _subconjunto_do_principal(resultados: Sequence[Mapping], nome: str, retidos: frozenset[str],
                              blocos: Collection[str]) -> tuple:
    """Linhas do principal nos canais retidos, só dos blocos que o braço de canais não vistos de fato avaliou."""
    linhas, central, celulas = [], [], []
    for r in (r for r in resultados if r["bloco"] in blocos):
        mascara = [l["channel_id"] in retidos for l in r["dobra"].avaliacao]
        linhas += [l for l, m in zip(r["dobra"].avaliacao, mascara, strict=True) if m]
        central += list(np.asarray(r["previsoes"][nome]["central_log"])[mascara])
        celulas += [c for c, m in zip(_celulas(r["dobra"].avaliacao, r["bloco"]), mascara, strict=True) if m]
    return linhas, central, celulas


def braco_nao_vistos(fornecedor: Fornecedor, cfg: Configuracao, base: Mapping, resultados_principal: Sequence[Mapping],
                     consulta: ConsultaAberta | None) -> dict:
    saida = {"sorteios": []}
    agregado = {nome: {"linhas": [], "central": [], "celulas": []} for nome in MODELOS_SERVIDOS}
    inscritos_agregado = _AcumuladoInscritos()
    for d, sorteio in enumerate(cfg.sorteios, start=1):
        retidos = frozenset(sorteio["canais_retidos"])
        acumulado = {nome: {"linhas": [], "central": [], "celulas": []} for nome in MODELOS_SERVIDOS}
        inscritos_sorteio = _AcumuladoInscritos()
        sem_avaliacao, puladas, blocos_avaliados, contextos = [], [], [], {}
        for k, bloco in enumerate(cfg.blocos_avaliados, start=1):
            dobra, manifesto = _rodar(fornecedor, espec_nao_visto(cfg, d, k, retidos), consulta)
            contextos[bloco] = {"n_treino": len(dobra.treino), "n_avaliacao_total": len(dobra.avaliacao),
                                "reconciliacao": manifesto.get("reconciliacao_anotacoes")}
            vazados = sorted({l["channel_id"] for l in dobra.treino if l["channel_id"] in retidos})
            if vazados:
                raise ValueError(f"canal retido no treino do sorteio {d}, {bloco}: {vazados}")
            avaliacao = [l for l in dobra.avaliacao if l["channel_id"] in retidos]
            if not avaliacao:
                sem_avaliacao.append(bloco)
                continue
            if not _treino_suficiente(dobra):
                puladas.append({"bloco": bloco, "n_treino": len(dobra.treino), "motivo": "treino_insuficiente"})
                continue
            blocos_avaliados.append(bloco)
            if cfg.inscritos is not None:
                resultado_inscritos = bi.prever_dobra(dobra.treino, avaliacao, cfg.inscritos)
                for destino in (inscritos_sorteio, inscritos_agregado):
                    destino.acrescentar(f"s{d}|{bloco}", *resultado_inscritos)
            for nome, spec in especs_servidos(base).items():
                central = db.prever_central(db.treinar_q50(dobra.treino, spec, cfg.parametros), avaliacao)
                for destino in (acumulado[nome], agregado[nome]):
                    destino["linhas"] += avaliacao
                    destino["central"] += list(central["central_log"])
                    destino["celulas"] += _celulas(avaliacao, bloco, f"|s{d}")
        saida["sorteios"].append({
            "sorteio": d, "semente": sorteio["semente"], "canais_retidos": sorted(retidos),
            "blocos_sem_avaliacao": sem_avaliacao, "dobras_puladas": puladas, "contextos": contextos,
            "por_modelo": {nome: _avaliar_retidos(acumulado[nome], cfg) for nome in MODELOS_SERVIDOS},
            "baseline_inscritos": _baseline_inscritos_retidos(acumulado, inscritos_sorteio, cfg),
            "principal_nos_mesmos_canais": {
                nome: _avaliar_retidos(dict(zip(("linhas", "central", "celulas"),
                                                _subconjunto_do_principal(resultados_principal, nome, retidos,
                                                                blocos_avaliados))), cfg)
                for nome in MODELOS_SERVIDOS}})
    saida["agregado_dos_sorteios"] = {nome: _avaliar_retidos(agregado[nome], cfg) for nome in MODELOS_SERVIDOS}
    saida["baseline_inscritos_agregado"] = _baseline_inscritos_retidos(agregado, inscritos_agregado, cfg)
    return saida


class _AcumuladoInscritos:
    """Ajuste e previsão do baseline 4 por dobra do braço de canais não vistos, na ordem das linhas acumuladas."""

    def __init__(self) -> None:
        self.previsto: list[np.ndarray] = []
        self.ajustes: dict[str, bi.AjusteInscritos] = {}
        self.cortadas: dict[str, int] = {}

    def acrescentar(self, rotulo: str, ajuste: bi.AjusteInscritos, previsto: np.ndarray, cortadas: int) -> None:
        self.ajustes[rotulo], self.cortadas[rotulo] = ajuste, cortadas
        self.previsto.append(previsto)


def _baseline_inscritos_retidos(acumulado: Mapping[str, Mapping], inscritos: _AcumuladoInscritos,
                                cfg: Configuracao) -> dict:
    if cfg.inscritos is None:
        return bi.bloco_nao_avaliado("tabela de inscritos não fornecida à avaliação (`inscritos` ausente na "
                                     "configuração)")
    linhas = acumulado[MODELOS_SERVIDOS[0]]["linhas"]
    if not linhas:
        return bi.bloco_nao_avaliado("nenhuma linha de canal retido avaliada")
    ids = {}
    for nome in MODELOS_SERVIDOS:
        pop, _ = dv.montar_populacao(linhas, acumulado[nome]["central"], celulas=acumulado[nome]["celulas"])
        ids[nome] = bi.hash_dos_ids(pop.ids)
    return bi.montar_bloco(
        inscritos.ajustes, inscritos.cortadas, linhas, np.concatenate(inscritos.previsto),
        {nome: acumulado[nome]["central"] for nome in MODELOS_SERVIDOS},
        celulas=acumulado[MODELOS_SERVIDOS[0]]["celulas"], ids_do_principal=ids,
        n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)


def _avaliar_retidos(acumulado: Mapping, cfg: Configuracao) -> dict:
    if not acumulado["linhas"]:
        return {"n_videos": 0, "aviso": "nenhuma linha avaliada para estes canais"}
    pop, exclusoes = dv.montar_populacao(acumulado["linhas"], acumulado["central"], celulas=acumulado["celulas"])
    unicos = len({l["video_id"] for l in acumulado["linhas"]})
    return {"exclusoes": exclusoes, "n_videos_unicos": unicos} | dv.avaliar(
        pop, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)


# --- sensibilidade (exploratória) ---------------------------------------------------------------------------------------


def braco_sensibilidade(fornecedor: Fornecedor, cfg: Configuracao, base: Mapping,
                        resultados_principal: Sequence[Mapping], consulta: ConsultaAberta | None) -> dict:
    principal = {l["video_id"]: (l, c, cel) for r in resultados_principal
                 for l, c, cel in zip(r["dobra"].avaliacao, r["previsoes"]["N4"]["central_log"],
                                      _celulas(r["dobra"].avaliacao, r["bloco"]), strict=True)}
    saida = {"natureza": "exploratória; nunca correção; não influencia seleção, limiar nem promoção"}
    spec = especs_servidos(base)["N4"]
    for rotulo in ("S1", "S2"):
        linhas, central, celulas, puladas = [], [], [], []
        for k, bloco in enumerate(cfg.blocos_avaliados, start=1):
            dobra, _ = _rodar(fornecedor, espec_sensibilidade(cfg, rotulo, k), consulta)
            if not _treino_suficiente(dobra):
                puladas.append({"bloco": bloco, "n_treino": len(dobra.treino), "motivo": "treino_insuficiente"})
                continue
            previsto = db.prever_central(db.treinar_q50(dobra.treino, spec, cfg.parametros), dobra.avaliacao)
            linhas += dobra.avaliacao
            central += list(previsto["central_log"])
            celulas += _celulas(dobra.avaliacao, bloco)
        if not linhas:
            saida[rotulo] = {"dobras_puladas": puladas, "aviso": "nenhuma dobra com treino suficiente: braço sem resultado"}
            continue
        pop, exclusoes = dv.montar_populacao(linhas, central, celulas=celulas)
        saida[rotulo] = {"exclusoes": exclusoes, "dobras_puladas": puladas,
                         "proprio": dv.avaliar(pop, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente)}
        comuns = [(l, c, cel) for l, c, cel in zip(linhas, central, celulas, strict=True) if l["video_id"] in principal]
        multi, _ = dv.montar_populacao_multi(
            [principal[l["video_id"]][0] for l, _, _ in comuns],
            {"principal": [principal[l["video_id"]][1] for l, _, _ in comuns], rotulo: [c for _, c, _ in comuns]},
            celulas=[cel for _, _, cel in comuns], com_baselines=True)
        saida[rotulo]["contra_o_principal"] = dv.avaliar(multi, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente,
                                                         contrastes=[(rotulo, "principal")])
    return saida


# --- ablação ----------------------------------------------------------------------------------------------------------------


def braco_ablacao(fornecedor: Fornecedor, cfg: Configuracao, base: Mapping, consulta: ConsultaAberta | None,
                  hashes_congelados: Mapping[str, str] | None = None) -> dict:
    espec = espec_ablacao(cfg)
    dobra, manifesto = _rodar(fornecedor, espec, consulta)
    _exigir_treino(dobra, espec)
    previsoes = {nivel: list(db.prever_central(db.treinar_q50(dobra.treino, spec, cfg.parametros),
                                               dobra.avaliacao)["central_log"])
                 for nivel, spec in especs_ablacao(base, hashes_congelados).items()}
    pop, exclusoes = dv.montar_populacao_multi(dobra.avaliacao, previsoes, com_baselines=True)
    contrastes = [*CONTRASTES_ABLACAO, *((nivel, BASE_DO_ACASO) for nivel in NIVEIS_ABLACAO)]
    avaliacao = dv.avaliar(pop, n_repeticoes=cfg.n_repeticoes, semente=cfg.semente, contrastes=contrastes)
    leituras = {}
    for chave, contraste in avaliacao["contrastes"].items():
        ic = contraste["spearman_intra_canal"]["bootstrap"]["ic95"]
        leituras[chave] = gn.leitura_do_ganho(ic) | {
            "menor_efeito_detectavel_aprox": None if ic is None else (ic[1] - ic[0]) / 2}
    elegiveis = avaliacao["spearman_intra_canal"]["E4"]
    return {"n_treino": len(dobra.treino), "n_avaliacao": len(dobra.avaliacao), "exclusoes": exclusoes,
            "mesmas_linhas_para_todos_os_niveis": True, "niveis": list(NIVEIS_ABLACAO),
            "sha256_dataset": manifesto.get("sha256_dataset"),
            "canais_elegiveis_spearman": elegiveis["elegiveis"],
            "canais_excluidos": {"por_tamanho": elegiveis["excluidos_por_tamanho"],
                                 "sem_variacao_no_real": elegiveis["excluidos_sem_variacao_no_real"]},
            "avaliacao": avaliacao, "leitura_do_ganho_spearman_intra_canal": leituras,
            "nao_avalia": "a Camada 1 local do TAPI §6.3 (desvio declarado, não resolvido)"}


# --- execução -------------------------------------------------------------------------------------------------------------------


def gravador_de_bracos(pasta: Path) -> Callable[[str, Mapping], None]:
    """`salvar` que grava cada braço em `<pasta>/<nome>.json`, sem sobrescrever (modo exclusivo)."""
    pasta = Path(pasta)

    def salvar(nome: str, dado: Mapping) -> None:
        pasta.mkdir(parents=True, exist_ok=True)
        with open(pasta / f"{nome}.json", "x", encoding="utf-8") as saida:
            json.dump(rl.sanear(dado), saida, ensure_ascii=False, indent=1)

    return salvar


def _braco(nome: str, funcao: Callable[[], object], salvar: Callable[[str, Mapping], None] | None,
           gravavel: Callable[[object], Mapping] = lambda r: r) -> object:
    """Roda um braço, grava o resultado ao terminar e, se falhar, grava o erro antes de propagar."""
    try:
        resultado = funcao()
    except Exception as erro:
        if salvar:
            salvar(f"{nome}.erro", {"braco": nome, "tipo": type(erro).__name__, "mensagem": str(erro)})
        raise
    if salvar:
        salvar(nome, gravavel(resultado))
    return resultado


def executar(fornecedor: Fornecedor, cfg: Configuracao, *, consulta: ConsultaAberta | None,
             hashes_ablacao: Mapping[str, str] | None = None,
             salvar: Callable[[str, Mapping], None] | None = None) -> dict:
    """Roda os quatro braços e devolve o relatório (JSON-serializável com `relatorio.sanear`).

    Com a consulta ao teste aberta, a configuração tem de ser a congelada e os hashes da ablação são obrigatórios.
    `salvar(nome, dado)` recebe cada braço assim que ele termina (e o erro, se falhar): a consulta é única e nada que
    já foi calculado pode se perder por falha de um braço posterior."""
    if consulta is not None:
        if not isinstance(consulta, ConsultaAberta):
            raise PortaoFechado("a consulta ao teste só vale como a prova devolvida por abrir_consulta_ao_teste")
        exigir_configuracao_congelada(cfg, consulta.definicoes)
        if hashes_ablacao is None:
            raise ValueError("com a consulta ao teste aberta, os hashes congelados da ablação (E1 a E4) são "
                             "obrigatórios")
    resultados, base, principal = _braco(
        "principal", lambda: _principal(fornecedor, cfg, consulta), salvar, gravavel=lambda r: r[2])
    erros: list[Exception] = []

    def independente(nome: str, funcao: Callable[[], Mapping]) -> Mapping:
        """Sensibilidade, canais não vistos e ablação só dependem do principal: a falha de um não derruba os outros."""
        try:
            return _braco(nome, funcao, salvar)
        except Exception as erro:
            erros.append(erro)
            return {"erro": {"tipo": type(erro).__name__, "mensagem": str(erro)}}

    relatorio = {
        "configuracao": {"blocos_avaliados": list(cfg.blocos_avaliados), "blocos_anteriores": list(cfg.blocos_anteriores),
                         "parametros": dict(cfg.parametros), "n_repeticoes": cfg.n_repeticoes, "semente": cfg.semente,
                         "coleta": cfg.coleta.isoformat(), "baseline_inscritos_fornecido": cfg.inscritos is not None},
        "principal": principal,
        "canais_nao_vistos": independente("canais_nao_vistos", lambda: braco_nao_vistos(
            fornecedor, cfg, base, resultados, consulta)),
        "sensibilidade": independente("sensibilidade", lambda: braco_sensibilidade(
            fornecedor, cfg, base, resultados, consulta)),
        "ablacao": independente("ablacao", lambda: braco_ablacao(fornecedor, cfg, base, consulta, hashes_ablacao)),
    }
    if erros:
        raise erros[0]  # todos os braços já foram gravados (ou tiveram o erro gravado); o primeiro erro sobe
    return relatorio


def _principal(fornecedor: Fornecedor, cfg: Configuracao, consulta: ConsultaAberta | None) -> tuple:
    resultados, base = braco_principal(fornecedor, cfg, consulta)
    return resultados, base, avaliar_principal(resultados, cfg)


# ──── Driver da avaliação final: recalcula do disco todos os hashes, congela, abre o portão e executa (rev
BIBLIOTECAS = ("lightgbm", "scikit-learn", "numpy", "scipy", "pyarrow")


@dataclass(frozen=True)
class Caminhos:
    produto: Path  # raiz do produto: os módulos hasheados e o uv.lock são relativos a ela
    pasta_amostra: Path  # amostra=v2 (amostra.json e cache/)
    pasta_lim: Path  # lim=v1 (amostra.json e cache/)
    arquivo_particao: Path
    pasta_base: Path  # curadoria=v1
    pasta_legenda_capa: Path
    cache_embeddings: Path
    pasta_dataset: Path  # dataset final de desenvolvimento (dataset.parquet e manifesto.json)
    pasta_protocolo: Path  # protocolo_final.json, revisão, marcadores e resultados
    registro_global: Path  # uma linha por consulta consumida, por partição
    destino: Path  # contextos materializados


def _lista(pasta: Path) -> list[dict]:
    return json.loads((Path(pasta) / "amostra.json").read_text(encoding="utf-8"))["lista"]


def ids_de_teste(c: Caminhos) -> list[str]:
    return sorted(x["video_id"] for x in _lista(c.pasta_amostra) if str(x["bloco"]).startswith(PREFIXO_TESTE))


def ids_da_lim(c: Caminhos) -> list[str]:
    return sorted(x["video_id"] for x in _lista(c.pasta_lim))


def _canais(c: Caminhos) -> dict[str, tuple[str, str]]:
    return {x["channel_id"]: (x["setor"], x["tipo_canal"]) for x in _lista(c.pasta_amostra)}


def _manifesto_do_dataset(c: Caminhos) -> dict:
    manifesto = json.loads((Path(c.pasta_dataset) / ds.ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))
    real = hashlib.sha256((Path(c.pasta_dataset) / ds.ARQUIVO_DATASET).read_bytes()).hexdigest()
    if real != manifesto.get("sha256_dataset"):
        raise pf.PortaoFechado("o dataset em disco não confere com o hash do seu manifesto")
    return manifesto


def hashes_atuais(c: Caminhos) -> dict:
    """Tudo que a abertura do portão reconfere, recalculado dos arquivos; nenhum rótulo é lido."""
    manifesto = _manifesto_do_dataset(c)
    especificacoes = {nivel: fa.hash_da_especificacao(espec)
                      for nivel, espec in especs_ablacao(manifesto["especificacao"]).items()}
    entradas = hashes_das_entradas(
        caches_de_anotacao=[(Path(c.pasta_amostra) / "cache", ids_de_teste(c)),
                            (Path(c.pasta_lim) / "cache", ids_da_lim(c))],
        pasta_legenda_capa=c.pasta_legenda_capa, cache_embeddings=c.cache_embeddings,
        tabela_inscritos=Path(c.pasta_base) / bi.TABELA_CANAIS_RESTRITA, uv_lock=Path(c.produto) / "uv.lock")
    return {"hash_dataset_atual": manifesto["sha256_dataset"],
            "hash_amostra_atual": hash_do_arquivo(Path(c.pasta_amostra) / "amostra.json"),
            "hash_particao_atual": hash_do_arquivo(c.arquivo_particao),
            "hash_lim_atual": hash_do_arquivo(Path(c.pasta_lim) / "amostra.json"),
            "bibliotecas_atuais": {b: metadados.version(b) for b in BIBLIOTECAS},
            "hashes_especificacoes_atuais": especificacoes, "hashes_entradas_atuais": dict(entradas)}


def configuracao_do_projeto(*, coleta: datetime, arvore: Caminhos) -> Configuracao:
    """A configuração do protocolo: defaults do projeto, os três sorteios de canais retidos e a tabela de inscritos."""
    return Configuracao(coleta=coleta, sorteios=cr.tres_sorteios(_canais(arvore)),
                           inscritos=bi.carregar_inscritos(arvore.pasta_base))


def anotacoes_ausentes(c: Caminhos) -> dict[str, list[str]]:
    return {"teste": hash_das_anotacoes(Path(c.pasta_amostra) / "cache", ids_de_teste(c))["ausentes"],
            "lim": hash_das_anotacoes(Path(c.pasta_lim) / "cache", ids_da_lim(c))["ausentes"]}


def extras_de_arquivos(arquivos: Mapping[str, Path]) -> dict[str, dict]:
    """Artefatos que sustentam decisões (pré-registro, veredito, limiar da LIM): nome do arquivo e SHA-256 dos bytes."""
    return {nome: {"arquivo": Path(caminho).name, "sha256": hash_do_arquivo(Path(caminho))}
            for nome, caminho in arquivos.items()}


def congelar_o_protocolo(c: Caminhos, cfg: Configuracao, *, extras: Mapping | None = None) -> Path:
    """Congela só com a população completa: anotação de teste ou LIM ausente mantém o teste fechado."""
    ausentes = anotacoes_ausentes(c)
    if ausentes["teste"] or ausentes["lim"]:
        raise pf.PortaoFechado(f"anotações ausentes: {len(ausentes['teste'])} de teste e {len(ausentes['lim'])} da "
                               "LIM; o protocolo só é congelado com a população completa (ou com a decisão de Pedro "
                               "sobre as falhas finais)")
    h = hashes_atuais(c)
    protocolo = pf.montar_protocolo(
        raiz=c.produto, modulos=list(pf.MODULOS_EXIGIDOS), hash_dataset=h["hash_dataset_atual"],
        hash_amostra=h["hash_amostra_atual"], hash_particao=h["hash_particao_atual"], hash_lim=h["hash_lim_atual"],
        especificacoes=h["hashes_especificacoes_atuais"], entradas=h["hashes_entradas_atuais"],
        sorteios=cfg.sorteios, bibliotecas=h["bibliotecas_atuais"], coleta_utc=cfg.coleta.isoformat(),
        extras=extras)
    return pf.congelar(protocolo, c.pasta_protocolo)


def abrir(c: Caminhos, cfg: Configuracao) -> pf.ConsultaAberta:
    """Fase 1: recalcula todos os hashes, confere a configuração e cria o marcador. Nenhum rótulo é lido."""
    return pf.abrir_consulta_ao_teste(
        c.pasta_protocolo, raiz=c.produto, registro_global=c.registro_global,
        validar_configuracao=lambda definicoes: exigir_configuracao_congelada(cfg, definicoes), **hashes_atuais(c))


ARQUIVO_COPIA_DO_CACHE = "proveniencia_cache_embeddings.copia.json"
PASTA_CACHE_DE_TRABALHO = "cache_embeddings_trabalho"


def _arquivos(pasta: Path) -> dict[str, str]:
    """Caminho relativo -> SHA-256 de cada arquivo da pasta."""
    return {a.relative_to(pasta).as_posix(): hash_do_arquivo(a) for a in sorted(Path(pasta).rglob("*")) if a.is_file()}


def cache_de_trabalho(c: Caminhos) -> Path:
    """Cópia gravável do cache de embeddings: a execução final escreve nela e NUNCA no cache de origem, cujo hash
    pertence ao protocolo congelado. Na retomada técnica a cópia existente é reaproveitada (nada é recopiado)."""
    origem, trabalho = Path(c.cache_embeddings), Path(c.destino) / PASTA_CACHE_DE_TRABALHO
    if trabalho.exists():
        return trabalho
    Path(c.destino).mkdir(parents=True, exist_ok=True)
    hash_origem = hash_da_pasta(origem)
    parcial = Path(tempfile.mkdtemp(prefix=".cache_embeddings.", suffix=".parcial", dir=c.destino))
    try:
        shutil.copytree(origem, parcial, dirs_exist_ok=True)
        parcial.rename(trabalho)
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    with open(Path(c.destino) / ARQUIVO_COPIA_DO_CACHE, "x", encoding="utf-8") as saida:
        json.dump({"origem": str(origem), "trabalho": str(trabalho), "sha256_origem_na_copia": hash_origem,
                   "copiado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}, saida, indent=1)
    return trabalho


def proveniencia_do_cache(c: Caminhos) -> dict:
    """Separa o que é entrada congelada do que a execução gerou, e diz se a origem continua intacta."""
    origem, trabalho = Path(c.cache_embeddings), Path(c.destino) / PASTA_CACHE_DE_TRABALHO
    copia = json.loads((Path(c.destino) / ARQUIVO_COPIA_DO_CACHE).read_text(encoding="utf-8"))
    congelados, no_trabalho = _arquivos(origem), _arquivos(trabalho)
    agora = hash_da_pasta(origem)
    return {"origem": str(origem), "trabalho": str(trabalho), "sha256_origem_na_copia": copia["sha256_origem_na_copia"],
            "sha256_origem_agora": agora, "origem_intacta": agora == copia["sha256_origem_na_copia"],
            "arquivos_congelados": sorted(congelados),
            "arquivos_gerados_na_execucao": sorted(set(no_trabalho) - set(congelados)),
            "arquivos_congelados_alterados_na_copia": sorted(a for a in set(no_trabalho) & set(congelados)
                                                             if no_trabalho[a] != congelados[a])}


def exigir_origem_do_cache_intacta(c: Caminhos) -> None:
    if not proveniencia_do_cache(c)["origem_intacta"]:
        raise pf.PortaoFechado("o cache de embeddings de origem mudou depois da cópia de trabalho: a entrada "
                               "congelada foi alterada")


def _gravar_proveniencia(c: Caminhos, *, retomada: bool) -> Path:
    n = len(list(Path(c.pasta_protocolo).glob("retomada_tecnica_*.json")))
    nome = f"proveniencia_cache_embeddings_{'retomada_%02d' % n if retomada else 'original'}.json"
    arquivo = Path(c.destino) / nome
    with open(arquivo, "x", encoding="utf-8") as saida:
        json.dump(proveniencia_do_cache(c) | {"registrado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
                  saida, ensure_ascii=False, indent=1)
    return arquivo


def abrir_retomada(c: Caminhos, cfg: Configuracao, *, modulos_reparados: Sequence[str], motivo: str,
                   autorizacao: str, declaracao: str) -> pf.ConsultaAberta:
    """Retomada técnica: só com método, alvo, seleção e população inalterados; o marcador original fica como está."""
    return pf.abrir_retomada_tecnica(
        c.pasta_protocolo, raiz=c.produto, registro_global=c.registro_global, modulos_reparados=modulos_reparados,
        motivo=motivo, autorizacao=autorizacao, declaracao_metodo_inalterado=declaracao,
        validar_configuracao=lambda definicoes: exigir_configuracao_congelada(cfg, definicoes), **hashes_atuais(c))


def pasta_de_resultados(c: Caminhos, *, retomada: bool = False) -> Path:
    """`resultados` da tentativa original; `resultados_retomada_NN` da retomada mais recente (nunca sobrescreve)."""
    if not retomada:
        return Path(c.pasta_protocolo) / "resultados"
    n = len(list(Path(c.pasta_protocolo).glob("retomada_tecnica_*.json")))
    if n == 0:
        raise ValueError("não há registro de retomada técnica neste protocolo")
    return Path(c.pasta_protocolo) / f"resultados_retomada_{n:02d}"


def executar_final(c: Caminhos, cfg: Configuracao, *, codificador, carregar_anotacoes,
                   retomada: Mapping | None = None) -> dict:
    """Consulta única: abre o portão e roda os quatro braços, gravando cada um ao terminar.

    Com `retomada` (módulos reparados, motivo, autorização, declaração), segue o caminho da retomada técnica: os
    resultados da tentativa original ficam intocados e os novos vão para `resultados_retomada_NN`."""
    consulta = abrir_retomada(c, cfg, **retomada) if retomada else abrir(c, cfg)
    fornecedor = FornecedorReal(
        pasta_base=c.pasta_base, arquivo_particao=c.arquivo_particao, destino=c.destino,
        pasta_legenda_capa=c.pasta_legenda_capa, arquivo_amostra=Path(c.pasta_amostra) / "amostra.json",
        carregar_anotacoes=carregar_anotacoes, codificador=codificador, cache_embeddings=cache_de_trabalho(c),
        consulta=consulta)
    hashes_ablacao = hashes_atuais(c)["hashes_especificacoes_atuais"]
    try:
        relatorio = executar(fornecedor, cfg, consulta=consulta, hashes_ablacao=hashes_ablacao,
                                salvar=gravador_de_bracos(pasta_de_resultados(c, retomada=bool(retomada))))
    finally:
        _gravar_proveniencia(c, retomada=bool(retomada))  # também na falha: o que foi gerado fica distinguido
    exigir_origem_do_cache_intacta(c)
    return relatorio


# --- linha de comando -------------------------------------------------------------------------------------------------


def _caminhos_padrao(produto: Path, dataset: Path, protocolo: Path) -> Caminhos:
    snap = "snapshot=20261014T021437Z"
    anot = produto / "data/annotations/gemini" / snap
    feat = produto / "data/features" / snap
    return Caminhos(
        produto=produto, pasta_amostra=anot / "amostra=v2", pasta_lim=anot / "lim=v1",
        arquivo_particao=produto / "data/processed" / snap / "particao=v1/particao.json",
        pasta_base=produto / "data/processed" / snap / "curadoria=v1", pasta_legenda_capa=feat / "legenda_capa=v1",
        cache_embeddings=feat / "cache_embeddings_transcricao", pasta_dataset=dataset, pasta_protocolo=protocolo,
        registro_global=protocolo.parent / "consultas_ao_teste.jsonl", destino=protocolo / "contextos")


def _coleta(c: Caminhos) -> datetime:
    manifesto = _manifesto_do_dataset(c)
    return datetime.strptime(manifesto["entradas"]["base"]["snapshot_id"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def main(argv: Sequence[str] | None = None) -> int:
    from src.config import RAIZ_PROJETO

    parser = argparse.ArgumentParser(prog="python -m src.avaliacao.execucao_final", description=__doc__)
    parser.add_argument("etapa", choices=("hashes", "congelar", "revisao", "executar", "retomar"))
    parser.add_argument("--dataset", type=Path, required=True, help="pasta do dataset final de desenvolvimento")
    parser.add_argument("--protocolo", type=Path, required=True, help="pasta do protocolo (nova, para congelar)")
    parser.add_argument("--decisao", choices=pf.DECISOES_VALIDAS)
    parser.add_argument("--resumo", default="")
    parser.add_argument("--aceite", default=None, help="aceite por escrito de Pedro (obrigatório em WARN)")
    parser.add_argument("--consulta-unica-confirmada", action="store_true",
                        help="exigida em `executar`: lê os rótulos de teste uma única vez por partição")
    parser.add_argument("--motivo", help="retomar: o que falhou e o que foi reparado")
    parser.add_argument("--autorizacao", help="retomar: a autorização de Pedro para a retomada técnica")
    parser.add_argument("--declaracao", help="retomar: declaração de que método, alvo, seleção e população não mudaram")
    parser.add_argument("--modulos-reparados", nargs="+", help="retomar: caminhos dos módulos reparados")
    parser.add_argument("--registrar", action="append", default=[], metavar="NOME=ARQUIVO",
                        help="congelar: artefato de decisão a registrar por hash (repetível)")
    parser.add_argument("--retomada-tecnica-confirmada", action="store_true",
                        help="exigida em `retomar`: relê rótulos de teste, sem nova consulta nem nova seleção")
    args = parser.parse_args(argv)
    c = _caminhos_padrao(RAIZ_PROJETO, args.dataset, args.protocolo)
    try:
        if args.etapa == "hashes":
            print(json.dumps(hashes_atuais(c), ensure_ascii=False, indent=1))
        elif args.etapa == "congelar":
            extras = extras_de_arquivos(dict(item.split("=", 1) for item in args.registrar))
            print(congelar_o_protocolo(c, configuracao_do_projeto(coleta=_coleta(c), arvore=c), extras=extras))
        elif args.etapa == "revisao":
            print(pf.registrar_revisao_mle(c.pasta_protocolo, decisao=args.decisao, resumo=args.resumo,
                                           aceite_por_escrito=args.aceite))
        elif args.etapa == "retomar":
            faltam = [n for n in ("motivo", "autorizacao", "declaracao", "modulos_reparados") if not getattr(args, n)]
            if faltam or not args.retomada_tecnica_confirmada:
                print(f"`retomar` exige {faltam or 'nenhum argumento a mais'} e --retomada-tecnica-confirmada: a "
                      "retomada técnica relê rótulos de teste sem mudar método, alvo, seleção nem população.",
                      file=sys.stderr)
                return 2
            cfg = configuracao_do_projeto(coleta=_coleta(c), arvore=c)
            executar_final(c, cfg, codificador=leg.CodificadorTexto(),
                           carregar_anotacoes=ds._anotacoes_do_cache(c.pasta_amostra),
                           retomada={"modulos_reparados": args.modulos_reparados, "motivo": args.motivo,
                                     "autorizacao": args.autorizacao, "declaracao": args.declaracao})
        else:
            if not args.consulta_unica_confirmada:
                print("`executar` lê rótulos de teste uma única vez: repita com --consulta-unica-confirmada.",
                      file=sys.stderr)
                return 2
            cfg = configuracao_do_projeto(coleta=_coleta(c), arvore=c)
            executar_final(c, cfg, codificador=leg.CodificadorTexto(),
                           carregar_anotacoes=ds._anotacoes_do_cache(c.pasta_amostra))
    except (pf.PortaoFechado, ValueError, FileNotFoundError, FileExistsError) as erro:
        print(f"Recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
