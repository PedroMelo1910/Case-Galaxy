"""Módulo consolidado. Reúne:
- Dados do refit final: contexto "final", histórico (N1) e datasets de vídeo, imagem e texto (D-D7, ARQUITETURA §6.5).
- Cobertura do procedimento de avaliação final, lida do resultado gravado e gravada nos pacotes servidos.
- Ajuste e empacotamento dos modelos servidos do refit final (D-D7, ARQUITETURA §6.5).
- Pacote do preditor audiovisual local: ajuste final, gravação imutável e verificação de recarga.
- Limiar de largura de baixa confiança do preditor audiovisual local, ligado à identidade do artefato exato.
- Verificação de um pacote final em processo novo: recarga bit a bit, identidade e aditividade do SHAP."""

import argparse
import io
import json
import math
import platform
import shutil
import sys
import tempfile
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, UTC
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from src.avaliacao import experimentos as af, final as cf, limiar as dfx, limiar as dl
from src.config import RAIZ_PROJETO
from src.dados import dataset as dm, dataset as ds, historico as hs, particoes
from src.extracao import modelos as especificacao_extrator
from src.extracao.rubrica import gravar_json_atomico
from src.modelos import modalidades as mm, modalidades as rat, treino as fa, treino as fp, treino as md
from src.proveniencia import (
    iso_utc as _iso,
    sha256_arquivo,
    sha256_arquivo as _sha256_arquivo,
    sha256_bytes,
    sha256_ids as _sha256_ids,
)


# ──── Dados do refit final: contexto "final", histórico (N1) e datasets de vídeo, imagem e texto (D-D7, AR
SNAPSHOT = "snapshot=20261014T021437Z"
NOME_CONTEXTO = "final"
PERMITIDOS = particoes.BLOCOS
PASTA_N1 = "desenvolvimento_final_historico=v1"
PASTA_CACHE_EMBEDDINGS = "desenvolvimento_final_cache_embeddings=v1"
NOMES_DE_PASTA = {"video": "desenvolvimento_final=v1", "imagem": "desenvolvimento_final_imagem=v1",
                  "texto": "desenvolvimento_final_texto=v1"}
ARQUIVO_REGISTRO = "REGISTRO_REFIT.json"
FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"


class ErroRefit(ValueError):
    """População, contexto ou dado incompatível com o refit final."""


def caminho_relativo(caminho: Path, raiz: Path = RAIZ_PROJETO) -> str:
    """Caminho em relação à raiz do projeto (portátil entre máquinas); fora da raiz, o caminho como está."""
    caminho = Path(caminho)
    try:
        return caminho.resolve().relative_to(Path(raiz).resolve()).as_posix()
    except ValueError:
        return caminho.as_posix()


# --- contexto e N1 (puros) ---------------------------------------------------------------------------------------


def especificacao_do_contexto() -> cf.EspecContexto:
    """Contexto sem bloco avaliado, sem canal excluído e sem intervalo; só descreve, nunca abre o portão."""
    return cf.EspecContexto(nome=NOME_CONTEXTO, bloco_corte=PERMITIDOS[-1], avaliados=(), permitidos=PERMITIDOS,
                            defasagem_dias=hs.DEFASAGEM_PADRAO, intervalo_dias=0)


def contexto_final(posts: Sequence[hs.Post], ids: Collection[str]) -> hs.ContextoTreino:
    """`treino_ate_utc` = a última publicação entre os alvos da amostra; sem canais excluídos."""
    publicacoes = [p.published_at for p in posts if p.video_id in ids]
    if not publicacoes:
        raise ErroRefit("nenhum vídeo da amostra na base: não há corte de treino")
    return hs.ContextoTreino(NOME_CONTEXTO, max(publicacoes), frozenset())


def n1_do_refit(posts: Sequence[hs.Post], canais: Mapping[str, hs.Canal], atribuicao: Mapping[str, str],
                ids: Sequence[str], *, ids_lim: Collection[str] = ()) -> tuple[list[dict], hs.ContextoTreino]:
    """Linhas do N1 de todos os alvos da amostra no contexto final; recusa LIM na amostra e linha que não seja treino."""
    ids_set = set(ids)
    if len(ids_set) != len(ids):
        raise ErroRefit("ids repetidos na amostra")
    interseccao = sorted(ids_set & set(ids_lim))
    if interseccao:
        raise ErroRefit(f"a LIM deve ficar fora do ajuste; {len(interseccao)} vídeos aparecem também na amostra "
                        f"(ex.: {interseccao[:3]})")
    sem_bloco = sorted(v for v in ids_set if atribuicao.get(v) not in PERMITIDOS)
    if sem_bloco:
        raise ErroRefit(f"{len(sem_bloco)} vídeos da amostra sem bloco da partição (ex.: {sem_bloco[:3]})")
    ctx = contexto_final(posts, ids_set)
    linhas = cf.n1_do_contexto(posts, canais, atribuicao, ctx, avaliados=(), permitidos=PERMITIDOS,
                               defasagem_dias=hs.DEFASAGEM_PADRAO, ids_amostra=ids_set)
    if len(linhas) != len(ids_set):
        raise ErroRefit(f"o N1 final tem {len(linhas)} linhas para {len(ids_set)} vídeos da amostra")
    fora_do_treino = sorted({l["uso_no_contexto"] for l in linhas} - {"treino"})
    if fora_do_treino:
        raise ErroRefit(f"no contexto final toda linha deve ser treino; encontrado {fora_do_treino}")
    return linhas, ctx


def contar_exclusoes(linhas: Sequence[Mapping]) -> dict:
    """Linhas que não entram no ajuste por falta de referência ou de alvo (contadas, nunca preenchidas)."""
    sem_ref = sum(l.get("ref_log") is None for l in linhas)
    sem_alvo = sum(l.get("y_rel") is None for l in linhas)
    sem_algum = sum(l.get("ref_log") is None or l.get("y_rel") is None for l in linhas)
    return {"n_linhas": len(linhas), "sem_ref_log": sem_ref, "sem_y_rel": sem_alvo, "excluidas_do_ajuste": sem_algum,
            "aptas_ao_ajuste": len(linhas) - sem_algum}


# --- cache de embeddings -----------------------------------------------------------------------------------------


def preparar_cache(origem: Path, destino: Path) -> dict:
    """Copia as entradas de embeddings ainda ausentes em `destino`; a origem nunca é alterada."""
    origem, destino = Path(origem), Path(destino)
    if not origem.is_dir():
        raise ErroRefit(f"cache de embeddings de origem inexistente: {origem}")
    destino.mkdir(parents=True, exist_ok=True)
    copiadas = 0
    for arquivo in sorted(origem.glob("*.json")):
        alvo = destino / arquivo.name
        if not alvo.exists():
            shutil.copy2(arquivo, alvo)
            copiadas += 1
    return {"origem": caminho_relativo(origem), "destino": caminho_relativo(destino), "copiadas": copiadas,
            "entradas_no_destino": len(list(destino.glob("*.json")))}


# --- orquestração ------------------------------------------------------------------------------------------------


def _ler_ids(arquivo: Path) -> list[str]:
    return [x["video_id"] for x in json.loads(Path(arquivo).read_text(encoding="utf-8"))["lista"]]


def materializar(*, pasta_base: Path, arquivo_particao: Path, pasta_legenda_capa: Path, pasta_amostra: Path,
                 arquivo_lim: Path, cache_origem: Path, destino: Path, codificador, carregar_anotacoes=None) -> dict:
    """N1 final → datasets de vídeo, imagem e texto → registro do refit. Idempotente."""
    pasta_base, destino = Path(pasta_base), Path(destino)
    arquivo_amostra = Path(pasta_amostra) / "amostra.json"
    ids = _ler_ids(arquivo_amostra)
    ids_lim = _ler_ids(arquivo_lim)
    particao = particoes.ler_particao(arquivo_particao, pasta_base=pasta_base)
    posts, canais = hs.carregar_base(pasta_base)
    linhas, ctx = n1_do_refit(posts, canais, particao["atribuicao"], ids, ids_lim=ids_lim)
    exclusoes = contar_exclusoes(linhas)
    pasta_n1 = cf._escrever_n1(linhas, destino / PASTA_N1, pasta_base=pasta_base, arquivo_particao=arquivo_particao,
                               particao=particao, ctx=ctx, espec=especificacao_do_contexto(), posts=posts,
                               canais=canais)
    cache = preparar_cache(cache_origem, destino / PASTA_CACHE_EMBEDDINGS)
    anotacoes = (carregar_anotacoes or ds._anotacoes_do_cache(pasta_amostra))
    if len(anotacoes(ids)) != len(ids):
        raise ErroRefit("há vídeos da amostra sem anotação válida; o refit final exige as anotações completas")
    comuns = {"tabela_n1": pasta_n1 / "n1.parquet", "manifesto_n1": pasta_n1 / "manifesto.json",
              "pasta_legenda_capa": Path(pasta_legenda_capa), "arquivo_amostra": arquivo_amostra,
              "destino_raiz": destino}
    pastas = {"video": ds.montar(**comuns, carregar_anotacoes=anotacoes, codificador=codificador,
                                 cache_embeddings=destino / PASTA_CACHE_EMBEDDINGS,
                                 nome_da_pasta=NOMES_DE_PASTA["video"])}
    for modalidade in ("imagem", "texto"):
        pastas[modalidade] = dm.montar_dataset_modalidades(modalidade=modalidade, **comuns, nome_da_pasta=NOMES_DE_PASTA[modalidade])
    registro = _registro(pastas, pasta_n1, exclusoes, ctx, ids, ids_lim, cache,
                         cache_final=destino / PASTA_CACHE_EMBEDDINGS)
    gravar_registro(destino / NOMES_DE_PASTA["video"] / ARQUIVO_REGISTRO, registro)
    return registro


def _resumo_dataset(pasta: Path) -> dict:
    manifesto = json.loads((pasta / ds.ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))
    return {"pasta": caminho_relativo(pasta), "sha12": pasta.name, "sha256_dataset": manifesto["sha256_dataset"],
            "sha256_manifesto": _sha256_arquivo(pasta / ds.ARQUIVO_MANIFESTO), "n_linhas": manifesto["n_linhas"],
            "uso_no_contexto": manifesto["populacao"]["uso_no_contexto"], "n_canais": manifesto["populacao"]["n_canais"],
            "reconciliacao": manifesto.get("reconciliacao_anotacoes", manifesto.get("reconciliacao")),
            "retencao": manifesto["retencao"]}


def _registro(pastas: Mapping[str, Path], pasta_n1: Path, exclusoes: Mapping, ctx: hs.ContextoTreino,
              ids: Sequence[str], ids_lim: Sequence[str], cache: Mapping, *, cache_final: Path) -> dict:
    return {
        "tipo": "registro_do_refit_final", "gerado_em_utc": _iso(datetime.now(UTC)),
        "contexto": {"nome": ctx.nome, "treino_ate_utc": _iso(ctx.treino_ate_utc), "canais_excluidos": [],
                     "defasagem_dias": hs.DEFASAGEM_PADRAO,
                     "regra": "treino_ate_utc = última publicação entre os alvos da amostra; toda linha é treino"},
        "populacao": {"amostra_n": len(ids), "amostra_ids_sha256": _sha256_ids(ids), "lim_n": len(ids_lim),
                      "lim_ids_sha256": _sha256_ids(sorted(ids_lim)), "lim_fora_do_ajuste": True},
        "exclusoes_por_referencia_ou_alvo": dict(exclusoes),
        "historico": {"pasta": caminho_relativo(pasta_n1), "sha256_n1": _sha256_arquivo(pasta_n1 / "n1.parquet"),
                      "sha256_manifesto": _sha256_arquivo(pasta_n1 / "manifesto.json")},
        "datasets": {m: _resumo_dataset(p) for m, p in pastas.items()},
        "cache_de_embeddings": dict(cache) | {"pasta_final": caminho_relativo(cache_final)},
        "consulta_ao_teste": "nenhuma: o refit lê rótulos só como alvo do ajuste; o portão não é aberto nem tocado",
    }


def gravar_registro(arquivo: Path, registro: Mapping) -> Path:
    """Grava o registro uma vez; um registro idêntico em conteúdo reaproveita o existente (sem `gerado_em_utc`)."""
    arquivo = Path(arquivo)
    if arquivo.exists():
        anterior = json.loads(arquivo.read_text(encoding="utf-8"))
        anterior.pop("gerado_em_utc", None)
        atual = {k: v for k, v in registro.items() if k != "gerado_em_utc"}
        atual = json.loads(json.dumps(atual))
        anterior_cmp = {k: v for k, v in anterior.items() if k != "cache_de_embeddings"}
        atual_cmp = {k: v for k, v in atual.items() if k != "cache_de_embeddings"}
        if anterior_cmp != atual_cmp:
            raise ErroRefit(f"o registro do refit já existe com outro conteúdo e não será sobrescrito: {arquivo}")
        return arquivo
    gravar_json_atomico(arquivo, dict(registro))
    return arquivo


def main(argv: Sequence[str] | None = None) -> int:
    from src.extracao import texto as leg

    features = RAIZ_PROJETO / "data/features" / SNAPSHOT
    parser = argparse.ArgumentParser(prog="python -m src.modelos.refit refit_dados", description=__doc__)
    parser.add_argument("--base", type=Path, default=RAIZ_PROJETO / "data/processed" / SNAPSHOT / "curadoria=v1")
    parser.add_argument("--particao", type=Path,
                        default=RAIZ_PROJETO / "data/processed" / SNAPSHOT / "particao=v1/particao.json")
    parser.add_argument("--legenda-capa", type=Path, default=features / "legenda_capa=v1")
    parser.add_argument("--amostra", type=Path, default=RAIZ_PROJETO / "data/annotations/gemini" / SNAPSHOT / "amostra=v2")
    parser.add_argument("--lim", type=Path,
                        default=RAIZ_PROJETO / "data/annotations/gemini" / SNAPSHOT / "lim=v1/amostra.json")
    parser.add_argument("--cache-origem", type=Path, required=True,
                        help="cache de embeddings de transcrição a copiar (nunca alterado)")
    parser.add_argument("--destino", type=Path, default=features)
    args = parser.parse_args(argv)
    try:
        registro = materializar(pasta_base=args.base, arquivo_particao=args.particao,
                                pasta_legenda_capa=args.legenda_capa, pasta_amostra=args.amostra, arquivo_lim=args.lim,
                                cache_origem=args.cache_origem, destino=args.destino, codificador=leg.CodificadorTexto())
    except (ErroRefit, ds.ErroDataset, particoes.ErroParticao, ValueError, KeyError, OSError) as erro:
        print(f"Refit recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps(registro, ensure_ascii=False, indent=1, default=str))
    return 0


# ──── Cobertura do procedimento de avaliação final, lida do resultado gravado e gravada nos pacotes servid
NIVEL_DO_PROCEDIMENTO = {"video": "N4", "imagem": "N3", "texto": "N2"}
ROTULO_DO_MODELO = {
    "video": "N4 (vídeo, modelo completo)", "imagem": "N3 (imagem: histórico, legenda e capa)",
    "texto": "N2 (texto: histórico e legenda)"}
AVISO_DO_REFIT = (
    "O modelo servido foi retreinado com todos os vídeos anotados da amostra, inclusive os de teste: esta cobertura "
    "foi medida pelo procedimento de avaliação congelado em 452 vídeos de teste e NÃO foi medida neste artefato. "
    "Métricas calculadas sobre os vídeos do ajuste não são desempenho independente. Não há cobertura garantida para "
    "vídeos novos.")


class ErroCobertura(ValueError):
    """Resultado da avaliação ausente, sem o bloco esperado ou fora do contrato."""


def _bloco(principal: Mapping, nivel: str) -> tuple[Mapping, Mapping]:
    try:
        empilhado = principal[nivel]["empilhado"]
        return empilhado["cobertura"], empilhado["cobertura_por_setor"]
    except (KeyError, TypeError) as falha:
        raise ErroCobertura(f"principal.json sem {nivel}.empilhado.cobertura / cobertura_por_setor") from falha


def declaracao_do_procedimento(arquivo_principal: Path, modalidade: str, *, protocolo_sha256: str | None) -> dict:
    """Declaração (texto e números) do modelo da modalidade, com a origem explícita no procedimento de avaliação."""
    if modalidade not in NIVEL_DO_PROCEDIMENTO:
        raise ErroCobertura(f"modalidade sem cobertura de procedimento: {modalidade!r}")
    nivel = NIVEL_DO_PROCEDIMENTO[modalidade]
    principal = json.loads(Path(arquivo_principal).read_text(encoding="utf-8"))
    total, por_setor = _bloco(principal, nivel)
    declaracao = dfx.declarar("teste_final", modelo=ROTULO_DO_MODELO[modalidade], total=total, por_setor=por_setor)
    return declaracao | {
        "aviso": f"{declaracao['aviso']} {AVISO_DO_REFIT}",
        "origem": "procedimento_de_avaliacao_congelado", "nivel_do_procedimento": nivel,
        "medida_no_artefato_servido": False, "garantia_de_cobertura": False,
        "n_canais": total.get("n_canais"), "n_videos_de_teste": total["n"],
        "procedimento": {"consulta": "única, 18/10/2026", "arquivo": Path(arquivo_principal).name,
                         "sha256_arquivo": sha256_arquivo(arquivo_principal), "protocolo_sha256": protocolo_sha256,
                         "metodo_do_intervalo": total.get("metodo_do_intervalo")},
    }


def conferir(declaracao: Mapping) -> None:
    """Falha se a declaração gravada deixar de afirmar a origem no procedimento ou prometer garantia."""
    if (declaracao.get("origem") != "procedimento_de_avaliacao_congelado"
            or declaracao.get("medida_no_artefato_servido") is not False
            or declaracao.get("garantia_de_cobertura") is not False
            or declaracao.get("nivel_nominal") != dfx.NIVEL_NOMINAL):
        raise ErroCobertura("a declaração de cobertura não descreve o procedimento de avaliação sem garantia")


# ──── Ajuste e empacotamento dos modelos servidos do refit final (D-D7, ARQUITETURA §6.5).
VERSAO = "1"
PASTA_RAIZ = "final=v1"
MODALIDADES = ("video", "imagem", "texto")
REFERENCIAS_APROVADAS = {  # pacotes de desenvolvimento aprovados: listas de atributos e parâmetros que o refit repete
    "video": "provisorio=v1/1f4ef5b5eac0-c62b81db", "imagem": "provisorio=v1/2a4739584168-c62b81db",
    "texto": "provisorio=v1/18be8a9f7e86-c62b81db"}
TIPO_DO_DATASET = {"video": "dataset_desenvolvimento", "imagem": "dataset_modalidade", "texto": "dataset_modalidade"}
MODULOS_DO_AJUSTE = (
    "src/modelos/treino.py", "src/modelos/modalidades.py", "src/modelos/treino.py", "src/modelos/preprocessamento.py",
    "src/modelos/treino.py", "src/dados/dataset.py", "src/dados/dataset.py", "src/dados/historico.py",
    "src/modelos/refit.py", "src/modelos/refit.py", "src/modelos/refit.py", "src/avaliacao/final.py",
    "src/avaliacao/limiar.py", "src/avaliacao/limiar.py", "src/avaliacao/limiar.py")
BIBLIOTECAS = ("lightgbm", "scikit-learn", "numpy", "scipy", "pyarrow", "joblib", "torch", "transformers")
ARQUIVO_PACOTE, ARQUIVO_IDS, ARQUIVO_PREDICOES = "pacote.json", "ids.json", "predicoes_em_memoria.npz"
ARQUIVO_MANIFESTO = "manifesto_arquivos.json"
ARQUIVO_PREPROCESSADOR = "preprocessador.joblib"
PRAZO_DAS_ANOTACOES_DOS_ALVOS = "progresso.json"
LIMITACOES = [
    "Refit final: o modelo foi ajustado com todos os vídeos anotados da amostra congelada, inclusive os 452 vídeos de "
    "teste; a LIM ficou fora. Não existe conjunto independente para este artefato.",
    "A cobertura da faixa é a do procedimento de avaliação (consulta única, 452 vídeos), não foi medida neste artefato "
    "e não é garantia para vídeos novos; ficou abaixo dos 80% pretendidos.",
    "Métricas calculadas sobre os vídeos do ajuste não são desempenho independente e nunca devem ser reportadas como tal.",
    "Avaliação retrospectiva aproximada: curtidas lidas na coleta, histórico com defasagem de 30 dias, 16 canais, 82% "
    "automotivo; a Camada 1 local do TAPI §6.3 não foi atendida (desvio declarado).",
    "Os limites da faixa são o mínimo e o máximo das três saídas não negativas, com o q50 central (sem CQR).",
]


class ErroPacote(ValueError):
    """Dataset, população, especificação ou pacote incompatível com o refit final."""


def sha256_ids(ids: Sequence[str]) -> str:
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def dependencias(bibliotecas: tuple[str, ...] = BIBLIOTECAS) -> dict:
    return {"python": platform.python_version(), "plataforma": platform.platform(),
            **{nome: version(nome) for nome in bibliotecas}}


def hashes_dos_modulos(raiz: Path = RAIZ_PROJETO) -> dict:
    return {m: sha256_arquivo(Path(raiz) / m) for m in MODULOS_DO_AJUSTE}


def chave_dos_parametros(parametros: Mapping) -> str:
    return sha256_bytes(json.dumps(dict(parametros), sort_keys=True).encode())[:8]


def id_do_pacote(modalidade: str, manifesto: Mapping, parametros: Mapping) -> str:
    return f"{modalidade}-{manifesto['sha256_dataset'][:12]}-{chave_dos_parametros(parametros)}"


# --- conferências antes do ajuste ----------------------------------------------------------------------------------


def conferir_dataset(modalidade: str, linhas: Sequence[Mapping], manifesto: Mapping) -> md.EspecificacaoAtributos:
    """Dataset do refit: tipo e modalidade certos, contexto "final", todas as linhas `treino`; devolve a especificação."""
    if modalidade not in MODALIDADES:
        raise ErroPacote(f"modalidade desconhecida: {modalidade!r}")
    if manifesto.get("tipo") != TIPO_DO_DATASET[modalidade]:
        raise ErroPacote(f"tipo de dataset inesperado para {modalidade}: {manifesto.get('tipo')!r}")
    if modalidade != "video" and manifesto.get("modalidade") != modalidade:
        raise ErroPacote(f"o dataset é da modalidade {manifesto.get('modalidade')!r}, não de {modalidade!r}")
    contexto = manifesto["entradas"]["contexto"]
    if contexto["nome"] != "final" or contexto["canais_excluidos"]:
        raise ErroPacote("o refit exige o contexto 'final' sem canais excluídos")
    usos = {l["uso_no_contexto"] for l in linhas}
    if usos != {md.USO_TREINO}:
        raise ErroPacote(f"no refit final toda linha deve ser treino; encontrado {sorted(usos)}")
    espec = md.EspecificacaoAtributos(**manifesto["especificacao"])
    if modalidade != "video":
        if fa.hash_da_especificacao(espec) != manifesto["especificacao_sha256"] or "duracao_s" in espec.escalares:
            raise ErroPacote("a especificação do dataset não é a operacional da modalidade (sem duracao_s)")
    return espec


def conferir_contra_referencia(espec: md.EspecificacaoAtributos, pasta_referencia: Path) -> dict:
    """Mesmas listas de atributos, na mesma ordem, e exatamente a configuração A do pacote aprovado de referência."""
    meta = json.loads((Path(pasta_referencia) / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    ref = meta["especificacao"]
    if (list(espec.escalares) != list(ref["escalares"]) or list(espec.categoricas) != list(ref["categoricas"])
            or dict(espec.embeddings) != dict(ref["embeddings"])):
        raise ErroPacote("as listas de atributos diferem das do pacote aprovado de referência")
    parametros = mm.parametros_de_referencia(pasta_referencia)
    return {"pasta": caminho_relativo(pasta_referencia), "sha256_metadados": sha256_arquivo(
        Path(pasta_referencia) / md.ARQUIVO_METADADOS), "sha256_modelo": meta["sha256_modelo"],
        "parametros": parametros}


def aptas_ao_ajuste(linhas: Sequence[Mapping]) -> tuple[list[dict], list[str]]:
    """(linhas com `ref_log` e `y_rel`, ids excluídos por falta deles); os excluídos são contados, nunca preenchidos."""
    aptas = [dict(l) for l in linhas if l.get("ref_log") is not None and l.get("y_rel") is not None]
    ids_aptas = {l["video_id"] for l in aptas}
    return aptas, [l["video_id"] for l in linhas if l["video_id"] not in ids_aptas]


# --- identidades de entrada ----------------------------------------------------------------------------------------


def identidade_das_anotacoes(pasta_cache: Path, ids: Sequence[str]) -> dict:
    """Identidade do conjunto de anotações usadas: chave de compatibilidade e SHA-256 do conjunto de arquivos."""
    linhas, chaves = [], set()
    for video_id in sorted(ids):
        bruto = (Path(pasta_cache) / f"{video_id}.json").read_bytes()
        chaves.add(json.loads(bruto)["chave_compatibilidade"])
        linhas.append(f"{video_id}:{sha256_bytes(bruto)}")
    if len(chaves) != 1:
        raise ErroPacote(f"anotações com chaves de compatibilidade diferentes: {len(chaves)}")
    return {"n": len(linhas), "chave_compatibilidade": chaves.pop(), "sha256_conjunto": sha256_bytes("\n".join(linhas).encode())}


def fontes_de_dados(manifesto: Mapping, pasta_dataset: Path, pasta_legenda_capa: Path, pasta_base: Path,
                    pasta_amostra: Path, anotacoes: Mapping | None) -> dict:
    fontes = {
        "dataset": {"pasta": caminho_relativo(pasta_dataset), "sha256": manifesto["sha256_dataset"],
                    "sha256_manifesto": sha256_arquivo(Path(pasta_dataset) / ds.ARQUIVO_MANIFESTO),
                    "versao": manifesto["versao"], "tipo": manifesto["tipo"]},
        "historico_n1": manifesto["entradas"]["historico"],
        "base_curada": {"pasta": caminho_relativo(pasta_base), "snapshot_id": manifesto["entradas"]["base"]["snapshot_id"],
                        "sha256_manifesto": sha256_arquivo(Path(pasta_base) / "MANIFESTO_SHA256.txt")
                        if (Path(pasta_base) / "MANIFESTO_SHA256.txt").exists() else None,
                        "sha256_manifesto_registrado": manifesto["entradas"]["base"]["sha256_manifesto"]},
        "particao": manifesto["entradas"]["particao"],
        "amostra": manifesto["entradas"]["amostra"],
        "legenda_capa": {"pasta": caminho_relativo(pasta_legenda_capa),
                         "sha256_legenda": sha256_arquivo(Path(pasta_legenda_capa) / "legenda.parquet"),
                         "sha256_capa": sha256_arquivo(Path(pasta_legenda_capa) / "capa.parquet"),
                         "sha256_manifesto": sha256_arquivo(Path(pasta_legenda_capa) / "manifesto.json")},
    }
    if anotacoes is not None:
        fontes["anotacoes_do_gemini"] = anotacoes | {"progresso_sha256": sha256_arquivo(
            Path(pasta_amostra) / PRAZO_DAS_ANOTACOES_DOS_ALVOS)}
    if "codificador_transcricao" in manifesto["entradas"]:
        fontes["codificador_transcricao"] = manifesto["entradas"]["codificador_transcricao"]
    return fontes


def retencao_herdada(manifesto: Mapping, pasta_amostra: Path, pasta_lim: Path) -> dict:
    """Menor prazo do dataset e os prazos de cada origem; o pacote derivado não vive além do menor deles."""
    progresso = json.loads((Path(pasta_amostra) / PRAZO_DAS_ANOTACOES_DOS_ALVOS).read_text(encoding="utf-8"))
    lim = json.loads((Path(pasta_lim) / "amostra.json").read_text(encoding="utf-8"))
    prazos = {"dataset_menor_prazo": manifesto["retencao"]["apagar_ate_utc"],
              "anotacoes_dos_alvos": progresso["retencao"]["apagar_ate_utc"],
              "lim_legenda_capa_e_base": lim["retencao"]["apagar_ate_utc"]}
    return prazos | {"apagar_ate_utc": min(prazos.values()),
                     "regra": "dados derivados herdam o menor prazo; modelo e metadados do pacote devem ser revistos "
                              "ou apagados com os dados de origem (D-B9)"}


# --- ajuste ----------------------------------------------------------------------------------------------------------


def _serializar_preprocessador(modelo: md.ModeloTreinado) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(modelo.pipeline.named_steps["pre"], buffer)
    return buffer.getvalue()


def _salvar_modelo(modelo: md.ModeloTreinado, pasta: Path, metadados: Mapping) -> dict:
    md.salvar(modelo, pasta, metadados=metadados)
    meta = json.loads((pasta / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    (pasta / ARQUIVO_PREPROCESSADOR).write_bytes(_serializar_preprocessador(modelo))
    return {"pasta": pasta.name, "sha256_modelo": meta["sha256_modelo"],
            "sha256_metadados": sha256_arquivo(pasta / md.ARQUIVO_METADADOS),
            "sha256_preprocessador": sha256_arquivo(pasta / ARQUIVO_PREPROCESSADOR)}


def _populacao(aptas: Sequence[Mapping], excluidos: Sequence[str], manifesto: Mapping) -> dict:
    ids = [l["video_id"] for l in aptas]
    canais = sorted({l["channel_id"] for l in aptas})
    por_setor: dict[str, int] = {}
    for l in aptas:
        por_setor[str(l["setor"])] = por_setor.get(str(l["setor"]), 0) + 1
    return {"n_linhas": len(ids), "n_canais": len(canais), "por_setor": dict(sorted(por_setor.items())),
            "por_bloco": dict(sorted({b: sum(l["bloco"] == b for l in aptas) for b in {l["bloco"] for l in aptas}}.items())),
            "sha256_ids": sha256_ids(ids), "sha256_canais": sha256_ids(canais),
            "excluidos_por_falta_de_referencia_ou_alvo": {"n": len(excluidos), "sha256_ids": sha256_ids(sorted(excluidos))},
            "reconciliacao_do_dataset": manifesto.get("reconciliacao_anotacoes", manifesto.get("reconciliacao")),
            "n_treino": len(ids), "n_validacao": 0,
            "nota": "n_validacao = 0: o refit não tem conjunto de validação nem de avaliação"}


def _metadados_comuns(modalidade: str, nivel: str, manifesto: Mapping, populacao: Mapping, retencao: Mapping) -> dict:
    return {"status": "final_refit", "modalidade": modalidade, "nivel_do_modelo": nivel,
            "dataset": {"sha256": manifesto["sha256_dataset"], "versao": manifesto["versao"],
                        "entradas": manifesto["entradas"]},
            "populacao": dict(populacao), "retencao": dict(retencao), "limitacoes": LIMITACOES,
            "semente": md.SEMENTE,
            "alvo": "y_rel = log1p(curtidas) − ref_log; previsão absoluta = ref_log + saída, cortada em 0"}


def treinar_modelos(aptas: Sequence[Mapping], espec: md.EspecificacaoAtributos, parametros: Mapping) -> dict:
    q50 = md.treinar(aptas, espec, parametros=parametros)
    return {"q50": q50} | fp.treinar_companheiros(aptas, q50)


def nome_para_nivel(modalidade: str) -> str:
    return "N4: modelo completo de vídeo (refit final)" if modalidade == "video" else (
        fa.nivel_do_modelo(modalidade).replace("modelo reduzido; desenvolvimento, provisório", "modelo reduzido; refit final"))


def treinar_pacote(modalidade: str, pasta_dataset: Path, saida_raiz: Path, *, raiz_projeto: Path = RAIZ_PROJETO,
                   arquivo_principal: Path, arquivo_sha_protocolo: Path, pasta_referencia: Path | None = None) -> dict:
    """Ajusta q10/q50/q90, grava o pacote imutável e confere a recarga no mesmo processo. Idempotente."""
    raiz = Path(raiz_projeto)
    linhas, manifesto = ds.carregar_dataset(pasta_dataset)
    espec = conferir_dataset(modalidade, linhas, manifesto)
    referencia = Path(pasta_referencia) if pasta_referencia else raiz / "data/models" / REFERENCIAS_APROVADAS[modalidade]
    ref = conferir_contra_referencia(espec, referencia)
    parametros = ref["parametros"]
    destino = Path(saida_raiz) / PASTA_RAIZ / id_do_pacote(modalidade, manifesto, parametros)
    if destino.exists():
        return verificar_integridade(destino) | {"reaproveitado": True}
    aptas, excluidos = aptas_ao_ajuste(linhas)
    if not aptas:
        raise ErroPacote("nenhuma linha apta ao ajuste")
    lim = {x["video_id"] for x in json.loads((raiz / "data/annotations/gemini" / SNAPSHOT / "lim=v1" / "amostra.json")
                                              .read_text(encoding="utf-8"))["lista"]}
    if lim & {l["video_id"] for l in aptas}:
        raise ErroPacote("a LIM não pode entrar no ajuste")
    modelos = treinar_modelos(aptas, espec, parametros)
    nivel = nome_para_nivel(modalidade)
    populacao = _populacao(aptas, excluidos, manifesto)
    pasta_amostra, pasta_lim = raiz / "data/annotations/gemini" / SNAPSHOT / "amostra=v2", raiz / "data/annotations/gemini" / SNAPSHOT / "lim=v1"
    retencao = retencao_herdada(manifesto, pasta_amostra, pasta_lim)
    comuns = _metadados_comuns(modalidade, nivel, manifesto, populacao, retencao)
    parcial = Path(tempfile.mkdtemp(prefix=".pacote.", suffix=".parcial", dir=_garantir(destino.parent)))
    try:
        registro_modelos = {}
        for nome, modelo in modelos.items():
            extra = {"tipo": f"modelo_final_{nome}", "alfa": {"q10": fp.ALFA_INFERIOR, "q50": fp.ALFA_CENTRAL,
                                                               "q90": fp.ALFA_SUPERIOR}[nome]}
            registro_modelos[nome] = _salvar_modelo(modelo, parcial / nome, comuns | extra)
        previsoes = {nome: modelo.prever_relativo(aptas) for nome, modelo in modelos.items()}
        np.savez(parcial / ARQUIVO_PREDICOES, **previsoes)
        ids = [l["video_id"] for l in aptas]
        (parcial / ARQUIVO_IDS).write_text(json.dumps({"ajuste": ids, "excluidos": sorted(excluidos)}), encoding="utf-8")
        identidade = dl.identidade_do_pacote({n: r["sha256_modelo"] for n, r in registro_modelos.items()})
        pacote = _registro_do_pacote(
            modalidade, nivel, destino.name, manifesto, populacao, retencao, registro_modelos, identidade, ref, parametros,
            raiz, pasta_dataset, arquivo_principal, arquivo_sha_protocolo, aptas, espec, modelos["q50"])
        (parcial / ARQUIVO_PACOTE).write_text(json.dumps(pacote, ensure_ascii=False, indent=1), encoding="utf-8")
        _gravar_manifesto_de_arquivos(parcial)
        parcial.rename(destino)
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    reconferencia = conferir_recarga(destino, aptas, previsoes)
    return {"pacote": destino.as_posix(), "id": destino.name, "identidade_sha256": identidade,
            "n_linhas": len(aptas), "recarga_identica_no_processo": reconferencia, "reaproveitado": False}


def _contexto(manifesto: Mapping) -> dict:
    contexto = manifesto["entradas"]["contexto"]
    return {"nome": contexto["nome"], "treino_ate_utc": contexto["treino_ate_utc"],
            "canais_excluidos": list(contexto["canais_excluidos"]),
            "defasagem_dias": manifesto["entradas"]["defasagem_dias"],
            "regra": "treino_ate_utc = última publicação entre os alvos da amostra; toda linha é treino; sem intervalo "
                     "excluído, sem bloco avaliado e sem canal retido"}


def _garantir(pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _registro_do_pacote(modalidade, nivel, id_, manifesto, populacao, retencao, registro_modelos, identidade, ref,
                        parametros, raiz, pasta_dataset, arquivo_principal, arquivo_sha_protocolo, aptas, espec,
                        q50) -> dict:
    pasta_lc = raiz / "data/features" / SNAPSHOT / "legenda_capa=v1"
    pasta_base = raiz / "data/processed" / SNAPSHOT / "curadoria=v1"
    pasta_amostra = raiz / "data/annotations/gemini" / SNAPSHOT / "amostra=v2"
    anotacoes = identidade_das_anotacoes(pasta_amostra / "cache", [l["video_id"] for l in aptas]) \
        if modalidade == "video" else None
    protocolo = Path(arquivo_sha_protocolo).read_text(encoding="utf-8").split()[0]
    declaracao = declaracao_do_procedimento(arquivo_principal, modalidade, protocolo_sha256=protocolo)
    conferir(declaracao)
    return {
        "tipo": "pacote_final", "versao": VERSAO, "id": id_, "gerado_em_utc": _iso(datetime.now(UTC)),
        "modalidade": modalidade, "nivel_do_modelo": nivel, "versao_servida": f"{PASTA_RAIZ}/{id_}",
        "modelos": registro_modelos, "identidade_sha256": identidade,
        "populacao": populacao,
        "contexto": _contexto(manifesto),
        "especificacao": {
            "escalares": list(espec.escalares), "categoricas": list(espec.categoricas),
            "embeddings": dict(espec.embeddings), "sha256": fa.hash_da_especificacao(espec),
            "atributos_de_saida": q50.nomes_atributos, "n_atributos_de_saida": len(q50.nomes_atributos),
            "duracao_s_e_preditor": "duracao_s" in espec.escalares,
            "referencia_aprovada": ref | {"parametros": "idênticos à CONFIGURACAO_A"}},
        "parametros": parametros, "sementes": {"lightgbm": parametros["random_state"], "preprocessador": md.SEMENTE,
                                              "pca_componentes_por_bloco": ds.COMPONENTES_PCA},
        "regra_da_faixa": "q50 central; limites = mínimo e máximo das três saídas reconstruídas e não negativas; sem CQR",
        "alvo": "y_rel = log1p(curtidas) − ref_log; previsão = ref_log + saída, cortada em 0",
        "cobertura_do_procedimento": declaracao,
        "fontes": fontes_de_dados(manifesto, pasta_dataset, pasta_lc, pasta_base, pasta_amostra, anotacoes),
        "codigo": {"modulos_sha256": hashes_dos_modulos(raiz)}, "dependencias": dependencias(),
        "retencao": retencao, "limitacoes": LIMITACOES,
        "lim": {"fora_do_ajuste": True, "uso": "somente a largura da faixa do vídeo (limiar de baixa confiança), sem rótulos"
                if modalidade == "video" else "não usada: imagem e texto não têm limiar (sinal sempre ligado)"},
        "verificacao_de_recarga": {"predicoes_em_memoria": ARQUIVO_PREDICOES,
                                   "hash": f"em {ARQUIVO_MANIFESTO} (gravado por último)"},
    }


def _gravar_manifesto_de_arquivos(pasta: Path, excluir: tuple[str, ...] = ()) -> None:
    """Manifesto SHA-256 dos arquivos do pacote; `excluir` lista pastas de primeiro nível que ficam fora (ex.: `limiar`)."""
    arquivos = {p.relative_to(pasta).as_posix(): sha256_arquivo(p)
                for p in sorted(pasta.rglob("*")) if p.is_file() and p.name != ARQUIVO_MANIFESTO
                and p.relative_to(pasta).parts[0] not in excluir}
    (pasta / ARQUIVO_MANIFESTO).write_text(json.dumps({"arquivos": arquivos}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")


# --- integridade e recarga ---------------------------------------------------------------------------------------


def verificar_integridade(pasta: Path) -> dict:
    """Todos os arquivos do pacote conferem com o manifesto de arquivos e os modelos com o `pacote.json`."""
    pasta = Path(pasta)
    manifesto = json.loads((pasta / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))["arquivos"]
    divergentes = [nome for nome, esperado in manifesto.items()
                   if not (pasta / nome).is_file() or sha256_arquivo(pasta / nome) != esperado]
    atuais = {p.relative_to(pasta).as_posix() for p in pasta.rglob("*") if p.is_file() and p.name != ARQUIVO_MANIFESTO
              and not p.relative_to(pasta).parts[0] == "limiar"}
    extras = sorted(atuais - set(manifesto))
    if divergentes or extras:
        raise ErroPacote(f"pacote {pasta.name} não confere com o manifesto: divergentes={divergentes[:3]} extras={extras[:3]}")
    pacote = json.loads((pasta / ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    for nome, registro in pacote["modelos"].items():
        if sha256_arquivo(pasta / nome / md.ARQUIVO_MODELO) != registro["sha256_modelo"]:
            raise ErroPacote(f"modelo {nome} do pacote {pasta.name} não confere com o hash registrado")
    identidade = dl.identidade_do_pacote({n: r["sha256_modelo"] for n, r in pacote["modelos"].items()})
    if identidade != pacote["identidade_sha256"]:
        raise ErroPacote("a identidade do pacote não confere com os hashes dos três modelos")
    return {"pacote": pasta.as_posix(), "id": pasta.name, "identidade_sha256": identidade,
            "n_linhas": pacote["populacao"]["n_linhas"], "integridade": "ok"}


def conferir_recarga(pasta: Path, aptas: Sequence[Mapping], previsoes_em_memoria: Mapping[str, np.ndarray]) -> bool:
    """Recarrega do disco (no mesmo processo) e exige previsões idênticas bit a bit nas mesmas linhas."""
    for nome, esperado in previsoes_em_memoria.items():
        recarregado = md.carregar(Path(pasta) / nome)
        if not np.array_equal(recarregado.prever_relativo(aptas), esperado):
            raise ErroPacote(f"o modelo {nome} recarregado não reproduz as previsões")
    return True


def main_refit_treino(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.modelos.refit refit_treino", description=__doc__)
    parser.add_argument("--modalidade", choices=MODALIDADES, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="pasta do dataset final da modalidade")
    parser.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "data" / "models")
    parser.add_argument("--principal", type=Path, required=True, help="principal.json da avaliação final")
    parser.add_argument("--sha-protocolo", type=Path, required=True, help="protocolo_final.sha256")
    args = parser.parse_args(argv)
    try:
        resultado = treinar_pacote(args.modalidade, args.dataset, args.saida, arquivo_principal=args.principal,
                                   arquivo_sha_protocolo=args.sha_protocolo)
    except (ErroPacote, ds.ErroDataset, ErroCobertura, md.ErroModelo, ValueError, KeyError, OSError) as erro:
        print(f"Refit recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    return 0


# ──── Pacote do preditor audiovisual local: ajuste final, gravação imutável e verificação de recarga.
PASTA_RAIZ_refit_audiovisual = "final=v2"
ROTA = "video_local"
ROTULO_AVALIACAO = "exploratória / teste reutilizado"
BIBLIOTECAS_AUDIOVISUAL = ("lightgbm", "scikit-learn", "numpy", "scipy", "pyarrow", "joblib")
NIVEL_NOMINAL = 0.8
LIMITACOES_refit_audiovisual = [
    "O modelo foi ajustado com todas as linhas completas da amostra ativa, inclusive as do teste temporal já exposto; "
    "a LIM ficou fora. Não existe conjunto independente para este artefato.",
    "A avaliação descritiva foi feita uma vez no teste temporal já exposto (exploratória / teste reutilizado); a "
    "população pode diferir da original por substituição de IDs e por ausência de extração local.",
    "A cobertura da faixa descreve o procedimento de avaliação e não foi medida neste artefato; não é garantia para "
    "vídeos novos.",
    "Os atributos locais vêm de uma extração offline com modelos pequenos; nenhum deles é validado como medida de "
    "qualidade criativa, e embeddings podem codificar estilo do canal.",
    "Métricas calculadas sobre as linhas do ajuste não são desempenho independente.",
]


class ErroPacoteLocal(ValueError):
    """Entrada, população ou pacote incompatível com o contrato do preditor local."""


def identidade_do_extrator() -> dict:
    """Identidade do extrator que produz os atributos: SHA-256 da configuração e dos pesos fixados por revisão."""
    config = especificacao_extrator.configuracao()
    return {"versao": config["versao"], "identidade_sha256": especificacao_extrator.identidade_da_configuracao(config),
            "modelos": config["modelos"], "bibliotecas": config["bibliotecas"]}


FONTES_DE_RETENCAO_OBRIGATORIAS = ("dataset_v2", "base_de_posts", "extracao_local")
FONTES_DE_RETENCAO_OPCIONAIS = ("anotacoes_gemini", "lim")


def _instante(valor: str) -> datetime:
    instante = datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
    if instante.tzinfo is None:
        raise ErroPacoteLocal(f"prazo de retenção sem fuso horário: {valor!r}")
    return instante.astimezone(UTC)


def retencao_minima(prazos: Mapping[str, str]) -> dict:
    """Menor prazo entre as fontes explícitas; registra quem impôs o mínimo. Copiar dados não estende retenção.

    Todo prazo é lido como instante com fuso (data sem fuso é recusada). `dataset_v2`, `base_de_posts` e
    `extracao_local` são obrigatórios; `anotacoes_gemini` e `lim` entram quando usadas."""
    faltam = [f for f in FONTES_DE_RETENCAO_OBRIGATORIAS if f not in prazos]
    desconhecidas = sorted(set(prazos) - set(FONTES_DE_RETENCAO_OBRIGATORIAS) - set(FONTES_DE_RETENCAO_OPCIONAIS))
    if faltam or desconhecidas:
        raise ErroPacoteLocal(f"fontes de retenção: faltam {faltam}; desconhecidas {desconhecidas}")
    instantes = {fonte: _instante(valor) for fonte, valor in prazos.items()}
    quem = min(instantes, key=lambda f: (instantes[f], f))
    return dict(prazos) | {"apagar_ate_utc": instantes[quem].strftime("%Y-%m-%dT%H:%M:%SZ"), "imposto_por": quem,
                           "regra": "o artefato derivado herda o menor prazo entre as fontes"}


def atualizar_retencao(retencao: Mapping, **novos_prazos: str) -> dict:
    """Recalcula o mínimo depois de acrescentar fontes (ex.: o prazo da LIM no passo do limiar)."""
    prazos = {k: v for k, v in retencao.items() if k in FONTES_DE_RETENCAO_OBRIGATORIAS + FONTES_DE_RETENCAO_OPCIONAIS}
    return retencao_minima(prazos | novos_prazos)


def duracao_s_do_arquivo(duracao_medida_s: float) -> int:
    """Convenção de `duracao_s` do modelo: inteiro por teto da duração medida no arquivo (igual à sonda do agente)."""
    return math.ceil(duracao_medida_s - 1e-9)


AVISOS_OBRIGATORIOS = {
    "superou_m0_em_erro": "O ganho de erro sobre M0 (histórico e contexto, sem atributos locais) não foi demonstrado: o "
                          "preditor local é servido sem evidência de melhora sobre a referência de M0.",
    "superou_media_historica_em_erro": "O preditor não superou a média histórica do canal em erro com evidência; a "
                                       "referência do canal deve ser exibida ao lado.",
    "ordenacao_demonstrada": "Melhora de erro; ordenação não demonstrada: não se afirma que o preditor ordene vídeos "
                             "melhor que M0.",
}


AVISO_ATALHO_DE_CANAL = ("O ganho de erro sobre M0 pode refletir a identificação do canal pelos atributos locais, e não "
                         "o conteúdo do vídeo.")


def avisos_da_selecao(selecao: Mapping) -> list[str]:
    """Aviso de cada condição falsa na seleção; imposto por código e gravado em `limitacoes`."""
    avisos = [texto for chave, texto in AVISOS_OBRIGATORIOS.items() if not selecao.get(chave)]
    if selecao.get("atalho_de_canal_sinalizado"):
        avisos.append(selecao.get("aviso_de_atalho_de_canal") or AVISO_ATALHO_DE_CANAL)
    return avisos


def declaracao_de_cobertura(cobertura: Mapping | None, por_setor: Mapping | None, *, resultado_arquivo: Path | None,
                            protocolo_sha256: str | None) -> dict:
    """Declaração no formato dos pacotes finais, com a origem no procedimento de avaliação exploratória."""
    base = {"nome": "faixa de previsão", "contexto": "teste_reutilizado_exploratorio", "nivel_nominal": NIVEL_NOMINAL,
            "origem": "procedimento_de_avaliacao_congelado", "medida_no_artefato_servido": False,
            "garantia_de_cobertura": False, "rotulo": ROTULO_AVALIACAO}
    if cobertura is None:
        return base | {"cobertura_observada": None, "n_dentro": None, "n": None, "abaixo_do_pretendido": None,
                       "por_setor": [], "aviso": "Cobertura da faixa indisponível: a avaliação exploratória não foi "
                       "gravada para este artefato; a faixa não tem garantia."}
    observada = cobertura["n_dentro"] / cobertura["n"]
    abaixo = observada < NIVEL_NOMINAL
    setores = [{"setor": s, "texto": f"{s}: {r['n_dentro']} de {r['n']} vídeos ({100 * r['cobertura']:.1f}%), "
                                      f"{r['n_canais']} canais", "ic_estado": r.get("ic_estado")}
               for s, r in sorted((por_setor or {}).items())]
    aviso = (f"Faixa de previsão do preditor local, com nível pretendido de {NIVEL_NOMINAL:.0%}. Na avaliação "
             f"{ROTULO_AVALIACAO}, {cobertura['n_dentro']} de {cobertura['n']} vídeos ({100 * observada:.1f}%) ficaram "
             f"dentro dela ({'abaixo do pretendido' if abaixo else 'no nível pretendido ou acima'}). Descreve o "
             "procedimento, não o artefato servido, e não é garantia para vídeos novos.")
    return base | {
        "cobertura_observada": observada, "n_dentro": cobertura["n_dentro"], "n": cobertura["n"],
        "ic95_bootstrap_por_canal": cobertura.get("ic95_bootstrap_por_canal"), "n_canais": cobertura.get("n_canais"),
        "abaixo_do_pretendido": abaixo, "por_setor": setores, "aviso": aviso,
        "procedimento": {"arquivo": None if resultado_arquivo is None else Path(resultado_arquivo).name,
                         "sha256_arquivo": None if resultado_arquivo is None else sha256_arquivo(resultado_arquivo),
                         "protocolo_sha256": protocolo_sha256}}


def _ordem_dos_atributos(modelo: md.ModeloTreinado) -> dict:
    pre = modelo.pipeline.named_steps["pre"]
    descartadas = set(pre.colunas_descartadas_)
    return {"entrada_candidata": af.ordem_de_saida(modelo.especificacao),
            "entrada_efetiva": [c for c in af.ordem_de_saida(modelo.especificacao) if c not in descartadas],
            "descartadas_por_constancia": sorted(descartadas),
            "categorias_locais_permitidas": dict(pre.categorias_permitidas_),
            "saida_do_preprocessador": list(pre.nomes_saida_)}


def _dimensoes(modelo: md.ModeloTreinado, dimensao_original: Mapping[str, int]) -> dict:
    pre = modelo.pipeline.named_steps["pre"]
    return {"n_colunas_de_saida": len(pre.nomes_saida_),
            "embeddings": {c: {"dimensao_original": dimensao_original.get(c), "componentes_pca": k}
                           for c, k in modelo.especificacao.embeddings.items()}}


def id_do_pacote_refit_audiovisual(id12: str, config_id: str, parametros: Mapping) -> str:
    return f"{ROTA}-{id12}-{config_id}_{chave_dos_parametros(parametros)}"


def treinar_pacote_local(config_id: str, linhas_refit: Sequence[Mapping], total: Mapping, *, saida_raiz: Path,
                         selecao: Mapping, ids_lim: Sequence[str], id12: str, fontes: Mapping,
                         prazos_de_retencao: Mapping[str, str],
                         dimensao_original: Mapping[str, int], avaliacao: Mapping | None = None,
                         cobertura: Mapping | None = None, ids_excluidos: Sequence[str] = (), treino_ate_utc: str | None = None,
                         parametros: Mapping = rat.PARAMETROS, extrator: Mapping | None = None) -> dict:
    """Ajusta q10/q50/q90, grava o pacote imutável e confere a recarga. Reexecutar não sobrescreve.

    `selecao` é o registro de seleção do resultado do desenvolvimento (verificado por hash pelo chamador): a
    configuração tem de ser a escolhida nele, e os avisos de "não superou" e "ordenação não demonstrada" entram em
    `limitacoes` por código quando as condições são falsas."""
    if config_id != selecao.get("escolhida"):
        raise ErroPacoteLocal(f"{config_id} difere da configuração escolhida no desenvolvimento: {selecao.get('escolhida')!r}")
    if config_id not in af.ELEGIVEIS_PARA_SERVIR:
        raise ErroPacoteLocal(f"{config_id} não pode ser servida; elegíveis: {list(af.ELEGIVEIS_PARA_SERVIR)}")
    espec = af.especificacao_da_configuracao(config_id, total)
    af.conferir_sem_gemini(espec)
    destino = Path(saida_raiz) / PASTA_RAIZ_refit_audiovisual / id_do_pacote_refit_audiovisual(id12, config_id, parametros)
    if destino.exists():
        raise ErroPacoteLocal(f"o pacote já existe e não será sobrescrito: {destino.name}")
    aptas = [dict(l, uso_no_contexto="treino") for l in linhas_refit
             if l.get("ref_log") is not None and l.get("y_rel") is not None]
    if not aptas:
        raise ErroPacoteLocal("nenhuma linha apta ao ajuste")
    if set(ids_lim) & {l["video_id"] for l in aptas}:
        raise ErroPacoteLocal("a LIM não pode entrar no ajuste")
    modelos = rat.treinar_quantis(aptas, espec, af.colunas_locais_da_especificacao(espec), parametros,
                                  contador=rat.ContadorDeAjustes(len(rat.ALFAS)), rotulo=f"refit/{config_id}")
    pre_q50 = modelos["q50"].pipeline.named_steps["pre"]
    af.conferir_sem_gemini(espec, pre_q50.nomes_saida_)
    ids = [l["video_id"] for l in aptas]
    retencao = retencao_minima(prazos_de_retencao)
    nivel = f"Preditor local de vídeo {config_id}: {af.CONFIGURACOES[config_id].descricao} (refit; exploratório)"
    avisos = avisos_da_selecao(selecao)
    limitacoes = [*LIMITACOES_refit_audiovisual, *avisos]
    comuns = {"status": "final_refit", "modalidade": "video", "rota": ROTA, "nivel_do_modelo": nivel,
              "dataset": dict(fontes.get("dataset", {})),
              "populacao": {"n_treino": len(ids), "n_validacao": 0, "n_linhas": len(ids)},
              "retencao": retencao, "limitacoes": limitacoes, "semente": md.SEMENTE,
              "alvo": "y_rel = log1p(curtidas) − ref_log; previsão absoluta = ref_log + saída, cortada em 0"}
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = Path(tempfile.mkdtemp(prefix=".pacote.", suffix=".parcial", dir=destino.parent))
    try:
        registro = {n: _salvar_modelo(m, parcial / n, comuns | {"tipo": f"modelo_local_{n}", "alfa": rat.ALFAS[n]})
                    for n, m in modelos.items()}
        previsoes = {n: m.prever_relativo(aptas) for n, m in modelos.items()}
        np.savez(parcial / ARQUIVO_PREDICOES, **previsoes)
        (parcial / ARQUIVO_IDS).write_text(json.dumps({"ajuste": ids, "excluidos": sorted(ids_excluidos)}),
                                              encoding="utf-8")
        identidade = dl.identidade_do_pacote({n: r["sha256_modelo"] for n, r in registro.items()})
        pacote = {
            "tipo": "pacote_final", "versao": VERSAO, "id": destino.name, "gerado_em_utc": _iso(datetime.now(UTC)),
            "modalidade": "video", "rota": ROTA, "usa_gemini": False, "config_id": config_id, "nivel_do_modelo": nivel,
            "versao_servida": f"{PASTA_RAIZ_refit_audiovisual}/{destino.name}", "modelos": registro, "identidade_sha256": identidade,
            "populacao": comuns["populacao"] | {
                "n_canais": len({l["channel_id"] for l in aptas}), "sha256_ids": sha256_ids(ids),
                "excluidos": {"n": len(ids_excluidos), "sha256_ids": sha256_ids(sorted(ids_excluidos))}},
            "contexto": {"nome": "final", "treino_ate_utc": treino_ate_utc,
                         "regra": "histórico e referências no contexto final; toda linha é treino"},
            "extrator": {"config_id": config_id} | dict(extrator or identidade_do_extrator()),
            "familias": list(af.CONFIGURACOES[config_id].familias),
            "ordem_atributos": _ordem_dos_atributos(modelos["q50"]),
            "dimensoes": _dimensoes(modelos["q50"], dimensao_original),
            "especificacao": {"escalares": list(espec.escalares), "categoricas": list(espec.categoricas),
                              "embeddings": dict(espec.embeddings), "sha256": fa.hash_da_especificacao(espec),
                              "atributos_de_saida": modelos["q50"].nomes_atributos,
                              "n_atributos_de_saida": len(modelos["q50"].nomes_atributos),
                              "colunas_locais_excluidas": af.colunas_locais_excluidas()},
            "parametros": dict(parametros), "sementes": {"lightgbm": parametros["random_state"],
                                                          "preprocessador": md.SEMENTE},
            "regra_da_faixa": "q50 central; limites = mínimo e máximo das três saídas reconstruídas e não negativas; "
                              "a correção de cruzamento preserva o q50",
            "alvo": comuns["alvo"],
            "cobertura_do_procedimento": cobertura if cobertura is not None else declaracao_de_cobertura(
                None, None, resultado_arquivo=None, protocolo_sha256=None),
            "selecao": {k: selecao.get(k) for k in (
                "escolhida", "regra", "mae_log", "mae_por_dobra", "mae_macro_por_canal", "desempate",
                "criterio_de_superou", "superou_m0_em_erro", "superou_media_historica_em_erro", "ordenacao_vs_m0",
                "ordenacao_demonstrada", "texto_obrigatorio", "sonda_de_canal", "atalho_de_canal_sinalizado")},
            "superou_m0_em_erro": bool(selecao.get("superou_m0_em_erro")),
            "superou_media_historica_em_erro": bool(selecao.get("superou_media_historica_em_erro")),
            "ordenacao_demonstrada": bool(selecao.get("ordenacao_demonstrada")),
            "avisos_obrigatorios": avisos, "texto_de_ordenacao": selecao.get("texto_obrigatorio"),
            "convencao_de_entrada": {"duracao_s": "inteiro por teto da duração medida no arquivo "
                                                   "(igual a SondaVideo.duracao_inteira_s)"},
            "avaliacao": dict(avaliacao) if avaliacao else {"estado": "indisponivel", "rotulo": ROTULO_AVALIACAO},
            "limiar_largura": {"estado": "indisponivel", "id_artefato": identidade,
                               "motivo": "ainda não calculado para este artefato; confiança insuficiente/indisponível"},
            "fontes": dict(fontes), "dependencias": dependencias(BIBLIOTECAS_AUDIOVISUAL), "retencao": retencao, "limitacoes": limitacoes,
            "lim": {"fora_do_ajuste": True, "ids_sha256": sha256_ids(sorted(ids_lim)), "n": len(ids_lim),
                    "uso": "somente a largura da faixa (dispersão relativa na LIM), sem rótulos"},
            "verificacao_de_recarga": {"predicoes_em_memoria": ARQUIVO_PREDICOES},
        }
        (parcial / ARQUIVO_PACOTE).write_text(json.dumps(pacote, ensure_ascii=False, indent=1), encoding="utf-8")
        _gravar_manifesto_de_arquivos(parcial, excluir=("limiar",))
        for nome, esperado in previsoes.items():  # recarga conferida ANTES de o pacote existir com o nome final
            if not np.array_equal(md.carregar(parcial / nome).prever_relativo(aptas), esperado):
                raise ErroPacoteLocal(f"o modelo {nome} recarregado do diretório parcial não reproduz as previsões")
        parcial.rename(destino)
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    return {"pacote": destino.as_posix(), "id": destino.name, "identidade_sha256": identidade, "n_linhas": len(ids),
            "recarga_identica_no_processo": True}


def prever_exibido(modelos: Mapping[str, md.ModeloTreinado], linhas: Sequence[Mapping]) -> dict:
    """Curtidas e log exibidos: q50 central, limites mínimo/máximo, nunca negativos."""
    return rat.prever_com_faixa(modelos, linhas)


# ──── Limiar de largura de baixa confiança do preditor audiovisual local, ligado à identidade do artefato 
COLUNAS_ROTULO = tuple(hs.COLUNAS_ALVO)


def linhas_da_lim(ids: Sequence[str], posts: Sequence[hs.Post], canais: Mapping[str, hs.Canal],
                  ctx: hs.ContextoTreino, tabela_local: Mapping[str, Mapping], *,
                  defasagem_dias: int = hs.DEFASAGEM_PADRAO) -> list[dict]:
    """Linhas de inferência da LIM no contexto final: histórico, atributos locais e nenhuma coluna de rótulo."""
    ids = list(ids)
    alvos = [p for p in posts if p.video_id in set(ids)]
    if len(alvos) != len(ids):
        raise ErroPacoteLocal(f"{len(ids) - len(alvos)} vídeos da LIM fora da base de posts")
    faltam = sorted(v for v in ids if v not in tabela_local)
    if faltam:
        raise ErroPacoteLocal(f"{len(faltam)} vídeos da LIM sem atributos locais (ex.: {faltam[:3]})")
    n1 = {l["video_id"]: l for l in hs.calcular(posts, canais, alvos, ctx, defasagem_dias=defasagem_dias)}
    sem_referencia = sorted(i for i in ids if n1[i]["ref_log"] is None)
    if sem_referencia:
        raise ErroPacoteLocal(f"{len(sem_referencia)} vídeos da LIM sem referência de histórico ou de setor "
                              f"(ex.: {sem_referencia[:3]}): a largura da faixa não é definida")
    juntas = [{**{c: v for c, v in tabela_local[i].items() if c not in n1[i]}, **n1[i]} for i in ids]
    return dl.sem_rotulos(juntas)


def _ic_do_p80(larguras: np.ndarray, percentil: float, n_repeticoes: int = 2000, semente: int = 20261015) -> list[float]:
    """IC95 percentílico do p80 por reamostragem dos vídeos da LIM (a LIM tem poucos canais: aproximado)."""
    rng = np.random.default_rng(semente)
    p80 = [np.percentile(rng.choice(larguras, size=len(larguras), replace=True), percentil) for _ in range(n_repeticoes)]
    return [float(x) for x in np.percentile(p80, [2.5, 97.5])]


def anexar_limiar(pasta_pacote: Path, linhas_lim: Sequence[Mapping], prazo_lim: str) -> dict:
    """Calcula o p80 das larguras da LIM com os modelos DESTE pacote e o grava vinculado à identidade.

    Confere que os IDs da LIM são os registrados no pacote e disjuntos do ajuste; recalcula a retenção com o prazo da
    LIM. O valor é uma dispersão relativa da faixa na LIM, não uma confiança calibrada."""
    pasta = Path(pasta_pacote)
    verificar_integridade(pasta)
    pacote = json.loads((pasta / ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    if pacote.get("limiar_largura", {}).get("estado") != "indisponivel":
        raise ErroPacoteLocal("o pacote já tem limiar de largura e não será alterado")
    if len(linhas_lim) != dl.N_LIM:
        raise ErroPacoteLocal(f"a LIM tem {dl.N_LIM} vídeos e o limiar só é definido com todos; recebi {len(linhas_lim)}")
    ids = [l["video_id"] for l in linhas_lim]
    if sha256_ids(sorted(ids)) != pacote.get("lim", {}).get("ids_sha256"):
        raise ErroPacoteLocal("os IDs da LIM diferem dos registrados no pacote")
    ajuste = set(json.loads((pasta / ARQUIVO_IDS).read_text(encoding="utf-8"))["ajuste"])
    if ajuste & set(ids):
        raise ErroPacoteLocal("a LIM não é disjunta do ajuste")
    modelos = {q: md.carregar(pasta / q) for q in dl.QUANTIS}
    identidade = dl.identidade_do_pacote({q: pacote["modelos"][q]["sha256_modelo"] for q in dl.QUANTIS})
    if identidade != pacote["identidade_sha256"]:
        raise ErroPacoteLocal("a identidade do pacote não confere com os hashes dos modelos")
    larguras = dl.larguras_do_pacote(modelos["q10"], modelos["q50"], modelos["q90"], linhas_lim)
    limiar = dl.calcular_limiar("servido", identidade, [l["video_id"] for l in linhas_lim], larguras)
    arquivo = dl.gravar_limiar(limiar, pasta / "limiar")
    pacote["retencao"] = atualizar_retencao(pacote["retencao"], lim=prazo_lim)
    pacote["limiar_largura"] = {
        "estado": "calculado", "nome": "dispersão relativa da faixa na LIM (não é confiança calibrada)",
        "ic95_do_p80_por_reamostragem": _ic_do_p80(np.asarray(larguras, dtype=float), limiar.percentil),
        "valor": limiar.valor, "n_lim": limiar.n, "id_artefato": identidade,
        "percentil": limiar.percentil, "ids_sha256": limiar.ids_sha256, "arquivo": f"limiar/{arquivo.name}",
        "sha256_arquivo": sha256_arquivo(arquivo), "papel": limiar.papel,
        "nota": dl.NOTA_DA_LIM, "diagnostico_de_desenvolvimento": "não calculado para este artefato"}
    (pasta / ARQUIVO_PACOTE).write_text(json.dumps(pacote, ensure_ascii=False, indent=1), encoding="utf-8")
    _gravar_manifesto_de_arquivos(pasta, excluir=("limiar",))
    verificar_integridade(pasta)
    return pacote["limiar_largura"]


def ler_limiar_vinculado(pasta_pacote: Path) -> dl.LimiarDeLargura | None:
    """O limiar do pacote, se existir e pertencer à identidade dele; `LimiarIncompativel` caso contrário."""
    pasta = Path(pasta_pacote)
    pacote = json.loads((pasta / ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    registro = pacote.get("limiar_largura", {})
    if registro.get("estado") != "calculado":
        return None
    limiar = dl.ler_limiar(pasta / registro["arquivo"])
    dl._exigir_compativel(limiar, pacote["identidade_sha256"])
    return limiar


# ──── Verificação de um pacote final em processo novo: recarga bit a bit, identidade e aditividade do SHAP
TOLERANCIA_SHAP = 1e-9
REF_LOG_BAIXA = -50.0  # força saída bruta negativa: a linha sintética é cortada em zero
QUANTIS = ("q10", "q50", "q90")


def _aditividade(q50: md.ModeloTreinado, linhas: Sequence[Mapping]) -> dict:
    explicacao = fp.contribuicoes_q50(q50, linhas)
    bruta = q50.prever_relativo(linhas)
    reconstruida = explicacao["contribuicoes"].sum(axis=1) + explicacao["base"]
    erro = np.abs(reconstruida - bruta)
    return {"n": len(linhas), "erro_maximo": float(erro.max()), "tolerancia": TOLERANCIA_SHAP,
            "aditivo": bool(erro.max() <= TOLERANCIA_SHAP), "base": explicacao["base"],
            "n_cortadas_em_zero": int(explicacao["cortada_em_zero"].sum()),
            "cortadas_em_zero": [bool(c) for c in explicacao["cortada_em_zero"]]}


def _linhas_cortadas(linhas: Sequence[Mapping], n: int = 5) -> list[dict]:
    return [dict(l) | {"ref_log": REF_LOG_BAIXA} for l in linhas[:n]]


def verificar_com_linhas(pasta: Path, linhas: Sequence[Mapping]) -> dict:
    """Núcleo testável: `linhas` são as do ajuste, na ordem de `ids.json` (`ajuste`)."""
    pasta = Path(pasta)
    integridade = verificar_integridade(pasta)
    pacote = json.loads((pasta / ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    modelos = {n: md.carregar(pasta / n) for n in QUANTIS}
    guardadas = np.load(pasta / ARQUIVO_PREDICOES)
    iguais = {n: bool(np.array_equal(modelos[n].prever_relativo(linhas), guardadas[n])) for n in QUANTIS}
    identidade = dl.identidade_do_pacote({n: pacote["modelos"][n]["sha256_modelo"] for n in QUANTIS})
    pre = {n: _preprocessador_confere(pasta / n, modelos[n]) for n in QUANTIS}
    reais = _aditividade(modelos["q50"], linhas)
    sinteticas = _aditividade(modelos["q50"], _linhas_cortadas(linhas))
    faixa = fp.prever_faixa(_linhas_cortadas(linhas), *(modelos[n].prever_relativo(_linhas_cortadas(linhas))
                                                       for n in QUANTIS))
    ok = (all(iguais.values()) and identidade == pacote["identidade_sha256"] and all(pre.values())
          and reais["aditivo"] and sinteticas["aditivo"] and all(sinteticas["cortadas_em_zero"])
          and bool(faixa["cortada_em_zero"].all()) and bool((faixa["central_log"] == 0).all()))
    return {
        "verificado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "pacote": pasta.name,
        "integridade": integridade["integridade"], "n_linhas": len(linhas),
        "previsoes_identicas_bit_a_bit": iguais, "identidade_sha256": identidade,
        "identidade_confere": identidade == pacote["identidade_sha256"], "preprocessador_confere": pre,
        "shap_linhas_do_ajuste": {k: v for k, v in reais.items() if k != "cortadas_em_zero"},
        "shap_linhas_sinteticas_cortadas_em_zero": {k: v for k, v in sinteticas.items() if k != "cortadas_em_zero"}
        | {"todas_declaradas_cortadas": all(sinteticas["cortadas_em_zero"]),
           "valor_exibido_central_log_zero": bool((faixa["central_log"] == 0).all())},
        "escala_do_shap": "y_rel em log (saída bruta do q50); o valor exibido passa por + ref_log, corte em 0 e expm1",
        "aprovado": ok,
    }


def _preprocessador_confere(pasta_modelo: Path, modelo: md.ModeloTreinado) -> bool:
    """O preprocessador serializado ao lado do modelo é o mesmo que está dentro do pipeline carregado."""
    arquivo = joblib.load(pasta_modelo / ARQUIVO_PREPROCESSADOR)
    dentro = modelo.pipeline.named_steps["pre"]
    return (list(arquivo.nomes_saida_) == list(dentro.nomes_saida_) and arquivo.categorias_ == dentro.categorias_
            and all(np.array_equal(arquivo.pca_[b].components_, dentro.pca_[b].components_)
                    for b in dentro.pca_ if dentro.pca_[b] is not None))


def linhas_do_ajuste(pasta: Path, raiz: Path = RAIZ_PROJETO) -> list[dict]:
    """Linhas do dataset registrado no pacote, na ordem do ajuste; o dataset e a lista de IDs são reconferidos."""
    pacote = json.loads((Path(pasta) / ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    linhas, manifesto = ds.carregar_dataset(Path(raiz) / pacote["fontes"]["dataset"]["pasta"])
    if manifesto["sha256_dataset"] != pacote["fontes"]["dataset"]["sha256"]:
        raise ErroPacote("o dataset em disco não é o registrado no pacote")
    ajuste = json.loads((Path(pasta) / ARQUIVO_IDS).read_text(encoding="utf-8"))["ajuste"]
    por_id = {l["video_id"]: l for l in linhas}
    if sha256_ids(ajuste) != pacote["populacao"]["sha256_ids"]:
        raise ErroPacote("a lista de IDs do ajuste não confere com o hash do pacote")
    return [por_id[v] for v in ajuste]


def verificar_pacote(pasta: Path, raiz: Path = RAIZ_PROJETO) -> dict:
    return verificar_com_linhas(pasta, linhas_do_ajuste(pasta, raiz))


def main_refit_verificacao(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.modelos.refit refit_verificacao", description=__doc__)
    parser.add_argument("--pacote", type=Path, required=True)
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args(argv)
    try:
        resultado = verificar_pacote(args.pacote)
    except (ErroPacote, md.ErroModelo, ds.ErroDataset, ValueError, KeyError, OSError) as erro:
        print(f"Verificação recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    texto = json.dumps(resultado, ensure_ascii=False, indent=1)
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        with open(args.saida, "x", encoding="utf-8") as arquivo:  # nunca sobrescreve
            arquivo.write(texto)
    print(texto)
    return 0 if resultado["aprovado"] else 1


def _script_refit_dados() -> None:
    sys.exit(main())


def _script_refit_treino() -> None:
    sys.exit(main_refit_treino())


def _script_refit_verificacao() -> None:
    sys.exit(main_refit_verificacao())


_COMANDOS = {
    "refit_dados": _script_refit_dados,
    "refit_treino": _script_refit_treino,
    "refit_verificacao": _script_refit_verificacao,
}


def _despachar() -> None:
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in _COMANDOS:
        raise SystemExit("uso: python -m src.modelos.refit {" + "|".join(_COMANDOS) + "} [argumentos]")
    comando = sys.argv.pop(1)
    sys.argv[0] = f"{sys.argv[0]} {comando}"
    _COMANDOS[comando]()


if __name__ == "__main__":
    _despachar()
