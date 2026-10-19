"""Dados do refit final: contexto "final", histórico (N1) e datasets de vídeo, imagem e texto (D-D7, ARQUITETURA §6.5).

O refit é um treino pré-determinado, não uma avaliação nem uma consulta ao teste: ajusta com TODOS os vídeos-alvo
anotados da amostra congelada (blocos de treino, validação, intervalo excluído e teste), com a LIM inteira fora. Por isso
o contexto "final" não tem bloco avaliado, não tem canal excluído e tem `treino_ate_utc` igual à última publicação
entre os alvos da amostra: toda linha é `treino` e nenhuma cai no intervalo excluído. As curtidas dos blocos de teste
só entram aqui como rótulo do ajuste; nenhuma métrica de desempenho é calculada (e a cobertura gravada nos pacotes vem
do procedimento de avaliação já encerrado, não do artefato).

Nada aqui abre o portão da consulta ao teste (`exigir_portao`, `ConsultaAberta`, marcadores `consulta_teste*`): a
consulta única já foi consumida e este módulo não a repete. O histórico e as referências são recalculados com o mesmo
código do desenvolvimento (`historico.calcular`, defasagem de 30 dias) e os datasets pelos mesmos montadores
(`dataset_desenvolvimento.montar`, `dataset_modalidades.montar`). Linhas sem `ref_log` ou `y_rel` ficam fora do ajuste
e são contadas. Idempotente: o mesmo conteúdo reaproveita a pasta existente e nada é sobrescrito.

Uso (de `Entrega Galaxy`, com HF_HUB_OFFLINE=1): `python -m src.refit_dados`.
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from src import dataset_desenvolvimento as ds
from src import dataset_modalidades as dm
from src import historico as hs
from src import particoes
from src.avaliacao import contextos_finais as cf
from src.cache_gemini import gravar_json_atomico
from src.config import RAIZ_PROJETO

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


def _iso(instante: datetime) -> str:
    return instante.astimezone(UTC).strftime(FORMATO_UTC)


def _sha256_arquivo(arquivo: Path) -> str:
    return hashlib.sha256(Path(arquivo).read_bytes()).hexdigest()


def _sha256_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


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
        pastas[modalidade] = dm.montar(modalidade=modalidade, **comuns, nome_da_pasta=NOMES_DE_PASTA[modalidade])
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
    from src import extracao_legenda as leg

    features = RAIZ_PROJETO / "data/features" / SNAPSHOT
    parser = argparse.ArgumentParser(prog="python -m src.refit_dados", description=__doc__)
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


if __name__ == "__main__":
    sys.exit(main())
