"""Driver da avaliação final: recalcula do disco todos os hashes, congela, abre o portão e executa (revisão ML, C1).

Este módulo faz parte do hash do protocolo. Nada aqui lê rótulo de teste por conta própria: os hashes vêm de bytes e de
manifestos; as curtidas de teste só são lidas dentro de `contextos_finais.materializar_contexto`, e só com a prova do
portão (`ConsultaAberta`). Etapas, cada uma um subcomando:

    hashes     mostra os hashes recalculados do disco (nenhum rótulo é lido)
    congelar   monta e grava `protocolo_final.json` + SHA-256 (depois, registrar a revisão do mle-reviewer)
    revisao    registra a passagem única do `ecc:mle-reviewer`
    executar   confere TUDO de novo, abre o portão e roda os quatro braços (consulta única por partição)
    retomar    retomada técnica depois da consulta iniciada: só módulos de orquestração, com registro ligado à
               tentativa original; resultados novos em `resultados_retomada_NN`, os originais ficam intocados
"""

import argparse
import hashlib
import importlib.metadata as metadados
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src import dataset_desenvolvimento as ds
from src import extracao_legenda as leg
from src import familias_atributos as fa
from src.avaliacao import baseline_inscritos as bi
from src.avaliacao import canais_retidos as cr
from src.avaliacao import entradas_finais as ef
from src.avaliacao import final as fn
from src.avaliacao import fornecedor_real as fr
from src.avaliacao import protocolo_final as pf

BIBLIOTECAS = ("lightgbm", "scikit-learn", "numpy", "scipy", "pyarrow")
PREFIXO_TESTE = "teste"


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
                      for nivel, espec in fn.especs_ablacao(manifesto["especificacao"]).items()}
    entradas = ef.hashes_das_entradas(
        caches_de_anotacao=[(Path(c.pasta_amostra) / "cache", ids_de_teste(c)),
                            (Path(c.pasta_lim) / "cache", ids_da_lim(c))],
        pasta_legenda_capa=c.pasta_legenda_capa, cache_embeddings=c.cache_embeddings,
        tabela_inscritos=Path(c.pasta_base) / bi.TABELA_CANAIS_RESTRITA, uv_lock=Path(c.produto) / "uv.lock")
    return {"hash_dataset_atual": manifesto["sha256_dataset"],
            "hash_amostra_atual": ef.hash_do_arquivo(Path(c.pasta_amostra) / "amostra.json"),
            "hash_particao_atual": ef.hash_do_arquivo(c.arquivo_particao),
            "hash_lim_atual": ef.hash_do_arquivo(Path(c.pasta_lim) / "amostra.json"),
            "bibliotecas_atuais": {b: metadados.version(b) for b in BIBLIOTECAS},
            "hashes_especificacoes_atuais": especificacoes, "hashes_entradas_atuais": dict(entradas)}


def configuracao_do_projeto(*, coleta: datetime, arvore: Caminhos) -> fn.Configuracao:
    """A configuração do protocolo: defaults do projeto, os três sorteios de canais retidos e a tabela de inscritos."""
    return fn.Configuracao(coleta=coleta, sorteios=cr.tres_sorteios(_canais(arvore)),
                           inscritos=bi.carregar_inscritos(arvore.pasta_base))


def anotacoes_ausentes(c: Caminhos) -> dict[str, list[str]]:
    return {"teste": ef.hash_das_anotacoes(Path(c.pasta_amostra) / "cache", ids_de_teste(c))["ausentes"],
            "lim": ef.hash_das_anotacoes(Path(c.pasta_lim) / "cache", ids_da_lim(c))["ausentes"]}


def extras_de_arquivos(arquivos: Mapping[str, Path]) -> dict[str, dict]:
    """Artefatos que sustentam decisões (pré-registro, veredito, limiar da LIM): nome do arquivo e SHA-256 dos bytes."""
    return {nome: {"arquivo": Path(caminho).name, "sha256": ef.hash_do_arquivo(Path(caminho))}
            for nome, caminho in arquivos.items()}


def congelar_o_protocolo(c: Caminhos, cfg: fn.Configuracao, *, extras: Mapping | None = None) -> Path:
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


def abrir(c: Caminhos, cfg: fn.Configuracao) -> pf.ConsultaAberta:
    """Fase 1: recalcula todos os hashes, confere a configuração e cria o marcador. Nenhum rótulo é lido."""
    return pf.abrir_consulta_ao_teste(
        c.pasta_protocolo, raiz=c.produto, registro_global=c.registro_global,
        validar_configuracao=lambda definicoes: fn.exigir_configuracao_congelada(cfg, definicoes), **hashes_atuais(c))


ARQUIVO_COPIA_DO_CACHE = "proveniencia_cache_embeddings.copia.json"
PASTA_CACHE_DE_TRABALHO = "cache_embeddings_trabalho"


def _arquivos(pasta: Path) -> dict[str, str]:
    """Caminho relativo -> SHA-256 de cada arquivo da pasta."""
    return {a.relative_to(pasta).as_posix(): ef.hash_do_arquivo(a) for a in sorted(Path(pasta).rglob("*")) if a.is_file()}


def cache_de_trabalho(c: Caminhos) -> Path:
    """Cópia gravável do cache de embeddings: a execução final escreve nela e NUNCA no cache de origem, cujo hash
    pertence ao protocolo congelado. Na retomada técnica a cópia existente é reaproveitada (nada é recopiado)."""
    origem, trabalho = Path(c.cache_embeddings), Path(c.destino) / PASTA_CACHE_DE_TRABALHO
    if trabalho.exists():
        return trabalho
    Path(c.destino).mkdir(parents=True, exist_ok=True)
    hash_origem = ef.hash_da_pasta(origem)
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
    agora = ef.hash_da_pasta(origem)
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


def abrir_retomada(c: Caminhos, cfg: fn.Configuracao, *, modulos_reparados: Sequence[str], motivo: str,
                   autorizacao: str, declaracao: str) -> pf.ConsultaAberta:
    """Retomada técnica: só com método, alvo, seleção e população inalterados; o marcador original fica como está."""
    return pf.abrir_retomada_tecnica(
        c.pasta_protocolo, raiz=c.produto, registro_global=c.registro_global, modulos_reparados=modulos_reparados,
        motivo=motivo, autorizacao=autorizacao, declaracao_metodo_inalterado=declaracao,
        validar_configuracao=lambda definicoes: fn.exigir_configuracao_congelada(cfg, definicoes), **hashes_atuais(c))


def pasta_de_resultados(c: Caminhos, *, retomada: bool = False) -> Path:
    """`resultados` da tentativa original; `resultados_retomada_NN` da retomada mais recente (nunca sobrescreve)."""
    if not retomada:
        return Path(c.pasta_protocolo) / "resultados"
    n = len(list(Path(c.pasta_protocolo).glob("retomada_tecnica_*.json")))
    if n == 0:
        raise ValueError("não há registro de retomada técnica neste protocolo")
    return Path(c.pasta_protocolo) / f"resultados_retomada_{n:02d}"


def executar_final(c: Caminhos, cfg: fn.Configuracao, *, codificador, carregar_anotacoes,
                   retomada: Mapping | None = None) -> dict:
    """Consulta única: abre o portão e roda os quatro braços, gravando cada um ao terminar.

    Com `retomada` (módulos reparados, motivo, autorização, declaração), segue o caminho da retomada técnica: os
    resultados da tentativa original ficam intocados e os novos vão para `resultados_retomada_NN`."""
    consulta = abrir_retomada(c, cfg, **retomada) if retomada else abrir(c, cfg)
    fornecedor = fr.FornecedorReal(
        pasta_base=c.pasta_base, arquivo_particao=c.arquivo_particao, destino=c.destino,
        pasta_legenda_capa=c.pasta_legenda_capa, arquivo_amostra=Path(c.pasta_amostra) / "amostra.json",
        carregar_anotacoes=carregar_anotacoes, codificador=codificador, cache_embeddings=cache_de_trabalho(c),
        consulta=consulta)
    hashes_ablacao = hashes_atuais(c)["hashes_especificacoes_atuais"]
    try:
        relatorio = fn.executar(fornecedor, cfg, consulta=consulta, hashes_ablacao=hashes_ablacao,
                                salvar=fn.gravador_de_bracos(pasta_de_resultados(c, retomada=bool(retomada))))
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
