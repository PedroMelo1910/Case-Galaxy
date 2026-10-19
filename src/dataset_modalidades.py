"""Datasets de desenvolvimento reduzidos: texto (histórico e legenda) e imagem (histórico, legenda e capa).

Não dependem do Gemini. A população é a da amostra congelada com `uso_no_contexto` igual a `treino` (publicado até o
corte temporal) ou `avaliacao`; o intervalo excluído, os blocos de teste e a LIM nunca entram. Os helpers de
montagem, tipos e classificação de colunas são os do dataset completo, para que a linhagem seja a mesma.

`duracao_s` continua na tabela como coluna de auditoria, mas não é preditor nestes modelos (ver `familias_atributos`).
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src import dataset_desenvolvimento as ds
from src import familias_atributos as fa
from src import snapshot
from src.config import RAIZ_PROJETO

VERSAO = "1"
SNAPSHOT = "snapshot=20261014T021437Z"
LIMITACOES = [
    ("Provisório: população da amostra congelada com uso `treino` ou `avaliacao`; maior que a do modelo completo (que "
     "exige anotação do Gemini), então os resultados não são comparáveis a ele nem constituem ablação multimodal."),
    "Sem informação do vídeo: sem transcrição, sem atributos do Gemini e sem a duração como preditor.",
    "A thumbnail é a lida na coleta e pode ter sido trocada depois da publicação.",
    "Retrospectivo: as curtidas lidas na coleta são acumuladas e não um valor exato aos 30 dias.",
]
NAO_INCLUIDO = ["blocos de teste", "intervalo excluído", "LIM", "estatísticas do próprio vídeo", "atributos do Gemini"]


def _sha256_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def _sha256_arquivo(arquivo: Path) -> str:
    return _sha256_bytes(Path(arquivo).read_bytes())


def _juntar(ids: Sequence[str], fontes: Mapping[str, Mapping[str, Mapping]]) -> list[dict]:
    """Uma linha por vídeo, na ordem de `ids`. Fonte faltante ou coluna repetida é erro, nunca linha descartada."""
    linhas = []
    for video_id in ids:
        ausentes = [nome for nome, tabela in fontes.items() if video_id not in tabela]
        if ausentes:
            raise ds.ErroDataset(f"{video_id}: sem linha em {ausentes}")
        linha: dict = {}
        for tabela in fontes.values():
            novas = {c: v for c, v in tabela[video_id].items() if c != "video_id" or not linha}
            repetidas = (set(novas) & set(linha)) - {"video_id"}
            if repetidas:
                raise ds.ErroDataset(f"{video_id}: colunas repetidas entre fontes: {sorted(repetidas)}")
            linha |= novas
        linhas.append(linha)
    return linhas


def _papeis_originais(tabelas: Sequence[pa.Table]) -> dict[str, str]:
    """Papel de cada coluna das fontes, na ordem de junção (a primeira fonte que traz a coluna vale)."""
    papeis: dict[str, str] = {}
    for tabela in tabelas:
        for coluna, papel in ds._papeis_da_tabela(tabela.schema).items():
            papeis.setdefault(coluna, papel)
    return papeis


def _sem_preditores_nao_operacionais(papeis: Mapping[str, str]) -> dict[str, str]:
    """`duracao_s` fica na tabela, mas como auditoria: não é preditor nestes modelos."""
    return {c: ("auditoria" if c in fa.COLUNAS_NAO_PREDITORAS and p == "atributo" else p) for c, p in papeis.items()}


def montar(*, modalidade: str, tabela_n1: Path, manifesto_n1: Path, pasta_legenda_capa: Path, arquivo_amostra: Path,
           destino_raiz: Path, nome_da_pasta: str | None = None) -> Path:
    """Monta e grava (sem sobrescrever) o dataset reduzido da modalidade; idempotente para o mesmo conteúdo.

    `nome_da_pasta` (padrão `desenvolvimento_<modalidade>=v<versão>`) permite gravar o dataset do refit final."""
    fa.declaracao_da_modalidade(modalidade)  # recusa modalidade sem modelo reduzido, antes de ler qualquer coisa
    usa_capa = "capa" in fa.MODALIDADES[modalidade].blocos
    amostra = json.loads(Path(arquivo_amostra).read_text(encoding="utf-8"))
    ordem = [x["video_id"] for x in amostra["lista"]]
    tab_n1 = pq.read_table(tabela_n1)
    reconciliacao = ds.reconciliar(tab_n1.to_pylist(), anotados=ordem, ordem_amostra=ordem)
    ids = reconciliacao["ids"]
    if not ids:
        raise ds.ErroDataset("nenhuma linha da amostra permitida ao desenvolvimento")
    pasta_lc = Path(pasta_legenda_capa)
    tabelas = {"historico": tab_n1, "legenda": pq.read_table(pasta_lc / "legenda.parquet")}
    if usa_capa:
        tabelas["capa"] = pq.read_table(pasta_lc / "capa.parquet")
    linhas = _juntar(ids, {nome: ds._por_id(tabela) for nome, tabela in tabelas.items()})

    originais = _papeis_originais(list(tabelas.values()))
    originais = {c: originais[c] for c in linhas[0]}
    papeis = _sem_preditores_nao_operacionais(originais)
    tipos_fonte = {c.name: c.type for t in tabelas.values() for c in t.schema}
    tipos = {c: ds._tipo_da_coluna(c, tipos_fonte) for c in papeis}
    base = ds._classificar(originais, tipos)
    especificacao = fa.especificacao_da_modalidade(base, modalidade)
    experimental = fa.especificacao_ate(base, fa.MODALIDADES[modalidade].blocos)  # a congelada: com `duracao_s`
    spec_dict = {"escalares": list(especificacao.escalares), "categoricas": list(especificacao.categoricas),
                 "embeddings": dict(especificacao.embeddings)}
    esquema = pa.schema([pa.field(c, tipos[c], metadata={"papel_coluna": papeis[c]}) for c in papeis],
                        metadata={"tabela": "dataset_modalidade", "versao": VERSAO, "modalidade": modalidade})
    saida = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([{c: l.get(c) for c in papeis} for l in linhas], schema=esquema), saida)
    dados = saida.getvalue().to_pybytes()
    nome = nome_da_pasta or f"desenvolvimento_{modalidade}=v{VERSAO}"
    pasta = Path(destino_raiz) / nome / _sha256_bytes(dados)[:12]
    if pasta.exists():
        return pasta  # mesmo conteúdo já gravado: idempotente, nada é sobrescrito
    manifesto = _manifesto(modalidade, linhas, papeis, spec_dict, (especificacao, experimental), reconciliacao, ids,
                           dados, Path(tabela_n1), Path(manifesto_n1), pasta_lc, amostra, usa_capa)
    pasta.parent.mkdir(parents=True, exist_ok=True)
    parcial = Path(tempfile.mkdtemp(prefix=".dataset.", suffix=".parcial", dir=pasta.parent))
    try:
        (parcial / ds.ARQUIVO_DATASET).write_bytes(dados)
        (parcial / ds.ARQUIVO_MANIFESTO).write_text(json.dumps(manifesto, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
        parcial.rename(pasta)
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    return pasta


def _manifesto(modalidade, linhas, papeis, spec_dict, especificacoes, reconciliacao, ids, dados, tabela_n1,
               manifesto_n1, pasta_lc, amostra, usa_capa) -> dict:
    operacional, experimental = especificacoes
    man_n1 = json.loads(manifesto_n1.read_text(encoding="utf-8"))
    man_lc = json.loads((pasta_lc / "manifesto.json").read_text(encoding="utf-8"))
    retencoes = [man_n1["retencao"]["apagar_ate_utc"], man_lc["retencao"]["apagar_ate_utc"],
                 amostra["retencao"]["apagar_ate_utc"]]
    por_papel = {p: [c for c, q in papeis.items() if q == p] for p in
                 ("identificacao", "alvo", "referencia", "atributo", "auditoria", "explicacao")}
    entradas = {
        "base": man_n1["base"], "particao": man_n1["particao"], "contexto": man_n1["contexto"],
        "defasagem_dias": man_n1["defasagem_dias"],
        "historico": {"sha256": _sha256_arquivo(tabela_n1), "sha256_manifesto": _sha256_arquivo(manifesto_n1)},
        "legenda": {"sha256": _sha256_arquivo(pasta_lc / "legenda.parquet")},
        "legenda_capa": {"sha256_manifesto": _sha256_arquivo(pasta_lc / "manifesto.json")},
        "amostra": {"sha256_lista": amostra["sha256_lista"], "n_planejado": amostra["selecionados"]},
    }
    if usa_capa:
        entradas["capa"] = {"sha256": _sha256_arquivo(pasta_lc / "capa.parquet")}
    estados = ("leg_estado", "capa_estado_imagem", "cor_estado", "rosto_estado", "ocr_estado", "clip_estado",
               "estado_referencia")
    return {
        "versao": VERSAO, "tipo": "dataset_modalidade", "modalidade": modalidade,
        "gerado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256_dataset": _sha256_bytes(dados), "n_linhas": len(linhas), "entradas": entradas,
        "reconciliacao": {
            "na_amostra": reconciliacao["anotados"], "fora_do_n1": reconciliacao["fora_do_n1_teste_ou_lim"],
            "excluidos_por_uso": reconciliacao["excluidos_por_uso"],
            "sem_alvo_ou_referencia": reconciliacao["sem_alvo_ou_referencia"], "por_uso": reconciliacao["por_uso"],
            "n_ids": len(ids), "sha256_ids": _sha256_bytes("\n".join(ids).encode("utf-8")),
            "nota": "fora_do_n1 são os blocos de teste protegidos, contados só por ID; nenhum rótulo foi lido"},
        "populacao": {
            "uso_no_contexto": ds._contar(linhas, "uso_no_contexto"), "setor": ds._contar(linhas, "setor"),
            "tipo_canal": ds._contar(linhas, "tipo_canal"), "n_canais": len({l["channel_id"] for l in linhas}),
            "canais_por_uso": {u: len({l["channel_id"] for l in linhas if l["uso_no_contexto"] == u})
                               for u in ds.USOS_PERMITIDOS}},
        "estados_de_extracao": {c: ds._contar(linhas, c) for c in estados if c in papeis},
        "ausencias_nos_atributos": {c: sum(l.get(c) is None for l in linhas) for c in papeis
                                    if papeis[c] == "atributo"},
        "colunas_por_papel": por_papel, "especificacao": spec_dict,
        "especificacao_sha256": fa.hash_da_especificacao(operacional),
        "especificacao_experimental_sha256": fa.hash_da_especificacao(experimental),
        "adaptacao_operacional": fa.declaracao_da_modalidade(modalidade),
        "politica_de_ausencia": "ausência permanece nula; sem imputação por zero, média ou vetor fabricado. O "
        "LightGBM trata NaN nativamente; PCA e códigos de categoria são ajustados só no treino do pipeline.",
        "codigo": snapshot.estado_do_codigo(),
        "retencao": {"apagar_ate_utc": min(retencoes), "herdada_de": "o menor prazo entre histórico, legenda/capa e "
                     "amostra de anotação"},
        "limitacoes": LIMITACOES, "nao_incluido": NAO_INCLUIDO,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """0 = dataset gravado; 2 = entrada recusada."""
    parser = argparse.ArgumentParser(prog="python -m src.dataset_modalidades", description=__doc__)
    parser.add_argument("--modalidade", choices=sorted(fa.MODALIDADES), required=True)
    parser.add_argument("--n1", type=Path, default=RAIZ_PROJETO / "data/features" / SNAPSHOT
                        / "historico=v1/defasagem=30/contexto=validacao")
    parser.add_argument("--legenda-capa", type=Path,
                        default=RAIZ_PROJETO / "data/features" / SNAPSHOT / "legenda_capa=v1")
    parser.add_argument("--amostra", type=Path,
                        default=RAIZ_PROJETO / "data/annotations/gemini" / SNAPSHOT / "amostra=v2")
    parser.add_argument("--destino", type=Path, default=RAIZ_PROJETO / "data/features" / SNAPSHOT)
    args = parser.parse_args(argv)
    try:
        pasta = montar(modalidade=args.modalidade, tabela_n1=args.n1 / "n1.parquet",
                       manifesto_n1=args.n1 / "manifesto.json", pasta_legenda_capa=args.legenda_capa,
                       arquivo_amostra=args.amostra / "amostra.json", destino_raiz=args.destino)
    except (ds.ErroDataset, fa.ErroFamilia, ValueError, KeyError, OSError) as erro:
        print(f"Dataset recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({"pasta": str(pasta)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
