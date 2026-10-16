"""Dataset de desenvolvimento: junta histórico, legenda, capa e atributos do Gemini por `video_id`.

Só entram linhas que o contexto de desenvolvimento permite: `treino` (publicado até o corte) para ajustar e
`avaliacao` para validar. Intervalo excluído, canais retidos, blocos de teste e LIM nunca são selecionados.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src import extracao_legenda as leg
from src import rubrica as rb
from src import snapshot
from src.cache_gemini import gravar_json_atomico
from src.config import RAIZ_PROJETO

USOS_PERMITIDOS = ("treino", "avaliacao")


class ErroDataset(ValueError):
    """Entrada incoerente com o contrato do dataset de desenvolvimento."""


def _tem_alvo(linha: Mapping) -> bool:
    return linha.get("y_rel") is not None and linha.get("ref_log") is not None


def reconciliar(linhas_n1: Iterable[Mapping], *, anotados: Iterable[str], ordem_amostra: Sequence[str]) -> dict:
    """Reconcilia as anotações com as linhas do N1 e devolve os IDs aptos, na ordem congelada da amostra."""
    anotados, posicao = set(anotados), {v: i for i, v in enumerate(ordem_amostra)}
    fora = sorted(anotados - posicao.keys())
    if fora:
        raise ErroDataset(f"anotações fora da amostra congelada: {fora[:5]}")
    por_id = {linha["video_id"]: linha for linha in linhas_n1}
    aptos, usos, sem_alvo, excluidos = [], Counter(), 0, Counter()
    for video_id in sorted(anotados):
        linha = por_id.get(video_id)
        if linha is None:
            excluidos["fora_do_n1_teste_ou_lim"] += 1
        elif linha["uso_no_contexto"] not in USOS_PERMITIDOS:
            excluidos[linha["uso_no_contexto"]] += 1
        elif not _tem_alvo(linha):
            sem_alvo += 1
            usos[linha["uso_no_contexto"]] += 1
        else:
            aptos.append(video_id)
            usos[linha["uso_no_contexto"]] += 1
    return {
        "ids": sorted(aptos, key=posicao.__getitem__),
        "anotados": len(anotados),
        "fora_do_n1_teste_ou_lim": excluidos.pop("fora_do_n1_teste_ou_lim", 0),
        "excluidos_por_uso": dict(sorted(excluidos.items())),
        "sem_alvo_ou_referencia": sem_alvo,
        "por_uso": dict(usos),
    }


def selecionar_ids(linhas_n1: Iterable[Mapping], *, anotados: Iterable[str], ordem_amostra: Sequence[str]) -> list[str]:
    return reconciliar(linhas_n1, anotados=anotados, ordem_amostra=ordem_amostra)["ids"]


# --- atributos do Gemini ----------------------------------------------------------------------------------------



NOTAS = tuple(c.nome for c in rb.CAMPOS_COMUNS if c.tipo == "nota")
SIM_NAO = tuple(dict.fromkeys(c.nome for campos in (*rb.CAMPOS_POR_SETOR.values(), rb.CAMPOS_COMUNS) for c in campos
                              if c.tipo == "sim_nao"))
CATEGORICAS_GEMINI = ("tom_narracao",)
TEXTO_LIVRE = ("hook_primeiros_3s", "cor_dominante", "elementos_video_top5", "elementos_audio_top5",
               "transcricao_texto", "transcricao_idioma")
COLUNAS_TRANSCRICAO = ("gem_transc_status", "gem_transc_n_caracteres", "gem_transc_n_palavras")
COLUNA_EMBEDDING_TRANSCRICAO = "gem_transc_emb"
COLUNA_ESTADO_EMBEDDING = "gem_transc_emb_estado"

COLUNAS_GEMINI = (
    *(f"gem_{c}" for c in (*NOTAS, *SIM_NAO, *CATEGORICAS_GEMINI)), *COLUNAS_TRANSCRICAO,
)
PAPEL_DAS_COLUNAS: dict[str, str] = {
    **dict.fromkeys(COLUNAS_GEMINI, "atributo"),
    COLUNA_EMBEDDING_TRANSCRICAO: "atributo",
    COLUNA_ESTADO_EMBEDDING: "auditoria",
    **{f"exp_{c}": "explicacao" for c in TEXTO_LIVRE},
}
PROIBIDAS_COMO_ATRIBUTO = frozenset({
    "video_id", "channel_id", "published_at", "bloco", "uso_no_contexto", "y_log", "y_rel", "ref_log",
    "pred_media_aritmetica", "pred_mediana", "pred_mme_log", "curtidas", "visualizacoes", "comentarios",
    "inscritos", "total_videos", "topic_details", "topicDetails", "mes",
})


def conferir_papeis(atributos: Sequence[str]) -> None:
    """Recusa atributo proibido (identificação, rótulo, referência, estatística do próprio vídeo) ou repetido."""
    invalidas = [c for c in atributos if c in PROIBIDAS_COMO_ATRIBUTO]
    if invalidas or len(set(atributos)) != len(atributos):
        raise ErroDataset(f"colunas não permitidas como atributo: {invalidas or 'repetidas'}")


def _palavras(texto: str) -> int:
    return sum(any(c.isalnum() for c in parte) for parte in texto.split())


def linha_gemini(anotacao: Mapping, setor: str) -> dict:
    """Colunas do Gemini de um vídeo. Desconhecido fica nulo (nunca 0); campo de outro setor fica nulo."""
    linha: dict = {c: None for c in (*COLUNAS_GEMINI, COLUNA_EMBEDDING_TRANSCRICAO)}
    for campo in (*NOTAS, *SIM_NAO, *CATEGORICAS_GEMINI):
        linha[f"gem_{campo}"] = anotacao.get(campo)
    status = anotacao["transcricao_status"]
    texto = leg.normalizar_texto(anotacao.get("transcricao_texto"))
    linha["gem_transc_status"] = status
    if status != "sem_fala" and texto:
        linha["gem_transc_n_caracteres"], linha["gem_transc_n_palavras"] = len(texto), _palavras(texto)
    for campo in TEXTO_LIVRE:
        valor = anotacao.get(campo)
        linha[f"exp_{campo}"] = json.dumps(valor, ensure_ascii=False) if isinstance(valor, (list, tuple)) else valor
    linha["setor_anotacao"] = setor
    linha[COLUNA_ESTADO_EMBEDDING] = "sem_texto" if status == "sem_fala" or not texto else None
    return linha


def _chave_texto(identificacao: Mapping, texto: str) -> str:
    base = json.dumps(identificacao, sort_keys=True, ensure_ascii=False) + "\n" + texto
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def embeddings_transcricao(textos: Mapping[str, str], codificador: leg.Codificador, pasta_cache: Path) -> dict:
    """Embedding da transcrição com o mesmo codificador da legenda; cache por SHA-256 do texto e do codificador.

    Texto vazio não é codificado (`sem_texto`); falha do codificador fica explícita e não entra no cache.
    """
    pasta_cache, identificacao = Path(pasta_cache), codificador.identificacao()
    resultados: dict[str, dict] = {}
    pendentes: dict[str, list[str]] = {}
    for video_id, bruto in textos.items():
        texto = leg.normalizar_texto(bruto)
        if not texto:
            resultados[video_id] = {"estado": "sem_texto", "vetor": None}
            continue
        arquivo = pasta_cache / f"{_chave_texto(identificacao, texto)}.json"
        if arquivo.exists():
            try:
                dados = json.loads(arquivo.read_text(encoding="utf-8"))
                vetor = leg.validar_vetor(dados["vetor"], leg.DIMENSAO_TEXTO)
            except (ValueError, KeyError, OSError) as falha:
                raise ErroDataset(f"cache de embedding ilegível: {arquivo.name} ({type(falha).__name__})") from falha
            resultados[video_id] = {"estado": "ok", "vetor": vetor}
        else:
            pendentes.setdefault(texto, []).append(video_id)
    unicos = list(pendentes)
    try:
        vetores = codificador.codificar(unicos) if unicos else []
        if len(vetores) != len(unicos):
            raise ErroDataset("o codificador devolveu menos vetores que textos")
        erro = None
    except Exception as falha:  # noqa: BLE001 - falha registrada com o tipo, sem vetor inventado
        vetores, erro = [], type(falha).__name__
    for posicao, texto in enumerate(unicos):
        for video_id in pendentes[texto]:
            if erro is not None:
                resultados[video_id] = {"estado": "falha_extrator", "vetor": None, "erro": erro}
                continue
            vetor = leg.validar_vetor(vetores[posicao].vetor, leg.DIMENSAO_TEXTO)
            resultados[video_id] = {"estado": "ok", "vetor": vetor}
        if erro is None:
            gravar_json_atomico(pasta_cache / f"{_chave_texto(identificacao, texto)}.json",
                                {"codificador": identificacao, "vetor": list(vetores[posicao].vetor)})
    return resultados


# --- junção -----------------------------------------------------------------------------------------------------


def juntar(ids: Sequence[str], *, n1: Mapping[str, Mapping], legenda: Mapping[str, Mapping],
           capa: Mapping[str, Mapping], gemini: Mapping[str, Mapping]) -> list[dict]:
    """Uma linha por vídeo, na ordem de `ids`. Fonte faltante ou setor divergente é erro, nunca linha descartada."""
    linhas = []
    for video_id in ids:
        faltando = [nome for nome, fonte in (("historico", n1), ("legenda", legenda), ("capa", capa),
                                             ("gemini", gemini)) if video_id not in fonte]
        if faltando:
            raise ErroDataset(f"{video_id}: sem linha em {faltando}")
        linha = dict(n1[video_id])
        if gemini[video_id]["setor_anotacao"] != linha["setor"]:
            raise ErroDataset(f"{video_id}: setor da anotação difere do histórico")
        for fonte in (legenda[video_id], capa[video_id], {k: v for k, v in gemini[video_id].items()
                                                          if k != "setor_anotacao"}):
            repetidas = (set(fonte) & set(linha)) - {"video_id"}
            if repetidas:
                raise ErroDataset(f"{video_id}: colunas repetidas entre fontes: {sorted(repetidas)}")
            linha |= {c: v for c, v in fonte.items() if c != "video_id"}
        linhas.append(linha)
    return linhas



# --- montagem, gravação e leitura ---------------------------------------------------------------------------------




VERSAO = "1"
COMPONENTES_PCA = 8
ARQUIVO_DATASET, ARQUIVO_MANIFESTO = "dataset.parquet", "manifesto.json"
BLOCOS_EMBEDDING = {"leg_emb": leg.DIMENSAO_TEXTO, "capa_emb": 512, COLUNA_EMBEDDING_TRANSCRICAO: leg.DIMENSAO_TEXTO}
_PAPEL_DA_FONTE = {"auxiliar": "explicacao"}
LIMITACOES = [
    ("Provisório: população só dos vídeos anotados pelo Gemini e permitidos ao desenvolvimento; não é amostra "
     "aleatória nova nem a população final de treino."),
    "Atributos do Gemini são observações do conteúdo feitas por um modelo de visão; erros de anotação entram como ruído.",
    "A thumbnail é a lida na coleta e pode ter sido trocada depois da publicação.",
    "Retrospectivo: as curtidas lidas na coleta são acumuladas e não um valor exato aos 30 dias.",
]


def _sha256_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def _papeis_da_tabela(esquema: pa.Schema) -> dict[str, str]:
    return {c.name: _PAPEL_DA_FONTE.get(p := c.metadata[b"papel_coluna"].decode(), p) for c in esquema}


def _por_id(tabela: pa.Table) -> dict[str, dict]:
    return {linha["video_id"]: linha for linha in tabela.to_pylist()}


def _tipo_da_coluna(nome: str, tipos_fonte: Mapping[str, pa.DataType]) -> pa.DataType:
    if nome in BLOCOS_EMBEDDING:
        return pa.list_(pa.float32(), BLOCOS_EMBEDDING[nome])
    if nome in tipos_fonte:
        return tipos_fonte[nome]
    texto = nome.startswith("exp_") or nome in (COLUNA_ESTADO_EMBEDDING, "gem_transc_status", "setor_anotacao")
    texto = texto or any(nome == f"gem_{c}" for c in (*SIM_NAO, *CATEGORICAS_GEMINI))
    return pa.string() if texto else pa.float64()


def _classificar(papeis: Mapping[str, str], tipos: Mapping[str, pa.DataType]) -> dict:
    atributos = [c for c, p in papeis.items() if p == "atributo"]
    conferir_papeis(atributos)
    embeddings = {c: COMPONENTES_PCA for c in atributos if c in BLOCOS_EMBEDDING}
    categoricas = [c for c in atributos if c not in embeddings and pa.types.is_string(tipos[c])]
    escalares = [c for c in atributos if c not in embeddings and c not in categoricas]
    return {"escalares": escalares, "categoricas": categoricas, "embeddings": embeddings}


def montar(*, tabela_n1: Path, manifesto_n1: Path, pasta_legenda_capa: Path, arquivo_amostra: Path,
           carregar_anotacoes: Callable[[Sequence[str]], Mapping[str, Mapping]], codificador: leg.Codificador,
           cache_embeddings: Path, destino_raiz: Path) -> Path:
    """Monta e grava (sem sobrescrever) o dataset de desenvolvimento das linhas anotadas e permitidas."""
    amostra = json.loads(Path(arquivo_amostra).read_text(encoding="utf-8"))
    ordem = [x["video_id"] for x in amostra["lista"]]
    anotacoes = carregar_anotacoes(ordem)  # só as válidas e compatíveis: id → {"setor", "anotacao"}
    tab_n1 = pq.read_table(tabela_n1)
    reconciliacao = reconciliar(tab_n1.to_pylist(), anotados=list(anotacoes), ordem_amostra=ordem)
    ids = reconciliacao["ids"]
    if not ids:
        raise ErroDataset("nenhuma linha anotada e permitida ao desenvolvimento")
    tab_leg = pq.read_table(Path(pasta_legenda_capa) / "legenda.parquet")
    tab_capa = pq.read_table(Path(pasta_legenda_capa) / "capa.parquet")
    n1, legenda, capa = _por_id(tab_n1), _por_id(tab_leg), _por_id(tab_capa)
    transcricoes = embeddings_transcricao(
        {v: ("" if anotacoes[v]["anotacao"]["transcricao_status"] == "sem_fala"
             else anotacoes[v]["anotacao"].get("transcricao_texto", "")) for v in ids},
        codificador, Path(cache_embeddings))
    falhas = [v for v, r in transcricoes.items() if r["estado"] == "falha_extrator"]
    if falhas:
        raise ErroDataset(f"codificador falhou em {len(falhas)} transcrições; dataset não gravado")
    gemini = {}
    for v in ids:
        linha = linha_gemini(anotacoes[v]["anotacao"], anotacoes[v]["setor"])
        emb = transcricoes[v]
        linha[COLUNA_EMBEDDING_TRANSCRICAO] = emb["vetor"]
        linha[COLUNA_ESTADO_EMBEDDING] = emb["estado"]
        gemini[v] = linha
    linhas = juntar(ids, n1=n1, legenda=legenda, capa=capa, gemini=gemini)

    papeis = {**_papeis_da_tabela(tab_n1.schema), **_papeis_da_tabela(tab_leg.schema),
              **_papeis_da_tabela(tab_capa.schema), **PAPEL_DAS_COLUNAS, "setor_anotacao": "auditoria"}
    papeis = {c: papeis[c] for c in linhas[0]}
    tipos_fonte = {c.name: c.type for t in (tab_n1, tab_leg, tab_capa) for c in t.schema}
    tipos = {c: _tipo_da_coluna(c, tipos_fonte) for c in papeis}
    especificacao = _classificar(papeis, tipos)
    esquema = pa.schema([pa.field(c, tipos[c], metadata={"papel_coluna": papeis[c]}) for c in papeis],
                        metadata={"tabela": "dataset_desenvolvimento", "versao": VERSAO})
    saida = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([{c: l.get(c) for c in papeis} for l in linhas], schema=esquema), saida)
    dados = saida.getvalue().to_pybytes()
    pasta = Path(destino_raiz) / f"desenvolvimento=v{VERSAO}" / _sha256_bytes(dados)[:12]
    if pasta.exists():
        return pasta  # mesmo conteúdo já gravado: idempotente, nada é sobrescrito
    manifesto = _manifesto(linhas, papeis, especificacao, reconciliacao, ids, dados, tabela_n1, manifesto_n1,
                           pasta_legenda_capa, amostra, codificador)
    pasta.parent.mkdir(parents=True, exist_ok=True)
    parcial = Path(tempfile.mkdtemp(prefix=".dataset.", suffix=".parcial", dir=pasta.parent))
    try:
        (parcial / ARQUIVO_DATASET).write_bytes(dados)
        (parcial / ARQUIVO_MANIFESTO).write_text(json.dumps(manifesto, ensure_ascii=False, indent=1), encoding="utf-8")
        parcial.rename(pasta)
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    return pasta


def _contar(linhas: Sequence[Mapping], coluna: str) -> dict:
    return dict(sorted(Counter(str(l.get(coluna)) for l in linhas).items()))


def _manifesto(linhas, papeis, especificacao, reconciliacao, ids, dados, tabela_n1, manifesto_n1,
               pasta_legenda_capa, amostra, codificador) -> dict:
    man_n1 = json.loads(Path(manifesto_n1).read_text(encoding="utf-8"))
    man_lc = json.loads((Path(pasta_legenda_capa) / "manifesto.json").read_text(encoding="utf-8"))
    retencoes = [man_n1["retencao"]["apagar_ate_utc"], man_lc["retencao"]["apagar_ate_utc"],
                 amostra["retencao"]["apagar_ate_utc"]]
    por_papel = {p: [c for c, q in papeis.items() if q == p] for p in
                 ("identificacao", "alvo", "referencia", "atributo", "auditoria", "explicacao")}
    return {
        "versao": VERSAO, "tipo": "dataset_desenvolvimento", "gerado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256_dataset": _sha256_bytes(dados), "n_linhas": len(linhas),
        "entradas": {
            "base": man_n1["base"], "particao": man_n1["particao"], "contexto": man_n1["contexto"],
            "defasagem_dias": man_n1["defasagem_dias"],
            "historico": {"sha256": _sha256_bytes(Path(tabela_n1).read_bytes()),
                          "sha256_manifesto": _sha256_bytes(Path(manifesto_n1).read_bytes())},
            "legenda_capa": {"tabelas": man_lc["tabelas"], "populacao": man_lc["populacao"],
                             "configuracoes": {k: (v or {}).get("chave") for k, v in man_lc["configuracoes"].items()}},
            "amostra": {"sha256_lista": amostra["sha256_lista"], "n_planejado": amostra["selecionados"]},
            "codificador_transcricao": codificador.identificacao(),
        },
        "reconciliacao_anotacoes": {k: v for k, v in reconciliacao.items() if k != "ids"}
        | {"n_ids": len(ids), "sha256_ids": _sha256_bytes("\n".join(ids).encode("utf-8"))},
        "populacao": {
            "uso_no_contexto": _contar(linhas, "uso_no_contexto"), "setor": _contar(linhas, "setor"),
            "tipo_canal": _contar(linhas, "tipo_canal"), "n_canais": len({l["channel_id"] for l in linhas}),
            "canais_por_uso": {u: len({l["channel_id"] for l in linhas if l["uso_no_contexto"] == u})
                               for u in USOS_PERMITIDOS},
        },
        "estados_de_extracao": {c: _contar(linhas, c) for c in
                                ("leg_estado", "capa_estado_imagem", "cor_estado", "rosto_estado", "ocr_estado",
                                 "clip_estado", COLUNA_ESTADO_EMBEDDING, "gem_transc_status", "estado_referencia")
                                if c in papeis},
        "ausencias_nos_atributos": {c: sum(l.get(c) is None for l in linhas) for c in papeis
                                    if papeis[c] == "atributo"},
        "colunas_por_papel": por_papel, "especificacao": especificacao,
        "politica_de_ausencia": "ausência permanece nula; sem imputação por zero, média ou vetor fabricado. O "
        "LightGBM trata NaN nativamente; PCA e códigos de categoria são ajustados só no treino do pipeline.",
        "codigo": snapshot.estado_do_codigo(),
        "retencao": {"apagar_ate_utc": min(retencoes), "herdada_de": "o menor prazo entre histórico, legenda/capa e "
                     "amostra de anotação; cópias reaproveitadas do piloto podem ter prazo anterior"},
        "limitacoes": LIMITACOES,
        "nao_incluido": ["blocos de teste", "intervalo excluído", "LIM", "estatísticas do próprio vídeo"],
    }


def carregar_dataset(pasta: Path) -> tuple[list[dict], dict]:
    pasta = Path(pasta)
    manifesto = json.loads((pasta / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))
    dados = (pasta / ARQUIVO_DATASET).read_bytes()
    if _sha256_bytes(dados) != manifesto["sha256_dataset"]:
        raise ErroDataset("dataset não confere com o hash do manifesto")
    return pq.read_table(pa.BufferReader(dados)).to_pylist(), manifesto


def _anotacoes_do_cache(pasta_amostra: Path) -> Callable[[Sequence[str]], dict]:
    from src import anotacao
    from src.cache_gemini import CacheAnotacoes

    def carregar(ordem: Sequence[str]) -> dict:
        cache = CacheAnotacoes(Path(pasta_amostra) / "cache", anotacao.configuracao())
        saida = {}
        for video_id in ordem:
            resultado = cache.ler(video_id)
            if resultado is not None and resultado.estado == "ok" and resultado.anotacao is not None:
                saida[video_id] = {"setor": str(resultado.setor), "anotacao": dict(resultado.anotacao)}
        return saida

    return carregar


def main(argv: Sequence[str] | None = None) -> int:
    """0 = dataset gravado; 2 = entrada recusada."""
    snap = "snapshot=20261014T021437Z"
    parser = argparse.ArgumentParser(prog="python -m src.dataset_desenvolvimento", description=__doc__)
    parser.add_argument("--n1", type=Path, default=RAIZ_PROJETO / "data/features" / snap
                        / "historico=v1/defasagem=30/contexto=validacao")
    parser.add_argument("--legenda-capa", type=Path, required=True, help="pasta materializada de legenda e capa")
    parser.add_argument("--amostra", type=Path, default=RAIZ_PROJETO / "data/annotations/gemini" / snap / "amostra=v2")
    parser.add_argument("--destino", type=Path, default=RAIZ_PROJETO / "data/features" / snap)
    args = parser.parse_args(argv)
    try:
        pasta = montar(
            tabela_n1=args.n1 / "n1.parquet", manifesto_n1=args.n1 / "manifesto.json",
            pasta_legenda_capa=args.legenda_capa, arquivo_amostra=args.amostra / "amostra.json",
            carregar_anotacoes=_anotacoes_do_cache(args.amostra), codificador=leg.CodificadorTexto(),
            cache_embeddings=args.destino / "cache_embeddings_transcricao", destino_raiz=args.destino)
    except (ErroDataset, ValueError, KeyError, OSError) as erro:
        print(f"Dataset recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({"pasta": str(pasta)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
