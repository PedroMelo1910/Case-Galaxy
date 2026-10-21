"""Módulo consolidado. Reúne:
- Dataset versionado dos atributos audiovisuais locais (`avl_*`) para os 1.500 vídeos da amostra congelada.
- Manifesto de mídia de entrada: o portão objetivo para treinar um modelo supervisionado com atributos audiovisuais locais."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, UTC
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import RAIZ_PROJETO
from src.dados import dataset as ds
from src.extracao import pipeline as col
from src.extracao.modelos import identidade_da_configuracao
from src.proveniencia import sha256_bytes as _sha256_bytes


# ──── Dataset versionado dos atributos audiovisuais locais (`avl_*`) para os 1.500 vídeos da amostra conge
VERSAO = "1"
NOME = "audiovisual_local"
ARQUIVO_COMPLETO = "dataset_audiovisual.parquet"
ARQUIVO_DESENVOLVIMENTO = "desenvolvimento_audiovisual.parquet"
ARQUIVO_MANIFESTO = "manifesto.json"
SNAPSHOT = "snapshot=20261014T021437Z"
DISPONIVEL = "concluida"
COL_DISPONIBILIDADE, COL_MOTIVO = "avl_disponibilidade", "avl_motivo"
COLUNAS_DE_SITUACAO = (COL_DISPONIBILIDADE, COL_MOTIVO)
COLUNAS_ESTADO = (*col.COLUNAS_DE_ESTADO, *COLUNAS_DE_SITUACAO)
PAPEL_ESTADO = "estado_extracao"
USO_TESTE = "teste_protegido"
COLUNAS_DE_IDENTIFICACAO = ("video_id", "channel_id", "bloco", "uso_no_contexto", "setor", "tipo_canal")
# categoria do download no estado -> situação publicada; a categoria original fica em `avl_motivo`
SITUACAO_POR_CATEGORIA = {"bloqueio_acesso": "pendente_acesso_403", "indisponivel": "indisponivel",
                          "anti_bot": "bloqueio_anti_bot"}


COL_STATUS_AMOSTRA, COL_ORIGINAL_SUBSTITUIDO = "status_amostra", "original_substituido"
COLUNAS_DE_ORIGEM = (COL_STATUS_AMOSTRA, COL_ORIGINAL_SUBSTITUIDO)
STATUS_VALIDOS = frozenset({"original_extraido", "recuperado", "substituto", "pendente_acesso", "indisponivel"})
CLASSE_POR_PAPEL = {"atributo": "preditor", "alvo": "alvo", "referencia": "referencia_historica",
                    "identificacao": "identificador_ou_auditoria", "auditoria": "identificador_ou_auditoria",
                    PAPEL_ESTADO: "estado_de_extracao", "explicacao": "so_explicacao"}


class ErroDatasetAudiovisual(ds.ErroDataset):
    """Resultado, amostra ou configuração que não pode entrar no dataset."""


@dataclass(frozen=True)
class FonteLocal:
    """Uma pasta de resultados locais (atributos + estado do executor) e o universo de IDs que ela pode conter.

    Resultado de ID fora de `ids_permitidos` é erro; resultado de ID permitido mas fora da amostra ativa é ignorado
    (por exemplo, um candidato extraído e não aceito)."""

    nome: str
    pasta_atributos: Path
    pasta_estado: Path
    ids_permitidos: frozenset[str]


def _sha256_arquivo(arquivo: Path) -> str:
    return _sha256_bytes(Path(arquivo).read_bytes())


def _json(arquivo: Path) -> dict:
    return json.loads(Path(arquivo).read_text(encoding="utf-8"))


def situacao_da_extracao(estado: Mapping | None) -> tuple[str, str | None]:
    """(situação, motivo) de um vídeo a partir do estado do executor; `None` = nunca tentado."""
    if estado is None:
        return "nao_tentada", None
    if estado.get("extracao", {}).get("estado") == "concluida":
        return DISPONIVEL, None
    download = estado.get("download", {})
    categoria = download.get("categoria") or estado.get("extracao", {}).get("categoria")
    if download.get("estado") == "falha":
        return SITUACAO_POR_CATEGORIA.get(categoria, "falha_download"), categoria
    if download.get("estado") == "baixado":
        return "extracao_incompleta", categoria
    return "nao_tentada", categoria


def verificar_resultado(video_id: str, dados: Mapping, estado: Mapping, identidade: str) -> str | None:
    """Motivo da recusa de um resultado persistido, ou None. Vale para o vídeo, a configuração e o esquema."""
    if estado.get("video_id") != video_id:
        return "estado_de_outro_video"
    if estado.get("extracao", {}).get("estado") != "concluida":
        return "extracao_nao_concluida"
    if dados.get("sha256") != (estado.get("arquivo") or {}).get("sha256"):
        return "saida_de_outro_video"
    if dados.get("identidade") != identidade:
        return "configuracao_diferente"
    if estado.get("fonte") != f"https://www.youtube.com/watch?v={video_id}":
        return "fonte_ausente"
    if not dados.get("proveniencia") or not dados.get("etapas"):
        return "proveniencia_ou_etapas_ausentes"
    colunas = dados.get("colunas")
    if not isinstance(colunas, dict) or tuple(colunas) != col.COLUNAS:
        return "esquema_das_colunas_diferente"
    try:
        if col.linearizar({"etapas": dados["etapas"]}) != colunas:
            return "colunas_nao_derivam_das_etapas"
    except ValueError:
        return "estado_ou_valor_invalido"
    for coluna in col.COLUNAS_VETORIAIS:
        vetor = colunas[coluna]
        if vetor is not None and (not vetor or any(isinstance(v, bool) or not math.isfinite(v) for v in vetor)):
            return "embedding_invalido"
    return None


def ler_resultados(pasta_atributos: Path, pasta_estado: Path, ids_amostra: Sequence[str], identidade: str,
                   permitidos: frozenset[str] | None = None
                   ) -> tuple[dict[str, dict], dict[str, str], dict[str, dict]]:
    """(colunas por ID verificado, motivo por ID recusado, estado por ID). Arquivo fora da amostra é erro.

    Com `permitidos`, o universo aceito na pasta é esse conjunto (que deve conter `ids_amostra`); os resultados
    permitidos fora de `ids_amostra` não entram. Sem ele, só a própria amostra é aceita (comportamento da versão 1)."""
    if len(set(ids_amostra)) != len(ids_amostra):
        raise ErroDatasetAudiovisual("IDs repetidos na amostra")
    universo = set(ids_amostra) if permitidos is None else set(permitidos)
    if not set(ids_amostra) <= universo:
        raise ErroDatasetAudiovisual("amostra com IDs fora do universo permitido da fonte")
    arquivos = sorted(Path(pasta_atributos).glob("*.json"))
    fora = sorted(a.stem for a in arquivos if a.stem not in universo)
    if fora:
        raise ErroDatasetAudiovisual(f"resultados de vídeos fora da amostra congelada: {fora[:5]} ({len(fora)})")
    arquivos = [a for a in arquivos if a.stem in set(ids_amostra)]
    estados = {i: _json(a) for i in ids_amostra if (a := Path(pasta_estado) / f"{i}.json").is_file()}
    verificados: dict[str, dict] = {}
    recusados: dict[str, str] = {}
    for arquivo in arquivos:
        video_id = arquivo.stem
        estado = estados.get(video_id)
        dados = _json(arquivo)
        motivo = "sem_estado" if estado is None else verificar_resultado(video_id, dados, estado, identidade)
        if motivo:
            recusados[video_id] = motivo
        else:
            verificados[video_id] = dados["colunas"]
    return verificados, recusados, estados


def ler_resultados_fontes(fontes: Sequence[FonteLocal], ids_amostra: Sequence[str], identidade: str
                          ) -> tuple[dict[str, dict], dict[str, str], dict[str, dict]]:
    """Junta os resultados de várias fontes; cada ID da amostra pertence a no máximo uma delas."""
    donos: dict[str, str] = {}
    for fonte in fontes:
        for video_id in ids_amostra:
            if video_id in fonte.ids_permitidos:
                if video_id in donos:
                    raise ErroDatasetAudiovisual(f"{video_id}: presente em duas fontes ({donos[video_id]}, {fonte.nome})")
                donos[video_id] = fonte.nome
    verificados: dict[str, dict] = {}
    recusados: dict[str, str] = {}
    estados: dict[str, dict] = {}
    for fonte in fontes:
        ids = [i for i in ids_amostra if donos.get(i) == fonte.nome]
        v, r, e = ler_resultados(fonte.pasta_atributos, fonte.pasta_estado, ids, identidade, fonte.ids_permitidos)
        verificados |= v
        recusados |= r
        estados |= e
    return verificados, recusados, estados


def _linha_avl(video_id: str, verificados: Mapping[str, Mapping], estados: Mapping[str, Mapping]) -> dict:
    situacao, motivo = situacao_da_extracao(estados.get(video_id))
    if video_id in verificados:
        return {**verificados[video_id], COL_DISPONIBILIDADE: DISPONIVEL, COL_MOTIVO: None}
    if situacao == DISPONIVEL:  # concluída no estado mas sem resultado verificado: não vira dado
        situacao, motivo = "extracao_invalida", "resultado_nao_verificado"
    return {**dict.fromkeys(col.COLUNAS), COL_DISPONIBILIDADE: situacao, COL_MOTIVO: motivo}


def montar_linhas(lista_amostra: Sequence[Mapping], atribuicao: Mapping[str, str], uso_no_contexto: Mapping[str, str],
                  verificados: Mapping[str, Mapping], estados: Mapping[str, Mapping],
                  origem: Mapping[str, Mapping] | None = None) -> list[dict]:
    """Uma linha por vídeo da amostra, na ordem congelada; bloco e atribuição temporal vêm da partição, sem alteração."""
    linhas = []
    for item in lista_amostra:
        video_id = item["video_id"]
        if atribuicao.get(video_id) != item["bloco"]:
            raise ErroDatasetAudiovisual(f"{video_id}: bloco da amostra diferente da partição congelada")
        uso = USO_TESTE if item["bloco"].startswith("teste") else uso_no_contexto.get(video_id)
        if uso is None:
            raise ErroDatasetAudiovisual(f"{video_id}: sem uso no contexto de validação")
        extra = {}
        if origem is not None:
            marca = origem.get(video_id)
            if marca is None or marca.get("status") not in STATUS_VALIDOS:
                raise ErroDatasetAudiovisual(f"{video_id}: sem status válido na amostra ativa")
            extra = {COL_STATUS_AMOSTRA: marca["status"], COL_ORIGINAL_SUBSTITUIDO: marca.get("original_substituido")}
        linhas.append({"video_id": video_id, "channel_id": item["channel_id"], "bloco": item["bloco"],
                       "uso_no_contexto": uso, "setor": item["setor"], "tipo_canal": item["tipo_canal"],
                       "ordem_amostra": item["ordem"], **extra, **_linha_avl(video_id, verificados, estados)})
    return linhas


def _tipo(coluna: str, valores: Sequence) -> pa.DataType:
    if coluna in col.COLUNAS_VETORIAIS:
        return pa.list_(pa.float64())
    presentes = [v for v in valores if v is not None]
    if coluna in COLUNAS_ESTADO or coluna in COLUNAS_DE_ORIGEM or any(isinstance(v, str) for v in presentes):
        return pa.string()
    if presentes and all(isinstance(v, int) for v in presentes):
        return pa.int64()
    return pa.float64()


def papel_da_coluna(coluna: str) -> str:
    """Papel de uma coluna `avl_*`, conforme o contrato do extrator (`colunas.py`), sem convenção nova."""
    if coluna in COLUNAS_ESTADO:
        return PAPEL_ESTADO
    if coluna in col.COLUNAS_DE_IMPRESSAO_DIGITAL:
        return "auditoria"
    if coluna in col.COLUNAS_PREDITORAS:
        return "atributo"
    raise ErroDatasetAudiovisual(f"coluna sem papel definido: {coluna}")


def _tabela(linhas: Sequence[Mapping], papeis: Mapping[str, str], tipos_fixos: Mapping[str, pa.DataType],
            tabela: str, versao: str = VERSAO) -> pa.Table:
    tipos = {c: tipos_fixos.get(c) or _tipo(c, [l.get(c) for l in linhas]) for c in papeis}
    esquema = pa.schema([pa.field(c, tipos[c], metadata={"papel_coluna": papeis[c]}) for c in papeis],
                        metadata={"tabela": tabela, "versao": versao})
    return pa.Table.from_pylist([{c: l.get(c) for c in papeis} for l in linhas], schema=esquema)


def _bytes(tabela: pa.Table) -> bytes:
    saida = pa.BufferOutputStream()
    pq.write_table(tabela, saida)
    return saida.getvalue().to_pybytes()


def cobertura(linhas: Sequence[Mapping]) -> dict:
    """Concluídos/total por bloco, uso no contexto, setor, tipo de canal e canal; só contagem, sem olhar resultado."""
    def por(chave) -> dict:
        total, feitos = Counter(), Counter()
        for l in linhas:
            k = chave(l)
            total[k] += 1
            feitos[k] += l[COL_DISPONIBILIDADE] == DISPONIVEL
        return {"|".join(k) if isinstance(k, tuple) else str(k): {"concluidos": feitos[k], "total": total[k]}
                for k in sorted(total, key=str)}
    return {"situacao": dict(Counter(l[COL_DISPONIBILIDADE] for l in linhas)),
            "por_bloco": por(lambda l: l["bloco"]), "por_uso_no_contexto": por(lambda l: l["uso_no_contexto"]),
            "por_setor": por(lambda l: (l["uso_no_contexto"], l["setor"])),
            "por_tipo_canal": por(lambda l: (l["uso_no_contexto"], l["tipo_canal"])),
            "por_canal": por(lambda l: (l["uso_no_contexto"], l["channel_id"]))}


def _desenvolvimento(pasta: Path, linhas: Sequence[Mapping], atribuicao: Mapping[str, str],
                     papeis_avl: Mapping[str, str], versao: str = VERSAO) -> tuple[pa.Table, dict[str, str], dict]:
    """Linhas permitidas do dataset de desenvolvimento vigente + atributos locais; teste e intervalo nunca entram."""
    manifesto = _json(Path(pasta) / "manifesto.json")
    arquivo = Path(pasta) / "dataset.parquet"
    if _sha256_arquivo(arquivo) != manifesto["sha256_dataset"]:
        raise ErroDatasetAudiovisual("hash do dataset de desenvolvimento diferente do manifesto")
    base = pq.read_table(arquivo)
    por_id = {l["video_id"]: l for l in linhas}
    novas = (*col.COLUNAS, *COLUNAS_DE_SITUACAO,
             *(c for c in COLUNAS_DE_ORIGEM if linhas and c in linhas[0]))
    saida = []
    for linha in base.to_pylist():
        item = por_id.get(linha["video_id"])
        if item is None:
            raise ErroDatasetAudiovisual(f"{linha['video_id']}: fora da amostra congelada")
        if linha["uso_no_contexto"] not in ds.USOS_PERMITIDOS or str(linha["bloco"]).startswith("teste"):
            raise ErroDatasetAudiovisual(f"{linha['video_id']}: linha não permitida ao desenvolvimento")
        if linha["bloco"] != atribuicao[linha["video_id"]] or linha["uso_no_contexto"] != item["uso_no_contexto"]:
            raise ErroDatasetAudiovisual(f"{linha['video_id']}: atribuição temporal diferente da congelada")
        saida.append({**linha, **{c: item[c] for c in novas}})
    ids_dev = [l["video_id"] for l in saida]
    esperados = {l["video_id"] for l in linhas if l["uso_no_contexto"] in ds.USOS_PERMITIDOS}
    if len(ids_dev) != manifesto["n_linhas"] or len(set(ids_dev)) != len(ids_dev) or set(ids_dev) != esperados:
        raise ErroDatasetAudiovisual("desenvolvimento incompleto ou com IDs repetidos: deve ser exatamente treino + "
                                     "avaliação da partição, uma linha por ID, no total do manifesto")
    repetidas = sorted(set(base.column_names) & set(novas))
    if repetidas:
        raise ErroDatasetAudiovisual(f"colunas repetidas entre o desenvolvimento e o audiovisual: {repetidas}")
    papeis = {f.name: (f.metadata or {}).get(b"papel_coluna", b"auditoria").decode() for f in base.schema}
    papeis |= {c: papeis_avl[c] for c in novas}
    return (_tabela(saida, papeis, {f.name: f.type for f in base.schema}, "desenvolvimento_audiovisual", versao),
            papeis, manifesto)


def especificacao_candidata(tabela: pa.Table) -> dict:
    """Escalares, categóricas e embeddings que o esquema real permite como preditores locais (não é seleção)."""
    esc, cat, emb = [], [], {}
    for campo in tabela.schema:
        if (campo.metadata or {}).get(b"papel_coluna") != b"atributo" or not campo.name.startswith(col.PREFIXO):
            continue
        if pa.types.is_list(campo.type):
            emb[campo.name] = len(next(v for v in tabela.column(campo.name).to_pylist() if v is not None))
        elif pa.types.is_string(campo.type):
            cat.append(campo.name)
        else:
            esc.append(campo.name)
    return {"escalares": esc, "categoricas": cat, "embeddings": emb}


def _dimensao_unica(verificados: Mapping[str, Mapping]) -> int | None:
    dims = {len(v[c]) for v in verificados.values() for c in col.COLUNAS_VETORIAIS if v[c] is not None}
    if len(dims) > 1:
        raise ErroDatasetAudiovisual(f"embeddings com dimensões diferentes: {sorted(dims)}")
    return dims.pop() if dims else None


def classificar_colunas(papeis: Mapping[str, str]) -> dict[str, list[str]]:
    """Colunas por classe de uso: preditor, alvo, referência, identificador/auditoria, estado de extração, explicação."""
    saida: dict[str, list[str]] = {}
    for coluna, papel in papeis.items():
        saida.setdefault(CLASSE_POR_PAPEL.get(papel, "identificador_ou_auditoria"), []).append(coluna)
    return saida


LIMITACOES_COMUNS = (
    "A tabela completa não traz alvo nem referência; o teste protegido aparece só por ID e bloco.",
    "O teste final já foi consultado e o modelo servido foi refeito com os alvos originais: novos experimentos com "
    "estes dados são desenvolvimento/exploratórios, não um teste final novo e independente.",
)


def _limitacoes(versao: str, linhas: Sequence[Mapping]) -> list[str]:
    """Limitações da versão: a 1 é parcial por construção; a 2 descreve a cobertura e as trocas realmente feitas."""
    if versao == "1":
        return ["População parcial: só os vídeos extraídos; a ordem de extração pode tornar a disponibilidade "
                "desigual, e a parcial não é uma amostra aleatória representativa sem evidência.", *LIMITACOES_COMUNS]
    concluidos = sum(l[COL_DISPONIBILIDADE] == DISPONIVEL for l in linhas)
    trocas = sum(l.get(COL_STATUS_AMOSTRA) == "substituto" for l in linhas)
    return [f"Cobertura da amostra ativa: {concluidos} de {len(linhas)} vídeos com extração local verificada; "
            f"{trocas} substituto(s) entram no lugar de originais inacessíveis e mudam a população em relação à amostra "
            "original: diferenças de resultado associadas a esses IDs não são do modelo.",
            "Os resultados locais vêm de arquivos de vídeo baixados do YouTube em ~360p; a qualidade da transcrição de "
            "fala não foi medida contra referência. Estado de fala: transcrita, áudio sem fala detectada e transcrição "
            "incerta são estados distintos; transcrição incerta não gera medidas de fala.",
            *LIMITACOES_COMUNS]


def _manifesto(linhas, verificados, estados, papeis, dev_papeis, identidade, dados, amostra, particao, dev_manifesto,
               n_dev, tabela_dev, fontes, versao: str = VERSAO, retencoes_extras: Mapping[str, str] | None = None,
               extras: Mapping | None = None) -> dict:
    ids_ok = sorted(verificados)
    completo, desenvolvimento = dados
    retencoes = {"amostra": amostra["retencao"]["apagar_ate_utc"], "particao": particao["retencao"]["apagar_ate_utc"],
                 "historico": _json(fontes["manifesto_n1"])["retencao"]["apagar_ate_utc"],
                 "desenvolvimento": dev_manifesto["retencao"]["apagar_ate_utc"], **(retencoes_extras or {})}
    return {
        "versao": versao, "tipo": "dataset_audiovisual_local", **(extras or {}),
        "gerado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256_completo": _sha256_bytes(completo), "sha256_desenvolvimento": _sha256_bytes(desenvolvimento),
        "n_linhas_completo": len(linhas), "n_linhas_desenvolvimento": n_dev,
        "extracao": {"identidade_da_configuracao": identidade, "dimensao_embedding": _dimensao_unica(verificados),
                     "n_verificados": len(ids_ok), "sha256_ids_verificados": _sha256_bytes("\n".join(ids_ok).encode()),
                     "sha256_resultados": _sha256_bytes("\n".join(
                         f"{i}:{estados[i]['arquivo']['sha256']}" for i in ids_ok).encode()),
                     "estados_por_etapa": {c: dict(Counter(l[c] for l in linhas if l[c] is not None))
                                           for c in col.COLUNAS_DE_ESTADO}},
        "entradas": {"amostra": {"sha256_lista": amostra["sha256_lista"],
                                 "sha256_arquivo": _sha256_arquivo(fontes["arquivo_amostra"])},
                     "particao": {"sha256_atribuicao": particao["sha256_atribuicao"],
                                  "sha256_arquivo": _sha256_arquivo(fontes["arquivo_particao"])},
                     "historico": {"sha256": _sha256_arquivo(fontes["tabela_n1"])},
                     "desenvolvimento": {"sha256": dev_manifesto["sha256_dataset"],
                                         "n_linhas": dev_manifesto["n_linhas"]},
                     "codigo": {"colunas.py": _sha256_arquivo(Path(col.__file__)),
                                "dataset_audiovisual_local.py": _sha256_arquivo(Path(__file__))}},
        "ids": {"n_amostra": len(linhas),
                "sha256_ids_amostra": _sha256_bytes("\n".join(l["video_id"] for l in linhas).encode())},
        "cobertura": cobertura(linhas),
        "colunas_por_papel": {p: [c for c, q in papeis.items() if q == p] for p in
                              ("identificacao", "auditoria", "atributo", PAPEL_ESTADO)},
        "papeis_do_desenvolvimento": dict(Counter(dev_papeis.values())),
        **({"classificacao_das_colunas_do_desenvolvimento": classificar_colunas(dev_papeis)}
           if versao != "1" else {}),
        "especificacao_candidata_sem_selecao": especificacao_candidata(tabela_dev),
        "ausencias_nos_preditores_locais": {c: sum(l[c] is None for l in linhas) for c, p in papeis.items()
                                            if p == "atributo"},
        "politica_de_ausencia": "ausência permanece nula: vídeo não extraído não vira zero, sem fala, sem rosto ou sem "
        "objetos. Imputação, PCA, codificação e seleção ficam dentro de cada dobra do treino.",
        "retencao": {"apagar_ate_utc": min(retencoes.values()), "herdada_de": retencoes,
                     "regra": "o menor prazo entre as fontes; este dataset não estende nenhum prazo"},
        "limitacoes": _limitacoes(versao, linhas),
    }


def montar(*, pasta_atributos: Path, pasta_estado: Path, arquivo_amostra: Path, arquivo_particao: Path,
           tabela_n1: Path, manifesto_n1: Path, pasta_desenvolvimento: Path, destino_raiz: Path,
           identidade: str | None = None, versao: str = VERSAO, fontes: Sequence[FonteLocal] | None = None,
           origem: Mapping[str, Mapping] | None = None, retencoes_extras: Mapping[str, str] | None = None,
           extras: Mapping | None = None) -> Path:
    """Monta e grava (sem sobrescrever) o dataset; idempotente para o mesmo conteúdo. Devolve a pasta.

    Sem os argumentos opcionais reproduz a versão 1. `fontes` substitui o par `pasta_atributos`/`pasta_estado` quando os
    resultados vêm de mais de uma pasta; `origem` (ID -> {status, original_substituido}) acrescenta a origem de cada
    linha na amostra; `retencoes_extras` entra no menor prazo; `extras` é copiado para o topo do manifesto."""
    identidade = identidade or identidade_da_configuracao()
    amostra, particao = _json(arquivo_amostra), _json(arquivo_particao)
    ids = [x["video_id"] for x in amostra["lista"]]
    if len(ids) != amostra["selecionados"]:
        raise ErroDatasetAudiovisual("tamanho da lista diferente do congelado na amostra")
    n1 = {r["video_id"]: r["uso_no_contexto"] for r in pq.read_table(
        tabela_n1, columns=["video_id", "uso_no_contexto"]).to_pylist()}
    if fontes:
        verificados, recusados, estados = ler_resultados_fontes(fontes, ids, identidade)
    else:
        verificados, recusados, estados = ler_resultados(pasta_atributos, pasta_estado, ids, identidade)
    if recusados:
        raise ErroDatasetAudiovisual(f"resultados recusados: {dict(list(recusados.items())[:5])} ({len(recusados)})")
    linhas = montar_linhas(amostra["lista"], particao["atribuicao"], n1, verificados, estados, origem)
    papeis = {**dict.fromkeys(COLUNAS_DE_IDENTIFICACAO, "identificacao"), "ordem_amostra": "auditoria",
              **({c: "auditoria" for c in COLUNAS_DE_ORIGEM} if origem is not None else {}),
              **{c: papel_da_coluna(c) for c in (*col.COLUNAS, *COLUNAS_DE_SITUACAO)}}
    completa = _tabela(linhas, papeis, {"ordem_amostra": pa.int64()}, "dataset_audiovisual_completo", versao)
    desenvolvimento, dev_papeis, dev_manifesto = _desenvolvimento(
        pasta_desenvolvimento, linhas, particao["atribuicao"], papeis, versao)
    dados = (_bytes(completa), _bytes(desenvolvimento))
    pasta = Path(destino_raiz) / f"{NOME}=v{versao}" / _sha256_bytes(b"".join(dados))[:12]
    if pasta.exists():
        return pasta
    manifesto = _manifesto(linhas, verificados, estados, papeis, dev_papeis, identidade, dados, amostra, particao,
                           dev_manifesto, desenvolvimento.num_rows, desenvolvimento,
                           {"arquivo_amostra": arquivo_amostra, "arquivo_particao": arquivo_particao,
                            "tabela_n1": tabela_n1, "manifesto_n1": manifesto_n1},
                           versao, retencoes_extras, extras)
    pasta.parent.mkdir(parents=True, exist_ok=True)
    parcial = Path(tempfile.mkdtemp(prefix=".dataset.", suffix=".parcial", dir=pasta.parent))
    try:
        (parcial / ARQUIVO_COMPLETO).write_bytes(dados[0])
        (parcial / ARQUIVO_DESENVOLVIMENTO).write_bytes(dados[1])
        (parcial / ARQUIVO_MANIFESTO).write_text(json.dumps(manifesto, ensure_ascii=False, indent=1), encoding="utf-8")
        parcial.rename(pasta)
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    return pasta


def main(argv: Sequence[str] | None = None) -> int:
    """0 = dataset gravado; 2 = entrada recusada."""
    base = RAIZ_PROJETO / "data"
    verif = RAIZ_PROJETO.parent / "VERIFICACOES/audiovisual_local/piloto_mesmos_1500"
    parser = argparse.ArgumentParser(prog="python -m src.dados.dataset_audiovisual", description=__doc__)
    parser.add_argument("--atributos", type=Path, default=verif / "atributos")
    parser.add_argument("--estado", type=Path, default=verif / "estado")
    parser.add_argument("--amostra", type=Path,
                        default=base / "annotations/gemini" / SNAPSHOT / "amostra=v2/amostra.json")
    parser.add_argument("--particao", type=Path, default=base / "processed" / SNAPSHOT / "particao=v1/particao.json")
    parser.add_argument("--n1", type=Path, default=base / "features" / SNAPSHOT
                        / "historico=v1/defasagem=30/contexto=validacao")
    parser.add_argument("--desenvolvimento", type=Path,
                        default=base / "features" / SNAPSHOT / "desenvolvimento=v2/1f4ef5b5eac0")
    parser.add_argument("--destino", type=Path, default=base / "features" / SNAPSHOT)
    args = parser.parse_args(argv)
    try:
        pasta = montar(pasta_atributos=args.atributos, pasta_estado=args.estado, arquivo_amostra=args.amostra,
                       arquivo_particao=args.particao, tabela_n1=args.n1 / "n1.parquet",
                       manifesto_n1=args.n1 / "manifesto.json", pasta_desenvolvimento=args.desenvolvimento,
                       destino_raiz=args.destino)
    except (ds.ErroDataset, ValueError, KeyError, OSError) as erro:
        print(f"Dataset recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({"pasta": str(pasta)}, ensure_ascii=False))
    return 0


# ──── Manifesto de mídia de entrada: o portão objetivo para treinar um modelo supervisionado com atributos
COLUNAS_DO_MANIFESTO = ("video_id", "arquivo", "sha256", "canal_id", "setor", "published_at", "collected_at",
                        "curtidas_observadas", "uso", "fonte_url", "licenca", "autorizacao", "retencao_ate",
                        "observacao")
SETORES = ("beleza", "automotivo")
USO_TREINO = "treino"
USO_FUNCIONAL = "teste_funcional_somente"
IDADE_MINIMA_DIAS = 30  # TAPI §5: só vídeos com ao menos 30 dias na coleta (métricas estabilizadas)

MIN_VIDEOS_TREINO = 600
MIN_CANAIS = 14
MIN_SETORES = 2
MIN_CANAIS_POR_SETOR = 3
MIN_VIDEOS_POR_SETOR = 50

_ID_VIDEO = re.compile(r"^[A-Za-z0-9_-]{11}$")
_OBRIGATORIOS_TREINO = ("video_id", "arquivo", "sha256", "canal_id", "setor", "published_at", "collected_at",
                        "curtidas_observadas", "fonte_url", "licenca", "autorizacao", "retencao_ate")
_OBRIGATORIOS_FUNCIONAL = ("arquivo", "sha256", "fonte_url", "licenca", "autorizacao", "retencao_ate")
_PROIBIDOS_FUNCIONAL = ("video_id", "canal_id", "curtidas_observadas", "published_at", "collected_at")


@dataclass(frozen=True)
class ResumoManifesto:
    n_linhas: int
    n_treino_validos: int
    n_funcional: int
    canais_distintos: int
    por_setor: dict[str, dict[str, int]]
    invalidas: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Prontidao:
    pronto: bool
    motivos: list[str]


def _vazios(linha: dict, campos: tuple[str, ...]) -> list[str]:
    return [c for c in campos if not (linha.get(c) or "").strip()]


def _data_com_fuso(valor: str, campo: str, motivos: list[str]) -> datetime | None:
    try:
        data = datetime.fromisoformat(valor.strip())
    except ValueError:
        motivos.append(f"{campo} não é uma data ISO-8601")
        return None
    if data.tzinfo is None:
        motivos.append(f"{campo} sem fuso horário")
        return None
    return data


def _motivos_de_arquivo(linha: dict, raiz: Path, verificar_arquivos: bool) -> list[str]:
    destino = (raiz / linha["arquivo"].strip()).resolve()
    if not destino.is_relative_to(raiz.resolve()):
        return ["arquivo fora da raiz de mídia"]
    if not verificar_arquivos:
        return []
    if not destino.is_file():
        return ["arquivo não encontrado"]
    resumo = hashlib.sha256()
    with destino.open("rb") as arquivo:
        while bloco := arquivo.read(1_048_576):
            resumo.update(bloco)
    return [] if resumo.hexdigest() == linha["sha256"].strip().lower() else ["sha256 diferente do arquivo"]


def _motivos_de_datas(linha: dict, hoje: datetime) -> list[str]:
    motivos: list[str] = []
    publicado = _data_com_fuso(linha["published_at"], "published_at", motivos)
    coletado = _data_com_fuso(linha["collected_at"], "collected_at", motivos)
    if publicado and coletado:
        if coletado < publicado:
            motivos.append("collected_at anterior a published_at")
        elif (coletado - publicado).days < IDADE_MINIMA_DIAS:
            motivos.append(f"vídeo com menos de {IDADE_MINIMA_DIAS} dias na coleta")
    try:
        if date.fromisoformat(linha["retencao_ate"].strip()) < hoje.date():
            motivos.append("retenção vencida (retencao_ate já passou)")
    except ValueError:
        motivos.append("retencao_ate não é uma data ISO-8601")
    return motivos


def _motivos_de_treino(linha: dict, raiz: Path, hoje: datetime, verificar_arquivos: bool) -> list[str]:
    vazios = _vazios(linha, _OBRIGATORIOS_TREINO)
    if vazios:
        return [f"campos obrigatórios vazios: {', '.join(vazios)}"]
    motivos = []
    if linha["setor"].strip() not in SETORES:
        motivos.append(f"setor fora do recorte ({', '.join(SETORES)})")
    if not _ID_VIDEO.fullmatch(linha["video_id"].strip()):
        motivos.append("video_id fora do formato de 11 caracteres do YouTube")
    curtidas = linha["curtidas_observadas"].strip()
    if not (curtidas.isascii() and curtidas.isdigit()):  # `isdigit` sozinho aceita dígitos Unicode de outros alfabetos
        motivos.append("curtidas_observadas não é um inteiro maior ou igual a zero")
    return motivos + _motivos_de_datas(linha, hoje) + _motivos_de_arquivo(linha, raiz, verificar_arquivos)


def _motivos_funcional(linha: dict, raiz: Path, hoje: datetime, verificar_arquivos: bool) -> list[str]:
    proibidos = [c for c in _PROIBIDOS_FUNCIONAL if (linha.get(c) or "").strip()]
    if proibidos:
        return [f"funcional: não pode ter {', '.join(proibidos)} (não há resultado observado para este arquivo)"]
    vazios = _vazios(linha, _OBRIGATORIOS_FUNCIONAL)
    if vazios:
        return [f"funcional: campos obrigatórios vazios: {', '.join(vazios)}"]
    motivos = _motivos_de_arquivo(linha, raiz, verificar_arquivos)
    try:
        if date.fromisoformat(linha["retencao_ate"].strip()) < hoje.date():
            motivos.append("retenção vencida (retencao_ate já passou)")
    except ValueError:
        motivos.append("retencao_ate não é uma data ISO-8601")
    return motivos


def carregar_manifesto(caminho_csv: Path, raiz_midia: Path, *, hoje: datetime | None = None,
                       verificar_arquivos: bool = True) -> ResumoManifesto:
    """Valida cada linha do CSV; linhas inválidas ficam em `invalidas` com os motivos e não entram na contagem."""
    hoje = hoje or datetime.now(UTC)
    with Path(caminho_csv).open(newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        faltando = [c for c in COLUNAS_DO_MANIFESTO if c not in (leitor.fieldnames or [])]
        if faltando:
            raise ValueError(f"manifesto sem as colunas: {', '.join(faltando)}")
        linhas = list(leitor)
    invalidas: list[dict] = []
    treino: list[dict] = []
    n_funcional = 0
    vistos: dict[str, int] = {}
    for numero, linha in enumerate(linhas, start=1):
        uso = (linha.get("uso") or "").strip()
        if uso == USO_TREINO:
            motivos = _motivos_de_treino(linha, Path(raiz_midia), hoje, verificar_arquivos)
        elif uso == USO_FUNCIONAL:
            motivos = _motivos_funcional(linha, Path(raiz_midia), hoje, verificar_arquivos)
        else:
            motivos = [f"uso desconhecido: {uso!r} (esperado {USO_TREINO} ou {USO_FUNCIONAL})"]
        chaves = [f"sha256:{linha['sha256'].strip().lower()}"] if (linha.get("sha256") or "").strip() else []
        if uso == USO_TREINO and (linha.get("video_id") or "").strip():
            chaves.append(f"video_id:{linha['video_id'].strip()}")
        repetida = next((vistos[c] for c in chaves if c in vistos), None) if not motivos else None
        if repetida is not None:
            motivos = [f"duplicata de conteúdo ou de video_id da linha {repetida}"]
        if motivos:
            invalidas.append({"linha": numero, "motivos": motivos})
            continue
        vistos.update({c: numero for c in chaves})
        if uso == USO_TREINO:
            treino.append(linha)
        else:
            n_funcional += 1
    por_setor: dict[str, dict[str, set]] = {}
    for linha in treino:
        entrada = por_setor.setdefault(linha["setor"].strip(), {"videos": set(), "canais": set()})
        entrada["videos"].add(linha["video_id"].strip())
        entrada["canais"].add(linha["canal_id"].strip())
    return ResumoManifesto(n_linhas=len(linhas), n_treino_validos=len(treino), n_funcional=n_funcional,
                           canais_distintos=len({exemplo["canal_id"].strip() for exemplo in treino}),
                           por_setor={s: {"videos": len(v["videos"]), "canais": len(v["canais"])}
                                      for s, v in por_setor.items()}, invalidas=invalidas)


def avaliar_prontidao(resumo: ResumoManifesto) -> Prontidao:
    """Compara a cobertura com os pisos pré-registrados e diz exatamente o que falta."""
    motivos: list[str] = []
    if resumo.n_treino_validos < MIN_VIDEOS_TREINO:
        motivos.append(f"vídeos de treino válidos: {resumo.n_treino_validos} (mínimo {MIN_VIDEOS_TREINO})")
    if resumo.canais_distintos < MIN_CANAIS:
        motivos.append(f"canais distintos: {resumo.canais_distintos} (mínimo {MIN_CANAIS})")
    if len(resumo.por_setor) < MIN_SETORES:
        motivos.append(f"setores cobertos: {len(resumo.por_setor)} (mínimo {MIN_SETORES})")
    for setor, cobertura in resumo.por_setor.items():
        if cobertura["canais"] < MIN_CANAIS_POR_SETOR or cobertura["videos"] < MIN_VIDEOS_POR_SETOR:
            motivos.append(f"setor {setor}: {cobertura['videos']} vídeos em {cobertura['canais']} canais "
                           f"(mínimo {MIN_VIDEOS_POR_SETOR} vídeos e {MIN_CANAIS_POR_SETOR} canais)")
    return Prontidao(pronto=not motivos, motivos=motivos)


if __name__ == "__main__":
    sys.exit(main())
