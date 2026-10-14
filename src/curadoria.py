"""Organiza o snapshot imutável da coleta definitiva nas tabelas curadas `videos` e `canais`, em Parquet.

Entrada: `data/raw/snapshot=<id>/`, conferida pelo manifesto SHA-256 e só lida, nunca alterada.
Saída: `data/processed/snapshot=<id>/curadoria=v<VERSAO>/`, fora do Git. Os arquivos são gravados numa pasta
temporária e só então renomeados; uma base existente nunca é sobrescrita.

Ficam os canais elegíveis, pela mesma regra da coleta, e neles os vídeos elegíveis nos papéis alvo, só histórico e
amostra LIM. Visualizações, comentários, inscritos e totais do canal ficam em tabelas próprias, separadas das que
servem aos atributos. A data-limite de retenção é a do snapshot.

Exemplo:
    uv run python -m src.curadoria --snapshot data/raw/snapshot=20261014T021437Z
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ValidationError

from src import coleta, snapshot
from src.config import RAIZ_PROJETO
from src.contratos import CanalColetado, MotivoExclusaoCanal, Papel, VideoColetado, url_do_video

VERSAO = "1"
DIRETORIO_PADRAO = RAIZ_PROJETO / "data" / "processed"
ARQUIVOS_SNAPSHOT = ("bruto_canais.json", "bruto_videos.json", "canais.jsonl", "proveniencia.json", "videos.jsonl")
PAPEIS_MANTIDOS = (Papel.ALVO, Papel.SO_HISTORICO, Papel.AMOSTRA_LIM)
PROVENIENCIA = "proveniencia.json"
TEXTO = pa.string()
DATA_UTC = pa.timestamp("us", tz="UTC")
USO_RESTRITO = "descritivo; nunca entra como atributo do modelo"


def _campo(nome: str, tipo: pa.DataType, descricao: str, *, anulavel: bool = True) -> pa.Field:
    return pa.field(nome, tipo, nullable=anulavel, metadata={"descricao": descricao})


ESQUEMA_VIDEOS = pa.schema(
    [
        _campo("video_id", TEXTO, "ID do vídeo no YouTube", anulavel=False),
        _campo("channel_id", TEXTO, "ID do canal; chave de canais.parquet", anulavel=False),
        _campo("snapshot_id", TEXTO, "snapshot de origem", anulavel=False),
        _campo("papel", TEXTO, "alvo: alvo supervisionado; so_historico: só referência do canal; amostra_lim: menos de "
               "30 dias na coleta, nunca alvo nem parte de treino, validação ou teste", anulavel=False),
        _campo("published_at", DATA_UTC, "publicação do vídeo", anulavel=False),
        _campo("collected_at", DATA_UTC, "observação: início da execução da coleta", anulavel=False),
        _campo("titulo", TEXTO, "título observado, sem alteração", anulavel=False),
        _campo("descricao", TEXTO, "descrição observada, sem alteração", anulavel=False),
        _campo("tags", pa.list_(TEXTO), "tags observadas, na ordem da API; lista vazia quando não há", anulavel=False),
        _campo("categoria_id", TEXTO, "categoria informada pela API"),
        _campo("idioma_padrao", TEXTO, "idioma padrão informado pela API"),
        _campo("idioma_audio_padrao", TEXTO, "idioma do áudio informado pela API"),
        _campo("duracao_s", pa.int32(), "duração em segundos"),
        _campo("largura_player", pa.int32(), "largura do player informada pela API"),
        _campo("altura_player", pa.int32(), "altura do player informada pela API"),
        _campo("formato", TEXTO, "classificação aproximada pela duração e pela proporção do player"),
        _campo("privacidade", TEXTO, "status de privacidade observado"),
        _campo("transmissao", TEXTO, "situação de transmissão observada"),
        _campo("curtidas", pa.int64(), "curtidas observadas na coleta; nula quando ocultas, nunca zero por ausência"),
        _campo("thumbnail_url", TEXTO, "endereço da thumbnail informado pela API; a imagem não é baixada"),
        _campo("url_video", TEXTO, "endereço público do vídeo", anulavel=False),
    ],
    metadata={"tabela": "videos", "versao_curadoria": VERSAO},
)
ESQUEMA_CANAIS = pa.schema(
    [
        _campo("channel_id", TEXTO, "ID do canal no YouTube", anulavel=False),
        _campo("snapshot_id", TEXTO, "snapshot de origem", anulavel=False),
        _campo("collected_at", DATA_UTC, "observação: início da execução da coleta", anulavel=False),
        _campo("titulo", TEXTO, "título observado do canal"),
        _campo("setor", TEXTO, "setor atribuído pela lista curada", anulavel=False),
        _campo("tipo_canal", TEXTO, "marca ou criador, pela lista curada", anulavel=False),
        _campo("pais_api", TEXTO, "país informado pela API; nulo quando ausente, nunca preenchido"),
        _campo("pais_aceito_por_excecao", pa.bool_(), "país ausente aceito pela lista fechada da exceção",
               anulavel=False),
        _campo("playlist_uploads", TEXTO, "playlist de uploads lida na coleta"),
        _campo("videos_alvo", pa.int32(), "vídeos com papel alvo nesta base", anulavel=False),
        _campo("videos_so_historico", pa.int32(), "vídeos com papel só histórico nesta base", anulavel=False),
        _campo("videos_amostra_lim", pa.int32(), "vídeos da amostra LIM nesta base", anulavel=False),
    ],
    metadata={"tabela": "canais", "versao_curadoria": VERSAO},
)
ESQUEMA_VIDEOS_RESTRITAS = pa.schema(
    [
        _campo("video_id", TEXTO, "chave de videos.parquet", anulavel=False),
        _campo("visualizacoes", pa.int64(), "visualizações observadas; " + USO_RESTRITO),
        _campo("comentarios", pa.int64(), "comentários observados; " + USO_RESTRITO),
    ],
    metadata={"tabela": "videos_estatisticas_restritas", "uso": USO_RESTRITO, "versao_curadoria": VERSAO},
)
ESQUEMA_CANAIS_RESTRITAS = pa.schema(
    [
        _campo("channel_id", TEXTO, "chave de canais.parquet", anulavel=False),
        _campo("inscritos", pa.int64(), "inscritos atuais; só no baseline 4 do relatório e na faixa de tamanho da "
               "ficha; " + USO_RESTRITO),
        _campo("total_videos", pa.int64(), "total de vídeos do canal; " + USO_RESTRITO),
    ],
    metadata={"tabela": "canais_estatisticas_restritas", "uso": USO_RESTRITO, "versao_curadoria": VERSAO},
)
TABELAS = (
    "videos.parquet",
    "canais.parquet",
    "videos_estatisticas_restritas.parquet",
    "canais_estatisticas_restritas.parquet",
)
LIMITACOES = (
    "IDs únicos não provam ausência de repostagens: o mesmo criativo pode ter sido publicado com outro ID.",
    "Textos e curtidas são os valores observados na coleta e podem ter sido editados ou acumulados depois da "
    "publicação; a avaliação temporal é uma reconstrução retrospectiva aproximada.",
    "A classificação de Short é aproximada: duração de até 180 s e player vertical ou quadrado informados pela API.",
    "amostra_lim fica fora dos alvos supervisionados e de treino, validação e teste.",
    "Tabelas *_estatisticas_restritas: visualizações, comentários, inscritos e totais do canal nunca entram como "
    "atributo; inscritos só no baseline 4 do relatório e na faixa de tamanho da ficha.",
)


class ErroCuradoria(ValueError):
    """Snapshot ou base inconsistente. Nada é gravado."""


@dataclass(frozen=True)
class SnapshotLido:
    pasta: Path
    proveniencia: dict
    canais: tuple[CanalColetado, ...]
    videos: tuple[VideoColetado, ...]
    situacao: dict[str, MotivoExclusaoCanal | None]
    arquivos_sha256: dict[str, str]
    sha256_manifesto: str


@dataclass(frozen=True)
class BaseCurada:
    origem: SnapshotLido
    tabelas: dict[str, pa.Table]
    contagens: dict


def _sha256(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def _ler_conferido(pasta: Path, esperados: Collection[str]) -> tuple[dict[str, bytes], bytes]:
    """Conteúdo dos arquivos conferido pelo manifesto; recusa arquivo faltando, a mais ou com SHA-256 divergente."""
    arquivo = pasta / snapshot.ARQUIVO_MANIFESTO
    if not arquivo.is_file():
        raise ErroCuradoria(f"{pasta.name}: manifesto ausente")
    manifesto = arquivo.read_bytes()
    listados: dict[str, str] = {}
    for linha in manifesto.decode("utf-8").splitlines():
        sha, separador, nome = linha.partition("  ")
        if not separador:
            raise ErroCuradoria(f"{pasta.name}: linha inválida no manifesto")
        if nome in listados:
            raise ErroCuradoria(f"{pasta.name}: arquivo repetido no manifesto: {nome}")
        listados[nome] = sha
    presentes = {p.name for p in pasta.iterdir()} - {snapshot.ARQUIVO_MANIFESTO}
    if set(listados) != set(esperados) or presentes != set(esperados):
        raise ErroCuradoria(
            f"{pasta.name}: arquivos diferentes do manifesto (esperados {sorted(esperados)}, listados "
            f"{sorted(listados)}, presentes {sorted(presentes)})"
        )
    conteudo = {nome: (pasta / nome).read_bytes() for nome in sorted(esperados)}
    divergentes = [nome for nome, dados in conteudo.items() if _sha256(dados) != listados[nome]]
    if divergentes:
        raise ErroCuradoria(f"{pasta.name}: SHA-256 diverge do manifesto em {divergentes}")
    return conteudo, manifesto


def _registros[M: BaseModel](dados: bytes, modelo: type[M], nome: str) -> tuple[M, ...]:
    registros = []
    for numero, linha in enumerate(dados.decode("utf-8").splitlines(), start=1):
        try:
            registros.append(modelo.model_validate_json(linha))
        except ValidationError as erro:
            raise ErroCuradoria(f"{nome}, linha {numero}: registro inválido ({erro.error_count()} erro(s))") from None
    return tuple(registros)


def _repetidos(ids: Sequence[str]) -> list[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def _conferir_referencias(prov: dict, canais: Sequence[CanalColetado], videos: Sequence[VideoColetado]) -> None:
    """IDs únicos, vídeos ligados a canais coletados, mesma observação e papel coerente com a data de publicação."""
    for entidade, ids in (("canal", [c.channel_id for c in canais]), ("vídeo", [v.video_id for v in videos])):
        if repetidos := _repetidos(ids):
            raise ErroCuradoria(f"{entidade} repetido: {repetidos[:5]}")
    ids_canais = {c.channel_id for c in canais}
    if orfaos := sorted({v.channel_id for v in videos} - ids_canais):
        raise ErroCuradoria(f"vídeos de canal não coletado: {orfaos[:5]}")
    if ids_canais != {linha["channel_id"] for linha in prov["canais"]}:
        raise ErroCuradoria("canais de canais.jsonl diferentes dos canais da proveniência")
    coletado_em = datetime.fromisoformat(prov["coletado_em_utc"])
    if any(r.snapshot_id != prov["snapshot_id"] or r.collected_at != coletado_em for r in (*canais, *videos)):
        raise ErroCuradoria("registros com snapshot_id ou instante de observação diferentes da proveniência")
    janela = coleta.Janela(coletado_em)
    if incoerentes := [v.video_id for v in videos if janela.papel(v.published_at) is not v.papel]:
        raise ErroCuradoria(f"papel incoerente com a data de publicação: {incoerentes[:5]}")


def _situacao_pela_regra(
    prov: dict, canais: tuple[CanalColetado, ...], videos: tuple[VideoColetado, ...]
) -> dict[str, MotivoExclusaoCanal | None]:
    """Reaplica `coleta.avaliar_canais` aos dados do snapshot e confere com a situação gravada na proveniência.

    Só os campos que a regra lê são reconstruídos: canais, vídeos, falhas por canal e leituras parciais.
    """
    linhas = {linha["channel_id"]: linha for linha in prov["canais"]}
    parciais = {
        channel_id: coleta.UploadsDoCanal(ids=(), itens_sem_data=0, paginas_lidas=linha["paginas_lidas"],
                                          interrompida=True)
        for channel_id, linha in linhas.items()
        if linha["amostra_parcial"]
    }
    resultado = coleta.ResultadoColeta(
        snapshot_id=prov["snapshot_id"],
        coletado_em=datetime.fromisoformat(prov["coletado_em_utc"]),
        canais=canais,
        videos=videos,
        canais_nao_encontrados=(),
        videos_indisponiveis=(),
        videos_invalidos=(),
        itens_playlist_sem_data=0,
        unidades_cota=0,
        falhas_por_canal=dict(prov["falhas"]["falhas_por_canal"]),
        uploads_por_canal=parciais,
    )
    situacao = coleta.avaliar_canais(resultado)
    divergentes = sorted(c for c, m in situacao.items() if (str(m) if m else "elegivel") != linhas[c]["situacao"])
    if divergentes:
        raise ErroCuradoria(f"situação dos canais diverge da regra de elegibilidade: {divergentes}")
    return situacao


def ler_snapshot(pasta: Path) -> SnapshotLido:
    """Lê e confere um snapshot de coleta definitiva: manifesto, registros, referências e situação dos canais."""
    pasta = Path(pasta)
    conteudo, manifesto = _ler_conferido(pasta, ARQUIVOS_SNAPSHOT)
    try:
        prov = json.loads(conteudo[PROVENIENCIA])
        if prov.get("tipo") != "coleta":
            tipo = prov.get("tipo")
            raise ErroCuradoria(f"{pasta.name}: só snapshot de coleta definitiva vira base (tipo {tipo!r})")
        if pasta.name != f"snapshot={prov['snapshot_id']}":
            raise ErroCuradoria(f"{pasta.name}: pasta não corresponde ao snapshot_id {prov['snapshot_id']}")
        canais = _registros(conteudo["canais.jsonl"], CanalColetado, "canais.jsonl")
        videos = _registros(conteudo["videos.jsonl"], VideoColetado, "videos.jsonl")
        _conferir_referencias(prov, canais, videos)
        situacao = _situacao_pela_regra(prov, canais, videos)
    except ErroCuradoria:
        raise
    except (KeyError, TypeError, ValueError) as erro:
        raise ErroCuradoria(f"{pasta.name}: proveniência incompleta ou inválida ({type(erro).__name__}: {erro})") from None
    return SnapshotLido(
        pasta=pasta,
        proveniencia=prov,
        canais=canais,
        videos=videos,
        situacao=situacao,
        arquivos_sha256={nome: _sha256(dados) for nome, dados in conteudo.items()},
        sha256_manifesto=_sha256(manifesto),
    )


def _linha_video(v: VideoColetado) -> dict:
    return {
        "video_id": v.video_id,
        "channel_id": v.channel_id,
        "snapshot_id": v.snapshot_id,
        "papel": str(v.papel),
        "published_at": v.published_at.astimezone(UTC),
        "collected_at": v.collected_at.astimezone(UTC),
        "titulo": v.titulo,
        "descricao": v.descricao,
        "tags": list(v.tags),
        "categoria_id": v.categoria_id,
        "idioma_padrao": v.idioma_padrao,
        "idioma_audio_padrao": v.idioma_audio_padrao,
        "duracao_s": v.duracao_s,
        "largura_player": v.largura_player,
        "altura_player": v.altura_player,
        "formato": str(v.formato),
        "privacidade": v.privacidade,
        "transmissao": v.transmissao,
        "curtidas": v.curtidas,
        "thumbnail_url": v.thumbnail_url,
        "url_video": url_do_video(v.video_id),
    }


def _linha_canal(c: CanalColetado, por_papel: Counter) -> dict:
    return {
        "channel_id": c.channel_id,
        "snapshot_id": c.snapshot_id,
        "collected_at": c.collected_at.astimezone(UTC),
        "titulo": c.titulo,
        "setor": str(c.setor),
        "tipo_canal": str(c.tipo_canal),
        "pais_api": c.pais,
        "pais_aceito_por_excecao": coleta.pais_aceito_por_excecao(c),
        "playlist_uploads": c.playlist_uploads,
        "videos_alvo": por_papel[c.channel_id, Papel.ALVO],
        "videos_so_historico": por_papel[c.channel_id, Papel.SO_HISTORICO],
        "videos_amostra_lim": por_papel[c.channel_id, Papel.AMOSTRA_LIM],
    }


def _contagens(snap: SnapshotLido, canais: list[CanalColetado], videos: list[VideoColetado]) -> dict:
    """Antes e depois da curadoria, com o motivo de cada exclusão; recusa contas que não fecham."""
    elegiveis = {c.channel_id for c in canais}
    pela_regra = Counter(str(v.motivo_exclusao) for v in snap.videos if not v.elegivel)
    em_excluidos = Counter(str(v.papel) for v in snap.videos if v.elegivel and v.channel_id not in elegiveis)
    if len(snap.videos) != sum(pela_regra.values()) + sum(em_excluidos.values()) + len(videos):
        raise ErroCuradoria("as contagens de vídeos antes e depois da curadoria não fecham")
    setor = {c.channel_id: str(c.setor) for c in canais}
    return {
        "antes": {
            "canais": len(snap.canais),
            "videos": len(snap.videos),
            "videos_por_papel": dict(Counter(str(v.papel) for v in snap.videos)),
        },
        "canais_excluidos_por_motivo": dict(Counter(str(m) for m in snap.situacao.values() if m)),
        "videos_excluidos_pela_regra_do_video": dict(pela_regra),
        "videos_elegiveis_em_canais_excluidos": dict(em_excluidos),
        "depois": {
            "canais": len(canais),
            "canais_por_setor_tipo": dict(sorted(Counter(f"{c.setor}/{c.tipo_canal}" for c in canais).items())),
            "videos": len(videos),
            "videos_por_papel": dict(Counter(str(v.papel) for v in videos)),
            "videos_por_setor_e_papel": dict(sorted(Counter(f"{setor[v.channel_id]}/{v.papel}" for v in videos)
                                                    .items())),
        },
    }


def curar(snap: SnapshotLido) -> BaseCurada:
    """Canais elegíveis e seus vídeos elegíveis nos papéis mantidos, em ordem estável."""
    canais = sorted((c for c in snap.canais if snap.situacao[c.channel_id] is None), key=lambda c: c.channel_id)
    elegiveis = {c.channel_id for c in canais}
    videos = sorted(
        (v for v in snap.videos if v.channel_id in elegiveis and v.elegivel and v.papel in PAPEIS_MANTIDOS),
        key=lambda v: (v.channel_id, v.published_at, v.video_id),
    )
    por_papel = Counter((v.channel_id, v.papel) for v in videos)
    tabelas = {
        "videos.parquet": pa.Table.from_pylist([_linha_video(v) for v in videos], schema=ESQUEMA_VIDEOS),
        "canais.parquet": pa.Table.from_pylist([_linha_canal(c, por_papel) for c in canais], schema=ESQUEMA_CANAIS),
        "videos_estatisticas_restritas.parquet": pa.Table.from_pylist(
            [{"video_id": v.video_id, "visualizacoes": v.visualizacoes, "comentarios": v.comentarios} for v in videos],
            schema=ESQUEMA_VIDEOS_RESTRITAS,
        ),
        "canais_estatisticas_restritas.parquet": pa.Table.from_pylist(
            [{"channel_id": c.channel_id, "inscritos": c.inscritos, "total_videos": c.total_videos} for c in canais],
            schema=ESQUEMA_CANAIS_RESTRITAS,
        ),
    }
    return BaseCurada(origem=snap, tabelas=tabelas, contagens=_contagens(snap, canais, videos))


def proveniencia(base: BaseCurada, hashes: dict[str, str]) -> dict:
    """Origem conferida, versão, código, tabelas, contagens, exceção de país e retenção herdada."""
    snap, prov = base.origem, base.origem.proveniencia
    codigo = snapshot.estado_do_codigo()
    return {
        "versao_curadoria": VERSAO,
        "tipo": "base_curada",
        "gerado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "origem": {
            "snapshot_id": prov["snapshot_id"],
            "pasta": snap.pasta.name,
            "tipo": prov["tipo"],
            "completo": prov["completo"],
            "coletado_em_utc": prov["coletado_em_utc"],
            "terminado_em_utc": prov["terminado_em_utc"],
            "manifesto": snapshot.ARQUIVO_MANIFESTO,
            "sha256_manifesto": snap.sha256_manifesto,
            "arquivos_sha256": snap.arquivos_sha256,
            "commit_da_coleta": prov["codigo"]["commit_git"],
            "lista_canais_sha256": prov["lista_canais"]["sha256"],
            "janela_publicacao_utc": prov["janela_publicacao_utc"],
        },
        "regras": {
            "canais": "elegíveis pela regra da coleta (coleta.avaliar_canais), conferida com a proveniência",
            "videos": "sem motivo de exclusão e com papel em " + ", ".join(str(p) for p in PAPEIS_MANTIDOS),
            "minimo_shorts_por_canal": prov["parametros"]["minimo_shorts_por_canal"],
            "pais_exigido": prov["parametros"]["pais_exigido"],
            "canais_excecao_pais": prov["parametros"]["canais_excecao_pais"],
        },
        "codigo": {**codigo, "dependencias": {**codigo["dependencias"], "pyarrow": pa.__version__}},
        "tabelas": {
            nome: {"linhas": tabela.num_rows, "sha256": hashes[nome]} for nome, tabela in base.tabelas.items()
        },
        "contagens": base.contagens,
        "excecao_pais_aplicada": prov["excecao_pais_aplicada"],
        "retencao": {**prov["retencao"], "herdada_do_snapshot": True},
        "limitacoes": list(LIMITACOES),
    }


def _bytes_parquet(tabela: pa.Table) -> bytes:
    saida = pa.BufferOutputStream()
    pq.write_table(tabela, saida)
    return saida.getvalue().to_pybytes()


def _gravar(pasta: Path, nome: str, dados: bytes) -> None:
    (pasta / nome).write_bytes(dados)


def salvar_base(base: BaseCurada, *, destino: Path = DIRETORIO_PADRAO) -> Path:
    """Grava a base numa pasta nova e recusa sobrescrever.

    Tudo é montado em memória, gravado numa pasta temporária e só então renomeado para o nome final: uma falha no
    meio não deixa base parcial com aparência de completa. Feito para uma execução por vez: duas curadorias
    simultâneas do mesmo snapshot não são coordenadas entre si.
    """
    pai = Path(destino) / base.origem.pasta.name
    final = pai / f"curadoria=v{VERSAO}"
    if final.exists():
        raise FileExistsError(f"base já existe e não será sobrescrita: {pai.name}/{final.name}")
    arquivos = {nome: _bytes_parquet(tabela) for nome, tabela in base.tabelas.items()}
    hashes = {nome: _sha256(dados) for nome, dados in arquivos.items()}
    arquivos[PROVENIENCIA] = json.dumps(proveniencia(base, hashes), ensure_ascii=False, indent=1).encode("utf-8")
    manifesto = "".join(f"{_sha256(dados)}  {nome}\n" for nome, dados in sorted(arquivos.items()))
    arquivos[snapshot.ARQUIVO_MANIFESTO] = manifesto.encode("utf-8")
    pai.mkdir(parents=True, exist_ok=True)
    temporaria = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".parcial", dir=pai))
    try:
        for nome, dados in arquivos.items():
            _gravar(temporaria, nome, dados)
        temporaria.rename(final)
    except OSError as erro:
        shutil.rmtree(temporaria, ignore_errors=True)
        if final.exists():
            raise FileExistsError(f"base já existe e não será sobrescrita: {pai.name}/{final.name}") from erro
        raise
    return final


def ler_base(pasta: Path) -> dict[str, pa.Table]:
    """Tabelas de uma base curada, só depois de conferir o manifesto."""
    conteudo, _ = _ler_conferido(Path(pasta), (*TABELAS, PROVENIENCIA))
    return {nome: pq.read_table(pa.BufferReader(conteudo[nome])) for nome in TABELAS}


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.curadoria", description="Organiza o snapshot da coleta definitiva nas tabelas curadas."
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="pasta snapshot=<id> da coleta definitiva")
    parser.add_argument("--destino", type=Path, default=DIRETORIO_PADRAO, help="pasta das bases curadas")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """0 = base gravada; 2 = snapshot recusado; 3 = base já existe; 4 = falha ao gravar."""
    args = _argumentos(argv)
    try:
        base = curar(ler_snapshot(args.snapshot))
    except (OSError, ErroCuradoria) as erro:
        print(f"Snapshot recusado ({type(erro).__name__}): {erro}. Nada foi gravado.", file=sys.stderr)
        return 2
    try:
        pasta = salvar_base(base, destino=args.destino)
    except FileExistsError as erro:
        print(f"{erro}.", file=sys.stderr)
        return 3
    except OSError as erro:
        print(f"Falha ao gravar a base ({type(erro).__name__}: {erro}).", file=sys.stderr)
        return 4
    resumo = {
        "pasta": str(pasta),
        "snapshot_id": base.origem.proveniencia["snapshot_id"],
        **base.contagens["depois"],
        "retencao_apagar_ate_utc": base.origem.proveniencia["retencao"]["apagar_ate_utc"],
    }
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
