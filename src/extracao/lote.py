"""Executa a extração local de legenda e capa sobre a amostra congelada, em etapas retomáveis."""

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import RAIZ_PROJETO
from src.dados import amostra as am, curadoria, snapshot
from src.extracao import imagem as capa, texto as leg
from src.extracao.rubrica import gravar_json_atomico
from src.proveniencia import agora_utc as _agora, sha256_bytes as _sha256


# ──── Executa a extração local de legenda e capa sobre a amostra congelada, em etapas retomáveis.
VERSAO = "1"
ETAPAS_IMAGEM = ("cor", "rosto", "ocr", "clip")
ETAPAS = ("legenda", "baixar", *ETAPAS_IMAGEM)
ARQUIVO_LEGENDA, ARQUIVO_CAPA, ARQUIVO_MANIFESTO = "legenda.parquet", "capa.parquet", "manifesto.json"
DIRETORIO_PADRAO = RAIZ_PROJETO / "data" / "features"
COLUNAS_FONTE = ["video_id", "titulo", "descricao", "tags", "thumbnail_url"]
INTERVALO_DOWNLOAD_S = 0.2
TENTATIVAS_DOWNLOAD = 3
LIMITACOES = [
    "A thumbnail é a lida da URL registrada na coleta e pode ter sido trocada depois da publicação; o download agora "
    "não renova a retenção dos dados de origem.",
    "Extrator funcionando não é evidência de que o atributo melhora a previsão; nada foi medido contra curtidas.",
    "O CLIP ViT-B/32 é descrito na ficha como resultado de pesquisa, com uso implantado fora do escopo previsto; usado "
    "aqui só como extrator de atributos em projeto acadêmico.",
    "BlazeFace de curto alcance foi feito para imagens tipo selfie; rostos pequenos ou de perfil podem não ser "
    "detectados. Rosto não é identificado.",
    "A rota capa + Gemini não implementa a camada local de vídeo do TAPI (frames, Whisper, movimento, cortes): desvio "
    "continua declarado.",
]


class ErroExtracao(ValueError):
    """Entrada, população ou etapa incompatível."""


class CacheIncompativel(ErroExtracao):
    """Pasta de trabalho criada com outra configuração."""


@dataclass(frozen=True)
class Item:
    video_id: str
    titulo: str | None
    descricao: str | None
    tags: tuple[str, ...]
    thumbnail_url: str | None


def _chave(conteudo: dict) -> str:
    return _sha256(json.dumps(conteudo, sort_keys=True, ensure_ascii=False).encode("utf-8"))[:16]


def _videos_conferidos(pasta_base: Path) -> tuple[pa.Table, str]:
    linhas = (pasta_base / snapshot.ARQUIVO_MANIFESTO).read_text(encoding="utf-8").splitlines()
    registrados = {nome: valor for valor, nome in (linha.split("  ", 1) for linha in linhas if linha.strip())}
    dados = (pasta_base / "videos.parquet").read_bytes()
    if registrados.get("videos.parquet") != _sha256(dados):
        raise ErroExtracao("videos.parquet não confere com o manifesto da base")
    return pq.read_table(pa.BufferReader(dados), columns=COLUNAS_FONTE), registrados["videos.parquet"]


def ler_populacao(pasta_base: Path, arquivo_amostra: Path) -> list[Item]:
    """Itens da amostra congelada, na ordem registrada, só com texto e URL da thumbnail."""
    pasta_base = Path(pasta_base)
    amostra = am.ler_amostra(Path(arquivo_amostra))
    if amostra["base"]["sha256_manifesto"] != _sha256((pasta_base / snapshot.ARQUIVO_MANIFESTO).read_bytes()):
        raise ErroExtracao("a amostra foi sorteada sobre outra base")
    tabela, _ = _videos_conferidos(pasta_base)
    fonte = {v["video_id"]: v for v in tabela.to_pylist()}
    faltando = [x["video_id"] for x in amostra["lista"] if x["video_id"] not in fonte]
    if faltando:
        raise ErroExtracao(f"vídeos da amostra ausentes na base: {faltando[:5]}")
    return [Item(v["video_id"], v["titulo"], v["descricao"], tuple(v["tags"] or ()), v["thumbnail_url"])
            for v in (fonte[x["video_id"]] for x in amostra["lista"])]


def filtrar_ids(itens: Sequence[Item], ids: Sequence[str]) -> list[Item]:
    """Subconjunto pedido, na ordem congelada da amostra; ID fora da amostra é recusado, nunca ignorado."""
    if isinstance(ids, (str, bytes)) or not all(isinstance(i, str) for i in ids):
        raise ErroExtracao("ids deve ser uma lista de strings")
    pedidos = set(ids)
    desconhecidos = sorted(pedidos - {i.video_id for i in itens})
    if desconhecidos:
        raise ErroExtracao(f"IDs fora da amostra congelada: {desconhecidos[:5]}")
    return [i for i in itens if i.video_id in pedidos]


class Trabalho:
    """Pasta de trabalho retomável: um JSON por vídeo e etapa, mais as imagens baixadas."""

    def __init__(self, pasta: Path) -> None:
        self.pasta = Path(pasta)

    def _arquivo(self, etapa: str, video_id: str) -> Path:
        return self.pasta / etapa / f"{video_id}.json"

    def imagem(self, video_id: str) -> Path:
        return self.pasta / "imagens" / f"{video_id}.img"

    def configuracao(self, etapa: str) -> dict | None:
        arquivo = self.pasta / etapa / "_configuracao.json"
        return json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else None

    def preparar(self, etapa: str, identificacao: dict) -> str:
        """Registra a configuração da etapa; recusa uma pasta criada com outra."""
        conteudo = {"etapa": etapa, "versao": VERSAO, "identificacao": identificacao,
                    "regras": capa.REGRAS if etapa in ETAPAS_IMAGEM else leg.REGRAS if etapa == "legenda" else {}}
        chave = _chave(conteudo)
        registrada = self.configuracao(etapa)
        if registrada is not None and registrada["chave"] != chave:
            raise CacheIncompativel(f"etapa {etapa}: a pasta de trabalho foi criada com outra configuração")
        if registrada is None:
            gravar_json_atomico(self.pasta / etapa / "_configuracao.json", conteudo | {"chave": chave,
                                                                                       "criada_em_utc": _agora()})
        return chave

    def ler(self, etapa: str, video_id: str) -> dict | None:
        arquivo = self._arquivo(etapa, video_id)
        return json.loads(arquivo.read_text(encoding="utf-8"))["resultado"] if arquivo.exists() else None

    def gravar(self, etapa: str, video_id: str, chave: str, resultado: dict) -> None:
        gravar_json_atomico(self._arquivo(etapa, video_id), {"video_id": video_id, "chave": chave,
                                                            "gravado_em_utc": _agora(), "resultado": resultado})

    def pendentes(self, etapa: str, itens: Sequence[Item]) -> list[Item]:
        return [i for i in itens if not self._arquivo(etapa, i.video_id).exists()]


def etapa_legenda(itens: Sequence[Item], trabalho: Trabalho, codificador: leg.Codificador, *, lote: int = 32) -> Counter:
    chave = trabalho.preparar("legenda", codificador.identificacao())
    pendentes, estados = trabalho.pendentes("legenda", itens), Counter()
    for inicio in range(0, len(pendentes), lote):
        parte = pendentes[inicio:inicio + lote]
        for item, resultado in zip(parte, leg.extrair_legendas([(i.titulo, i.descricao, i.tags) for i in parte],
                                                                codificador), strict=True):
            trabalho.gravar("legenda", item.video_id, chave, resultado)
            estados[resultado["estado"]] += 1
    return estados


def _baixar(url: str, cliente: httpx.Client) -> dict:
    ultimo: dict = {}
    for tentativa in range(TENTATIVAS_DOWNLOAD):
        try:
            resposta = cliente.get(url, timeout=30, follow_redirects=True)
        except httpx.TransportError as erro:
            ultimo = {"estado": "falha_download", "erro": type(erro).__name__}
        else:
            if resposta.status_code == 200:
                return {"estado": "ok", "http_status": 200, "conteudo": resposta.content}
            ultimo = {"estado": "falha_download", "http_status": resposta.status_code}
            if resposta.status_code < 500 and resposta.status_code != 429:
                return ultimo
        if tentativa + 1 < TENTATIVAS_DOWNLOAD:
            time.sleep(2 ** tentativa)
    return ultimo


def etapa_baixar(itens: Sequence[Item], trabalho: Trabalho, cliente: httpx.Client, *,
                 intervalo: float = INTERVALO_DOWNLOAD_S) -> Counter:
    """Baixa só a thumbnail registrada; distingue imagem ausente, falha de download e imagem corrompida."""
    chave = trabalho.preparar("baixar", {"fonte": "thumbnail_url registrada na base curada",
                                         "tentativas": TENTATIVAS_DOWNLOAD})
    estados = Counter()
    for item in trabalho.pendentes("baixar", itens):
        if not item.thumbnail_url:
            resultado = {"estado": "sem_imagem"}
        else:
            baixado = _baixar(item.thumbnail_url, cliente)
            resultado = {k: v for k, v in baixado.items() if k != "conteudo"} | {"url": item.thumbnail_url}
            if baixado["estado"] == "ok":
                dados = baixado["conteudo"]
                resultado |= {"bytes": len(dados), "sha256": _sha256(dados)}
                try:
                    altura, largura = capa.decodificar_imagem(dados).shape[:2]
                    resultado |= {"largura": largura, "altura": altura}
                    trabalho.imagem(item.video_id).parent.mkdir(parents=True, exist_ok=True)
                    trabalho.imagem(item.video_id).write_bytes(dados)
                except capa.ImagemInvalida:
                    resultado["estado"] = "imagem_corrompida"
            if intervalo:
                time.sleep(intervalo)
        trabalho.gravar("baixar", item.video_id, chave, resultado)
        estados[resultado["estado"]] += 1
    return estados


def etapa_imagem(itens: Sequence[Item], trabalho: Trabalho, nome: str, extrator: Callable, *,
                 identificacao: dict | None = None) -> Counter:
    """Aplica um extrator à imagem já baixada e recortada; sem imagem, repete o estado do download."""
    if nome not in ETAPAS_IMAGEM:
        raise ErroExtracao(f"etapa de imagem desconhecida: {nome}")
    chave = trabalho.preparar(nome, identificacao if identificacao is not None else extrator.identificacao())
    estados = Counter()
    for item in trabalho.pendentes(nome, itens):
        baixado = trabalho.ler("baixar", item.video_id)
        if baixado is None:
            raise ErroExtracao(f"{item.video_id}: etapa baixar ainda não executada")
        if baixado["estado"] != "ok":
            resultado = {"estado": baixado["estado"]}
        else:
            try:
                imagem = capa.decodificar_imagem(trabalho.imagem(item.video_id))
            except (capa.ImagemInvalida, OSError):
                resultado = {"estado": "imagem_corrompida"}
            else:
                recortada, recorte = capa.remover_faixas_pretas(imagem)
                resultado = capa.aplicar_extrator(extrator, recortada) | {"recorte": recorte}
        trabalho.gravar(nome, item.video_id, chave, resultado)
        estados[resultado["estado"]] += 1
    return estados


# --- materialização ---------------------------------------------------------------------------------------------


def _campo(nome: str, tipo: pa.DataType, papel: str) -> pa.Field:
    return pa.field(nome, tipo, metadata={"papel_coluna": papel})


ESQUEMA_LEGENDA = pa.schema(
    [_campo("video_id", pa.string(), "identificacao"), _campo("leg_estado", pa.string(), "auditoria"),
     _campo("leg_n_tokens", pa.int32(), "auditoria"), _campo("leg_truncado", pa.bool_(), "auditoria")]
    + [_campo(c, pa.int32(), "atributo") for c in leg.CONTAGENS_LEGENDA]
    + [_campo("leg_emb", pa.list_(pa.float32(), leg.DIMENSAO_TEXTO), "atributo")],
    metadata={"tabela": "legenda", "versao": VERSAO})
_INTEIROS_CAPA = {"cor_dominante_r", "cor_dominante_g", "cor_dominante_b", "rosto_presente", "rosto_n",
                  "ocr_tem_texto", "ocr_n_caixas", "ocr_n_caracteres", "ocr_n_palavras"}
ESQUEMA_CAPA = pa.schema(
    [_campo("video_id", pa.string(), "identificacao"), _campo("capa_estado_imagem", pa.string(), "auditoria"),
     _campo("capa_recortada", pa.bool_(), "auditoria"), _campo("capa_motivo_recorte", pa.string(), "auditoria")]
    + [_campo(f"capa_caixa_{k}", pa.int32(), "auditoria") for k in ("x0", "y0", "x1", "y1")]
    + [_campo(f"{e}_estado", pa.string(), "auditoria") for e in ETAPAS_IMAGEM]
    + [_campo(c, pa.int32() if c in _INTEIROS_CAPA else pa.float64(), "atributo")
       for c in (*capa.MEDIDAS_COR, *capa.MEDIDAS_ROSTO, *capa.MEDIDAS_OCR)]
    + [_campo("capa_emb", pa.list_(pa.float32(), capa.DIMENSAO_IMAGEM), "atributo"),
       _campo("ocr_texto", pa.string(), "auxiliar")],
    metadata={"tabela": "capa", "versao": VERSAO})
_ATRIBUTOS_POR_ETAPA = {"cor": capa.MEDIDAS_COR, "rosto": capa.MEDIDAS_ROSTO, "ocr": (*capa.MEDIDAS_OCR, "ocr_texto"),
                        "clip": ("capa_emb",)}


def _linha_legenda(item: Item, r: dict) -> dict:
    """Contagens vêm do texto e valem mesmo sem embedding; só o vetor depende do codificador."""
    vetor = leg.validar_vetor(r["vetor"], leg.DIMENSAO_TEXTO) if r["estado"] == "ok" else None
    return {"video_id": item.video_id, "leg_estado": r["estado"], "leg_n_tokens": r["n_tokens"],
            "leg_truncado": r["truncado"], **{c: r[c] for c in leg.CONTAGENS_LEGENDA}, "leg_emb": vetor}


def _linha_capa(item: Item, trabalho: Trabalho) -> dict:
    linha = {"video_id": item.video_id, "capa_estado_imagem": trabalho.ler("baixar", item.video_id)["estado"]}
    recorte = None
    for etapa in ETAPAS_IMAGEM:
        r = trabalho.ler(etapa, item.video_id)
        linha[f"{etapa}_estado"] = r["estado"]
        recorte = recorte or r.get("recorte")
        for coluna in _ATRIBUTOS_POR_ETAPA[etapa]:
            linha[coluna] = r.get(coluna) if r["estado"] == "ok" else None
    if linha["capa_emb"] is not None:
        linha["capa_emb"] = capa.validar_vetor(linha["capa_emb"], capa.DIMENSAO_IMAGEM)
    if recorte is not None:
        caixa = recorte["caixa"]
        linha |= {"capa_recortada": recorte["recortado"], "capa_motivo_recorte": recorte["motivo"],
                  "capa_caixa_x0": caixa[0], "capa_caixa_y0": caixa[1], "capa_caixa_x1": caixa[2],
                  "capa_caixa_y1": caixa[3]}
    return linha


def _parquet(linhas: list[dict], esquema: pa.Schema) -> bytes:
    saida = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([{c: linha.get(c) for c in esquema.names} for linha in linhas],
                                        schema=esquema), saida)
    return saida.getvalue().to_pybytes()


def _contar(linhas: list[dict], coluna: str) -> dict:
    return dict(sorted(Counter(str(linha.get(coluna)) for linha in linhas).items()))


def _contagens(legenda: list[dict], imagens: list[dict], trabalho: Trabalho, itens: Sequence[Item]) -> dict:
    erros = Counter()
    for etapa in ETAPAS_IMAGEM:
        for item in itens:
            r = trabalho.ler(etapa, item.video_id)
            if r["estado"] == "falha_extrator":
                erros[f"{etapa}:{r.get('erro')}"] += 1
    return {
        "leg_estado": _contar(legenda, "leg_estado"), "leg_truncado": _contar(legenda, "leg_truncado"),
        "capa_estado_imagem": _contar(imagens, "capa_estado_imagem"),
        "capa_motivo_recorte": _contar(imagens, "capa_motivo_recorte"),
        **{f"{e}_estado": _contar(imagens, f"{e}_estado") for e in ETAPAS_IMAGEM},
        "erros_dos_extratores": dict(erros),
        "rosto_presente": _contar(imagens, "rosto_presente"), "ocr_tem_texto": _contar(imagens, "ocr_tem_texto"),
    }


def _gravar_pasta(final: Path, arquivos: dict[str, bytes]) -> Path:
    final.parent.mkdir(parents=True, exist_ok=True)
    temporaria = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".parcial", dir=final.parent))
    try:
        for nome, dados in arquivos.items():
            (temporaria / nome).write_bytes(dados)
        if final.exists():
            raise FileExistsError(f"saída já existe e não será sobrescrita: {final}")
        temporaria.rename(final)
    except BaseException:
        shutil.rmtree(temporaria, ignore_errors=True)
        raise
    return final


def materializar(pasta_base: Path, arquivo_amostra: Path, trabalho: Trabalho, *, destino: Path | None = None,
                 ids: Sequence[str] | None = None) -> Path:
    """Grava legenda.parquet, capa.parquet e o manifesto uma única vez.

    Sem `ids`, todos os IDs da amostra. Com `ids`, só esse subconjunto priorizado (na ordem congelada); a saída
    padrão ganha o hash dos IDs no nome e o manifesto registra o subconjunto.
    """
    pasta_base, arquivo_amostra = Path(pasta_base), Path(arquivo_amostra)
    itens = ler_populacao(pasta_base, arquivo_amostra)
    total_amostra = len(itens)
    if ids is not None:
        itens = filtrar_ids(itens, ids)
        if not itens:
            raise ErroExtracao("subconjunto vazio")
    sha_ids = _sha256("\n".join(i.video_id for i in itens).encode("utf-8"))
    for etapa in ETAPAS:
        faltando = len(trabalho.pendentes(etapa, itens))
        if faltando:
            raise ErroExtracao(f"etapa {etapa} incompleta: {faltando} vídeos sem resultado")
    amostra = am.ler_amostra(arquivo_amostra)
    prov = json.loads((pasta_base / curadoria.PROVENIENCIA).read_text(encoding="utf-8"))
    sufixo = f"_subconjunto_{sha_ids[:12]}" if ids is not None else ""
    final = Path(destino) if destino else (DIRETORIO_PADRAO / f"snapshot={amostra['base']['snapshot_id']}"
                                           / f"legenda_capa=v{VERSAO}{sufixo}")
    if final.exists():
        raise FileExistsError(f"saída já existe e não será sobrescrita: {final}")
    legenda = [_linha_legenda(i, trabalho.ler("legenda", i.video_id)) for i in itens]
    imagens = [_linha_capa(i, trabalho) for i in itens]
    arquivos = {ARQUIVO_LEGENDA: _parquet(legenda, ESQUEMA_LEGENDA), ARQUIVO_CAPA: _parquet(imagens, ESQUEMA_CAPA)}
    _, sha_videos = _videos_conferidos(pasta_base)
    manifesto = {
        "versao": VERSAO, "tipo": "atributos_locais_legenda_e_capa", "gerado_em_utc": _agora(),
        "base": amostra["base"] | {"sha256_videos_parquet": sha_videos, "colunas_lidas": COLUNAS_FONTE},
        "populacao": {"origem": f"{arquivo_amostra.parent.name}/{arquivo_amostra.name}", "n": len(itens),
                      "sha256_lista": amostra["sha256_lista"], "papel": "alvos congelados; LIM fora",
                      **({"subconjunto": {"n_amostra_congelada": total_amostra, "sha256_ids": sha_ids,
                                          "motivo": "vídeos priorizados para o desenvolvimento; o restante da "
                                          "amostra pode ser extraído depois pelo mesmo caminho"}}
                         if ids is not None else {}),
                      "escolha": "os 1.500 alvos congelados primeiro; não decide a população final "
                      "de treino de cada modelo"},
        "configuracoes": {e: trabalho.configuracao(e) for e in ETAPAS},
        "regras": {"legenda": leg.REGRAS, "capa": capa.REGRAS},
        "colunas": {nome: {papel: [c.name for c in esquema if c.metadata[b"papel_coluna"].decode() == papel]
                           for papel in ("identificacao", "atributo", "auditoria", "auxiliar")}
                    for nome, esquema in (("legenda", ESQUEMA_LEGENDA), ("capa", ESQUEMA_CAPA))},
        "tabelas": {nome: {"linhas": len(itens), "sha256": _sha256(dados)} for nome, dados in arquivos.items()},
        "contagens": _contagens(legenda, imagens, trabalho, itens),
        "codigo": snapshot.estado_do_codigo(),
        "retencao": {"apagar_ate_utc": prov["retencao"]["apagar_ate_utc"],
                     "herdada_de": f"snapshot={prov['origem']['snapshot_id']}",
                     "abrange": "tabelas, manifesto, thumbnails baixadas e pasta de trabalho; pesos públicos ficam "
                     "fora"},
        "limitacoes": LIMITACOES,
        "nao_incluido": ["PCA, imputação ou seleção de atributos (só dentro das dobras de treino)",
                         "transcrição do Gemini (família N4)", "LIM e os alvos fora da amostra congelada"],
    }
    arquivos[ARQUIVO_MANIFESTO] = json.dumps(manifesto, ensure_ascii=False, indent=1).encode("utf-8")
    return _gravar_pasta(final, arquivos)


# --- linha de comando -------------------------------------------------------------------------------------------


def _extrator(etapa: str) -> tuple[Callable, dict | None]:
    if etapa == "cor":
        import cv2

        return capa.medidas_cor, {"modelo": "OpenCV", "opencv": cv2.__version__}
    if etapa == "rosto":
        return capa.DetectorRosto(), None
    if etapa == "ocr":
        return capa.LeitorTexto(), None
    return capa.CodificadorImagem(), None


def main(argv: Sequence[str] | None = None) -> int:
    """0 = etapa concluída; 2 = entrada recusada; 3 = saída já existe."""
    parser = argparse.ArgumentParser(prog="python -m src.extracao.lote",
                                     description="Extração local de legenda e capa em etapas retomáveis.")
    parser.add_argument("--etapa", required=True, choices=(*ETAPAS, "materializar"))
    parser.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n>")
    parser.add_argument("--amostra", type=Path, required=True, help="amostra.json congelado")
    parser.add_argument("--trabalho", type=Path, help="pasta de trabalho (padrão: ao lado da saída)")
    parser.add_argument("--destino", type=Path, help="pasta final da materialização")
    parser.add_argument("--limite", type=int, help="só os primeiros N vídeos da ordem congelada (piloto)")
    parser.add_argument("--ids", type=Path, help="JSON com a lista de video_id a processar (subconjunto priorizado)")
    args = parser.parse_args(argv)
    inicio = time.monotonic()
    try:
        itens = ler_populacao(args.base, args.amostra)
        snapshot_id = am.ler_amostra(args.amostra)["base"]["snapshot_id"]
        trabalho = Trabalho(args.trabalho or DIRETORIO_PADRAO / f"snapshot={snapshot_id}"
                            / f"legenda_capa=v{VERSAO}_trabalho")
        ids = json.loads(args.ids.read_text(encoding="utf-8")) if args.ids else None
        itens = filtrar_ids(itens, ids) if ids is not None else itens
        itens = itens[:args.limite] if args.limite else itens
        if args.etapa == "materializar":
            pasta = materializar(args.base, args.amostra, trabalho, destino=args.destino, ids=ids)
            print(json.dumps({"pasta": str(pasta)}, ensure_ascii=False))
            return 0
        if args.etapa == "legenda":
            estados = etapa_legenda(itens, trabalho, leg.CodificadorTexto())
        elif args.etapa == "baixar":
            with httpx.Client(headers={"User-Agent": "casegalaxy-extracao/1"}) as cliente:
                estados = etapa_baixar(itens, trabalho, cliente)
        else:
            extrator, identificacao = _extrator(args.etapa)
            estados = etapa_imagem(itens, trabalho, args.etapa, extrator, identificacao=identificacao)
    except FileExistsError as erro:
        print(f"{erro}.", file=sys.stderr)
        return 3
    except (ValueError, KeyError, OSError) as erro:
        print(f"Extração recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({"etapa": args.etapa, "itens": len(itens), "novos": dict(estados),
                      "segundos": round(time.monotonic() - inicio, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
