"""Extração de atributos do criativo para a linha de inferência, com caches por conteúdo e configuração.

Os mesmos extratores do treino são usados na previsão (ARQUITETURA §5.5). Nenhum cache é chaveado só por
`video_id`: o vetor da legenda depende do texto composto e do codificador; as medidas da capa, dos bytes da imagem e
dos extratores. Legenda alterada, imagem trocada ou codificador diferente produzem outra chave e são recalculados.
O que não pôde ser extraído fica ausente (nulo), nunca zero, e o estado da etapa é devolvido para os avisos.
"""

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from src import dataset_desenvolvimento as ds
from src import extracao_capa as capa
from src import extracao_legenda as leg
from src import gemini_api as ga
from src import rubrica as rb
from src.agente.contratos import Codigo, ErroAgente, PedidoPrever, video_id_da_url  # noqa: F401
from src.cache_gemini import CacheAnotacoes, CacheCorrompido, CacheIncompativel, gravar_json_atomico
from src.contratos import Setor

ETAPAS_CAPA = ("cor", "rosto", "ocr", "clip")
MAX_DURACAO_PADRAO_S = 180
MAX_BYTES_IMAGEM = 20_000_000
logger = logging.getLogger(__name__)


def chave_conteudo(identificacao: Mapping, conteudo: str) -> str:
    """SHA-256 do conteúdo junto com a identificação de quem o processa."""
    base = json.dumps({"identificacao": identificacao, "conteudo": conteudo}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


class CacheConteudo:
    """Um JSON por chave de conteúdo; entrada ilegível conta como ausente e é recalculada."""

    def __init__(self, pasta: Path) -> None:
        self.pasta = Path(pasta)

    def _arquivo(self, chave: str) -> Path:
        return self.pasta / f"{chave}.json"

    def ler(self, chave: str) -> dict | None:
        arquivo = self._arquivo(chave)
        if not arquivo.is_file():
            return None
        try:
            return json.loads(arquivo.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def gravar(self, chave: str, conteudo: Mapping) -> None:
        """O cache é opcional: falha de gravação é registrada e não derruba a previsão."""
        try:
            gravar_json_atomico(self._arquivo(chave), dict(conteudo))
        except OSError as erro:
            logger.warning("cache não gravado (%s)", type(erro).__name__)


def _sha_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


class CodificadorEmCache:
    """Envolve um `leg.Codificador` com cache por (identificação do codificador, texto composto)."""

    def __init__(self, interno: leg.Codificador, cache: CacheConteudo) -> None:
        self.interno, self.cache = interno, cache
        self._identificacao: dict | None = None
        self.chamadas_ao_codificador = 0

    def identificacao(self) -> dict:
        if self._identificacao is None:
            self._identificacao = self.interno.identificacao()
        return self._identificacao

    def codificar(self, textos: Sequence[str]) -> list[leg.VetorTexto]:
        ident = self.identificacao()
        chaves = [chave_conteudo(ident, t) for t in textos]
        resultado: list[leg.VetorTexto | None] = [None] * len(textos)
        faltam = []
        for i, chave in enumerate(chaves):
            entrada = self.cache.ler(chave)
            try:
                if entrada is None or entrada.get("texto_sha256") != _sha_texto(textos[i]):
                    raise KeyError(chave)
                resultado[i] = leg.VetorTexto(leg.validar_vetor(entrada["vetor"], leg.DIMENSAO_TEXTO),
                                              int(entrada["n_tokens"]), bool(entrada["truncado"]))
            except (KeyError, ValueError, TypeError):
                faltam.append(i)
        if faltam:
            self.chamadas_ao_codificador += 1
            novos = self.interno.codificar([textos[i] for i in faltam])
            for i, v in zip(faltam, novos, strict=True):
                resultado[i] = v
                self.cache.gravar(chaves[i], {
                    "codificador": ident, "texto_sha256": _sha_texto(textos[i]),
                    "vetor": list(v.vetor), "n_tokens": v.n_tokens, "truncado": v.truncado})
        return [r for r in resultado if r is not None]


# --- capa ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtratoresCapa:
    """Os quatro extratores da capa: `cor` (função), `rosto`, `ocr` e `clip` (objetos com `identificacao`)."""

    cor: Callable[[np.ndarray], dict]
    rosto: Callable[[np.ndarray], dict]
    ocr: Callable[[np.ndarray], dict]
    clip: Callable[[np.ndarray], dict]
    identificacao_cor: Mapping = field(default_factory=lambda: {"modelo": "OpenCV"})

    def identificacao(self) -> dict:
        return {"cor": dict(self.identificacao_cor), "rosto": self.rosto.identificacao(),
                "ocr": self.ocr.identificacao(), "clip": self.clip.identificacao(), "versao_capa": capa.VERSAO}


def _linha_da_capa(estado_imagem: str, resultados: Mapping[str, dict]) -> dict:
    """Mesmas colunas de `capa.parquet` (extracao_legenda_capa._linha_capa), a partir dos resultados por etapa."""
    atributos = {"cor": capa.MEDIDAS_COR, "rosto": capa.MEDIDAS_ROSTO, "ocr": (*capa.MEDIDAS_OCR, "ocr_texto"),
                 "clip": ("capa_emb",)}
    linha: dict = {"capa_estado_imagem": estado_imagem}
    recorte = None
    for etapa in ETAPAS_CAPA:
        r = resultados[etapa]
        linha[f"{etapa}_estado"] = r["estado"]
        recorte = recorte or r.get("recorte")
        for coluna in atributos[etapa]:
            linha[coluna] = r.get(coluna) if r["estado"] == "ok" else None
    if linha["capa_emb"] is not None:
        linha["capa_emb"] = capa.validar_vetor(linha["capa_emb"], capa.DIMENSAO_IMAGEM)
    if recorte is not None:
        x0, y0, x1, y1 = recorte["caixa"]
        linha |= {"capa_recortada": recorte["recortado"], "capa_motivo_recorte": recorte["motivo"],
                  "capa_caixa_x0": x0, "capa_caixa_y0": y0, "capa_caixa_x1": x1, "capa_caixa_y1": y1}
    return linha


_COLUNAS_CAPA = {"capa_estado_imagem", *(f"{e}_estado" for e in ETAPAS_CAPA), *capa.MEDIDAS_COR, *capa.MEDIDAS_ROSTO,
                 *capa.MEDIDAS_OCR, "ocr_texto", "capa_emb"}


def _linha_valida(linha: object) -> bool:
    """Entrada de cache só vale com todas as colunas da capa e vetor de dimensão correta (ou ausente)."""
    if not isinstance(linha, dict) or not _COLUNAS_CAPA <= linha.keys():
        return False
    try:
        if linha["capa_emb"] is not None:
            capa.validar_vetor(linha["capa_emb"], capa.DIMENSAO_IMAGEM)
    except (ValueError, TypeError):
        return False
    return True


def extrair_capa(imagem: bytes | None, extratores: ExtratoresCapa, cache: CacheConteudo) -> dict:
    """Atributos da capa para os bytes da imagem; sem imagem, todas as etapas ficam `sem_imagem`."""
    if imagem is None:
        return _linha_da_capa("sem_imagem", {e: {"estado": "sem_imagem"} for e in ETAPAS_CAPA})
    chave = chave_conteudo(extratores.identificacao(), hashlib.sha256(imagem).hexdigest())
    guardado = cache.ler(chave)
    if guardado is not None and _linha_valida(guardado.get("linha")):
        return guardado["linha"]
    try:
        decodificada = capa.decodificar_imagem(imagem)
    except capa.ImagemInvalida:
        return _linha_da_capa("imagem_corrompida", {e: {"estado": "imagem_corrompida"} for e in ETAPAS_CAPA})
    recortada, recorte = capa.remover_faixas_pretas(decodificada)
    funcoes = {"cor": extratores.cor, "rosto": extratores.rosto, "ocr": extratores.ocr, "clip": extratores.clip}
    resultados = {e: capa.aplicar_extrator(f, recortada) | {"recorte": recorte} for e, f in funcoes.items()}
    linha = _linha_da_capa("ok", resultados)
    if all(resultados[e]["estado"] == "ok" for e in ETAPAS_CAPA):  # falha de extrator nunca é guardada como válida
        cache.gravar(chave, {"sha256_imagem": hashlib.sha256(imagem).hexdigest(), "linha": linha})
    return linha


# --- anotação do Gemini ---------------------------------------------------------------------------------------------


class FonteAnotacao(Protocol):
    def anotar(self, url: str | None, setor: Setor, duracao_s: int | None) -> tuple[dict | None, str]:
        """(anotação validada ou None, estado: ok | cache | indisponivel | falha_...)."""


@dataclass
class FonteAnotacaoGemini:
    """Cache de anotações por `video_id` + configuração; se faltar e houver cliente, uma chamada ao vídeo público.

    Sem cliente injetado (padrão), nenhuma chamada de rede acontece. A chamada ao vivo usa uma única tentativa e o
    mesmo pedido e esquema da rubrica; não grava no cache da amostra.
    """

    cache: CacheAnotacoes | None = None
    cliente: ga.ClienteGemini | None = None
    temperatura: float = 1.0
    max_tokens_saida: int = 16_384
    requisicoes_reais: int = 0  # cumulativo desde a criação da fonte; conta envios, não pedidos
    _trava: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def anotar(self, url: str | None, setor: Setor, duracao_s: int | None) -> tuple[dict | None, str]:
        video_id = video_id_da_url(url)
        if self.cache is not None and video_id is not None:
            try:
                guardado = self.cache.ler(video_id)
            except (CacheIncompativel, CacheCorrompido):
                guardado = None
            if guardado is not None and guardado.estado == "ok" and guardado.anotacao is not None:
                if guardado.setor != setor:
                    return None, "falha_setor_do_cache_diverge"
                return dict(guardado.anotacao), "cache"
        if self.cliente is None or url is None:
            return None, "indisponivel"
        return self._ao_vivo(url, setor, duracao_s)

    def _ao_vivo(self, url: str, setor: Setor, duracao_s: int | None) -> tuple[dict | None, str]:
        corpo = {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"fileUri": url, "mimeType": "video/mp4"}, "mediaProcessing": "STATIC"},
                {"text": rb.texto_do_pedido(setor)}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": self.temperatura,
                                 "maxOutputTokens": self.max_tokens_saida,
                                 "responseJsonSchema": rb.esquema_json(setor)},
        }
        try:
            with self._trava:
                self.requisicoes_reais += 1
            resposta = self.cliente.gerar(corpo, duracao_video_s=duracao_s or MAX_DURACAO_PADRAO_S)
        except ga.CotaDiariaEsgotada:
            return None, "falha_cota_esgotada"  # a previsão segue, com os atributos do Gemini ausentes e avisados
        except ga.ErroGemini as erro:
            return None, f"falha_{type(erro).__name__}"
        if resposta.situacao != "ok" or resposta.texto is None:
            return None, f"falha_resposta_{resposta.situacao}"
        try:
            return rb.validar_resposta(resposta.texto, setor).model_dump(mode="json"), "ok"
        except rb.RespostaInvalida as erro:
            return None, f"falha_invalida_{erro.categoria}"


def colunas_gemini(anotacao: Mapping | None, setor: str, codificador: leg.Codificador, pasta_cache: Path,
                   video_id: str) -> dict:
    """Colunas `gem_*` (e embedding da transcrição) de uma anotação; sem anotação, todas ausentes (nulas)."""
    if anotacao is None:
        return {c: None for c in (*ds.COLUNAS_GEMINI, ds.COLUNA_EMBEDDING_TRANSCRICAO)} | {
            ds.COLUNA_ESTADO_EMBEDDING: "sem_anotacao"}
    linha = ds.linha_gemini(anotacao, setor)
    texto = leg.normalizar_texto(anotacao.get("transcricao_texto"))
    if linha[ds.COLUNA_ESTADO_EMBEDDING] is None and texto:
        emb = ds.embeddings_transcricao({video_id: texto}, codificador, pasta_cache)[video_id]
        linha[ds.COLUNA_EMBEDDING_TRANSCRICAO] = emb["vetor"]
        linha[ds.COLUNA_ESTADO_EMBEDDING] = emb["estado"]
    return linha


# --- resolução da imagem da capa ---------------------------------------------------------------------------------


def bytes_da_capa(pedido: PedidoPrever, *, imagens_conferidas: Callable[[str], bytes | None] | None = None,
                  baixar: Callable[[str], bytes | None] | None = None) -> tuple[bytes | None, str]:
    """(bytes, origem): thumbnail do pedido, imagem conferida do vídeo, download injetado ou frame do arquivo."""
    if pedido.thumbnail is not None:
        return _ler_limitado(Path(pedido.thumbnail)), "thumbnail_do_pedido"
    criativo = pedido.creative
    if criativo.tipo == "imagem":
        return _ler_limitado(Path(criativo.arquivo)), "imagem_do_pedido"
    if criativo.tipo == "video" and criativo.url is not None:
        video_id = video_id_da_url(criativo.url)
        if video_id and imagens_conferidas and (dados := imagens_conferidas(video_id)) is not None:
            return dados, "thumbnail_conferida_do_video"
        if baixar is not None and video_id:
            return baixar(video_id), "thumbnail_baixada"
    if criativo.tipo == "video" and criativo.arquivo is not None:
        return _frame(Path(criativo.arquivo)), "frame_do_video"
    return None, "ausente"


def _ler_limitado(arquivo: Path) -> bytes:
    if arquivo.stat().st_size > MAX_BYTES_IMAGEM:
        raise ErroAgente(Codigo.MIDIA_INVALIDA, f"Imagem maior que {MAX_BYTES_IMAGEM // 1_000_000} MB.")
    return arquivo.read_bytes()


def _frame(arquivo: Path) -> bytes | None:
    import cv2

    captura = cv2.VideoCapture(str(arquivo))
    try:
        ok, quadro = captura.read()
        if not ok:
            return None
        ok, buffer = cv2.imencode(".png", quadro)
        return buffer.tobytes() if ok else None
    finally:
        captura.release()
