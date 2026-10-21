"""Módulo consolidado. Reúne:
- Extração de atributos do criativo para a linha de inferência, com caches por conteúdo e configuração.
- Rota local de previsão de vídeo: seleção da rota, extração audiovisual local e contexto do preditor."""

import dataclasses
import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Protocol

import numpy as np

from src.agente.contratos import aviso, Codigo, ErroAgente, MAX_BYTES_IMAGEM, PedidoPrever, video_id_da_url
from src.dados import dataset as ds, historico as hist
from src.dados.coleta import Setor
from src.dados.historico import BaseHistorica
from src.extracao import imagem as capa, pipeline as avc, rubrica as eg, rubrica as rb, texto as leg
from src.extracao.modelos import COMANDO_PREPARAR, ErroExtracao, ModeloAusente, ModeloCorrompido, sem_caminhos
from src.extracao.rubrica import CacheAnotacoes, CacheCorrompido, CacheIncompativel, gravar_json_atomico
from src.extracao.video_local import SondaVideo
from src.integracoes import gemini as ga
from src.modelos.pacote import identidade_do_extrator


# ──── Extração de atributos do criativo para a linha de inferência, com caches por conteúdo e configuração
ETAPAS_CAPA = ("cor", "rosto", "ocr", "clip")
MAX_DURACAO_PADRAO_S = 180
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


def colunas_da_legenda(extracao: Mapping) -> dict:
    """Colunas `leg_*` da linha de inferência a partir do resultado de `extracao_legenda.extrair_legenda`."""
    return {"leg_estado": extracao["estado"], "leg_n_tokens": extracao["n_tokens"],
            "leg_truncado": extracao["truncado"], **{c: extracao[c] for c in leg.CONTAGENS_LEGENDA},
            "leg_emb": extracao["vetor"]}


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
    cache_complementar: CacheAnotacoes | None = None  # lido só se o principal não tem o vídeo (ex.: anotações do de-para)
    _trava: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def anotar(self, url: str | None, setor: Setor, duracao_s: int | None) -> tuple[dict | None, str]:
        video_id = video_id_da_url(url)
        if video_id is not None:
            guardado, falhou = self._ler_cache(self.cache, video_id)
            if guardado is None and not falhou:  # o complementar só vale para vídeo realmente ausente do principal
                guardado, _ = self._ler_cache(self.cache_complementar, video_id)
            if guardado is not None and guardado.estado == "ok" and guardado.anotacao is not None:
                if guardado.setor != setor:
                    return None, "falha_setor_do_cache_diverge"
                return dict(guardado.anotacao), "cache"
        if self.cliente is None or url is None:
            return None, "indisponivel"
        return self._ao_vivo(url, setor, duracao_s)

    @staticmethod
    def _ler_cache(cache: CacheAnotacoes | None, video_id: str) -> tuple[object | None, bool]:
        """(anotação guardada ou None, falhou). Cache ilegível ou de outra configuração é avisado, nunca engolido em silêncio."""
        if cache is None:
            return None, False
        try:
            return cache.ler(video_id), False
        except (CacheIncompativel, CacheCorrompido) as erro:
            logger.warning("cache de anotações inutilizável para %s (%s): %s", video_id, type(erro).__name__, erro)
            return None, True

    def _ao_vivo(self, url: str, setor: Setor, duracao_s: int | None) -> tuple[dict | None, str]:
        config = eg.ConfiguracaoExtracao(modelo=getattr(self.cliente, "modelo", ""), temperatura=self.temperatura,
                                         max_tokens_saida=self.max_tokens_saida)
        corpo = eg.corpo_para_url(url, setor, config)  # o mesmo pedido da anotação em lote
        try:
            with self._trava:
                self.requisicoes_reais += 1
            resposta = self.cliente.gerar(corpo, duracao_video_s=duracao_s or MAX_DURACAO_PADRAO_S)
        except ga.CotaDiariaEsgotada:
            return None, "falha_cota_esgotada"  # o grafo segue pelo fallback para N3, avisado
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
    texto = ds.texto_para_embedding(anotacao)  # só a fala reconhecida; incerta, sem fala e falha não são codificadas
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


# ──── Rota local de previsão de vídeo: seleção da rota, extração audiovisual local e contexto do preditor.
ROTA_VIDEO_LOCAL = "video_local"
ROTA_VIDEO_GEMINI = "video_gemini"
ROTA_VIDEO_GEMINI_FALLBACK = "video_gemini_fallback_imagem"
ROTA_IMAGEM = "imagem"
ROTA_TEXTO = "texto"
FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"
PREFIXOS_LEGENDA = ("leg_",)
PREFIXOS_CAPA = ("cor_", "rosto_", "ocr_", "capa_")
ESTADOS_SEM_REDE_DO_GEMINI = frozenset({"cache", "indisponivel", "upload_nao_configurado", "nao_aplicavel"})

ROTULOS = {
    ROTA_VIDEO_LOCAL: "Previsão local de vídeo (atributos audiovisuais extraídos neste equipamento, sem Gemini)",
    ROTA_VIDEO_GEMINI: "Previsão de vídeo com anotação do Gemini",
    ROTA_VIDEO_GEMINI_FALLBACK: "Modelo de imagem (o vídeo ficou sem a anotação do Gemini)",
    ROTA_IMAGEM: "Previsão por imagem (histórico, legenda e capa)",
    ROTA_TEXTO: "Previsão por texto (histórico e legenda)",
}
EXPLICACAO_DETERMINISTICA = ("Texto determinístico montado a partir dos fatores SHAP do modelo (sem modelo de "
                             "linguagem); os números vêm do modelo.")
EXPLICACAO_REDIGIDA = "Texto redigido por um modelo de linguagem a partir dos fatores SHAP e validado contra eles."


def resolver_rota(servicos: object, pedido: PedidoPrever) -> str:
    """Rota que prevê o pedido, sem executar nada. `local` sem preditor local ativo é `MODELO_INDISPONIVEL`."""
    if pedido.modalidade != "video":
        return pedido.modalidade
    ativa = bool(getattr(servicos, "rota_local_ativa", False))
    if pedido.video_route == "gemini":
        return ROTA_VIDEO_GEMINI
    if pedido.video_route == "local":
        if not ativa:
            raise ErroAgente(Codigo.MODELO_INDISPONIVEL,
                             "A rota local de vídeo foi pedida, mas não há preditor local ativo neste serviço. Não há "
                             "volta ao modelo de imagem: ative o pacote local ou use `video_route = gemini`.",
                             {"rota": "local"})
        return ROTA_VIDEO_LOCAL
    arquivo = getattr(pedido.creative, "arquivo", None)
    return ROTA_VIDEO_LOCAL if arquivo is not None and ativa else ROTA_VIDEO_GEMINI


def descrever_rota(rota: str, *, versao_do_modelo: str, offline: bool) -> dict:
    """Bloco `route` da resposta: qual rota a produziu e se ela dependeu de rede."""
    return {"id": rota, "rotulo": ROTULOS[rota], "usa_gemini": rota in (ROTA_VIDEO_GEMINI,),
            "offline": offline, "versao_do_modelo": versao_do_modelo}


def usou_rede(*, rota: str, extracao: Mapping, origem_historico: str, explicacao_fonte: str,
              explicador_configurado: bool) -> bool:
    """A resposta dependeu de rede? Anotação ao vivo do Gemini, miniatura baixada, histórico consultado ao YouTube ou
    explicação por LLM (inclusive a tentada). Cache, arquivo local e texto fixo não usam rede."""
    gemini = extracao.get("gemini")
    anotou = rota in (ROTA_VIDEO_GEMINI, ROTA_VIDEO_GEMINI_FALLBACK) and gemini not in ESTADOS_SEM_REDE_DO_GEMINI
    return bool(anotou or extracao.get("capa_origem") == "thumbnail_baixada" or origem_historico == "consulta_atual"
                or explicacao_fonte == "llm" or explicador_configurado)


# --- extrator local ------------------------------------------------------------------------------------------------


class ExtratorLocalSobDemanda:
    """`ExtratorAudiovisual` criado na primeira extração (entrada recusada não carrega nada).

    A extração local nunca baixa pesos: `HF_HUB_OFFLINE` é ligado se o ambiente não o definiu. O cache é o do próprio
    extrator, por SHA-256 do conteúdo e identidade da configuração."""

    def __init__(self, fabrica: Callable[[], object] | None = None) -> None:
        self._fabrica, self._extrator = fabrica, None

    def extrair(self, caminho: object) -> dict:
        if self._extrator is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            if self._fabrica is not None:
                self._extrator = self._fabrica()
            else:
                from src.extracao.pipeline import ExtratorAudiovisual

                self._extrator = ExtratorAudiovisual()
        return self._extrator.extrair(caminho)


_CODIGO_DA_EXTRACAO = {"arquivo_ilegivel": Codigo.MIDIA_INVALIDA, "midia_invalida": Codigo.MIDIA_INVALIDA,
                       "limite_excedido": Codigo.ENTRADA_INVALIDA}


def erro_de_extracao(erro: ErroExtracao) -> ErroAgente:
    codigo = _CODIGO_DA_EXTRACAO.get(erro.codigo, Codigo.ERRO_INTERNO)
    return ErroAgente(codigo, erro.mensagem, {"motivo": erro.codigo, **erro.detalhes})


# --- atributos do modelo local -------------------------------------------------------------------------------------


def colunas_do_modelo(pacote) -> list[str]:
    espec = pacote.q50.especificacao
    return [*espec.escalares, *espec.categoricas, *espec.embeddings]


def necessidades(pacote) -> dict[str, bool]:
    """Blocos que o modelo local lê além dos `avl_*`: legenda e capa (locais) só se estiverem na especificação."""
    colunas = colunas_do_modelo(pacote)
    return {"legenda": any(c.startswith(PREFIXOS_LEGENDA) for c in colunas),
            "capa": any(c.startswith(PREFIXOS_CAPA) for c in colunas)}


def _avisos_da_extracao(resultado: Mapping) -> list[dict]:
    estados = resultado["resumo"]["estados"]
    avisos = []
    if estados.get("audio") == "sem_stream_audio":
        avisos.append(aviso("video_sem_audio", "O vídeo não tem faixa de áudio: os atributos de áudio e de fala ficam "
                            "ausentes, sem tratar como silêncio.", "info"))
    elif estados.get("fala") == "audio_sem_fala_detectada":
        avisos.append(aviso("sem_fala_detectada", "O áudio não tem fala detectada pelo transcritor local.", "info"))
    if estados.get("fala") == "transcricao_incerta":
        avisos.append(aviso("transcricao_incerta", "Transcrição local incerta: os atributos de fala ficam ausentes, "
                            "sem tratar como \"sem fala\"."))
    insuficientes = sorted(e for e, st in estados.items() if st == "dados_insuficientes")
    if insuficientes:
        avisos.append(aviso("etapa_com_dados_insuficientes",
                            f"Dados insuficientes para medir: {', '.join(insuficientes)}; esses atributos ficam "
                            "ausentes, sem valor inventado.", "info"))
    quadros = resultado["etapas"].get("quadros", {})
    if quadros.get("poucos_quadros"):
        avisos.append(aviso("poucos_quadros", "O vídeo rendeu menos quadros que o mínimo planejado para a extração."))
    do_cache, total = resultado["resumo"].get("do_cache", []), len(resultado["resumo"]["estados"]) - 1
    if do_cache:
        avisos.append(aviso("extracao_do_cache",
                            f"{len(do_cache)} de {total} etapas da extração vieram do cache (mesmo conteúdo do vídeo e "
                            "mesma configuração do extrator); nada foi recalculado.", "info"))
    if resultado["resumo"].get("cache_gravado") is False:
        avisos.append(aviso("cache_de_extracao_nao_gravado",
                            "O cache da extração não pôde ser gravado; o resultado vale, mas será recalculado.", "info"))
    return avisos


def _exigir_extrator_compativel(resultado: Mapping, sonda: SondaVideo, pacote) -> None:
    esperado = identidade_do_extrator(pacote.registro)
    if resultado.get("sha256") != sonda.sha256:
        raise ErroAgente(Codigo.MIDIA_INVALIDA, "O arquivo mudou durante o processamento; envie-o de novo.")
    if resultado.get("identidade") != esperado:
        raise ErroAgente(Codigo.MODELO_INDISPONIVEL,
                         "O extrator audiovisual instalado não é o do preditor local (configuração, modelos ou "
                         "bibliotecas diferentes): a extração de hoje não é comparável à do treino.",
                         {"identidade_esperada": esperado[:12], "identidade_encontrada": str(resultado.get("identidade"))[:12]})


def _exigir_vetores_compativeis(colunas: Mapping, pacote) -> None:
    pre = pacote.q50.pipeline.named_steps["pre"]
    for bloco in pacote.q50.especificacao.embeddings:
        pca, vetor = getattr(pre, "pca_", {}).get(bloco), colunas.get(bloco)
        if pca is not None and vetor is not None and len(vetor) != pca.n_features_in_:
            raise ErroAgente(Codigo.MODELO_INDISPONIVEL,
                             f"O vetor {bloco} tem {len(vetor)} valores; o preditor local foi treinado com "
                             f"{pca.n_features_in_}.", {"coluna": bloco})


def extrair_atributos_audiovisuais(servicos, sonda: SondaVideo) -> tuple[dict, dict, list[dict]]:
    """(colunas `avl_*`, bloco `extraction`, avisos) do vídeo, pela extração local com cache.

    Levanta `ErroAgente` para: pesos ausentes ou corrompidos, mídia ilegível, extrator de outra identidade, etapa que
    falhou (a política é a mesma do treino: falha recusa a previsão) e coluna que o modelo exige e a extração não deu."""
    pacote = servicos.pacote_local
    try:
        resultado = servicos.extrator_local.extrair(sonda.caminho)
    except ErroExtracao as erro:
        raise erro_de_extracao(erro) from None
    except (ModeloAusente, ModeloCorrompido) as erro:
        raise ErroAgente(Codigo.MODELO_INDISPONIVEL, sem_caminhos(str(erro)),
                         {"modelo": erro.nome, "comando": COMANDO_PREPARAR}) from None
    _exigir_extrator_compativel(resultado, sonda, pacote)
    colunas = resultado["colunas"]
    recusa = avc.motivos_de_recusa(colunas)
    if recusa:
        raise ErroAgente(Codigo.MODELO_INDISPONIVEL,
                         "A extração audiovisual local ficou incompleta; a previsão não é feita com atributos faltando "
                         "por falha (nem com o modelo de imagem no lugar).", {"etapas": recusa})
    faltam = [c for c in colunas_do_modelo(pacote) if c.startswith(avc.PREFIXO) and c not in colunas]
    if faltam:
        raise ErroAgente(Codigo.MODELO_INDISPONIVEL,
                         "O extrator não entrega colunas que o preditor local exige.", {"faltam": faltam[:10]})
    _exigir_vetores_compativeis(colunas, pacote)
    resumo = resultado["resumo"]
    extracao = {"rota": ROTA_VIDEO_LOCAL, "gemini": "nao_aplicavel", "transcricao": resultado["etapas"]["fala"]["estado"],
                "embedding_transcricao": None,
                "extrator": {"identidade_sha256": resultado["identidade"], "versao": resultado.get("versao")},
                "sha256_video": sonda.sha256, "etapas": dict(resumo["estados"]),
                "do_cache": list(resumo.get("do_cache", [])), "executadas": list(resumo.get("executadas", [])),
                "tempo_total_s": resumo.get("tempo_total_s")}
    return dict(colunas), extracao, _avisos_da_extracao(resultado)


# --- contexto do histórico -----------------------------------------------------------------------------------------


def base_da_rota(servicos, rota: str) -> BaseHistorica:
    """Base do histórico. Na rota local, o corte do setor é o contexto de treino gravado no pacote local (se houver);
    posts e canais são os mesmos do snapshot."""
    contexto = ((servicos.pacote_local.registro or {}).get("contexto") if rota == ROTA_VIDEO_LOCAL else None)
    if not contexto or not contexto.get("treino_ate_utc"):
        return servicos.base
    treino_ate = datetime.strptime(contexto["treino_ate_utc"], FORMATO_UTC).replace(tzinfo=UTC)
    return dataclasses.replace(servicos.base, contexto=hist.ContextoTreino(
        contexto["nome"], treino_ate, frozenset(contexto.get("canais_excluidos", ()))))
