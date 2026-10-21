"""Módulo consolidado. Reúne:
- Áudio local: decodificação com PyAV (sem ffmpeg no PATH) e medidas de energia.
- Fala local com faster-whisper e o contrato de estados que separa o que é ausência do que é falha."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from src.extracao.modelos import (
    AUDIO_ATIVO_ACIMA_DB,
    AUDIO_JANELA_S,
    AUDIO_PASSO_S,
    AUDIO_PISO_DB,
    AUDIO_SILENCIO_DIGITAL_DB,
    AUDIO_TAXA_HZ,
    MAX_DURACAO_S,
    OPCOES_FFMPEG,
    sem_caminhos,
    THREADS,
    WHISPER_BEAM,
    WHISPER_COMPRESSAO_MAX,
    WHISPER_COMPUTE,
    WHISPER_CONDICIONAR_NO_TEXTO,
    WHISPER_LOGPROB_MIN,
    WHISPER_NO_SPEECH_MAX,
    WHISPER_PROB_IDIOMA_MIN,
    WHISPER_TEMPERATURA,
    WHISPER_VAD,
)


# ──── Áudio local: decodificação com PyAV (sem ffmpeg no PATH) e medidas de energia.
PISO_DB = AUDIO_PISO_DB


@dataclass(frozen=True)
class AudioDecodificado:
    estado: str  # "ok" | "sem_stream_audio" | "falha_decodificacao"
    sinal: np.ndarray | None  # float32 mono em `taxa_hz`
    taxa_hz: int
    duracao_s: float | None
    erro: str | None = None


def _db(valor: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(valor, 1e-6))


def decodificar_audio(caminho: Path, *, max_duracao_s: float = MAX_DURACAO_S) -> AudioDecodificado:
    """Primeiro stream de áudio do arquivo, em mono 16 kHz float32, limitado a `max_duracao_s`."""
    limite = int(max_duracao_s * AUDIO_TAXA_HZ)
    try:
        with av.open(str(caminho), options=OPCOES_FFMPEG) as conteiner:
            fluxo = next((s for s in conteiner.streams if s.type == "audio"), None)
            if fluxo is None:
                return AudioDecodificado("sem_stream_audio", None, AUDIO_TAXA_HZ, None)
            reamostrador = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_TAXA_HZ)
            partes: list[np.ndarray] = []
            total = 0
            for quadro in conteiner.decode(fluxo):
                for saida in reamostrador.resample(quadro):
                    parte = saida.to_ndarray().reshape(-1)
                    partes.append(parte)
                    total += parte.size
                if total >= limite:
                    break
            for saida in reamostrador.resample(None):
                partes.append(saida.to_ndarray().reshape(-1))
    except Exception as erro:  # noqa: BLE001 - qualquer falha de leitura é registrada, nunca vira "sem áudio"
        return AudioDecodificado("falha_decodificacao", None, AUDIO_TAXA_HZ, None,
                                 f"{type(erro).__name__}: {sem_caminhos(str(erro))[:160]}")
    if not partes or sum(p.size for p in partes) == 0:
        return AudioDecodificado("falha_decodificacao", None, AUDIO_TAXA_HZ, None, "o stream de áudio não tem amostras")
    sinal = np.concatenate(partes).astype(np.float32)[:limite]
    return AudioDecodificado("ok", sinal, AUDIO_TAXA_HZ, sinal.size / AUDIO_TAXA_HZ)


def _janelas(sinal: np.ndarray, taxa_hz: int) -> np.ndarray:
    janela, passo = round(AUDIO_JANELA_S * taxa_hz), round(AUDIO_PASSO_S * taxa_hz)
    return np.lib.stride_tricks.sliding_window_view(sinal, janela)[::passo]


def medidas_energia(sinal: np.ndarray, taxa_hz: int = AUDIO_TAXA_HZ) -> dict:
    """Energia (dBFS), atividade, faixa dinâmica e espectro (centroide, planura) de um sinal mono; ValueError se inválido.

    Centroide e planura só existem para janelas ativas (acima de `AUDIO_ATIVO_ACIMA_DB`); sem elas são None.
    """
    x = np.asarray(sinal, dtype=np.float32)
    janela = round(AUDIO_JANELA_S * taxa_hz)
    if x.ndim != 1 or x.size < janela:
        raise ValueError("sinal vazio ou mais curto que uma janela de análise")
    if not np.all(np.isfinite(x)):
        raise ValueError("sinal com valores não finitos")
    quadros = _janelas(x, taxa_hz)
    niveis = np.maximum(_db(np.sqrt(np.mean(np.square(quadros, dtype=np.float32), axis=1))), PISO_DB)
    pico_db = float(max(_db(float(np.max(np.abs(x)))), PISO_DB))
    ativos = niveis > AUDIO_ATIVO_ACIMA_DB
    centroide = planura = None
    if ativos.any():
        espectro = np.abs(np.fft.rfft(quadros[ativos] * np.hanning(quadros.shape[1]), axis=1))
        frequencias = np.fft.rfftfreq(quadros.shape[1], 1.0 / taxa_hz)
        potencia = np.square(espectro) + 1e-12
        centroide = float(np.mean(np.sum(frequencias * espectro, axis=1) / np.maximum(np.sum(espectro, axis=1), 1e-12)))
        planura = float(np.mean(np.exp(np.mean(np.log(potencia), axis=1)) / np.mean(potencia, axis=1)))
    return {
        "rms_db_medio": float(np.mean(niveis)), "rms_db_std": float(np.std(niveis)), "pico_db": pico_db,
        "fracao_ativa": float(np.mean(ativos)),
        "faixa_dinamica_db": float(np.percentile(niveis, 95) - np.percentile(niveis, 10)),
        "centroide_hz": centroide, "planura": planura, "silencio_digital": bool(pico_db < AUDIO_SILENCIO_DIGITAL_DB),
    }


# ──── Fala local com faster-whisper e o contrato de estados que separa o que é ausência do que é falha.
@dataclass(frozen=True)
class Segmento:
    inicio_s: float
    fim_s: float
    texto: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


Transcritor = Callable[[np.ndarray], tuple[Sequence[Segmento], str | None, float | None]]


def _sem_valores(estado: str, **extra) -> dict:
    base = {"palavras": None, "palavras_por_s": None, "palavras_por_s_fala": None, "fracao_tempo_fala": None,
            "idioma": None, "prob_idioma": None}
    return {"estado": estado, **base, **extra}


def _sem_fala(motivo: str) -> dict:
    return _sem_valores("audio_sem_fala_detectada", motivo=motivo, palavras=0, palavras_por_s=0.0,
                        fracao_tempo_fala=0.0)


def _motivos_de_incerteza(segmentos: Sequence[Segmento], prob_idioma: float | None) -> list[str]:
    motivos = []
    if prob_idioma is None or prob_idioma < WHISPER_PROB_IDIOMA_MIN:
        motivos.append("idioma_incerto")
    if float(np.mean([s.avg_logprob for s in segmentos])) < WHISPER_LOGPROB_MIN:
        motivos.append("logprob_baixa")
    if float(np.mean([s.no_speech_prob for s in segmentos])) > WHISPER_NO_SPEECH_MAX:
        motivos.append("provavel_nao_fala")
    if max(s.compression_ratio for s in segmentos) > WHISPER_COMPRESSAO_MAX:
        motivos.append("repeticao_suspeita")
    return motivos


def analisar_fala(audio: AudioDecodificado, energia: dict | None, duracao_video_s: float,
                  transcritor: Transcritor) -> dict:
    """Estado e atributos de fala de um vídeo; o `transcritor` só é consultado quando há áudio com energia."""
    if audio.estado == "sem_stream_audio":
        return _sem_valores("sem_audio")
    if audio.estado != "ok" or audio.sinal is None:
        return _sem_valores("falha_extrator", erro="falha_decodificacao", mensagem=sem_caminhos(audio.erro or "")[:200])
    if energia and energia.get("silencio_digital"):
        return _sem_fala("silencio_digital")
    try:
        segmentos, idioma, prob_idioma = transcritor(audio.sinal)
    except Exception as erro:  # noqa: BLE001 - toda falha do transcritor é registrada, nunca vira "sem fala"
        return _sem_valores("falha_extrator", erro=type(erro).__name__, mensagem=sem_caminhos(str(erro))[:200])
    falados = [s for s in segmentos if s.texto.strip()]
    if not falados:
        return _sem_fala("sem_segmentos_de_fala")
    texto = " ".join(s.texto.strip() for s in falados)
    motivos = _motivos_de_incerteza(falados, prob_idioma)
    if motivos:
        return _sem_valores("transcricao_incerta", motivo=motivos, texto=texto)
    palavras = len(texto.split())
    segundos_de_fala = float(sum(max(s.fim_s - s.inicio_s, 0.0) for s in falados))
    return {"estado": "fala_transcrita", "palavras": palavras, "palavras_por_s": palavras / duracao_video_s,
            "palavras_por_s_fala": palavras / segundos_de_fala if segundos_de_fala > 0 else None,
            "fracao_tempo_fala": min(segundos_de_fala / duracao_video_s, 1.0), "idioma": idioma,
            "prob_idioma": float(prob_idioma), "texto": texto,
            "logprob_medio": float(np.mean([s.avg_logprob for s in falados]))}


class TranscritorWhisper:
    """faster-whisper `small` em CPU (int8), carregado no primeiro uso, só dos arquivos já em disco (sem rede)."""

    def __init__(self, threads: int = THREADS) -> None:
        self.threads, self._modelo = threads, None

    def _carregar(self) -> None:
        if self._modelo is not None:
            return
        from faster_whisper import WhisperModel

        from src.extracao.modelos import localizar

        self._modelo = WhisperModel(str(localizar("whisper")), device="cpu", compute_type=WHISPER_COMPUTE,
                                    cpu_threads=self.threads, num_workers=1, local_files_only=True)

    def liberar(self) -> None:
        self._modelo = None

    def __call__(self, sinal: np.ndarray) -> tuple[list[Segmento], str | None, float | None]:
        self._carregar()
        gerador, info = self._modelo.transcribe(sinal, beam_size=WHISPER_BEAM, vad_filter=WHISPER_VAD,
                                                temperature=WHISPER_TEMPERATURA,
                                                condition_on_previous_text=WHISPER_CONDICIONAR_NO_TEXTO)
        segmentos = [Segmento(s.start, s.end, s.text, s.avg_logprob, s.no_speech_prob, s.compression_ratio)
                     for s in gerador]  # o gerador é consumido aqui: toda a CPU é gasta dentro desta chamada
        return segmentos, info.language, info.language_probability
