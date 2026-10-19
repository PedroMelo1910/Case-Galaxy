"""Sonda de um vídeo local, ANTES de qualquer chamada de rede (ARQUITETURA §7: Short de até 180 s, vertical ou quadrado).

Confere, nesta ordem e sem enviar nada: tamanho (até 100 MB), assinatura do contêiner entre os tipos que o Gemini aceita,
decodificação do primeiro quadro (o contêiner precisa ser um vídeo de verdade), duração, orientação e SHA-256 do
conteúdo. A leitura usa o OpenCV, que já é dependência do projeto; nenhuma ferramenta nova.

Duração e orientação vêm do próprio arquivo e prevalecem sobre qualquer `duration_s` informado. Duração que não se
consegue medir é recusada (nunca se envia um arquivo cuja duração se ignora). Mensagens e detalhes nunca trazem caminho.
"""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from src.agente.contratos import Codigo, ErroAgente

MAX_BYTES_VIDEO = 100_000_000
MAX_DURACAO_S = 180
BYTES_ASSINATURA = 64
BLOCO_HASH_BYTES = 1_048_576
BRANDS_QUICKTIME = (b"qt  ",)
BRANDS_3GPP = (b"3gp", b"3g2")
# Tipos documentados em ai.google.dev/gemini-api/docs/video-understanding (conferido em 17/10/2026).
ASSINATURA_ASF = bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")


@dataclass(frozen=True)
class SondaVideo:
    """Resultado da sonda. `duracao_inteira_s` arredonda para CIMA: é o que a conta de segundos de vídeo usa."""

    caminho: Path
    sha256: str
    mime: str
    tamanho_bytes: int
    duracao_s: float
    largura: int
    altura: int

    @property
    def duracao_inteira_s(self) -> int:
        return math.ceil(self.duracao_s - 1e-9)


def mime_pela_assinatura(inicio: bytes) -> str | None:
    """Tipo MIME do Gemini pela assinatura do contêiner; None se não for um dos formatos suportados."""
    if inicio[4:8] == b"ftyp":
        marca = inicio[8:12]
        if marca in BRANDS_QUICKTIME:
            return "video/mov"
        if marca[:3] in BRANDS_3GPP:
            return "video/3gpp"
        return "video/mp4" if marca[:3] != b"hei" and marca not in (b"avif", b"mif1", b"msf1") else None
    if inicio[:4] == b"RIFF" and inicio[8:12] == b"AVI ":
        return "video/avi"
    if inicio[:4] == b"\x1aE\xdf\xa3":  # EBML: só WebM está na lista do Gemini, Matroska não
        return "video/webm" if b"webm" in inicio[:BYTES_ASSINATURA] else None
    if inicio[:3] == b"FLV":
        return "video/x-flv"
    if inicio[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"):
        return "video/mpeg"
    if inicio[:16] == ASSINATURA_ASF:
        return "video/wmv"
    return None


def _midia_invalida(mensagem: str, **detalhes: object) -> ErroAgente:
    return ErroAgente(Codigo.MIDIA_INVALIDA, mensagem, detalhes)


def _sha256_do_arquivo(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while bloco := arquivo.read(BLOCO_HASH_BYTES):
            resumo.update(bloco)
    return resumo.hexdigest()


def _medidas(caminho: Path) -> tuple[float, int, int]:
    """(duração em segundos, largura, altura) do primeiro quadro decodificado, já na orientação de exibição."""
    import cv2

    captura = cv2.VideoCapture(str(caminho))
    try:
        if not captura.isOpened():
            raise _midia_invalida("O arquivo não pôde ser aberto como vídeo.")
        ok, quadro = captura.read()
        if not ok or quadro is None:
            raise _midia_invalida("Nenhum quadro de vídeo pôde ser decodificado do arquivo.")
        quadros, fps = captura.get(cv2.CAP_PROP_FRAME_COUNT), captura.get(cv2.CAP_PROP_FPS)
    finally:
        captura.release()
    if not (math.isfinite(quadros) and math.isfinite(fps) and quadros > 0 and fps > 0):
        raise _midia_invalida("A duração do vídeo não pôde ser determinada; o arquivo não será enviado.")
    altura, largura = quadro.shape[:2]
    return quadros / fps, int(largura), int(altura)


def sondar_video(caminho: Path, *, max_bytes: int = MAX_BYTES_VIDEO, max_duracao_s: int = MAX_DURACAO_S) -> SondaVideo:
    """Valida o arquivo local e devolve a sonda; levanta `ErroAgente` (MIDIA_INVALIDA ou ENTRADA_INVALIDA)."""
    caminho = Path(caminho)
    try:
        tamanho = caminho.stat().st_size
        if tamanho > max_bytes:
            raise _midia_invalida(f"Vídeo maior que o limite de {max_bytes} bytes.", limite_bytes=max_bytes)
        with caminho.open("rb") as arquivo:
            inicio = arquivo.read(BYTES_ASSINATURA)
    except OSError:
        raise _midia_invalida("O arquivo de vídeo não pôde ser lido.") from None
    mime = mime_pela_assinatura(inicio)
    if mime is None:
        raise _midia_invalida("O arquivo não é um vídeo em um formato aceito (MP4, MOV, 3GPP, AVI, WebM, FLV, MPEG "
                              "ou WMV).")
    duracao, largura, altura = _medidas(caminho)
    if math.ceil(duracao - 1e-9) > max_duracao_s:  # a mesma regra da cobrança (`duracao_inteira_s`)
        raise ErroAgente(Codigo.ENTRADA_INVALIDA,
                         f"O vídeo tem mais de {max_duracao_s} s; o agente aceita Shorts de até {max_duracao_s} s.",
                         {"duracao_s": round(duracao, 1), "limite_s": max_duracao_s})
    if largura > altura:
        raise ErroAgente(Codigo.ENTRADA_INVALIDA, "O vídeo é horizontal; o agente aceita vídeo vertical ou quadrado.",
                         {"largura": largura, "altura": altura})
    try:
        sha256 = _sha256_do_arquivo(caminho)
    except OSError:
        raise _midia_invalida("O arquivo de vídeo não pôde ser lido.") from None
    return SondaVideo(caminho=caminho, sha256=sha256, mime=mime, tamanho_bytes=tamanho, duracao_s=duracao,
                      largura=largura, altura=altura)
