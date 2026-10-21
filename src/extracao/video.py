"""Módulo consolidado. Reúne:
- Sonda de mídia: duração, dimensões na orientação de exibição, fps, áudio e SHA-256, tudo do próprio arquivo.
- Sonda de um vídeo local, ANTES de qualquer chamada de rede (ARQUITETURA §7: Short de até 180 s, vertical ou quadrado).
- Plano de amostragem de quadros (TAPI §6.2): 5 a 8 quadros, com 2 ou 3 nos primeiros 3 s (o gancho).
- Leitura de quadros reais do vídeo: os amostrados (PNG sem perdas, em disco) e os pares próximos para o fluxo óptico.
- Cortes de cena com PySceneDetect (ContentDetector), com taxa normalizada pela duração.
- Medidas OpenCV dos quadros amostrados: cor, brilho, contraste, saturação e movimento.
- Detecção de objetos com YOLOX-tiny (ONNX Runtime, CPU): 80 classes COCO."""

import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from src.extracao.cache import sha256_arquivo
from src.extracao.imagem import decodificar_imagem, medidas_cor
from src.extracao.modelos import (
    CORTES_DURACAO_MIN_S,
    CORTES_LIMIAR,
    DadosInsuficientes,
    ErroExtracao,
    FARNEBACK,
    FPS_PADRAO,
    JANELA_GANCHO_S,
    LADO_MAX_QUADRO_PX,
    MAX_BYTES,
    MAX_DURACAO_S,
    MIN_QUADROS_TAPI,
    N_QUADROS,
    N_QUADROS_GANCHO,
    OPCOES_FFMPEG,
    OPENCV_FFMPEG_OPCOES,
    THREADS,
    YOLOX_CONFIANCA,
    YOLOX_ENTRADA,
    YOLOX_NMS,
)


# ──── Sonda de mídia: duração, dimensões na orientação de exibição, fps, áudio e SHA-256, tudo do próprio 
@dataclass(frozen=True)
class SondaMidia:
    """`largura` e `altura` já na orientação de exibição (a rotação do arquivo, se houver, está aplicada)."""

    sha256: str
    tamanho_bytes: int
    duracao_s: float
    largura: int
    altura: int
    fps: float | None
    orientacao: str  # "vertical" | "horizontal" | "quadrado"
    tem_audio: bool
    rotacao_graus: int
    codec_video: str


def dimensoes_exibidas(largura: int, altura: int, rotacao_graus: int) -> tuple[int, int]:
    """Dimensões na exibição: rotações de 90 e 270 graus trocam largura e altura."""
    return (altura, largura) if rotacao_graus % 360 in (90, 270) else (largura, altura)


def _orientacao(largura: int, altura: int) -> str:
    return "vertical" if altura > largura else "horizontal" if largura > altura else "quadrado"


def _arquivo(caminho: Path, max_bytes: int) -> int:
    try:
        if not caminho.is_file():
            raise ErroExtracao("arquivo_ilegivel", "O caminho informado não é um arquivo de vídeo existente.")
        tamanho = caminho.stat().st_size
    except OSError:
        raise ErroExtracao("arquivo_ilegivel", "O arquivo não pôde ser acessado.") from None
    if tamanho == 0:
        raise ErroExtracao("arquivo_ilegivel", "O arquivo está vazio.")
    if tamanho > max_bytes:
        raise ErroExtracao("limite_excedido", f"Arquivo maior que o limite de {max_bytes} bytes.",
                           {"limite_bytes": max_bytes})
    return tamanho


def _duracao(conteiner, fluxo) -> float | None:
    """Duração do STREAM de vídeo (é nele que os quadros existem); a do contêiner é o máximo entre os streams, e um
    áudio mais longo que o vídeo a inflaria. O contêiner só vale quando o stream não declara duração."""
    if fluxo.duration and fluxo.time_base:
        return float(fluxo.duration * fluxo.time_base)
    if conteiner.duration:
        return conteiner.duration / av.time_base
    return None


def sondar_midia(caminho: Path, *, max_bytes: int = MAX_BYTES, max_duracao_s: float = MAX_DURACAO_S) -> SondaMidia:
    """Valida o arquivo e devolve a sonda; levanta `ErroExtracao` com código em caso de falha."""
    caminho = Path(caminho)
    tamanho = _arquivo(caminho, max_bytes)
    try:
        with av.open(str(caminho), options=OPCOES_FFMPEG) as conteiner:
            fluxo = next(iter(conteiner.streams.video), None)
            if fluxo is None:
                raise ErroExtracao("midia_invalida", "O arquivo não tem stream de vídeo.")
            primeiro = next(conteiner.decode(fluxo), None)
            if primeiro is None:
                raise ErroExtracao("midia_invalida", "Nenhum quadro de vídeo pôde ser decodificado.")
            taxa = fluxo.average_rate or fluxo.guessed_rate
            fps = float(taxa) if taxa else None
            duracao = _duracao(conteiner, fluxo)
            rotacao = int(getattr(primeiro, "rotation", 0) or 0)
            largura, altura = dimensoes_exibidas(primeiro.width, primeiro.height, rotacao)
            tem_audio, codec = len(conteiner.streams.audio) > 0, fluxo.codec_context.name
    except ErroExtracao:
        raise
    except Exception:  # noqa: BLE001 - qualquer erro do FFmpeg (formato, truncamento) vira mídia inválida, sem rastro
        raise ErroExtracao("midia_invalida", "O arquivo não pôde ser lido como vídeo.") from None
    if duracao is None or duracao <= 0:
        raise ErroExtracao("midia_invalida", "A duração do vídeo não pôde ser determinada.")
    if duracao > max_duracao_s:
        raise ErroExtracao("limite_excedido", f"O vídeo tem mais de {max_duracao_s:g} s.",
                           {"duracao_s": round(duracao, 1), "limite_s": max_duracao_s})
    return SondaMidia(sha256=sha256_arquivo(caminho), tamanho_bytes=tamanho, duracao_s=duracao, largura=largura,
                      altura=altura, fps=fps, orientacao=_orientacao(largura, altura), tem_audio=tem_audio,
                      rotacao_graus=rotacao, codec_video=codec)


# ──── Plano de amostragem de quadros (TAPI §6.2): 5 a 8 quadros, com 2 ou 3 nos primeiros 3 s (o gancho).
@dataclass(frozen=True)
class PlanoAmostragem:
    """`n_gancho` conta os instantes que caem na janela do gancho; `poucos_quadros`: menos que o mínimo do TAPI."""

    instantes_s: tuple[float, ...]
    indices_quadro: tuple[int, ...]
    n_gancho: int
    poucos_quadros: bool


def _bruto(duracao_s: float, n_quadros: int, n_gancho: int, janela_s: float) -> list[float]:
    """Instantes ideais: o gancho no centro de fatias iguais da janela; o resto no centro de fatias iguais do restante."""
    if duracao_s <= janela_s:
        return [duracao_s * (k + 0.5) / n_quadros for k in range(n_quadros)]
    resto = n_quadros - n_gancho
    gancho = [janela_s * (k + 0.5) / n_gancho for k in range(n_gancho)]
    return gancho + [janela_s + (duracao_s - janela_s) * (k + 0.5) / resto for k in range(resto)]


def planejar_instantes(duracao_s: float, fps: float | None, *, n_quadros: int = N_QUADROS,
                       n_gancho: int = N_QUADROS_GANCHO, janela_gancho_s: float = JANELA_GANCHO_S) -> PlanoAmostragem:
    """Instantes (s) e índices dos quadros a ler; `ValueError` para duração inválida ou gancho maior que o total."""
    if not (isinstance(duracao_s, (int, float)) and math.isfinite(duracao_s) and duracao_s > 0):
        raise ValueError("duração do vídeo inválida")
    if n_quadros < 1 or n_gancho > n_quadros:
        raise ValueError("o gancho não pode ter mais quadros que o total")
    taxa = float(fps) if fps and math.isfinite(fps) and fps > 0 else FPS_PADRAO
    total_quadros = max(1, math.floor(duracao_s * taxa + 1e-9))
    indices: list[int] = []
    for instante in _bruto(duracao_s, n_quadros, n_gancho, janela_gancho_s):
        indice = min(max(0, math.floor(instante * taxa + 1e-9)), total_quadros - 1)
        if indice not in indices:  # um quadro só entra uma vez: clipes curtos têm menos de 8 quadros distintos
            indices.append(indice)
    indices.sort()
    instantes = tuple(round(i / taxa, 6) for i in indices)
    no_gancho = sum(1 for t in instantes if t < janela_gancho_s)
    return PlanoAmostragem(instantes_s=instantes, indices_quadro=tuple(indices), n_gancho=no_gancho,
                           poucos_quadros=len(instantes) < MIN_QUADROS_TAPI)


# ──── Leitura de quadros reais do vídeo: os amostrados (PNG sem perdas, em disco) e os pares próximos para
@dataclass(frozen=True)
class QuadroAmostrado:
    ordem: int
    indice_planejado: int
    instante_planejado_s: float
    instante_real_s: float
    caminho: Path
    largura: int
    altura: int


def rotacionar(imagem: np.ndarray, graus: int) -> np.ndarray:
    """Gira a imagem como a exibição (sentido anti-horário); 0 devolve o próprio objeto."""
    graus = graus % 360
    if graus == 0:
        return imagem
    if graus % 90:
        raise ValueError("só rotações em múltiplos de 90 graus")
    return np.rot90(imagem, k=graus // 90)


def ler_png(caminho: Path) -> np.ndarray:
    """Quadro RGB uint8 de um PNG (leitura por bytes: funciona com caminhos Unicode no Windows)."""
    return decodificar_imagem(Path(caminho))


def _dimensoes_reduzidas(largura: int, altura: int, lado_max_px: int) -> tuple[int, int]:
    escala = min(1.0, lado_max_px / max(largura, altura))
    return max(2, round(largura * escala)), max(2, round(altura * escala))


def _deslocamento_inicial_s(fluxo) -> float:
    return float(fluxo.start_time * fluxo.time_base) if fluxo.start_time is not None else 0.0


def _quadro_em(conteiner, fluxo, instante_s: float, tolerancia_s: float):
    """Primeiro quadro com instante >= `instante_s` - tolerância e seu instante; (None, None) se o vídeo acabar antes.

    Nunca devolve "o último quadro que existe" no lugar de um instante que o vídeo não tem: vários instantes além do
    fim virariam o mesmo quadro repetido e enviesariam toda média calculada sobre os quadros.
    """
    inicio = _deslocamento_inicial_s(fluxo)
    conteiner.seek(int((instante_s + inicio) / fluxo.time_base), stream=fluxo, backward=True, any_frame=False)
    for quadro in conteiner.decode(fluxo):
        if quadro.pts is None:
            continue
        instante_real = float(quadro.pts * fluxo.time_base) - inicio
        if instante_real >= instante_s - tolerancia_s:
            return quadro, instante_real
    return None, None


def _gravar_png(caminho: Path, rgb: np.ndarray) -> None:
    """PNG sem perdas, gravado de forma atômica (temporário no mesmo diretório e `os.replace`)."""
    gerado, codificado = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not gerado:
        raise OSError("falha ao codificar o quadro em PNG")
    descritor, nome = tempfile.mkstemp(dir=caminho.parent, prefix=caminho.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(codificado.tobytes())
        os.replace(nome, caminho)
    finally:
        Path(nome).unlink(missing_ok=True)


def extrair_quadros(caminho: Path, plano: PlanoAmostragem, destino: Path, *, rotacao_graus: int = 0,
                    lado_max_px: int = LADO_MAX_QUADRO_PX) -> list[QuadroAmostrado]:
    """Lê os quadros do plano e os grava em `destino`; levanta `ErroExtracao` se nenhum puder ser decodificado."""
    destino.mkdir(parents=True, exist_ok=True)
    quadros: list[QuadroAmostrado] = []
    try:
        with av.open(str(caminho), options=OPCOES_FFMPEG) as conteiner:
            fluxo = conteiner.streams.video[0]
            taxa = float(fluxo.average_rate or fluxo.guessed_rate or FPS_PADRAO)
            largura, altura = _dimensoes_reduzidas(fluxo.codec_context.width, fluxo.codec_context.height, lado_max_px)
            for ordem, (indice, instante) in enumerate(zip(plano.indices_quadro, plano.instantes_s, strict=True)):
                quadro, real = _quadro_em(conteiner, fluxo, instante, 0.5 / taxa)
                if quadro is None:
                    continue
                rgb = rotacionar(quadro.to_ndarray(width=largura, height=altura, format="rgb24"), rotacao_graus)
                arquivo = destino / f"{ordem:02d}_{indice:06d}.png"
                _gravar_png(arquivo, rgb)
                quadros.append(QuadroAmostrado(ordem, indice, instante, real, arquivo, rgb.shape[1], rgb.shape[0]))
    except ErroExtracao:
        raise
    except OSError:
        raise ErroExtracao("arquivo_ilegivel", "Falha de leitura ou de gravação ao extrair os quadros.") from None
    except Exception:  # noqa: BLE001 - erro do FFmpeg ao decodificar
        raise ErroExtracao("midia_invalida", "Falha ao decodificar quadros do vídeo.") from None
    if not quadros:
        raise ErroExtracao("midia_invalida", "Nenhum quadro pôde ser decodificado nos instantes planejados.")
    return quadros


def _quadro_apos(conteiner, fluxo, pts_inicial: int, salto: int):
    """O quadro `salto` posições depois do que tem `pts_inicial`; None se o arquivo acabar antes."""
    avancados, ultimo = 0, None
    for candidato in conteiner.decode(fluxo):
        if candidato.pts is None or candidato.pts <= pts_inicial:
            continue
        ultimo, avancados = candidato, avancados + 1
        if avancados == salto:
            return ultimo
    return None


def ler_pares_fluxo(caminho: Path, instantes_s: Sequence[float], *, salto: int, lado_px: int, rotacao_graus: int,
                    fps: float | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pares (quadro, quadro `salto` posições depois) em cinza reduzido; instantes sem par completo são omitidos."""
    pares: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        with av.open(str(caminho), options=OPCOES_FFMPEG) as conteiner:
            fluxo = conteiner.streams.video[0]
            taxa = float(fps or fluxo.average_rate or FPS_PADRAO)
            largura, altura = _dimensoes_reduzidas(fluxo.codec_context.width, fluxo.codec_context.height, lado_px)
            for instante in instantes_s:
                primeiro, _ = _quadro_em(conteiner, fluxo, instante, 0.5 / taxa)
                if primeiro is None:
                    continue
                seguinte = _quadro_apos(conteiner, fluxo, primeiro.pts, salto)
                if seguinte is None:
                    continue
                a, b = (rotacionar(q.to_ndarray(width=largura, height=altura, format="gray"), rotacao_graus)
                        for q in (primeiro, seguinte))
                pares.append((a, b))
    except Exception:  # noqa: BLE001 - erro do FFmpeg
        raise ErroExtracao("midia_invalida", "Falha ao decodificar quadros do vídeo.") from None
    return pares


# ──── Cortes de cena com PySceneDetect (ContentDetector), com taxa normalizada pela duração.
MARGEM_DURACAO_S = 1.0  # o cabeçalho do contêiner pode errar a duração; o detector nunca lê além dela mais esta margem


def contar_cortes(caminho: Path, duracao_s: float, fps: float | None) -> dict:
    """Cortes (n, por 10 s, no gancho) e duração das cenas; `ErroExtracao` se o vídeo não pôde ser percorrido."""
    from scenedetect import ContentDetector, SceneManager, open_video

    taxa = fps or FPS_PADRAO
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", OPENCV_FFMPEG_OPCOES)  # só o protocolo `file`
    try:
        video = open_video(str(caminho), backend="opencv")
        gerente = SceneManager()
        gerente.add_detector(ContentDetector(threshold=CORTES_LIMIAR, min_scene_len=max(1, math.ceil(CORTES_DURACAO_MIN_S * taxa))))
        gerente.detect_scenes(video, duration=float(duracao_s) + MARGEM_DURACAO_S, show_progress=False)
        cenas = gerente.get_scene_list(start_in_scene=True)
        quadros_lidos = video.frame_number
    except Exception:  # noqa: BLE001 - qualquer falha de leitura do PySceneDetect
        raise ErroExtracao("midia_invalida", "A detecção de cortes não conseguiu ler o vídeo.") from None
    if quadros_lidos <= 0 or not cenas:
        raise ErroExtracao("midia_invalida", "A detecção de cortes não leu nenhum quadro.")
    inicios = [inicio.seconds for inicio, _ in cenas]
    duracoes = [fim.seconds - inicio.seconds for inicio, fim in cenas]
    cortes = len(cenas) - 1
    return {"cortes_n": cortes, "cortes_por_10s": cortes / duracao_s * 10.0,
            "cena_media_s": sum(duracoes) / len(duracoes), "cena_max_frac": max(duracoes) / sum(duracoes),
            "cortes_gancho_n": sum(1 for inicio in inicios[1:] if inicio < JANELA_GANCHO_S)}


# ──── Medidas OpenCV dos quadros amostrados: cor, brilho, contraste, saturação e movimento.
def resumir_cor(imagens: Sequence[np.ndarray]) -> dict:
    """Média e desvio, entre os quadros RGB, de brilho, contraste e saturação; cor dominante média.

    Com um único quadro não há desvio entre quadros: os `*_std` são None (um 0,0 pareceria uma medida).
    """
    if not imagens:
        raise ValueError("nenhum quadro para medir")
    linhas = [medidas_cor(imagem) for imagem in imagens]

    def serie(chave: str) -> np.ndarray:
        return np.array([linha[chave] for linha in linhas], dtype=np.float64)

    def desvio(chave: str) -> float | None:
        return float(serie(chave).std()) if len(linhas) >= 2 else None

    return {
        "brilho_medio": float(serie("cor_brilho").mean()), "brilho_std": desvio("cor_brilho"),
        "contraste_medio": float(serie("cor_contraste").mean()), "contraste_std": desvio("cor_contraste"),
        "saturacao_medio": float(serie("cor_saturacao").mean()), "saturacao_std": desvio("cor_saturacao"),
        "cor_r": float(serie("cor_dominante_r").mean()), "cor_g": float(serie("cor_dominante_g").mean()),
        "cor_b": float(serie("cor_dominante_b").mean()), "cor_fracao_dominante": float(serie("cor_fracao_dominante").mean()),
    }


def _fluxo_normalizado(par: tuple[np.ndarray, np.ndarray]) -> float:
    """Magnitude média do fluxo, em pixels, dividida pelo maior lado da imagem (comparável entre resoluções)."""
    antes, depois = par
    fluxo = cv2.calcOpticalFlowFarneback(antes, depois, None, *FARNEBACK)
    return float(np.linalg.norm(fluxo, axis=2).mean() / max(antes.shape))


def _diferenca_entre_quadros(quadros: Sequence[np.ndarray]) -> float:
    forma = quadros[0].shape
    alinhados = [q if q.shape == forma else cv2.resize(q, (forma[1], forma[0])) for q in quadros]
    return float(np.mean([np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() / 255.0
                          for a, b in zip(alinhados, alinhados[1:], strict=False)]))


def resumir_movimento(pares: Sequence[tuple[np.ndarray, np.ndarray]], quadros_cinza: Sequence[np.ndarray]) -> dict:
    """Fluxo óptico (média, desvio, p90) dos pares e diferença média entre quadros amostrados consecutivos.

    Cada parte só existe se houver dados: sem pares o fluxo é None; com menos de dois quadros a diferença é None; sem
    nenhum dos dois, `DadosInsuficientes`.
    """
    if not pares and len(quadros_cinza) < 2:
        raise DadosInsuficientes("movimento exige ao menos um par de quadros próximos ou dois quadros amostrados")
    fluxos = np.array([_fluxo_normalizado(par) for par in pares]) if pares else None
    return {
        "fluxo_medio": None if fluxos is None else float(fluxos.mean()),
        "fluxo_std": None if fluxos is None else float(fluxos.std()),
        "fluxo_p90": None if fluxos is None else float(np.percentile(fluxos, 90)),
        "pares_n": len(pares),  # quantos pares foram de fato lidos (pode ser menos que os planejados)
        "diff_quadros_medio": _diferenca_entre_quadros(quadros_cinza) if len(quadros_cinza) >= 2 else None,
    }


# ──── Detecção de objetos com YOLOX-tiny (ONNX Runtime, CPU): 80 classes COCO.
CLASSES_COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
CLASSES_VEICULO = frozenset({"car", "motorcycle", "bus", "truck"})
STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class Deteccao:
    classe_id: int
    classe: str
    confianca: float
    area_frac: float  # área da caixa (recortada ao quadro) sobre a área do quadro
    caixa: tuple[float, float, float, float]  # x0, y0, x1, y1 no quadro original


def preparar_entrada(rgb: np.ndarray, tamanho: int = YOLOX_ENTRADA) -> tuple[np.ndarray, float]:
    """Tensor (1, 3, T, T) float32 em BGR 0-255 com letterbox 114 no canto superior esquerdo, e a razão de escala."""
    altura, largura = rgb.shape[:2]
    razao = min(tamanho / altura, tamanho / largura)
    nova = (max(1, int(largura * razao)), max(1, int(altura * razao)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tela = np.full((tamanho, tamanho, 3), 114, dtype=np.uint8)
    tela[: nova[1], : nova[0]] = cv2.resize(bgr, nova, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(tela.transpose(2, 0, 1)[None], dtype=np.float32), razao


def _grades(tamanho: int) -> tuple[np.ndarray, np.ndarray]:
    grades, passos = [], []
    for stride in STRIDES:
        lado = tamanho // stride
        x, y = np.meshgrid(np.arange(lado), np.arange(lado))
        grades.append(np.stack((x, y), axis=2).reshape(-1, 2))
        passos.append(np.full((lado * lado, 1), stride))
    return np.concatenate(grades).astype(np.float32), np.concatenate(passos).astype(np.float32)


def decodificar_saidas(saida: np.ndarray, forma_original: tuple[int, int], tamanho: int = YOLOX_ENTRADA, *,
                       confianca: float = YOLOX_CONFIANCA, nms: float = YOLOX_NMS) -> list[Deteccao]:
    """Detecções no quadro original a partir da saída bruta (1, N, 85) do YOLOX; NMS por classe."""
    altura, largura = forma_original
    previsoes = np.array(saida[0], dtype=np.float32, copy=True)
    grade, passo = _grades(tamanho)
    previsoes[:, :2] = (previsoes[:, :2] + grade) * passo
    previsoes[:, 2:4] = np.exp(previsoes[:, 2:4]) * passo
    pontuacoes = previsoes[:, 4:5] * previsoes[:, 5:]
    classes = pontuacoes.argmax(axis=1)
    melhores = pontuacoes[np.arange(len(classes)), classes]
    manter = melhores > confianca
    if not manter.any():
        return []
    razao = min(tamanho / altura, tamanho / largura)
    caixas, melhores, classes = previsoes[manter, :4], melhores[manter], classes[manter]
    xyxy = np.stack((caixas[:, 0] - caixas[:, 2] / 2, caixas[:, 1] - caixas[:, 3] / 2,
                     caixas[:, 0] + caixas[:, 2] / 2, caixas[:, 1] + caixas[:, 3] / 2), axis=1) / razao
    deteccoes: list[Deteccao] = []
    for classe in np.unique(classes):
        indices = np.flatnonzero(classes == classe)
        xywh = [[float(x0), float(y0), float(x1 - x0), float(y1 - y0)] for x0, y0, x1, y1 in xyxy[indices]]
        escolhidos = cv2.dnn.NMSBoxes(xywh, melhores[indices].astype(float).tolist(), confianca, nms)
        for posicao in np.array(escolhidos, dtype=int).reshape(-1):
            x0, y0, x1, y1 = (float(v) for v in xyxy[indices[posicao]])
            x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(largura)), min(y1, float(altura))
            area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0) / (largura * altura)
            deteccoes.append(Deteccao(int(classe), CLASSES_COCO[int(classe)], float(melhores[indices[posicao]]),
                                      float(area), (x0, y0, x1, y1)))
    return sorted(deteccoes, key=lambda d: -d.confianca)


def resumir_objetos(deteccoes_por_quadro: Sequence[Sequence[Deteccao]]) -> dict:
    """Presença e área de pessoa e de veículo, e quantidade, sobre os quadros em que o detector rodou.

    Um quadro sem detecção entra como observação (o detector rodou e não achou nada nas classes COCO); a ausência de
    quadros não é observação e levanta `ValueError`.
    """
    if not deteccoes_por_quadro:
        raise ValueError("nenhum quadro foi processado pelo detector")
    n = len(deteccoes_por_quadro)
    pessoa = [[d for d in q if d.classe == "person"] for q in deteccoes_por_quadro]
    veiculo = [[d for d in q if d.classe in CLASSES_VEICULO] for q in deteccoes_por_quadro]
    return {
        "obj_pessoa_frac_quadros": sum(1 for q in pessoa if q) / n,
        "obj_pessoa_area_max": max((d.area_frac for q in pessoa for d in q), default=0.0),
        "obj_veiculo_frac_quadros": sum(1 for q in veiculo if q) / n,
        "obj_n_max": max(len(q) for q in deteccoes_por_quadro),
        "obj_classes_distintas": len({d.classe for q in deteccoes_por_quadro for d in q}),
    }


class DetectorObjetos:
    """YOLOX-tiny em ONNX Runtime (CPU, threads limitadas), carregado no primeiro uso, só do arquivo já em disco."""

    def __init__(self, threads: int = THREADS) -> None:
        self.threads, self._sessao = threads, None

    def _carregar(self) -> None:
        if self._sessao is not None:
            return
        import onnxruntime as ort

        from src.extracao.modelos import localizar

        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads, opcoes.inter_op_num_threads = self.threads, 1
        self._sessao = ort.InferenceSession(str(localizar("yolox")), sess_options=opcoes,
                                            providers=["CPUExecutionProvider"])

    def liberar(self) -> None:
        self._sessao = None

    def __call__(self, rgb: np.ndarray) -> list[Deteccao]:
        self._carregar()
        entrada, _ = preparar_entrada(rgb)
        saida = self._sessao.run(None, {self._sessao.get_inputs()[0].name: entrada})[0]
        return decodificar_saidas(saida, rgb.shape[:2])
