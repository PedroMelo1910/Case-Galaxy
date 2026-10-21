"""Módulo consolidado. Reúne:
- Atributos locais da capa (C6; família N3), a partir de bytes ou de um arquivo de imagem.
- Rosto e texto na tela, por quadro, reaproveitando os extratores da capa (`src/extracao/imagem.py`)."""

import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

from src.extracao.modelos import (
    DIRETORIO_MODELOS,
    ErroExtracao,
    localizar,
    sem_caminhos,
    texto_bibliotecas_ausentes,
    THREADS,
    verificar_bibliotecas,
)
from src.extracao.texto import validar_vetor
from src.proveniencia import sha256_arquivo as _sha256_arquivo


# ──── Atributos locais da capa (C6; família N3), a partir de bytes ou de um arquivo de imagem.
VERSAO = "1"
LIMIAR_PRETO = 12
FRACAO_LINHA_PRETA = 0.98
AREA_MINIMA_RESTANTE = 0.15
LADO_MEDIDAS = 256
PASSO_COR = 64
CONFIANCA_MINIMA_OCR = 0.3
CONFIANCA_MINIMA_ROSTO = 0.5
IDIOMAS_OCR = ("pt", "en")
MODELO_IMAGEM = "openai/clip-vit-base-patch32"
DIMENSAO_IMAGEM = 512
URL_MODELO_ROSTO = ("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/"
                    "latest/blaze_face_short_range.tflite")
MEDIDAS_COR = ("cor_dominante_r", "cor_dominante_g", "cor_dominante_b", "cor_fracao_dominante", "cor_brilho",
               "cor_contraste", "cor_saturacao")
MEDIDAS_ROSTO = ("rosto_presente", "rosto_n", "rosto_maior_fracao_area")
MEDIDAS_OCR = ("ocr_tem_texto", "ocr_n_caixas", "ocr_n_caracteres", "ocr_n_palavras", "ocr_fracao_area")
REGRAS = {
    "recorte": f"remove só faixas externas contínuas em que ≥ {FRACAO_LINHA_PRETA:.0%} dos pixels têm todos os "
    f"canais ≤ {LIMIAR_PRETO}; sem recorte se a imagem é toda escura ou se sobraria menos de "
    f"{AREA_MINIMA_RESTANTE:.0%} da área",
    "cor": f"imagem reduzida (lado maior ≤ {LADO_MEDIDAS}, INTER_AREA); cor dominante = centro da célula mais "
    f"frequente na quantização de {PASSO_COR} níveis por canal RGB; brilho = média de V; saturação = média de S "
    "(HSV); contraste = desvio-padrão do cinza; valores de 0 a 1",
    "rosto": f"MediaPipe FaceDetector (BlazeFace curto alcance), confiança ≥ {CONFIANCA_MINIMA_ROSTO}; maior área "
    "relativa do retângulo do rosto",
    "ocr": f"EasyOCR {list(IDIOMAS_OCR)}, sem GPU; caixas com confiança ≥ {CONFIANCA_MINIMA_OCR}; área = soma dos "
    "retângulos alinhados aos eixos, limitada a 1",
    "embedding": f"{MODELO_IMAGEM}, image_embeds da projeção treinada ({DIMENSAO_IMAGEM} valores), sem normalizar",
}


class ImagemInvalida(ValueError):
    """Bytes vazios ou que não decodificam como imagem."""


def decodificar_imagem(origem: bytes | Path) -> np.ndarray:
    """Imagem RGB uint8 a partir de bytes ou de um arquivo; ImagemInvalida se não decodificar."""
    dados = Path(origem).read_bytes() if isinstance(origem, Path) else bytes(origem)
    if not dados:
        raise ImagemInvalida("imagem vazia")
    imagem = cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None or imagem.ndim != 3 or min(imagem.shape[:2]) < 1:
        raise ImagemInvalida("bytes não decodificam como imagem")
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def _limites(pretas: np.ndarray) -> tuple[int, int] | None:
    claras = np.flatnonzero(~pretas)
    return (int(claras[0]), int(claras[-1]) + 1) if claras.size else None


def remover_faixas_pretas(imagem: np.ndarray) -> tuple[np.ndarray, dict]:
    """Recorta faixas externas pretas; devolve a imagem e o registro do recorte."""
    altura, largura = imagem.shape[:2]
    preto = imagem.max(axis=2) <= LIMIAR_PRETO
    colunas = _limites(preto.mean(axis=0) >= FRACAO_LINHA_PRETA)
    linhas = _limites(preto.mean(axis=1) >= FRACAO_LINHA_PRETA)
    inteira = {"recortado": False, "caixa": (0, 0, largura, altura), "largura": largura, "altura": altura}
    if colunas is None or linhas is None:
        return imagem, inteira | {"motivo": "imagem_toda_escura"}
    (x0, x1), (y0, y1) = colunas, linhas
    if (x0, y0, x1, y1) == (0, 0, largura, altura):
        return imagem, inteira | {"motivo": "sem_faixas"}
    if (x1 - x0) * (y1 - y0) < AREA_MINIMA_RESTANTE * largura * altura:
        return imagem, inteira | {"motivo": "recorte_excessivo"}
    return imagem[y0:y1, x0:x1], {"recortado": True, "caixa": (x0, y0, x1, y1), "largura": largura,
                                  "altura": altura, "motivo": "faixas_pretas"}


def medidas_cor(imagem: np.ndarray) -> dict:
    """Cor dominante, fração, brilho, contraste e saturação (determinístico, sem sorteio)."""
    escala = LADO_MEDIDAS / max(imagem.shape[:2])
    pequena = imagem if escala >= 1 else cv2.resize(imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(pequena, cv2.COLOR_RGB2HSV)
    cinza = cv2.cvtColor(pequena, cv2.COLOR_RGB2GRAY)
    niveis = 256 // PASSO_COR
    celulas = (pequena // PASSO_COR).astype(np.int64)
    codigos = celulas[..., 0] * niveis * niveis + celulas[..., 1] * niveis + celulas[..., 2]
    contagem = np.bincount(codigos.ravel(), minlength=niveis ** 3)
    dominante = int(np.argmax(contagem))
    centro = [(dominante // niveis ** k) % niveis * PASSO_COR + PASSO_COR // 2 for k in (2, 1, 0)]
    return {
        "cor_dominante_r": centro[0], "cor_dominante_g": centro[1], "cor_dominante_b": centro[2],
        "cor_fracao_dominante": float(contagem[dominante] / codigos.size),
        "cor_brilho": float(hsv[..., 2].mean() / 255), "cor_contraste": float(cinza.std() / 255),
        "cor_saturacao": float(hsv[..., 1].mean() / 255),
    }


def resumo_rostos(caixas: Sequence[tuple[float, float, float, float]], forma: tuple[int, int]) -> dict:
    """Presença, quantidade e maior área relativa a partir de retângulos (x, y, largura, altura) em pixels."""
    area = forma[0] * forma[1]
    maior = max((max(w, 0) * max(h, 0) for _, _, w, h in caixas), default=0)
    return {"rosto_presente": int(bool(caixas)), "rosto_n": len(caixas),
            "rosto_maior_fracao_area": float(min(maior / area, 1.0))}


def resumo_ocr(deteccoes: Iterable[tuple], forma: tuple[int, int]) -> dict:
    """Medidas do texto lido com confiança mínima; o texto em si fica como coluna auxiliar."""
    validas = [(caixa, texto.strip(), conf) for caixa, texto, conf in deteccoes
               if conf >= CONFIANCA_MINIMA_OCR and texto.strip()]
    area = 0.0
    for caixa, _, _ in validas:
        xs, ys = [p[0] for p in caixa], [p[1] for p in caixa]
        area += (max(xs) - min(xs)) * (max(ys) - min(ys))
    textos = [texto for _, texto, _ in validas]
    return {"ocr_tem_texto": int(bool(textos)), "ocr_n_caixas": len(textos),
            "ocr_n_caracteres": sum(len(t) for t in textos), "ocr_n_palavras": sum(len(t.split()) for t in textos),
            "ocr_fracao_area": float(min(area / (forma[0] * forma[1]), 1.0)), "ocr_texto": " ".join(textos)}


def aplicar_extrator(extrator: Callable[[np.ndarray], dict], imagem: np.ndarray) -> dict:
    """Resultado com estado `ok`, ou `falha_extrator` com o tipo do erro e nenhum valor de atributo."""
    try:
        return {"estado": "ok", **extrator(imagem)}
    except Exception as erro:  # noqa: BLE001 - toda falha do extrator é registrada, nunca vira ausência
        return {"estado": "falha_extrator", "erro": type(erro).__name__, "mensagem": str(erro)[:200]}


class DetectorRosto:
    """MediaPipe FaceDetector em modo imagem; o modelo é baixado uma vez da URL oficial e identificado por SHA-256."""

    def __init__(self, url: str = URL_MODELO_ROSTO, diretorio: Path = DIRETORIO_MODELOS) -> None:
        self.url, self.arquivo = url, Path(diretorio) / "mediapipe" / Path(url).name
        self._detector = None

    def _carregar(self) -> None:
        if self._detector is not None:
            return
        import httpx
        import mediapipe as mp

        if not self.arquivo.exists():
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
            resposta = httpx.get(self.url, timeout=60, follow_redirects=True)
            resposta.raise_for_status()
            parcial = self.arquivo.with_name(self.arquivo.name + ".parcial")
            try:  # gravação atômica: uma queda no meio nunca deixa um .tflite truncado no lugar do modelo
                parcial.write_bytes(resposta.content)
                os.replace(parcial, self.arquivo)
            finally:
                parcial.unlink(missing_ok=True)
        opcoes = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.arquivo)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, min_detection_confidence=CONFIANCA_MINIMA_ROSTO)
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(opcoes)

    def identificacao(self) -> dict:
        self._carregar()
        import mediapipe as mp

        return {"modelo": "MediaPipe BlazeFace curto alcance", "url": self.url, "sha256": _sha256_arquivo(self.arquivo),
                "mediapipe": mp.__version__, "confianca_minima": CONFIANCA_MINIMA_ROSTO}

    def __call__(self, imagem: np.ndarray) -> dict:
        self._carregar()
        import mediapipe as mp

        resultado = self._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                   data=np.ascontiguousarray(imagem)))
        caixas = [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)
                  for d in resultado.detections]
        return resumo_rostos(caixas, imagem.shape[:2])


class LeitorTexto:
    """EasyOCR em CPU com português e inglês; pesos baixados pelo próprio EasyOCR na primeira execução."""

    def __init__(self, diretorio: Path = DIRETORIO_MODELOS, threads: int = THREADS) -> None:
        self.diretorio, self.threads = Path(diretorio) / "easyocr", threads
        self._leitor = None

    def _carregar(self) -> None:
        if self._leitor is not None:
            return
        import easyocr
        import torch

        torch.set_num_threads(self.threads)
        self.diretorio.mkdir(parents=True, exist_ok=True)
        self._leitor = easyocr.Reader(list(IDIOMAS_OCR), gpu=False, model_storage_directory=str(self.diretorio),
                                      verbose=False)

    def identificacao(self) -> dict:
        self._carregar()
        import easyocr

        pesos = {p.name: _sha256_arquivo(p) for p in sorted(self.diretorio.glob("*.pth"))}
        return {"modelo": "EasyOCR", "easyocr": easyocr.__version__, "idiomas": list(IDIOMAS_OCR),
                "confianca_minima": CONFIANCA_MINIMA_OCR, "pesos_sha256": pesos}

    def __call__(self, imagem: np.ndarray) -> dict:
        self._carregar()
        deteccoes = self._leitor.readtext(cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR))
        return resumo_ocr(deteccoes, imagem.shape[:2])


class CodificadorImagem:
    """CLIP ViT-B/32 (visão + projeção treinada) em CPU, carregado no primeiro uso."""

    def __init__(self, modelo: str = MODELO_IMAGEM, revisao: str | None = None, threads: int = THREADS) -> None:
        self.modelo, self.revisao, self.threads = modelo, revisao, threads
        self._processador = self._rede = None

    def _carregar(self) -> None:
        if self._rede is not None:
            return
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        torch.set_num_threads(self.threads)
        self._processador = CLIPImageProcessor.from_pretrained(self.modelo, revision=self.revisao)
        self._rede = CLIPVisionModelWithProjection.from_pretrained(self.modelo, revision=self.revisao).eval()

    def identificacao(self) -> dict:
        self._carregar()
        import torch
        import transformers

        return {"modelo": self.modelo, "revisao": getattr(self._rede.config, "_commit_hash", None),
                "dimensao": DIMENSAO_IMAGEM, "saida": "image_embeds (projeção treinada), sem normalizar",
                "torch": torch.__version__, "transformers": transformers.__version__}

    def __call__(self, imagem: np.ndarray) -> dict:
        self._carregar()
        import torch

        entrada = self._processador(images=imagem, return_tensors="pt")
        with torch.inference_mode():
            vetor = self._rede(**entrada).image_embeds[0].tolist()
        return {"capa_emb": validar_vetor(vetor, DIMENSAO_IMAGEM)}


# ──── Rosto e texto na tela, por quadro, reaproveitando os extratores da capa (`src/extracao/imagem.py`).
BIBLIOTECA_DO_MEDIAPIPE = "libGLESv2.so.2"  # biblioteca de sistema (Linux) que o MediaPipe carrega com ctypes


class RostoPorQuadro:
    """MediaPipe BlazeFace por quadro; o peso é lido do disco (a preparação o baixa uma vez)."""

    def __init__(self) -> None:
        self._detector: DetectorRosto | None = None

    def _exigir_biblioteca(self) -> None:
        """No Linux, o MediaPipe precisa de uma biblioteca do sistema; sem ela o erro vira uma instrução, não um OSError."""
        ausentes = [e.nome for e in verificar_bibliotecas((BIBLIOTECA_DO_MEDIAPIPE,)) if e.estado == "ausente"]
        if ausentes:
            raise ErroExtracao("biblioteca_ausente", "Biblioteca do sistema ausente para o detector de rosto:\n"
                               + "\n".join(texto_bibliotecas_ausentes(ausentes)))

    def __call__(self, rgb: np.ndarray) -> dict:
        if self._detector is None:
            self._exigir_biblioteca()
            localizar("rosto")  # o extrator de capa baixaria o peso se faltasse; a inferência nunca baixa nada
            self._detector = DetectorRosto(diretorio=DIRETORIO_MODELOS)
        return self._detector(rgb)

    def liberar(self) -> None:
        self._detector = None


class TextoPorQuadro:
    """EasyOCR (português e inglês, CPU) por quadro; os pesos são lidos do disco."""

    def __init__(self, threads: int = THREADS) -> None:
        self.threads, self._leitor = threads, None

    def __call__(self, rgb: np.ndarray) -> dict:
        if self._leitor is None:
            localizar("ocr")  # o EasyOCR baixaria os pesos se faltassem; a inferência nunca baixa nada
            self._leitor = LeitorTexto(diretorio=DIRETORIO_MODELOS, threads=self.threads)
        return self._leitor(rgb)

    def liberar(self) -> None:
        self._leitor = None


def _todos_ok(resultados: Sequence[dict]) -> list[dict]:
    """Os resultados por quadro, se todos `ok`; senão `ErroExtracao` com a contagem e o tipo das falhas."""
    if not resultados:
        raise ValueError("nenhum quadro foi processado")
    falhas = [r for r in resultados if r.get("estado") != "ok"]
    if falhas:
        tipos = sorted({str(f.get("erro", f.get("estado"))) for f in falhas})
        primeira = sem_caminhos(str(falhas[0].get("mensagem", "")))
        raise ErroExtracao("falha_extrator", f"{len(falhas)} de {len(resultados)} quadros falharam "
                           f"({', '.join(tipos)}). Primeira mensagem: {primeira}")
    return list(resultados)


def resumir_rostos(resultados: Sequence[dict], n_gancho: int) -> dict:
    """Fração de quadros com rosto, máximo de rostos e de área, e fração entre os quadros do gancho."""
    r = _todos_ok(resultados)
    gancho = r[:n_gancho] or r
    return {"rosto_frac_quadros": float(np.mean([q["rosto_presente"] for q in r])),
            "rosto_n_max": int(max(q["rosto_n"] for q in r)),
            "rosto_area_max": float(max(q["rosto_maior_fracao_area"] for q in r)),
            "rosto_gancho_frac": float(np.mean([q["rosto_presente"] for q in gancho]))}


def resumir_texto(resultados: Sequence[dict], n_gancho: int) -> dict:
    """Fração de quadros com texto, caracteres e área média, e fração entre os quadros do gancho.

    O texto lido (`ocr_texto`) fica como auxiliar para inspeção; não é atributo.
    """
    r = _todos_ok(resultados)
    gancho = r[:n_gancho] or r
    return {"ocr_frac_quadros": float(np.mean([q["ocr_tem_texto"] for q in r])),
            "ocr_caracteres_medio": float(np.mean([q["ocr_n_caracteres"] for q in r])),
            "ocr_area_media": float(np.mean([q["ocr_fracao_area"] for q in r])),
            "ocr_gancho_frac": float(np.mean([q["ocr_tem_texto"] for q in gancho])),
            "texto": " ".join(q["ocr_texto"] for q in r if q.get("ocr_texto"))}
