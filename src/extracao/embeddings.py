"""Embedding dos quadros do vídeo com SigLIP2 (encoder congelado, torre de visão, CPU)."""

from collections.abc import Sequence

import numpy as np

from src.extracao import modelos as esp
from src.extracao.modelos import LOTE_EMBEDDING, THREADS


# ──── Embedding dos quadros do vídeo com SigLIP2 (encoder congelado, torre de visão, CPU).
DIMENSAO = esp.MODELOS["siglip2"]["dimensao"]


class CodificadorQuadros:
    """SigLIP2 base-patch16-224 (torre de visão) em CPU, em lotes pequenos, só dos pesos já em disco."""

    def __init__(self, threads: int = THREADS, lote: int = LOTE_EMBEDDING) -> None:
        self.threads, self.lote = threads, lote
        self._processador = self._rede = None

    def _carregar(self) -> None:
        if self._rede is not None:
            return
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        from src.extracao.modelos import localizar

        torch.set_num_threads(self.threads)
        pasta = str(localizar("siglip2"))
        self._processador = AutoImageProcessor.from_pretrained(pasta, local_files_only=True)
        self._rede = SiglipVisionModel.from_pretrained(pasta, local_files_only=True, dtype=torch.float32).eval()

    def liberar(self) -> None:
        self._processador = self._rede = None

    def __call__(self, imagens: Sequence[np.ndarray]) -> np.ndarray:
        """Matriz (n, 768) float32, um vetor por quadro RGB."""
        self._carregar()
        import torch

        partes = []
        for inicio in range(0, len(imagens), self.lote):
            entrada = self._processador(images=list(imagens[inicio: inicio + self.lote]), return_tensors="pt")
            with torch.inference_mode():
                partes.append(self._rede(**entrada).pooler_output.float().numpy())
        return np.concatenate(partes).astype(np.float32)


def resumir_embeddings(matriz: np.ndarray, n_gancho: int) -> dict:
    """Média de todos os quadros e média dos quadros do gancho (os primeiros `n_gancho`), como listas de float."""
    matriz = np.asarray(matriz, dtype=np.float64)
    if matriz.ndim != 2 or matriz.shape[0] < 1 or matriz.shape[1] != DIMENSAO:
        raise ValueError(f"embedding com forma inesperada {matriz.shape}; esperado (n, {DIMENSAO})")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("embedding com valores não finitos")
    gancho = matriz[:n_gancho] if n_gancho >= 1 else matriz
    return {"emb_media": matriz.mean(axis=0).tolist(), "emb_gancho": gancho.mean(axis=0).tolist()}
