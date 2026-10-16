"""Pré-processador do modelo de desenvolvimento, em módulo próprio.

Fica fora de `modelo_desenvolvimento` de propósito: o modelo é gravado com pickle e a classe precisa ter um caminho
importável estável (`src.preprocessador`) mesmo quando o treino roda por `python -m src.modelo_desenvolvimento`.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

SEMENTE = 20261015


def _numero(valor) -> float:
    return float("nan") if valor is None else float(valor)


class Preprocessador(BaseEstimator, TransformerMixin):
    """Escalares, categorias ordinais e PCA por bloco de embedding; tudo aprendido só em `fit`."""

    def __init__(self, escalares=(), categoricas=(), embeddings=None, semente=SEMENTE):
        self.escalares, self.categoricas, self.embeddings, self.semente = escalares, categoricas, embeddings, semente

    def fit(self, X, y=None):
        self.categorias_ = {c: sorted({str(l[c]) for l in X if l.get(c) is not None}) for c in self.categoricas}
        self.pca_ = {}
        for bloco, k in (self.embeddings or {}).items():
            vetores = np.array([l[bloco] for l in X if l.get(bloco) is not None], dtype=np.float64)
            if len(vetores) < 2:
                self.pca_[bloco] = None
                continue
            self.pca_[bloco] = PCA(n_components=min(k, len(vetores) - 1, vetores.shape[1]), svd_solver="full",
                                   random_state=self.semente).fit(vetores)
        self.nomes_saida_ = [*self.escalares, *self.categoricas,
                             *(f"{b}_pc{i}" for b, k in (self.embeddings or {}).items() for i in range(k))]
        return self

    def transform(self, X):
        saida = np.full((len(X), len(self.nomes_saida_)), np.nan)
        for j, c in enumerate(self.escalares):
            saida[:, j] = [_numero(l.get(c)) for l in X]
        for j, c in enumerate(self.categoricas, start=len(self.escalares)):
            codigos = {valor: i for i, valor in enumerate(self.categorias_[c])}
            saida[:, j] = [codigos.get(str(l[c]), np.nan) if l.get(c) is not None else np.nan for l in X]
        coluna = len(self.escalares) + len(self.categoricas)
        for bloco, k in (self.embeddings or {}).items():
            pca = self.pca_[bloco]
            presentes = [i for i, l in enumerate(X) if l.get(bloco) is not None]
            if pca is not None and presentes:
                projecao = pca.transform(np.array([X[i][bloco] for i in presentes], dtype=np.float64))
                saida[np.ix_(presentes, range(coluna, coluna + projecao.shape[1]))] = projecao
            coluna += k
        return saida
