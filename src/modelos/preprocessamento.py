"""Módulo consolidado. Reúne:
- Pré-processador do modelo de desenvolvimento, em módulo próprio.
- Pré-processador do preditor audiovisual local: o do modelo anterior mais duas regras aprendidas só no treino."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


# ──── Pré-processador do modelo de desenvolvimento, em módulo próprio.
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


# ──── Pré-processador do preditor audiovisual local: o do modelo anterior mais duas regras aprendidas só n
OUTRO = "outro"
MIN_CANAIS_CATEGORIA = 3


def _presente(valor) -> bool:
    return valor is not None and not (isinstance(valor, float) and np.isnan(valor))


class PreprocessadorAudiovisual(BaseEstimator, TransformerMixin):
    """`Preprocessador` com consolidação de categorias locais raras e descarte de colunas locais constantes."""

    def __init__(self, escalares=(), categoricas=(), embeddings=None, semente=SEMENTE, colunas_locais=(),
                 min_canais_categoria=MIN_CANAIS_CATEGORIA):
        self.escalares, self.categoricas, self.embeddings, self.semente = escalares, categoricas, embeddings, semente
        self.colunas_locais, self.min_canais_categoria = colunas_locais, min_canais_categoria

    def _consolidar(self, linha):
        if not self.categorias_permitidas_:
            return linha
        nova = dict(linha)
        for coluna, permitidas in self.categorias_permitidas_.items():
            valor = linha.get(coluna)
            if _presente(valor):
                nova[coluna] = str(valor) if str(valor) in permitidas else OUTRO
        return nova

    def fit(self, X, y=None):
        locais = set(self.colunas_locais)
        self.categorias_permitidas_ = {}
        for coluna in (c for c in self.categoricas if c in locais):
            canais_por_valor: dict[str, set] = {}
            for linha in X:
                if _presente(linha.get(coluna)):
                    canais_por_valor.setdefault(str(linha[coluna]), set()).add(linha.get("channel_id"))
            self.categorias_permitidas_[coluna] = sorted(
                v for v, canais in canais_por_valor.items() if len(canais) >= self.min_canais_categoria)
        consolidadas = [self._consolidar(l) for l in X]
        constantes = []
        for coluna in (c for c in (*self.escalares, *self.categoricas) if c in locais):
            distintos = {str(l[coluna]) if coluna in self.categorias_permitidas_ else float(l[coluna])
                         for l in consolidadas if _presente(l.get(coluna))}
            if len(distintos) <= 1:
                constantes.append(coluna)
        self.colunas_descartadas_ = constantes
        self.interno_ = Preprocessador(
            [c for c in self.escalares if c not in constantes], [c for c in self.categoricas if c not in constantes],
            dict(self.embeddings or {}), semente=self.semente).fit(consolidadas)
        self.nomes_saida_ = self.interno_.nomes_saida_
        self.categorias_ = self.interno_.categorias_
        self.pca_ = self.interno_.pca_
        return self

    def transform(self, X):
        return self.interno_.transform([self._consolidar(l) for l in X])
