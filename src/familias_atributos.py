"""Famílias de atributos e especificações operacionais dos modelos de texto e de imagem.

As famílias são aninhadas: histórico e contexto → + legenda → + capa → + Gemini (modelo completo). As especificações
experimentais E1–E4 (congeladas em `resultado_experimento_familias.json`) são reproduzidas por `especificacao_ate` e
nunca são alteradas.

Adaptação operacional versionada: os modelos de texto (de E2) e de imagem (de E3) não usam `duracao_s` como preditor.
Um criativo de texto ou de imagem não tem duração de vídeo e o usuário normalmente não a informa; por isso treino,
validação e inferência partem do mesmo conjunto de preditores, sem ela. A exclusão é uma derivação, não uma nova busca.
"""

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from src.modelo_desenvolvimento import EspecificacaoAtributos

VERSAO = "1"
FAMILIAS = ("historico", "legenda", "capa", "gemini")
COLUNAS_NAO_PREDITORAS = ("duracao_s",)


class ErroFamilia(ValueError):
    """Modalidade sem modelo reduzido ou especificação incoerente."""


@dataclass(frozen=True)
class Modalidade:
    nome: str
    blocos: tuple[str, ...]
    origem_experimental: str
    nivel: str
    descricao: str


MODALIDADES: dict[str, Modalidade] = {
    "texto": Modalidade("texto", ("historico", "legenda"), "E2_historico_legenda", "N2", "histórico e legenda"),
    "imagem": Modalidade("imagem", ("historico", "legenda", "capa"), "E3_historico_legenda_capa", "N3",
                         "histórico, legenda e capa"),
}


def familia(coluna: str) -> str:
    """Família de um atributo do dataset; o que não é legenda, capa ou Gemini é histórico/contexto."""
    if coluna.startswith("leg_"):
        return "legenda"
    if coluna.startswith(("cor_", "rosto_", "ocr_", "capa_")):
        return "capa"
    if coluna.startswith("gem_"):
        return "gemini"
    return "historico"


def especificacao_ate(base: Mapping, blocos: Sequence[str], *,
                      excluidas: Collection[str] = ()) -> EspecificacaoAtributos:
    """Atributos das famílias pedidas, na ordem da base; `excluidas` ausentes da base são ignoradas."""
    desconhecidos = [b for b in blocos if b not in FAMILIAS]
    if desconhecidos:
        raise ErroFamilia(f"famílias desconhecidas: {desconhecidos}")

    def mantem(coluna: str) -> bool:
        return familia(coluna) in blocos and coluna not in excluidas

    return EspecificacaoAtributos(
        [c for c in base["escalares"] if mantem(c)], [c for c in base["categoricas"] if mantem(c)],
        {c: k for c, k in base["embeddings"].items() if mantem(c)})


def _modalidade(nome: str) -> Modalidade:
    if nome not in MODALIDADES:
        raise ErroFamilia(f"não há modelo reduzido para {nome!r}; modalidades: {sorted(MODALIDADES)}")
    return MODALIDADES[nome]


def especificacao_da_modalidade(base: Mapping, nome: str) -> EspecificacaoAtributos:
    """Especificação operacional (E2 ou E3 sem `duracao_s`), usada igual no treino, na validação e na inferência."""
    modalidade = _modalidade(nome)
    return especificacao_ate(base, modalidade.blocos, excluidas=COLUNAS_NAO_PREDITORAS)


def n_colunas(espec: EspecificacaoAtributos) -> int:
    """Colunas de saída do pré-processador: escalares, categóricas e componentes dos blocos de embedding."""
    return len(espec.escalares) + len(espec.categoricas) + sum(espec.embeddings.values())


def hash_da_especificacao(espec: EspecificacaoAtributos) -> str:
    """Mesma fórmula do registro experimental congelado (escalares, categóricas e embeddings, nesta ordem)."""
    conteudo = json.dumps([list(espec.escalares), list(espec.categoricas), dict(espec.embeddings)],
                          ensure_ascii=False)
    return hashlib.sha256(conteudo.encode()).hexdigest()


def nivel_do_modelo(nome: str) -> str:
    """Rótulo exibido na saída do agente e gravado no pacote; nunca chama o modelo reduzido de completo."""
    modalidade = _modalidade(nome)
    return f"{modalidade.nivel}: {modalidade.descricao} (modelo reduzido; desenvolvimento, provisório)"


def declaracao_da_modalidade(nome: str) -> dict:
    """Texto auditável da adaptação operacional, gravado nos metadados do dataset e do pacote."""
    modalidade = _modalidade(nome)
    return {
        "versao": VERSAO, "modalidade": modalidade.nome, "nivel": modalidade.nivel,
        "origem_experimental": modalidade.origem_experimental, "blocos": list(modalidade.blocos),
        "colunas_excluidas": list(COLUNAS_NAO_PREDITORAS),
        "motivo_da_exclusao": (
            "A duração do vídeo não existe para um criativo de texto ou de imagem e normalmente não é informada pelo "
            "usuário; treino, validação e inferência usam o mesmo conjunto de preditores, sem a duração."),
        "observacao": (
            f"Adaptação operacional: a especificação experimental {modalidade.origem_experimental} permanece congelada "
            "e inalterada; esta é uma derivação versionada, sem nova busca nem novo ajuste de parâmetros."),
    }
