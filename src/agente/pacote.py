"""Pacote de modelos servido pelo agente: q10, q50 e q90 verificados, cobertura observada e metadados.

Só o modelo completo (vídeo) tem artefatos verificados. Imagem e texto não têm modelo próprio neste pacote:
`modelo_para` devolve `MODELO_INDISPONIVEL` com a limitação temporária; os atributos do modelo completo nunca são
preenchidos com valores inventados.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from src import faixa_previsao as fp
from src import modelo_desenvolvimento as md
from src.agente.contratos import Codigo, ErroAgente

MODALIDADE_COM_MODELO = "video"
LIMITACAO_MODALIDADE = (
    "Limitação temporária deste pacote: ainda não há modelo treinado e verificado para {modalidade}. A previsão só "
    "existe para vídeo (modelo completo, em desenvolvimento). O modelo de {modalidade} permanece pendente para a "
    "entrega final."
)


@dataclass(frozen=True)
class Pacote:
    q10: md.ModeloTreinado
    q50: md.ModeloTreinado
    q90: md.ModeloTreinado
    pasta_q50: Path
    pasta_faixa: Path
    declaracao_faixa: dict
    metadados_q50: dict
    resultado_faixa: dict

    @property
    def versao(self) -> str:
        return f"{self.pasta_q50.parent.name}/{self.pasta_q50.name}"

    @property
    def identificacao(self) -> dict:
        return {
            "q50_artefato": self.pasta_q50.name, "faixa_artefato": self.pasta_faixa.name,
            "sha256_q50": self.metadados_q50["sha256_modelo"], "parametros": self.metadados_q50["parametros"],
            "n_treino": self.metadados_q50["populacao"]["n_treino"],
            "n_validacao": self.metadados_q50["populacao"]["n_validacao"],
            "dataset_sha256": self.metadados_q50["dataset"]["sha256"],
        }

    def modelo_para(self, modalidade: str) -> "Pacote":
        if modalidade != MODALIDADE_COM_MODELO:
            raise ErroAgente(Codigo.MODELO_INDISPONIVEL, LIMITACAO_MODALIDADE.format(modalidade=modalidade),
                             {"modalidade": modalidade, "modalidade_com_modelo": MODALIDADE_COM_MODELO})
        return self


def _json(arquivo: Path) -> dict:
    return json.loads(arquivo.read_text(encoding="utf-8"))


def carregar_pacote(pasta_q50: Path, pasta_faixa: Path) -> Pacote:
    """Carrega os três modelos (hash e versões conferidos por `md.carregar`) e confere o vínculo entre eles."""
    pasta_q50, pasta_faixa = Path(pasta_q50), Path(pasta_faixa)
    meta_q50 = _json(pasta_q50 / md.ARQUIVO_METADADOS)
    resultado = _json(pasta_faixa / "resultado.json")
    if not resultado.get("recarga_identica"):
        raise md.ErroModelo("a faixa não tem recarga idêntica verificada")
    modelos = {"q50": md.carregar(pasta_q50)}
    for nome in ("q10", "q90"):
        meta = _json(pasta_faixa / nome / md.ARQUIVO_METADADOS)
        if meta.get("q50_sha256_modelo") != meta_q50["sha256_modelo"]:
            raise md.ErroModelo(f"{nome} não foi treinado a partir do q50 informado")
        if meta.get("dataset_sha256") != meta_q50["dataset"]["sha256"]:
            raise md.ErroModelo(f"{nome} usa outro dataset que o q50")
        modelos[nome] = md.carregar(pasta_faixa / nome)
    espec = modelos["q50"].especificacao
    for nome in ("q10", "q90"):
        outra = modelos[nome].especificacao
        if (list(outra.escalares) != list(espec.escalares) or list(outra.categoricas) != list(espec.categoricas)
                or dict(outra.embeddings) != dict(espec.embeddings)):
            raise md.ErroModelo(f"{nome} usa lista de atributos diferente da do q50")
    faixa = resultado["faixa"]
    n = int(faixa["n"])
    n_dentro = round(float(faixa["cobertura"]) * n)
    return Pacote(modelos["q10"], modelos["q50"], modelos["q90"], pasta_q50, pasta_faixa,
                  fp.declaracao_da_faixa(n_dentro=n_dentro, n=n), meta_q50, resultado)
