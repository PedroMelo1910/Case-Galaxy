"""Pacote de modelos servido pelo agente: q10, q50 e q90 verificados, cobertura observada e metadados.

O vídeo usa o modelo completo. Imagem e texto usam os próprios pacotes reduzidos (ver `modelo_modalidades`), cada um
com a sua cobertura observada; o roteamento por modalidade fica em `Servicos.pacote_para`. Sem pacote configurado
para a modalidade, o erro é `MODELO_INDISPONIVEL`: os atributos do modelo completo nunca são preenchidos com valores
inventados.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src import faixa_previsao as fp
from src import modelo_desenvolvimento as md
from src.agente.contratos import Codigo, ErroAgente
from src.avaliacao import driver_lim as dl

MODALIDADE_COM_MODELO = "video"
NIVEL_VIDEO = "completo (desenvolvimento, provisório)"
LIMITACAO_MODALIDADE = (
    "Não há pacote de modelo configurado para {modalidade} neste serviço. O vídeo usa o modelo completo servido; "
    "imagem e texto usam pacotes próprios, carregados da pasta de modelos quando existem. Confira se o pacote de "
    "{modalidade} foi treinado e está na pasta esperada."
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
    registro: Mapping | None = None  # `pacote.json` do pacote final; None nos pacotes de desenvolvimento (rollback)
    limiar: dl.LimiarDeLargura | None = None  # só no vídeo final, vinculado à identidade dos três modelos
    identidade_sha256: str | None = None
    detalhe_do_limiar: Mapping | None = None  # origem e leitura do diagnóstico de desenvolvimento (do manifesto)

    @property
    def eh_final(self) -> bool:
        return self.registro is not None

    @property
    def versao(self) -> str:
        if self.registro is not None:
            return self.registro["versao_servida"]
        return f"{self.pasta_q50.parent.name}/{self.pasta_q50.name}"

    @property
    def modalidade(self) -> str:
        """`video` para o pacote servido (sem modalidade gravada); texto ou imagem nos pacotes reduzidos."""
        return self.metadados_q50.get("modalidade", MODALIDADE_COM_MODELO)

    @property
    def nivel(self) -> str:
        if self.registro is not None:
            return self.registro["nivel_do_modelo"]
        return NIVEL_VIDEO if self.modalidade == MODALIDADE_COM_MODELO else self.metadados_q50["nivel_do_modelo"]

    @property
    def identificacao(self) -> dict:
        extra = ({"identidade_sha256": self.identidade_sha256, "versao_servida": self.registro["versao_servida"],
                  "refit_final": True, "limiar_vinculado": self.limiar is not None}
                 if self.registro is not None else {})
        return extra | {
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


def carregar_pacote_final(pasta: Path, *, entrada_do_manifesto: Mapping | None = None) -> Pacote:
    """Pacote do refit final (`final=v1/<id>`): modelos conferidos por hash, identidade dos três e limiar vinculado.

    O limiar só é aceito se o arquivo for válido (hash do conteúdo), de papel `servido` e vinculado à identidade DESTE
    pacote (`LimiarIncompativel` caso contrário). A declaração de cobertura é a do procedimento de avaliação gravada no
    `pacote.json`, nunca uma medida do artefato."""
    from src import refit_cobertura as rc

    pasta = Path(pasta)
    registro = _json(pasta / "pacote.json")
    modelos = {n: md.carregar(pasta / n) for n in ("q10", "q50", "q90")}
    identidade = dl.identidade_do_pacote({n: registro["modelos"][n]["sha256_modelo"] for n in modelos})
    if identidade != registro["identidade_sha256"]:
        raise md.ErroModelo("a identidade do pacote não confere com os hashes dos três modelos")
    espec = modelos["q50"].especificacao
    for nome in ("q10", "q90"):
        outra = modelos[nome].especificacao
        if (list(outra.escalares) != list(espec.escalares) or list(outra.categoricas) != list(espec.categoricas)
                or dict(outra.embeddings) != dict(espec.embeddings)):
            raise md.ErroModelo(f"{nome} usa lista de atributos diferente da do q50")
    declaracao = registro["cobertura_do_procedimento"]
    rc.conferir(declaracao)
    limiar = _limiar_vinculado(pasta, identidade, (entrada_do_manifesto or {}).get("limiar"))
    detalhe = ((entrada_do_manifesto or {}).get("limiar") or {}).get("diagnostico_de_desenvolvimento")
    meta_q50 = _json(pasta / "q50" / md.ARQUIVO_METADADOS)
    return Pacote(modelos["q10"], modelos["q50"], modelos["q90"], pasta / "q50", pasta, declaracao, meta_q50,
                  {"recarga_identica": True, "cobertura_do_procedimento": declaracao}, registro=registro,
                  limiar=limiar, identidade_sha256=identidade, detalhe_do_limiar=detalhe)


def _limiar_vinculado(pasta: Path, identidade: str, registrado: Mapping | None) -> dl.LimiarDeLargura | None:
    if registrado is None:
        return None
    limiar = dl.ler_limiar(pasta / registrado["arquivo"])
    if limiar.papel != "servido":
        raise dl.LimiarIncompativel(f"o limiar do pacote servido deve ter papel 'servido', não {limiar.papel!r}")
    dl._exigir_compativel(limiar, identidade)
    return limiar
