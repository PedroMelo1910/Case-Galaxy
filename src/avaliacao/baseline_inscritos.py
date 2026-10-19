"""Baseline 4 do TAPI (D-D1): regressão de log(1 + curtidas) sobre log(1 + inscritos atuais), só no relatório.

Lê a tabela restrita de canais (única fonte de inscritos) e a usa só aqui. Os inscritos nunca viram atributo dos modelos
nem da especificação (D-D4, D-D8, V3): este módulo não devolve coluna para nenhum dataset.

Regras da avaliação:
- o ajuste usa SOMENTE as linhas de treino da dobra (uma reta por dobra, mínimos quadrados ordinários, uma linha = um
  peso); recusa linha que não seja de treino;
- a previsão vale para qualquer canal com inscritos (inclusive canal retido, cujos inscritos são metadado público na
  inferência); canal sem inscritos fica SEM previsão (NaN), nunca zero nem média;
- a previsão é cortada em 0 (domínio de log1p das curtidas) e as linhas cortadas são contadas;
- a comparação com as outras referências usa as linhas em que TODAS existem; a cobertura e a igualdade ou não com a
  população principal são informadas. Sem ajuste possível, o bloco diz `nao_avaliado` e o motivo; nada é inventado.

Limitação declarada: os inscritos são os atuais na coleta, aplicados a posts de datas anteriores (avaliação
retrospectiva), e há no máximo um valor de inscritos por canal (16 canais).
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src import curadoria
from src.avaliacao import bootstrap as bs
from src.avaliacao import desenvolvimento as dv

TABELA_CANAIS_RESTRITA = "canais_estatisticas_restritas.parquet"
USO_TREINO = "treino"
MIN_LINHAS_AJUSTE = 20  # mesmo mínimo técnico do treino das dobras (final.MIN_LINHAS_TREINO); não foi calibrado
MIN_CANAIS_AJUSTE = 3  # com dois canais a reta passa exatamente pelas duas médias: não há ajuste a avaliar
ESTADO_OK = "ok"

DEFINICAO = ("regressão linear (mínimos quadrados) de log1p(curtidas do alvo) sobre log1p(inscritos atuais do canal), "
             "ajustada só nas linhas de treino de cada dobra; previsão em log1p, cortada em 0; ausência de inscritos "
             "preservada (sem previsão)")
LIMITACOES = (
    ("Inscritos atuais (lidos na coleta) aplicados a posts de datas anteriores: avaliação retrospectiva; o tamanho do "
    "canal na data do post não é observado, e canais que cresceram ou encolheram desde então ficam desalinhados."),
    ("Há um único valor de inscritos por canal (no máximo 16 valores distintos): a reta é ajustada sobre poucos "
    "pontos e o coeficiente é frágil; é uma referência de contexto, não um previsor validado fora destes canais."),
    ("Os inscritos nunca entram como atributo dos modelos servidos nem de qualquer especificação: só esta referência "
    "os usa, e o ajuste olha apenas o treino de cada dobra."),
    ("Canal sem inscritos fica sem previsão (não vira zero nem média); a comparação com as outras referências vale "
    "só nas linhas em que todas existem, com a cobertura informada."),
)
CONTRASTES = (("modelo", "inscritos"), ("inscritos", "media_aritmetica"), ("inscritos", "mediana"),
              ("inscritos", "media_movel_exponencial"))


@dataclass(frozen=True)
class AjusteInscritos:
    estado: str
    coeficiente: float | None = None
    intercepto: float | None = None
    n_linhas: int = 0
    n_canais: int = 0
    n_linhas_sem_inscritos: int = 0
    canais_do_ajuste: tuple[str, ...] = field(default_factory=tuple)

    def resumo(self) -> dict:
        return {"estado": self.estado, "coeficiente": self.coeficiente, "intercepto": self.intercepto,
                "n_linhas": self.n_linhas, "n_canais": self.n_canais,
                "n_linhas_sem_inscritos": self.n_linhas_sem_inscritos, "canais_do_ajuste": list(self.canais_do_ajuste)}


# --- leitura da tabela restrita -----------------------------------------------------------------------------------


def carregar_inscritos(pasta_base: Path, *, verificar_manifesto: bool = True) -> dict[str, int | None]:
    """`channel_id -> inscritos atuais` da tabela restrita; nulo continua nulo. Com manifesto conferido por padrão."""
    if verificar_manifesto:
        tabela = curadoria.ler_base(Path(pasta_base))[TABELA_CANAIS_RESTRITA]
    else:
        tabela = pq.read_table(Path(pasta_base) / TABELA_CANAIS_RESTRITA)
    return {str(linha["channel_id"]): linha["inscritos"] for linha in tabela.select(["channel_id", "inscritos"])
            .to_pylist()}


def hash_da_tabela(pasta_base: Path) -> str:
    """SHA-256 dos bytes da tabela restrita (para o protocolo congelar a entrada do baseline 4)."""
    return hashlib.sha256((Path(pasta_base) / TABELA_CANAIS_RESTRITA).read_bytes()).hexdigest()


# --- ajuste e previsão ----------------------------------------------------------------------------------------------


def _x(inscritos: int | None) -> float | None:
    return None if inscritos is None or inscritos < 0 else math.log1p(inscritos)


def ajustar(treino: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> AjusteInscritos:
    """Reta de log1p(curtidas) sobre log1p(inscritos) nas linhas de treino com inscritos conhecidos."""
    for linha in treino:
        if linha.get("uso_no_contexto") != USO_TREINO:
            raise ValueError("o baseline 4 só ajusta com linhas de treino: encontrado uso_no_contexto = "
                             f"{linha.get('uso_no_contexto')!r}")
    pares = [(_x(inscritos.get(str(l["channel_id"]))), l["y_log"], str(l["channel_id"])) for l in treino
             if l.get("y_log") is not None and np.isfinite(l["y_log"])]
    com_x = [(x, y, c) for x, y, c in pares if x is not None]
    sem_inscritos = len(pares) - len(com_x)
    canais = tuple(sorted({c for _, _, c in com_x}))
    base = {"n_linhas": len(com_x), "n_canais": len(canais), "n_linhas_sem_inscritos": sem_inscritos,
            "canais_do_ajuste": canais}
    if len(com_x) < MIN_LINHAS_AJUSTE:
        return AjusteInscritos("linhas_insuficientes", **base)
    if len(canais) < MIN_CANAIS_AJUSTE:
        return AjusteInscritos("canais_insuficientes", **base)
    x = np.array([p[0] for p in com_x])
    y = np.array([p[1] for p in com_x])
    variancia = float(((x - x.mean()) ** 2).sum())
    if variancia <= 0.0:
        return AjusteInscritos("sem_variacao_nos_inscritos", **base)
    coeficiente = float(((x - x.mean()) * (y - y.mean())).sum() / variancia)
    return AjusteInscritos(ESTADO_OK, coeficiente, float(y.mean() - coeficiente * x.mean()), **base)


def _bruta(ajuste: AjusteInscritos, linhas: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> np.ndarray:
    if ajuste.estado != ESTADO_OK:
        return np.full(len(linhas), np.nan)
    saida = []
    for linha in linhas:
        x = _x(inscritos.get(str(linha["channel_id"])))
        saida.append(np.nan if x is None else ajuste.intercepto + ajuste.coeficiente * x)
    return np.array(saida, dtype=float)


def prever(ajuste: AjusteInscritos, linhas: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> np.ndarray:
    """Previsão em log1p, cortada em 0; NaN onde o ajuste não existe ou o canal não tem inscritos."""
    bruta = _bruta(ajuste, linhas, inscritos)
    return np.where(np.isnan(bruta), np.nan, np.maximum(bruta, 0.0))


def contar_cortadas(ajuste: AjusteInscritos, linhas: Sequence[Mapping], inscritos: Mapping[str, int | None]) -> int:
    bruta = _bruta(ajuste, linhas, inscritos)
    return int(np.sum(bruta[~np.isnan(bruta)] < 0.0))


# --- bloco de relatório ---------------------------------------------------------------------------------------------


def hash_dos_ids(ids: Sequence[str]) -> str:
    """Mesma fórmula de `relatorio._bloco` para `ids_sha256`: permite conferir população idêntica."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def bloco_nao_avaliado(motivo: str) -> dict:
    return {"estado": "nao_avaliado", "motivo": motivo, "definicao": DEFINICAO, "limitacoes": list(LIMITACOES),
            "desvio": "baseline 4 do TAPI não avaliado nesta população; nenhum valor foi inventado"}


def prever_dobra(treino: Sequence[Mapping], avaliacao: Sequence[Mapping],
                 inscritos: Mapping[str, int | None]) -> tuple[AjusteInscritos, np.ndarray, int]:
    """Ajuste só no treino da dobra e previsão nas linhas de avaliação: (ajuste, previsto, n cortadas em zero)."""
    ajuste = ajustar(treino, inscritos)
    return ajuste, prever(ajuste, avaliacao, inscritos), contar_cortadas(ajuste, avaliacao, inscritos)


def _cobertura(linhas: Sequence[Mapping], previsto: np.ndarray) -> dict:
    com = np.isfinite(previsto)
    sem_canais = sorted({str(l["channel_id"]) for l, ok in zip(linhas, com, strict=True) if not ok})
    return {"n_linhas_avaliacao": len(linhas), "n_com_previsao": int(com.sum()),
            "fracao": float(com.mean()) if len(linhas) else None, "canais_sem_inscritos": sem_canais}


def _avaliar_modelo(linhas: Sequence[Mapping], central: Sequence[float], previsto: np.ndarray,
                    celulas: Sequence[str] | None, n_repeticoes: int, semente: int) -> dict:
    pop, exclusoes = dv.montar_populacao_multi(linhas, {"modelo": central, "inscritos": previsto}, celulas=celulas,
                                               com_baselines=True)
    if not pop.ids:
        return {"estado": "sem_linhas_comuns", "exclusoes": exclusoes}
    return {"estado": "avaliado", "exclusoes": exclusoes,
            "populacao": {"n_avaliadas": len(pop.ids), "n_canais": len(set(pop.canais)),
                          "ids_sha256": hash_dos_ids(pop.ids)},
            "avaliacao": dv.avaliar(pop, n_repeticoes=n_repeticoes, semente=semente, contrastes=CONTRASTES)}


def montar_bloco(ajustes: Mapping[str, AjusteInscritos], cortadas: Mapping[str, int], linhas: Sequence[Mapping],
                 previsto: np.ndarray, centrais: Mapping[str, Sequence[float]], *,
                 celulas: Sequence[str] | None = None, ids_do_principal: Mapping[str, str] | None = None,
                 n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE) -> dict:
    """Bloco do baseline 4: modelo × inscritos × as três referências, nas linhas em que todas existem.

    `ajustes` e `cortadas` são por dobra; `linhas`, `previsto` e `centrais` já estão empilhados na ordem das dobras.
    `ids_do_principal` (modelo -> `ids_sha256` da população principal) permite afirmar se as linhas são as mesmas.
    """
    cobertura = _cobertura(linhas, previsto)
    if cobertura["n_com_previsao"] == 0:
        estados = sorted({a.estado for a in ajustes.values()})
        return bloco_nao_avaliado(f"nenhuma linha com previsão (estados do ajuste: {estados}; canais sem inscritos: "
                                  f"{cobertura['canais_sem_inscritos']})")
    por_modelo = {nome: _avaliar_modelo(linhas, central, previsto, celulas, n_repeticoes, semente)
                  for nome, central in centrais.items()}
    iguais = bool(ids_do_principal) and all(
        por_modelo[nome].get("populacao", {}).get("ids_sha256") == ids_do_principal.get(nome) for nome in por_modelo)
    return {"estado": "avaliado", "definicao": DEFINICAO, "limitacoes": list(LIMITACOES),
            "ajustes_por_dobra": {rotulo: a.resumo() | {"n_cortadas_em_zero_na_avaliacao": cortadas.get(rotulo, 0)}
                                  for rotulo, a in ajustes.items()},
            "cobertura": cobertura, "mesmas_linhas_que_o_principal": iguais, "por_modelo": por_modelo}
