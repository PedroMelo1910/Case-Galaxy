"""Modelo LightGBM de desenvolvimento: pré-processamento ajustado só no treino, previsão em log e artefato.

O alvo ajustado é `y_rel = log1p(curtidas) − ref_log`. A previsão absoluta em log é `ref_log + previsão`, cortada em 0
(curtidas não negativas) com a contagem dos cortes registrada. Ausências permanecem `NaN` (o LightGBM as trata
nativamente): não há imputação por zero ou média. PCA dos embeddings e códigos das categorias são aprendidos só com as
linhas de treino que chegam a `fit`, de modo que cada corte de uma busca refaz tudo com o seu próprio treino.
"""

import hashlib
import io
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.metricas import resumo_erros
from src.preprocessador import Preprocessador

VERSAO = "1"
SEMENTE = 20261015
ARQUIVO_MODELO, ARQUIVO_METADADOS = "modelo.joblib", "metadados.json"
USO_TREINO, USO_AVALIACAO = "treino", "avaliacao"
PARAMETROS_INICIAIS = {
    "objective": "quantile", "alpha": 0.5, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 7,
    "min_child_samples": 20, "reg_lambda": 1.0, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": SEMENTE, "n_jobs": 4, "deterministic": True, "force_row_wise": True, "verbose": -1,
}


CHAVES_DA_BUSCA = frozenset({"num_leaves", "min_child_samples", "reg_lambda"})  # grade aprovada: nada mais é sobrescrito


class ErroModelo(ValueError):
    """Entrada, população ou artefato incompatível com o contrato do modelo."""


@dataclass(frozen=True)
class EspecificacaoAtributos:
    escalares: Sequence[str]
    categoricas: Sequence[str]
    embeddings: Mapping[str, int]


def reconstruir_log(linhas: Sequence[Mapping], previsao_relativa: np.ndarray) -> tuple[np.ndarray, int]:
    """`ref_log + previsão relativa`, cortada em 0; devolve também quantas previsões foram cortadas."""
    bruto = np.array([l["ref_log"] for l in linhas], dtype=np.float64) + np.asarray(previsao_relativa, dtype=np.float64)
    return np.maximum(bruto, 0.0), int((bruto < 0).sum())


class ModeloTreinado:
    def __init__(self, pipeline: Pipeline, especificacao: EspecificacaoAtributos, parametros: Mapping,
                 categoricas_nativas: bool = False) -> None:
        self.pipeline, self.especificacao, self.parametros = pipeline, especificacao, dict(parametros)
        self.categoricas_nativas = categoricas_nativas  # False: categorias entram como códigos ordinais (padrão)

    @property
    def nomes_atributos(self) -> list[str]:
        return list(self.pipeline.named_steps["pre"].nomes_saida_)

    def prever_relativo(self, linhas: Sequence[Mapping], num_arvores: int | None = None) -> np.ndarray:
        """Saída relativa; `num_arvores` lê só as primeiras árvores (None = todas, o comportamento de sempre).

        Fora de 1..árvores treinadas é recusado: o LightGBM leria todas as árvores em silêncio (0 ou negativo) ou
        repetiria o modelo inteiro (acima do treinado), e o chamador acreditaria ter lido um prefixo."""
        if num_arvores is None:
            return self.pipeline.predict(list(linhas))
        treinadas = self.pipeline.named_steps["modelo"].booster_.current_iteration()
        if not 1 <= num_arvores <= treinadas:
            raise ErroModelo(f"num_arvores deve estar entre 1 e {treinadas} (árvores treinadas); recebido {num_arvores}")
        return self.pipeline.predict(list(linhas), num_iteration=num_arvores)

    def prever_log(self, linhas: Sequence[Mapping], num_arvores: int | None = None) -> tuple[np.ndarray, int]:
        return reconstruir_log(linhas, self.prever_relativo(linhas, num_arvores))


def construir_pipeline(especificacao: EspecificacaoAtributos, parametros: Mapping) -> Pipeline:
    from lightgbm import LGBMRegressor

    pre = Preprocessador(list(especificacao.escalares), list(especificacao.categoricas),
                         dict(especificacao.embeddings), semente=SEMENTE)
    return Pipeline([("pre", pre), ("modelo", LGBMRegressor(**parametros))])


def exigir_uso(linhas: Sequence[Mapping], uso: str) -> None:
    fora = sorted({str(l.get("uso_no_contexto")) for l in linhas} - {uso})
    if fora or not linhas:
        raise ErroModelo(f"esperado só uso_no_contexto = {uso!r}; recebido {fora or 'nenhuma linha'}")
    if any(l.get("y_rel") is None or l.get("ref_log") is None for l in linhas):
        raise ErroModelo("linha sem alvo ou referência; ela deve ser excluída antes (e contada)")


def treinar(linhas_treino: Sequence[Mapping], especificacao: EspecificacaoAtributos, *,
            parametros: Mapping = PARAMETROS_INICIAIS, categoricas_nativas: bool = False) -> ModeloTreinado:
    """`categoricas_nativas=True` declara ao LightGBM as colunas categóricas (as posições logo após os escalares na
    matriz do pré-processador), para dividir por conjuntos de categorias em vez de por códigos ordinais."""
    exigir_uso(linhas_treino, USO_TREINO)
    pipeline = construir_pipeline(especificacao, parametros)
    ajuste: dict = {}
    if categoricas_nativas and especificacao.categoricas:
        inicio = len(especificacao.escalares)
        ajuste["modelo__categorical_feature"] = list(range(inicio, inicio + len(especificacao.categoricas)))
    pipeline.fit(list(linhas_treino), np.array([l["y_rel"] for l in linhas_treino], dtype=np.float64), **ajuste)
    return ModeloTreinado(pipeline, especificacao, parametros, categoricas_nativas=bool(ajuste))  # só se declarou


def _curtidas(linhas: Sequence[Mapping]) -> list[int]:
    return [round(np.expm1(l["y_log"])) for l in linhas]


def avaliar(modelo: ModeloTreinado, linhas_validacao: Sequence[Mapping]) -> dict:
    """Modelo × média histórica aritmética, nas mesmas linhas (as sem baseline saem dos dois lados)."""
    exigir_uso(linhas_validacao, USO_AVALIACAO)
    previsto, cortadas = modelo.prever_log(linhas_validacao)
    usadas = [i for i, l in enumerate(linhas_validacao) if l.get("pred_media_aritmetica") is not None]
    comparadas = [linhas_validacao[i] for i in usadas]
    curtidas = _curtidas(comparadas)
    relatorio = {
        "n_validacao": len(linhas_validacao), "n_comparado": len(usadas), "previsoes_cortadas_em_zero": cortadas,
        "modelo": resumo_erros(curtidas, [float(previsto[i]) for i in usadas]),
        "baseline_media_aritmetica": resumo_erros(curtidas, [float(l["pred_media_aritmetica"]) for l in comparadas]),
    }
    pre = modelo.pipeline.named_steps["pre"]
    desconhecidas = {c: sum(l.get(c) is not None and str(l[c]) not in pre.categorias_[c] for l in linhas_validacao)
                     for c in modelo.especificacao.categoricas}
    por_setor = {}
    for setor in sorted({str(l["setor"]) for l in comparadas}):
        posicoes = [k for k, l in enumerate(comparadas) if str(l["setor"]) == setor]
        reais = [curtidas[k] for k in posicoes]
        por_setor[setor] = {
            "n": len(posicoes),
            "mae_log_modelo": resumo_erros(reais, [float(previsto[usadas[k]]) for k in posicoes])["mae_log"],
            "mae_log_baseline": resumo_erros(reais, [float(comparadas[k]["pred_media_aritmetica"])
                                                     for k in posicoes])["mae_log"],
        }
    return relatorio | {"por_setor": por_setor,
                        "categorias_nao_vistas_no_treino": {c: n for c, n in desconhecidas.items() if n}}


def _sha256(arquivo: Path) -> str:
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()


def salvar(modelo: ModeloTreinado, pasta: Path, *, metadados: Mapping) -> Path:
    """Grava uma vez (nunca sobrescreve) o modelo e seus metadados com versões e hash do arquivo."""
    pasta = Path(pasta)
    if pasta.exists():
        raise FileExistsError(f"artefato já existe e não será sobrescrito: {pasta}")
    final, pasta = pasta, Path(tempfile.mkdtemp(prefix=".artefato.", suffix=".parcial", dir=_garantir(pasta.parent)))
    try:
        _gravar_artefato(modelo, pasta, metadados)
        pasta.rename(final)
    except BaseException:
        shutil.rmtree(pasta, ignore_errors=True)
        raise
    return final


def _gravar_artefato(modelo: ModeloTreinado, pasta: Path, metadados: Mapping) -> None:
    import lightgbm
    import sklearn

    joblib.dump(modelo.pipeline, pasta / ARQUIVO_MODELO)
    conteudo = {
        "versao": VERSAO, "sha256_modelo": _sha256(pasta / ARQUIVO_MODELO), "parametros": modelo.parametros,
        "categoricas_nativas": modelo.categoricas_nativas,
        "especificacao": {k: list(v) if not isinstance(v, Mapping) else dict(v)
                          for k, v in asdict(modelo.especificacao).items()},
        "atributos_de_saida": modelo.nomes_atributos,
        "dependencias": {"python": sys.version.split()[0], "lightgbm": lightgbm.__version__,
                         "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                         "plataforma": platform.platform()},
    }
    colisoes = sorted(set(conteudo) & set(metadados))
    if colisoes:
        raise ErroModelo(f"os metadados não podem sobrescrever campos do próprio artefato: {colisoes}")
    (pasta / ARQUIVO_METADADOS).write_text(
        json.dumps({**conteudo, **dict(metadados)}, ensure_ascii=False, indent=1), encoding="utf-8")


def _garantir(pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def carregar(pasta: Path) -> ModeloTreinado:
    pasta = Path(pasta)
    meta = json.loads((pasta / ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    bytes_modelo = (pasta / ARQUIVO_MODELO).read_bytes()  # lido uma vez: o hash vale para o que é carregado
    if meta.get("sha256_modelo") != hashlib.sha256(bytes_modelo).hexdigest():
        raise ErroModelo("o arquivo do modelo não confere com o hash registrado (só carregue artefatos de pasta "
                         "confiável: o hash fica ao lado do arquivo e protege contra corrupção, não adulteração)")
    import lightgbm
    import sklearn

    atuais = {"lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__}
    registradas = meta.get("dependencias", {})
    divergentes = {k: (registradas.get(k), v) for k, v in atuais.items() if registradas.get(k) != v}
    if divergentes:
        raise ErroModelo(f"versões das dependências diferem das do treino: {divergentes}")
    espec = meta["especificacao"]
    return ModeloTreinado(joblib.load(io.BytesIO(bytes_modelo)),
                          EspecificacaoAtributos(espec["escalares"], espec["categoricas"], espec["embeddings"]),
                          meta["parametros"], categoricas_nativas=bool(meta.get("categoricas_nativas", False)))


# --- execução completa: treino, avaliação, artefato e verificação de recarga --------------------------------------

LIMITACOES = [
    "Provisório: configuração do LightGBM fixa; se há `busca`, ela veio de uma busca única feita em outra população "
    "(ver `busca.n_treino_da_busca`) e não foi repetida para esta.",
    "A comparação com a média histórica é preliminar, em desenvolvimento, sem intervalo de confiança e sem o teste final.",
    "A população é só a anotada pelo Gemini e permitida ao desenvolvimento; não é amostra aleatória nova.",
    "Alvo observado nas curtidas acumuladas na coleta, não um valor exato aos 30 dias; avaliação retrospectiva.",
]


def _parametros_da_busca(arquivo: Path, base: Mapping) -> tuple[dict, dict]:
    """Aplica sobre `base` só os parâmetros escolhidos pela busca; devolve também o registro de ligação."""
    bruto = Path(arquivo).read_bytes()
    conteudo = json.loads(bruto)
    escolhidos = conteudo["busca"]["escolhida"]["parametros"]
    fora_da_grade = sorted(set(escolhidos) - CHAVES_DA_BUSCA)
    if fora_da_grade:
        raise ErroModelo(f"a busca só pode alterar {sorted(CHAVES_DA_BUSCA)}; recebido também {fora_da_grade}")
    ligacao = {"arquivo": Path(arquivo).name, "sha256": hashlib.sha256(bruto).hexdigest(),
               "parametros_escolhidos": escolhidos, "n_treino_da_busca": conteudo.get("n_treino")}
    return dict(base) | escolhidos, ligacao


def executar(pasta_dataset: Path, saida_raiz: Path, *, parametros: Mapping = PARAMETROS_INICIAIS,
             busca: Path | None = None, anterior: Path | None = None) -> dict:
    """Treina no uso `treino`, avalia em `avaliacao`, grava o artefato e confere previsões idênticas após recarregar.

    `busca`: resultado do GridSearchCV cujos parâmetros escolhidos substituem os correspondentes (os demais ficam).
    `anterior`: artefato já treinado, avaliado nas mesmas linhas de validação para comparação.
    """
    from src import dataset_desenvolvimento as ds

    ligacao_busca = None
    if busca is not None:
        parametros, ligacao_busca = _parametros_da_busca(busca, parametros)
    linhas, manifesto = ds.carregar_dataset(pasta_dataset)
    chave_parametros = hashlib.sha256(json.dumps(dict(parametros), sort_keys=True).encode()).hexdigest()[:8]
    pasta_artefato = (Path(saida_raiz) / f"provisorio=v{VERSAO}"
                      / f"{manifesto['sha256_dataset'][:12]}-{chave_parametros}")
    if pasta_artefato.exists():
        arquivo = pasta_artefato / "resultado.json"
        anterior = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else None
        if not anterior or not anterior.get("recarga_identica"):
            raise ErroModelo(f"artefato incompleto ou sem recarga verificada em {pasta_artefato}; remova-o antes de "
                             "repetir")
        return anterior | {"reaproveitado": True}
    treino = [l for l in linhas if l["uso_no_contexto"] == USO_TREINO]
    validacao = [l for l in linhas if l["uso_no_contexto"] == USO_AVALIACAO]
    espec = EspecificacaoAtributos(**manifesto["especificacao"])
    modelo = treinar(treino, espec, parametros=parametros)
    relatorio = avaliar(modelo, validacao)
    ids = {"treino": [l["video_id"] for l in treino], "validacao": [l["video_id"] for l in validacao]}
    pasta = salvar(modelo, pasta_artefato, metadados={
        "tipo": "modelo_provisorio_q50", "status": "provisorio",
        "dataset": {"sha256": manifesto["sha256_dataset"], "versao": manifesto.get("versao"),
                                                     "pasta": str(Path(pasta_dataset)),
                                                     "entradas": manifesto["entradas"],
                                                     "reconciliacao": manifesto["reconciliacao_anotacoes"]},
        "populacao": {"n_treino": len(treino), "n_validacao": len(validacao),
                      "sha256_ids_treino": hashlib.sha256("\n".join(ids["treino"]).encode()).hexdigest(),
                      "sha256_ids_validacao": hashlib.sha256("\n".join(ids["validacao"]).encode()).hexdigest(),
                      **manifesto["populacao"]},
        "busca": ligacao_busca, "semente": SEMENTE, "alvo": "y_rel = log1p(curtidas) − ref_log; previsão absoluta = ref_log + previsão, "
        "cortada em 0", "retencao": manifesto["retencao"], "limitacoes": LIMITACOES,
    })
    (pasta / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    recarregado = carregar(pasta)
    antes, depois = modelo.prever_log(validacao)[0], recarregado.prever_log(validacao)[0]
    identico = bool(np.array_equal(antes, depois))
    resultado = {"artefato": str(pasta), "recarga_identica": identico, "n_previsoes_conferidas": len(validacao),
                 "avaliacao": relatorio, "n_treino": len(treino), "atributos_de_saida": len(modelo.nomes_atributos),
                 "parametros": dict(parametros), "busca": ligacao_busca}
    if anterior is not None:
        antigo = avaliar(carregar(anterior), validacao)
        if antigo["n_comparado"] != relatorio["n_comparado"]:
            raise ErroModelo("modelo anterior e novo não foram avaliados nas mesmas linhas")
        resultado["comparacao_mesmas_linhas"] = {
            "n": relatorio["n_comparado"], "anterior_artefato": str(anterior), "novo": relatorio["modelo"],
            "anterior": antigo["modelo"], "baseline_media_aritmetica": relatorio["baseline_media_aritmetica"],
            "por_setor_novo": relatorio["por_setor"], "por_setor_anterior": antigo["por_setor"]}
    if not identico:
        raise ErroModelo("o modelo recarregado não reproduz as previsões")
    (pasta / "resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from src.config import RAIZ_PROJETO

    parser = argparse.ArgumentParser(prog="python -m src.modelo_desenvolvimento", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="pasta do dataset de desenvolvimento")
    parser.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "data" / "models")
    parser.add_argument("--busca", type=Path, help="resultado do GridSearchCV com os parâmetros escolhidos")
    parser.add_argument("--anterior", type=Path, help="artefato anterior, avaliado nas mesmas linhas")
    args = parser.parse_args(argv)
    try:
        resultado = executar(args.dataset, args.saida, busca=args.busca, anterior=args.anterior)
    except (ValueError, KeyError, OSError) as erro:
        print(f"Treino recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
