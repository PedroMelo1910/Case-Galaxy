"""Módulo consolidado. Reúne:
- Modelo LightGBM de desenvolvimento: pré-processamento ajustado só no treino, previsão em log e artefato.
- Faixa de previsão q10–q90 em torno do q50 e explicação do q50 (provisório, desenvolvimento).
- Famílias de atributos e especificações operacionais dos modelos de texto e de imagem.
- Executor reutilizável do GridSearchCV: cortes cronológicos explícitos e intervalo em dias de calendário."""

import argparse
import hashlib
import io
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.avaliacao.metricas import resumo_erros
from src.modelos.preprocessamento import Preprocessador
from src.proveniencia import sha256_arquivo as _sha, sha256_arquivo as _sha256


# ──── Modelo LightGBM de desenvolvimento: pré-processamento ajustado só no treino, previsão em log e artef
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
    from src.dados import dataset as ds

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

    parser = argparse.ArgumentParser(prog="python -m src.modelos.treino modelo_desenvolvimento", description=__doc__)
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


# ──── Faixa de previsão q10–q90 em torno do q50 e explicação do q50 (provisório, desenvolvimento).
ALFA_INFERIOR, ALFA_CENTRAL, ALFA_SUPERIOR = 0.1, 0.5, 0.9
COBERTURA_NOMINAL = 0.8
TRANSFORMACOES = [
    "SHAP do q50 na escala do modelo: alvo y_rel = log1p(curtidas) − ref_log",
    "previsão em log = ref_log + saída do modelo",
    "corte em zero: previsão em log menor que 0 vira 0",
    "curtidas exibidas = arredondamento de expm1(previsão em log, depois do corte)",
    "a decomposição aditiva vale para a saída bruta; com corte em zero ela não descreve o valor exibido",
]


def corrigir_faixa(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray):
    """(inferior, central, superior, cruzou) em log. Central = q50; limites = mínimo e máximo das três saídas."""
    q10, q50, q90 = (np.asarray(x, dtype=np.float64) for x in (q10, q50, q90))
    inferior, superior = np.minimum(np.minimum(q10, q50), q90), np.maximum(np.maximum(q10, q50), q90)
    return inferior, q50, superior, (q10 > q50) | (q50 > q90)


def prever_faixa(linhas: Sequence[Mapping], rel_q10, rel_q50, rel_q90) -> dict:
    """Saídas relativas dos três modelos → valores exibidos (log e curtidas), corrigidos e comparados ao canal."""
    ref = np.array([l["ref_log"] for l in linhas], dtype=np.float64)
    brutos = [ref + np.asarray(r, dtype=np.float64) for r in (rel_q10, rel_q50, rel_q90)]
    q10, q50, q90 = (np.maximum(b, 0.0) for b in brutos)
    inferior, central, superior, cruzou = corrigir_faixa(q10, q50, q90)
    return {
        "q10_log": q10, "q50_log": q50, "q90_log": q90, "inferior_log": inferior, "central_log": central,
        "superior_log": superior, "cruzou": cruzou, "cortada_em_zero": brutos[1] < 0,
        "variacao_log_vs_canal": central - ref,
        "curtidas_central": np.array([round(np.expm1(v)) for v in central]),
        "curtidas_inferior": np.array([round(np.expm1(v)) for v in inferior]),
        "curtidas_superior": np.array([round(np.expm1(v)) for v in superior]),
    }


def _resumo(y, prev, mascara) -> dict:
    dentro = (y >= prev["inferior_log"]) & (y <= prev["superior_log"])
    largura = prev["superior_log"] - prev["inferior_log"]
    return {
        "n": int(mascara.sum()), "cobertura": float(dentro[mascara].mean()),
        "largura_log_media": float(largura[mascara].mean()), "largura_log_mediana": float(np.median(largura[mascara])),
        "largura_curtidas_mediana": float(np.median((prev["curtidas_superior"] - prev["curtidas_inferior"])[mascara])),
        "frequencia_de_cruzamento": float(prev["cruzou"][mascara].mean()),
        "abaixo_do_q10": float((y < prev["q10_log"])[mascara].mean()),
        "acima_do_q90": float((y > prev["q90_log"])[mascara].mean()),
    }


def avaliar_faixa(linhas: Sequence[Mapping], prev: Mapping) -> dict:
    """Cobertura e largura da faixa já corrigida e com corte em zero, no total e por setor."""
    y = np.array([l["y_log"] for l in linhas], dtype=np.float64)
    todas = np.ones(len(y), dtype=bool)
    setores = np.array([str(l["setor"]) for l in linhas])
    return _resumo(y, prev, todas) | {
        "cobertura_nominal": COBERTURA_NOMINAL,
        "por_setor": {s: _resumo(y, prev, setores == s) for s in sorted(set(setores))},
        "regra": "faixa = (mínimo, máximo) das três saídas com q50 mantido; valores depois do corte em zero",
    }


def contribuicoes_q50(modelo: ModeloTreinado, linhas: Sequence[Mapping]) -> dict:
    """TreeSHAP do próprio LightGBM (`pred_contrib`) para o q50, na escala do modelo, com a regra de transformação."""
    pre, regressor = modelo.pipeline.named_steps["pre"], modelo.pipeline.named_steps["modelo"]
    contrib = regressor.booster_.predict(pre.transform(list(linhas)), pred_contrib=True)
    soma = contrib.sum(axis=1)
    if not np.allclose(soma, modelo.prever_relativo(linhas), atol=1e-8):
        raise ErroModelo("as contribuições não somam a previsão do modelo")
    ref = np.array([l["ref_log"] for l in linhas], dtype=np.float64)
    cortada = (ref + soma) < 0
    return {
        "contribuicoes": contrib[:, :-1], "base": float(contrib[0, -1]), "nomes": list(pre.nomes_saida_),
        "saida_bruta_relativa": soma, "cortada_em_zero": cortada,
        "decomposicao_vale_para_o_exibido": ~cortada, "transformacoes": TRANSFORMACOES,
    }


def declaracao_da_faixa(*, n_dentro: int, n: int) -> dict:
    """Texto e números que acompanham a faixa: nível pretendido, cobertura observada e aviso.

    A faixa mantém o nome `q10–q90`; a cobertura observada não a renomeia nem é apresentada como o nível atingido.
    """
    observada = n_dentro / n
    abaixo = observada < COBERTURA_NOMINAL
    aviso = (f"Faixa de previsão q10–q90, com nível pretendido de {COBERTURA_NOMINAL:.0%}. Na validação de "
             f"desenvolvimento, {n_dentro} de {n} vídeos ({observada:.1%}) ficaram dentro dela: a cobertura ficou "
             "abaixo do pretendido. É um resultado de desenvolvimento, sem garantia para novos vídeos.") if abaixo else (
             f"Faixa de previsão q10–q90, com nível pretendido de {COBERTURA_NOMINAL:.0%}. Na validação de "
             f"desenvolvimento, {n_dentro} de {n} vídeos ({observada:.1%}) ficaram dentro dela. É um resultado de "
             "desenvolvimento, sem garantia para novos vídeos.")
    return {"nome": "faixa de previsão q10–q90", "nivel_nominal": COBERTURA_NOMINAL, "cobertura_observada": observada,
            "n_dentro": n_dentro, "n": n, "abaixo_do_pretendido": abaixo, "aviso": aviso}


def treinar_companheiros(treino: Sequence[Mapping], q50: ModeloTreinado) -> dict[str, ModeloTreinado]:
    """q10 e q90 com a população, os atributos, a representação das categorias e os parâmetros do q50; só o `alpha`
    muda."""
    return {nome: treinar(treino, q50.especificacao, parametros=q50.parametros | {"alpha": alfa},
                             categoricas_nativas=q50.categoricas_nativas)
            for nome, alfa in (("q10", ALFA_INFERIOR), ("q90", ALFA_SUPERIOR))}


def executar_faixa_previsao(pasta_dataset: Path, artefato_q50: Path, saida_raiz: Path) -> dict:
    """Treina q10 e q90 com a configuração do q50, grava os três vínculos e mede a faixa na validação."""
    from src.dados import dataset as ds

    linhas, manifesto = ds.carregar_dataset(pasta_dataset)
    treino = [l for l in linhas if l["uso_no_contexto"] == USO_TREINO]
    validacao = [l for l in linhas if l["uso_no_contexto"] == USO_AVALIACAO]
    q50 = carregar(artefato_q50)
    destino = Path(saida_raiz) / "faixa=v1" / f"{manifesto['sha256_dataset'][:12]}-{Path(artefato_q50).name}"
    if destino.exists():
        raise FileExistsError(f"já existe e não será sobrescrito: {destino}")
    modelos = {"q50": q50} | treinar_companheiros(treino, q50)
    for nome, alfa in (("q10", ALFA_INFERIOR), ("q90", ALFA_SUPERIOR)):
        salvar(modelos[nome], destino / nome, metadados={
            "tipo": f"modelo_provisorio_{nome}", "status": "provisorio", "alfa": alfa, "dataset_sha256": manifesto["sha256_dataset"],
            "q50_artefato": str(artefato_q50), "q50_sha256_modelo": _sha(Path(artefato_q50) / ARQUIVO_MODELO),
            "regra": "mesma população, atributos, pré-processamento e configuração do q50; só o objetivo muda"})
    recarregados = {n: (q50 if n == "q50" else carregar(destino / n)) for n in ("q10", "q50", "q90")}
    relativos = {n: modelos[n].prever_relativo(validacao) for n in modelos}
    identico = all(np.array_equal(relativos[n], recarregados[n].prever_relativo(validacao)) for n in modelos)
    if not identico:
        raise ErroModelo("recarga não reproduz as previsões dos quantis")
    prev = prever_faixa(validacao, relativos["q10"], relativos["q50"], relativos["q90"])
    avaliacao_q50 = avaliar(q50, validacao)
    exp = contribuicoes_q50(q50, validacao)
    media_abs = np.abs(exp["contribuicoes"]).mean(axis=0)
    ordem = np.argsort(-media_abs)[:12]
    resultado = {
        "destino": str(destino), "recarga_identica": identico, "n_treino": len(treino), "n_validacao": len(validacao),
        "faixa": avaliar_faixa(validacao, prev),
        "q50_preservado": {"mae_log": avaliacao_q50["modelo"]["mae_log"],
                           "central_igual_ao_q50": bool(np.array_equal(
                               prev["central_log"], reconstruir_log(validacao, relativos["q50"])[0]))},
        "variacao_vs_canal": {"mediana_log": float(np.median(prev["variacao_log_vs_canal"])),
                              "n_cortadas_em_zero": int(prev["cortada_em_zero"].sum())},
        "explicacao_q50": {"escala": "y_rel em log", "base": exp["base"], "transformacoes": exp["transformacoes"],
                           "n_linhas_cortadas_em_zero": int(exp["cortada_em_zero"].sum()),
                           "soma_confere_com_o_modelo": True,
                           "maiores_contribuicoes_medias_abs": {exp["nomes"][i]: float(media_abs[i]) for i in ordem}},
        "limitacoes": [f"provisório; {len(validacao)} linhas de validação; sem teste final",
                       "cobertura é observada, não garantida",
                       "TreeSHAP em atributos correlacionados divide contribuição; não é causal"],
    }
    (destino / "resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


def main_faixa_previsao(argv: Sequence[str] | None = None) -> int:
    from src.config import RAIZ_PROJETO

    parser = argparse.ArgumentParser(prog="python -m src.modelos.treino faixa_previsao", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--q50", type=Path, required=True, help="artefato do modelo central já treinado")
    parser.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "data" / "models")
    args = parser.parse_args(argv)
    try:
        resultado = executar_faixa_previsao(args.dataset, args.q50, args.saida)
    except (ValueError, KeyError, OSError) as erro:
        print(f"Faixa recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    return 0


# ──── Famílias de atributos e especificações operacionais dos modelos de texto e de imagem.
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


# ──── Executor reutilizável do GridSearchCV: cortes cronológicos explícitos e intervalo em dias de calendá
N_CORTES = 3
TREINO_MINIMO = 100
# Cortes escolhidos para a população de desenvolvimento atual (346 treino): manter os cortes 2 e 3 da
# divisão em 3, preservando seus limites. Trocar `n_cortes` por 2 recalcularia as janelas e não é a mesma escolha.
# Reconferir a viabilidade quando a população mudar, sem usar resultados de teste.
CORTES_APROVADOS = (2, 3)
# Desempate: tolerância absoluta na métrica (não é equivalência estatística).
TOLERANCIA_DESEMPATE = 0.005
INTERVALO_DIAS = 30
GRADE_PROPOSTA = {"num_leaves": [7, 15, 31], "min_child_samples": [10, 20], "reg_lambda": [0.0, 1.0]}


class ErroBusca(ValueError):
    """Entrada incompatível com o contrato da busca temporal."""


def combinacoes(grade: Mapping[str, Sequence]) -> list[dict]:
    nomes = list(grade)
    return [dict(zip(nomes, valores, strict=True)) for valores in product(*(grade[n] for n in nomes))]


def cortes_cronologicos(datas: Sequence[datetime], *, n_cortes: int = N_CORTES,
                        intervalo_dias: int = INTERVALO_DIAS, treino_minimo: int = TREINO_MINIMO,
                        cortes_mantidos: Sequence[int] | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pares (treino, validação) de índices. O bloco k+1 (por quantidade, em ordem de data) valida; o treino é tudo
    o que foi publicado até `início do bloco − intervalo_dias`. Treino ou validação vazios são recusados."""
    if any(d.tzinfo is None for d in datas):
        raise ErroBusca("datas sem fuso horário")
    ordem = sorted(range(len(datas)), key=lambda i: (datas[i], i))
    if len(ordem) < n_cortes + 1:
        raise ErroBusca("linhas insuficientes para os cortes pedidos")
    limites = np.linspace(0, len(ordem), n_cortes + 2).round().astype(int)
    cortes = []
    for k in range(1, n_cortes + 1):
        validacao = np.array(ordem[limites[k]:limites[k + 1]])
        inicio = min(datas[i] for i in validacao)
        treino = np.array([i for i in ordem if datas[i] <= inicio - timedelta(days=intervalo_dias)])
        if not treino.size or not validacao.size:
            raise ErroBusca(f"corte {k} ficou sem treino ou sem validação com intervalo de {intervalo_dias} dias")
        cortes.append((treino, validacao))
    if cortes_mantidos is not None:
        invalidos = [k for k in cortes_mantidos if not 1 <= k <= n_cortes]
        if invalidos:
            raise ErroBusca(f"cortes inexistentes na divisão de {n_cortes}: {invalidos}")
        cortes = [cortes[k - 1] for k in sorted(set(cortes_mantidos))]
    pequenos = [len(t) for t, _ in cortes if len(t) < max(treino_minimo, 1)]
    if pequenos:
        raise ErroBusca(f"treino com {pequenos} linhas, abaixo do mínimo de {treino_minimo}")
    return cortes


def descrever_cortes(datas: Sequence[datetime], cortes: Sequence[tuple[np.ndarray, np.ndarray]],
                     numeros: Sequence[int] | None = None) -> list[dict]:
    return [{
        "corte": (numeros[k - 1] if numeros else k), "n_treino": len(t), "n_validacao": len(v),
        "treino_ate": max(datas[i] for i in t).isoformat(), "validacao_de": min(datas[i] for i in v).isoformat(),
        "validacao_ate": max(datas[i] for i in v).isoformat(),
        "intervalo_dias": (min(datas[i] for i in v) - max(datas[i] for i in t)) / timedelta(days=1),
    } for k, (t, v) in enumerate(cortes, start=1)]


def escolher(combinacoes_resumo: Sequence[Mapping], *, tolerancia: float = TOLERANCIA_DESEMPATE) -> dict:
    """Entre as combinações até `tolerancia` (absoluta) acima do melhor MAE médio: menor `num_leaves`, depois maior
    `min_child_samples`, depois maior `reg_lambda`."""
    melhor = min(combinacoes_resumo, key=lambda c: c["mae_log_medio"])
    limite = melhor["mae_log_medio"] + tolerancia
    dentro = [c for c in combinacoes_resumo if c["mae_log_medio"] <= limite + 1e-12]
    p = "parametros"
    escolhida = min(dentro, key=lambda c: (c[p]["num_leaves"], -c[p].get("min_child_samples", 0), -c[p].get("reg_lambda", 0.0)))
    return {**escolhida, "melhor_bruto": melhor, "tolerancia_absoluta": tolerancia,
            "n_dentro_da_tolerancia": len(dentro)}


def executar_busca(linhas_treino: Sequence[Mapping], especificacao: EspecificacaoAtributos, *,
                   grade: Mapping[str, Sequence], grade_aprovada: bool,
                   parametros_base: Mapping = PARAMETROS_INICIAIS, n_cortes: int = N_CORTES,
                   intervalo_dias: int = INTERVALO_DIAS, treino_minimo: int = TREINO_MINIMO,
                   cortes_mantidos: Sequence[int] | None = CORTES_APROVADOS) -> dict:
    """GridSearchCV com cortes explícitos, MAE em log e sem reajuste final (`refit=False`)."""
    if not grade_aprovada:
        raise ErroBusca("a grade numérica ainda não foi aprovada; nada foi executado")
    from sklearn.model_selection import GridSearchCV

    exigir_uso(linhas_treino, USO_TREINO)
    linhas = list(linhas_treino)
    cortes = cortes_cronologicos([l["published_at"] for l in linhas], n_cortes=n_cortes,
                                 intervalo_dias=intervalo_dias, treino_minimo=treino_minimo,
                                 cortes_mantidos=cortes_mantidos)
    busca = GridSearchCV(construir_pipeline(especificacao, parametros_base),
                         {f"modelo__{k}": list(v) for k, v in grade.items()}, cv=cortes,
                         scoring="neg_mean_absolute_error", refit=False, n_jobs=1, error_score="raise")
    busca.fit(linhas, np.array([l["y_rel"] for l in linhas], dtype=np.float64))
    r = busca.cv_results_
    resumo = [{
        "parametros": {k.removeprefix("modelo__"): v for k, v in p.items()},
        "mae_log_medio": float(-r["mean_test_score"][i]),
        "mae_log_por_corte": [float(-r[f"split{k}_test_score"][i]) for k in range(len(cortes))],
        "posicao": int(r["rank_test_score"][i]),
    } for i, p in enumerate(r["params"])]
    return {"n_cortes": len(cortes), "combinacoes": resumo, "escolhida": escolher(resumo),
            "metrica": "MAE do alvo relativo em log; sem o corte em zero da previsão final e sem a exclusão de linhas "
            "sem baseline, portanto aproxima, e não é idêntico ao MAE reportado na validação"}


def main_busca_parametros(argv: Sequence[str] | None = None) -> int:
    """0 = ok; 2 = entrada recusada. Sem `--grade-aprovada`, apenas verifica os cortes (nada é ajustado)."""
    parser = argparse.ArgumentParser(prog="python -m src.modelos.treino busca_parametros", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="pasta do dataset de desenvolvimento")
    parser.add_argument("--saida", type=Path, help="pasta onde gravar (uma vez) o resultado da busca")
    parser.add_argument("--grade-aprovada", action="store_true",
                        help="só use depois de confirmar os valores da grade; sem isto, só verifica os cortes")
    args = parser.parse_args(argv)
    from src.dados import dataset as ds

    try:
        linhas, manifesto = ds.carregar_dataset(args.dataset)
        treino = [l for l in linhas if l["uso_no_contexto"] == USO_TREINO]
        datas = [l["published_at"] for l in treino]
        cortes = cortes_cronologicos(datas, cortes_mantidos=CORTES_APROVADOS)
        saida = {"n_treino": len(treino), "cortes": descrever_cortes(datas, cortes, CORTES_APROVADOS),
                 "cortes_aprovados": list(CORTES_APROVADOS), "treino_minimo": TREINO_MINIMO,
                 "grade_proposta": GRADE_PROPOSTA, "n_combinacoes": len(combinacoes(GRADE_PROPOSTA))}
        if args.grade_aprovada:
            espec = EspecificacaoAtributos(**manifesto["especificacao"])
            saida["busca"] = executar_busca(treino, espec, grade=GRADE_PROPOSTA, grade_aprovada=True)
    except (ValueError, KeyError, TypeError, OSError) as erro:
        print(f"Busca recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    if args.saida and "busca" in saida:
        destino = args.saida / f"busca=v1_{manifesto['sha256_dataset'][:12]}.json"
        if destino.exists():
            print(f"resultado já existe e não será sobrescrito: {destino}", file=sys.stderr)
            return 2
        args.saida.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1))
    return 0


def _script_modelo_desenvolvimento() -> None:
    sys.exit(main())


def _script_faixa_previsao() -> None:
    sys.exit(main_faixa_previsao())


def _script_busca_parametros() -> None:
    sys.exit(main_busca_parametros())


_COMANDOS = {
    "modelo_desenvolvimento": _script_modelo_desenvolvimento,
    "faixa_previsao": _script_faixa_previsao,
    "busca_parametros": _script_busca_parametros,
}


def _despachar() -> None:
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in _COMANDOS:
        raise SystemExit("uso: python -m src.modelos.treino {" + "|".join(_COMANDOS) + "} [argumentos]")
    comando = sys.argv.pop(1)
    sys.argv[0] = f"{sys.argv[0]} {comando}"
    _COMANDOS[comando]()


if __name__ == "__main__":
    _despachar()
