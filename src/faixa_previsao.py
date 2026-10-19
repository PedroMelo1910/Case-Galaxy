"""Faixa de previsão q10–q90 em torno do q50 e explicação do q50 (provisório, desenvolvimento).

Regras aprovadas por Pedro em 15/10:
- o q50 é a previsão central e nunca é substituído pela mediana das três saídas;
- q10 e q90 usam a mesma população, os mesmos atributos, o mesmo pré-processamento e a configuração registrada do
  q50, mudando só o objetivo do quantil; nenhuma nova busca;
- se houver cruzamento, a faixa passa a ser (mínimo das três saídas, máximo das três saídas) com o q50 mantido.
  Isto **substitui** a regra anterior de reordenação total: a faixa corrigida não é "quantis brutos";
- a explicação (SHAP) é só do q50, na escala do modelo (`y_rel` em log); o valor exibido passa por
  `+ ref_log`, corte em zero e `expm1`, e a decomposição aditiva vale para o valor bruto, não para o exibido quando
  houve corte;
- a comparação com o canal usa o valor exibido (depois do corte).
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src import modelo_desenvolvimento as md

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


def contribuicoes_q50(modelo: md.ModeloTreinado, linhas: Sequence[Mapping]) -> dict:
    """TreeSHAP do próprio LightGBM (`pred_contrib`) para o q50, na escala do modelo, com a regra de transformação."""
    pre, regressor = modelo.pipeline.named_steps["pre"], modelo.pipeline.named_steps["modelo"]
    contrib = regressor.booster_.predict(pre.transform(list(linhas)), pred_contrib=True)
    soma = contrib.sum(axis=1)
    if not np.allclose(soma, modelo.prever_relativo(linhas), atol=1e-8):
        raise md.ErroModelo("as contribuições não somam a previsão do modelo")
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


def _sha(arquivo: Path) -> str:
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()


def treinar_companheiros(treino: Sequence[Mapping], q50: md.ModeloTreinado) -> dict[str, md.ModeloTreinado]:
    """q10 e q90 com a população, os atributos, a representação das categorias e os parâmetros do q50; só o `alpha`
    muda."""
    return {nome: md.treinar(treino, q50.especificacao, parametros=q50.parametros | {"alpha": alfa},
                             categoricas_nativas=q50.categoricas_nativas)
            for nome, alfa in (("q10", ALFA_INFERIOR), ("q90", ALFA_SUPERIOR))}


def executar(pasta_dataset: Path, artefato_q50: Path, saida_raiz: Path) -> dict:
    """Treina q10 e q90 com a configuração do q50, grava os três vínculos e mede a faixa na validação."""
    from src import dataset_desenvolvimento as ds

    linhas, manifesto = ds.carregar_dataset(pasta_dataset)
    treino = [l for l in linhas if l["uso_no_contexto"] == md.USO_TREINO]
    validacao = [l for l in linhas if l["uso_no_contexto"] == md.USO_AVALIACAO]
    q50 = md.carregar(artefato_q50)
    destino = Path(saida_raiz) / "faixa=v1" / f"{manifesto['sha256_dataset'][:12]}-{Path(artefato_q50).name}"
    if destino.exists():
        raise FileExistsError(f"já existe e não será sobrescrito: {destino}")
    modelos = {"q50": q50} | treinar_companheiros(treino, q50)
    for nome, alfa in (("q10", ALFA_INFERIOR), ("q90", ALFA_SUPERIOR)):
        md.salvar(modelos[nome], destino / nome, metadados={
            "tipo": f"modelo_provisorio_{nome}", "status": "provisorio", "alfa": alfa, "dataset_sha256": manifesto["sha256_dataset"],
            "q50_artefato": str(artefato_q50), "q50_sha256_modelo": _sha(Path(artefato_q50) / md.ARQUIVO_MODELO),
            "regra": "mesma população, atributos, pré-processamento e configuração do q50; só o objetivo muda"})
    recarregados = {n: (q50 if n == "q50" else md.carregar(destino / n)) for n in ("q10", "q50", "q90")}
    relativos = {n: modelos[n].prever_relativo(validacao) for n in modelos}
    identico = all(np.array_equal(relativos[n], recarregados[n].prever_relativo(validacao)) for n in modelos)
    if not identico:
        raise md.ErroModelo("recarga não reproduz as previsões dos quantis")
    prev = prever_faixa(validacao, relativos["q10"], relativos["q50"], relativos["q90"])
    avaliacao_q50 = md.avaliar(q50, validacao)
    exp = contribuicoes_q50(q50, validacao)
    media_abs = np.abs(exp["contribuicoes"]).mean(axis=0)
    ordem = np.argsort(-media_abs)[:12]
    resultado = {
        "destino": str(destino), "recarga_identica": identico, "n_treino": len(treino), "n_validacao": len(validacao),
        "faixa": avaliar_faixa(validacao, prev),
        "q50_preservado": {"mae_log": avaliacao_q50["modelo"]["mae_log"],
                           "central_igual_ao_q50": bool(np.array_equal(
                               prev["central_log"], md.reconstruir_log(validacao, relativos["q50"])[0]))},
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


def main(argv: Sequence[str] | None = None) -> int:
    from src.config import RAIZ_PROJETO

    parser = argparse.ArgumentParser(prog="python -m src.faixa_previsao", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--q50", type=Path, required=True, help="artefato do modelo central já treinado")
    parser.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "data" / "models")
    args = parser.parse_args(argv)
    try:
        resultado = executar(args.dataset, args.q50, args.saida)
    except (ValueError, KeyError, OSError) as erro:
        print(f"Faixa recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
