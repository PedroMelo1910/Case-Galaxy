"""Executor reutilizável do GridSearchCV: cortes cronológicos explícitos e intervalo em dias de calendário.

Não usa `cv=3`/`cv=5` inteiros: cada corte é uma janela expansiva de treino e um bloco de validação posterior, separados
por pelo menos `intervalo_dias` dias corridos (não linhas). O pré-processamento (PCA, categorias) é refeito dentro de cada
corte porque o `Pipeline` é clonado a cada ajuste. A busca só roda com a grade marcada como aprovada; a grade numérica
proposta (`GRADE_PROPOSTA`) ainda aguarda confirmação de Pedro; a flag `--grade-aprovada` é uma trava de processo
(convenção), não de segurança.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np

from src import modelo_desenvolvimento as md

N_CORTES = 3
TREINO_MINIMO = 100
# Aprovado por Pedro em 15/10 para a população de desenvolvimento atual (346 treino): manter os cortes 2 e 3 da
# divisão em 3, preservando seus limites. Trocar `n_cortes` por 2 recalcularia as janelas e não é a mesma escolha.
# Reconferir a viabilidade quando a população mudar, sem usar resultados de teste.
CORTES_APROVADOS = (2, 3)
# Desempate aprovado por Pedro em 15/10: tolerância absoluta na métrica (não é equivalência estatística).
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


def executar_busca(linhas_treino: Sequence[Mapping], especificacao: md.EspecificacaoAtributos, *,
                   grade: Mapping[str, Sequence], grade_aprovada: bool,
                   parametros_base: Mapping = md.PARAMETROS_INICIAIS, n_cortes: int = N_CORTES,
                   intervalo_dias: int = INTERVALO_DIAS, treino_minimo: int = TREINO_MINIMO,
                   cortes_mantidos: Sequence[int] | None = CORTES_APROVADOS) -> dict:
    """GridSearchCV com cortes explícitos, MAE em log e sem reajuste final (`refit=False`)."""
    if not grade_aprovada:
        raise ErroBusca("a grade numérica ainda não foi aprovada; nada foi executado")
    from sklearn.model_selection import GridSearchCV

    md.exigir_uso(linhas_treino, md.USO_TREINO)
    linhas = list(linhas_treino)
    cortes = cortes_cronologicos([l["published_at"] for l in linhas], n_cortes=n_cortes,
                                 intervalo_dias=intervalo_dias, treino_minimo=treino_minimo,
                                 cortes_mantidos=cortes_mantidos)
    busca = GridSearchCV(md.construir_pipeline(especificacao, parametros_base),
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


def main(argv: Sequence[str] | None = None) -> int:
    """0 = ok; 2 = entrada recusada. Sem `--grade-aprovada`, apenas verifica os cortes (nada é ajustado)."""
    parser = argparse.ArgumentParser(prog="python -m src.busca_parametros", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="pasta do dataset de desenvolvimento")
    parser.add_argument("--saida", type=Path, help="pasta onde gravar (uma vez) o resultado da busca")
    parser.add_argument("--grade-aprovada", action="store_true",
                        help="só use depois de Pedro aprovar os valores da grade; sem isto, só verifica os cortes")
    args = parser.parse_args(argv)
    from src import dataset_desenvolvimento as ds

    try:
        linhas, manifesto = ds.carregar_dataset(args.dataset)
        treino = [l for l in linhas if l["uso_no_contexto"] == md.USO_TREINO]
        datas = [l["published_at"] for l in treino]
        cortes = cortes_cronologicos(datas, cortes_mantidos=CORTES_APROVADOS)
        saida = {"n_treino": len(treino), "cortes": descrever_cortes(datas, cortes, CORTES_APROVADOS),
                 "cortes_aprovados": list(CORTES_APROVADOS), "treino_minimo": TREINO_MINIMO,
                 "grade_proposta": GRADE_PROPOSTA, "n_combinacoes": len(combinacoes(GRADE_PROPOSTA))}
        if args.grade_aprovada:
            espec = md.EspecificacaoAtributos(**manifesto["especificacao"])
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


if __name__ == "__main__":
    sys.exit(main())
