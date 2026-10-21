"""Relatório de avaliação de desenvolvimento de um pacote de modelos (q50 + faixa q10/q90) sobre um dataset.

Lê só treino e validação de desenvolvimento; não carrega teste nem LIM. Cada resultado traz a população, as versões
(dataset, artefatos) e os métodos usados, para que números de populações diferentes não sejam comparados como
equivalentes. Não treina nada: apenas carrega artefatos já gravados.

    python -m src.avaliacao.relatorio --dataset <pasta> --q50 <artefato> --faixa <pasta q10/q90> --saida <relatorio.json>
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from src.dados import dataset as ds
from src.modelos import treino as fp
from src.modelos import treino as md
from src.avaliacao import experimentos as bi
from src.avaliacao import metricas as bs
from src.avaliacao import experimentos as cs
from src.avaliacao import limiar as dcl
from src.avaliacao import experimentos as dv
from src.avaliacao import experimentos as dg
from src.proveniencia import sha256_arquivo as _sha

BORDAS_IDADE = (0, 30, 60, 90, 120, 150, 180, 240, 300, 370)  # os blocos de teste têm 31 a 118 dias na coleta
BORDAS_HISTORICO = (0, 5, 10, 20, 21)


def prever_pacote(linhas: Sequence[dict], q10: md.ModeloTreinado, q50: md.ModeloTreinado,
                  q90: md.ModeloTreinado) -> dict:
    return fp.prever_faixa(linhas, q10.prever_relativo(linhas), q50.prever_relativo(linhas),
                           q90.prever_relativo(linhas))


def _alinhar(pop: dv.Populacao, previsao: dict, chave: str) -> np.ndarray:
    return np.asarray(previsao[chave])[list(pop.posicoes)]


def _rotulos(linhas: Sequence[dict], pop: dv.Populacao, coleta: datetime) -> dict[str, list[str]]:
    mantidas = [linhas[i] for i in pop.posicoes]
    idade = [dg.idade_em_dias(l["published_at"].astimezone(UTC), coleta) for l in mantidas]
    historico = [l["hist_n_posts"] if l.get("hist_n_posts") is not None else -1 for l in mantidas]
    rotulos = {
        "idade_do_rotulo": dg.rotular_faixas(idade, BORDAS_IDADE),
        "setor": [l["setor"] for l in mantidas],
        "canal": [str(l["channel_id"]) for l in mantidas],
        "historico_do_canal": dg.rotular_faixas(historico, BORDAS_HISTORICO, "posts"),
        "estado_da_referencia": [str(l.get("estado_referencia")) for l in mantidas],
    }
    if mantidas and "tipo_canal" in mantidas[0]:
        rotulos["tipo_canal"] = [str(l["tipo_canal"]) for l in mantidas]
    if mantidas and "gem_transc_status" in mantidas[0]:  # datasets reduzidos (texto, imagem) não têm transcrição
        rotulos["transcricao"] = [f"{l['gem_transc_status']}/{l['gem_transc_emb_estado']}" for l in mantidas]
    return rotulos


def _bloco(linhas: Sequence[dict], previsao: dict, coleta: datetime, *, n_repeticoes: int, semente: int,
           completo: bool, celulas: Sequence[str] | None = None) -> tuple[dict, dv.Populacao]:
    pop, exclusoes = dv.montar_populacao(linhas, previsao["central_log"], celulas=celulas)
    inferior, superior = _alinhar(pop, previsao, "inferior_log"), _alinhar(pop, previsao, "superior_log")
    setores, contagens = np.unique(pop.setores, return_counts=True)
    saida = {
        "populacao": {"n_linhas": len(linhas), "n_avaliadas": len(pop.ids), "exclusoes": exclusoes,
                      "n_canais": len(set(pop.canais)), "setores": dict(zip(setores.tolist(), contagens.tolist())),
                      "ids_sha256": hashlib.sha256("\n".join(pop.ids).encode()).hexdigest()},
        "cobertura": dv.cobertura(pop, inferior, superior, n_repeticoes=n_repeticoes, semente=semente),
        "cobertura_por_setor": cs.cobertura_por_setor(pop, inferior, superior, n_repeticoes=n_repeticoes,
                                                      semente=semente),
        "faixa": fp.avaliar_faixa([linhas[i] for i in pop.posicoes],
                                  {k: np.asarray(v)[list(pop.posicoes)] for k, v in previsao.items()}),
        "diagnosticos": {nome: dg.por_grupo(pop, r, inferior=inferior, superior=superior)
                         for nome, r in _rotulos(linhas, pop, coleta).items()},
    }
    if completo:
        saida["avaliacao"] = dv.avaliar(pop, n_repeticoes=n_repeticoes, semente=semente)
    else:
        saida["avaliacao_pontual"] = {m: dv._erros(pop, m) for m in dv.METODOS}
    return saida, pop


def bloco_baseline_inscritos(treino: Sequence[dict], validacao: Sequence[dict], central_log: Sequence[float],
                             inscritos: dict | None, *, n_repeticoes: int = bs.N_REPETICOES,
                             semente: int = bs.SEMENTE, rotulo: str = "validacao") -> dict:
    """Baseline 4 (TAPI, D-D1) sobre a validação de desenvolvimento: ajuste só no treino, mesmas linhas das bases."""
    if inscritos is None:
        return bi.bloco_nao_avaliado("tabela de inscritos não fornecida ao relatório")
    ajuste, previsto, cortadas = bi.prever_dobra(treino, validacao, inscritos)
    return bi.montar_bloco({rotulo: ajuste}, {rotulo: cortadas}, validacao, previsto, {"servido": central_log},
                           n_repeticoes=n_repeticoes, semente=semente)


def gerar(pasta_dataset: Path, q50: Path, pasta_faixa: Path, *, ids_validacao: Sequence[str] | None = None,
          n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE,
          inscritos: dict | None = None) -> tuple[dict, list[dict]]:
    linhas, manifesto = ds.carregar_dataset(pasta_dataset)
    modelos = {"q10": md.carregar(Path(pasta_faixa) / "q10"), "q50": md.carregar(q50),
               "q90": md.carregar(Path(pasta_faixa) / "q90")}
    coleta = datetime.strptime(manifesto["entradas"]["base"]["snapshot_id"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    treino = [l for l in linhas if l["uso_no_contexto"] == md.USO_TREINO]
    validacao = [l for l in linhas if l["uso_no_contexto"] == md.USO_AVALIACAO]
    prev_val, prev_trn = prever_pacote(validacao, **modelos), prever_pacote(treino, **modelos)
    relatorio = {
        "versoes": {
            "dataset": {"versao": manifesto["versao"], "sha256": manifesto["sha256_dataset"],
                        "pasta": str(pasta_dataset), "n_treino": len(treino), "n_validacao": len(validacao)},
            "artefatos": {n: {"pasta": str(p), "sha256_modelo": _sha(Path(p) / md.ARQUIVO_MODELO)} for n, p in
                          (("q10", Path(pasta_faixa) / "q10"), ("q50", Path(q50)), ("q90", Path(pasta_faixa) / "q90"))},
            "parametros_q50": modelos["q50"].parametros, "coleta_dos_rotulos_utc": coleta.isoformat(),
            "bibliotecas": {"numpy": np.__version__, "scipy": scipy.__version__},
        },
        "bootstrap": {"semente": semente, "n_repeticoes": n_repeticoes},
    }
    bloco_val, pop_val = _bloco(validacao, prev_val, coleta, n_repeticoes=n_repeticoes, semente=semente, completo=True)
    relatorio["validacao"] = bloco_val | {
        "declaracao_da_faixa": fp.declaracao_da_faixa(n_dentro=bloco_val["cobertura"]["n_dentro"],
                                                      n=bloco_val["cobertura"]["n"]),
        "declaracao_da_faixa_por_setor": dcl.declarar(
            "desenvolvimento", modelo="servido", total=bloco_val["cobertura"],
            por_setor=bloco_val["cobertura_por_setor"])}
    if ids_validacao is not None:
        pedidos = set(ids_validacao)
        idx = [i for i, l in enumerate(validacao) if l["video_id"] in pedidos]
        bloco_sub, _ = _bloco([validacao[i] for i in idx], {k: np.asarray(v)[idx] for k, v in prev_val.items()}, coleta,
                              n_repeticoes=n_repeticoes, semente=semente, completo=True)
        relatorio["validacao_subconjunto_dado"] = bloco_sub | {"n_ids_pedidos": len(pedidos), "n_encontrados": len(idx)}
    if inscritos is not None:  # baseline 4 é opcional: a tabela restrita só é lida quando o chamador a fornece
        relatorio["baseline_inscritos"] = bloco_baseline_inscritos(
            treino, validacao, prev_val["central_log"], inscritos, n_repeticoes=n_repeticoes, semente=semente)
    bloco_trn, _ = _bloco(treino, prev_trn, coleta, n_repeticoes=n_repeticoes, semente=semente, completo=False)
    relatorio["treino_dentro_da_amostra"] = {
        "aviso": "diagnóstico dentro da amostra de ajuste: não é evidência de generalização"} | bloco_trn
    previsoes = [{"video_id": pop_val.ids[k], "channel_id": str(pop_val.canais[k]), "uso": "avaliacao",
                  "y_log": float(pop_val.real_log[k]),
                  **{f"prev_{m}": float(pop_val.previsto[m][k]) for m in dv.METODOS},
                  "q10_log": float(prev_val["q10_log"][pop_val.posicoes[k]]),
                  "q90_log": float(prev_val["q90_log"][pop_val.posicoes[k]])} for k in range(len(pop_val.ids))]
    return relatorio, previsoes


def sanear(obj):
    """Converte tipos numpy e troca NaN/inf por `None` (indefinido), para o JSON ser válido (`allow_nan=False`)."""
    if isinstance(obj, dict):
        return {str(k): sanear(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [sanear(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.avaliacao.relatorio", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--q50", type=Path, required=True)
    parser.add_argument("--faixa", type=Path, required=True, help="pasta com q10/ e q90/")
    parser.add_argument("--saida", type=Path, required=True, help="relatório JSON (as previsões vão ao lado)")
    parser.add_argument("--ids", type=Path, help="JSON {'validacao': [...]} para avaliar também esse subconjunto")
    parser.add_argument("--base-curada", type=Path,
                        help="pasta curadoria=v1: acrescenta o baseline 4 (regressão sobre inscritos atuais)")
    args = parser.parse_args(argv)
    try:
        ids = json.loads(args.ids.read_text(encoding="utf-8"))["validacao"] if args.ids else None
        inscritos = bi.carregar_inscritos(args.base_curada) if args.base_curada else None
        relatorio, previsoes = gerar(args.dataset, args.q50, args.faixa, ids_validacao=ids, inscritos=inscritos)
    except (ValueError, KeyError, OSError) as erro:
        print(f"Relatório recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(sanear(relatorio), ensure_ascii=False, indent=1, allow_nan=False),
                          encoding="utf-8")
    args.saida.with_suffix(".previsoes.json").write_text(
        json.dumps(sanear(previsoes), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"relatorio": str(args.saida)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
