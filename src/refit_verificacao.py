"""Verificação de um pacote final em processo novo: recarga bit a bit, identidade e aditividade do SHAP (D-D7).

Roda em um processo que NÃO treinou o modelo: recarrega q10, q50 e q90 do disco (`modelo_desenvolvimento.carregar` confere
hash e versões), refaz as previsões nas mesmas linhas do ajuste e exige igualdade exata com as previsões em memória gravadas
no treino (`np.array_equal`). Confere a identidade do pacote (hashes dos três modelos), o preprocessador serializado e a
aditividade do TreeSHAP do q50 real (`faixa_previsao.contribuicoes_q50`, escala `y_rel` em log): |soma das contribuições +
base − saída bruta| ≤ 1e-9, inclusive em linhas cortadas em zero (as reais, se houver, e linhas sintéticas com referência
muito baixa), onde a decomposição vale para a saída bruta e não para o valor exibido.

Nada aqui mede desempenho: as previsões sobre as linhas do ajuste servem só para a igualdade entre processos.
Uso: `python -m src.refit_verificacao --pacote <pasta> [--saida resultado.json]`.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from src import dataset_desenvolvimento as ds
from src import faixa_previsao as fp
from src import modelo_desenvolvimento as md
from src import refit_treino as rt
from src.avaliacao import driver_lim as dl
from src.config import RAIZ_PROJETO

TOLERANCIA_SHAP = 1e-9
REF_LOG_BAIXA = -50.0  # força saída bruta negativa: a linha sintética é cortada em zero
QUANTIS = ("q10", "q50", "q90")


def _aditividade(q50: md.ModeloTreinado, linhas: Sequence[Mapping]) -> dict:
    explicacao = fp.contribuicoes_q50(q50, linhas)
    bruta = q50.prever_relativo(linhas)
    reconstruida = explicacao["contribuicoes"].sum(axis=1) + explicacao["base"]
    erro = np.abs(reconstruida - bruta)
    return {"n": len(linhas), "erro_maximo": float(erro.max()), "tolerancia": TOLERANCIA_SHAP,
            "aditivo": bool(erro.max() <= TOLERANCIA_SHAP), "base": explicacao["base"],
            "n_cortadas_em_zero": int(explicacao["cortada_em_zero"].sum()),
            "cortadas_em_zero": [bool(c) for c in explicacao["cortada_em_zero"]]}


def _linhas_cortadas(linhas: Sequence[Mapping], n: int = 5) -> list[dict]:
    return [dict(l) | {"ref_log": REF_LOG_BAIXA} for l in linhas[:n]]


def verificar_com_linhas(pasta: Path, linhas: Sequence[Mapping]) -> dict:
    """Núcleo testável: `linhas` são as do ajuste, na ordem de `ids.json` (`ajuste`)."""
    pasta = Path(pasta)
    integridade = rt.verificar_integridade(pasta)
    pacote = json.loads((pasta / rt.ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    modelos = {n: md.carregar(pasta / n) for n in QUANTIS}
    guardadas = np.load(pasta / rt.ARQUIVO_PREDICOES)
    iguais = {n: bool(np.array_equal(modelos[n].prever_relativo(linhas), guardadas[n])) for n in QUANTIS}
    identidade = dl.identidade_do_pacote({n: pacote["modelos"][n]["sha256_modelo"] for n in QUANTIS})
    pre = {n: _preprocessador_confere(pasta / n, modelos[n]) for n in QUANTIS}
    reais = _aditividade(modelos["q50"], linhas)
    sinteticas = _aditividade(modelos["q50"], _linhas_cortadas(linhas))
    faixa = fp.prever_faixa(_linhas_cortadas(linhas), *(modelos[n].prever_relativo(_linhas_cortadas(linhas))
                                                       for n in QUANTIS))
    ok = (all(iguais.values()) and identidade == pacote["identidade_sha256"] and all(pre.values())
          and reais["aditivo"] and sinteticas["aditivo"] and all(sinteticas["cortadas_em_zero"])
          and bool(faixa["cortada_em_zero"].all()) and bool((faixa["central_log"] == 0).all()))
    return {
        "verificado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "pacote": pasta.name,
        "integridade": integridade["integridade"], "n_linhas": len(linhas),
        "previsoes_identicas_bit_a_bit": iguais, "identidade_sha256": identidade,
        "identidade_confere": identidade == pacote["identidade_sha256"], "preprocessador_confere": pre,
        "shap_linhas_do_ajuste": {k: v for k, v in reais.items() if k != "cortadas_em_zero"},
        "shap_linhas_sinteticas_cortadas_em_zero": {k: v for k, v in sinteticas.items() if k != "cortadas_em_zero"}
        | {"todas_declaradas_cortadas": all(sinteticas["cortadas_em_zero"]),
           "valor_exibido_central_log_zero": bool((faixa["central_log"] == 0).all())},
        "escala_do_shap": "y_rel em log (saída bruta do q50); o valor exibido passa por + ref_log, corte em 0 e expm1",
        "aprovado": ok,
    }


def _preprocessador_confere(pasta_modelo: Path, modelo: md.ModeloTreinado) -> bool:
    """O preprocessador serializado ao lado do modelo é o mesmo que está dentro do pipeline carregado."""
    arquivo = joblib.load(pasta_modelo / rt.ARQUIVO_PREPROCESSADOR)
    dentro = modelo.pipeline.named_steps["pre"]
    return (list(arquivo.nomes_saida_) == list(dentro.nomes_saida_) and arquivo.categorias_ == dentro.categorias_
            and all(np.array_equal(arquivo.pca_[b].components_, dentro.pca_[b].components_)
                    for b in dentro.pca_ if dentro.pca_[b] is not None))


def linhas_do_ajuste(pasta: Path, raiz: Path = RAIZ_PROJETO) -> list[dict]:
    """Linhas do dataset registrado no pacote, na ordem do ajuste; o dataset e a lista de IDs são reconferidos."""
    pacote = json.loads((Path(pasta) / rt.ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    linhas, manifesto = ds.carregar_dataset(Path(raiz) / pacote["fontes"]["dataset"]["pasta"])
    if manifesto["sha256_dataset"] != pacote["fontes"]["dataset"]["sha256"]:
        raise rt.ErroPacote("o dataset em disco não é o registrado no pacote")
    ajuste = json.loads((Path(pasta) / rt.ARQUIVO_IDS).read_text(encoding="utf-8"))["ajuste"]
    por_id = {l["video_id"]: l for l in linhas}
    if rt.sha256_ids(ajuste) != pacote["populacao"]["sha256_ids"]:
        raise rt.ErroPacote("a lista de IDs do ajuste não confere com o hash do pacote")
    return [por_id[v] for v in ajuste]


def verificar_pacote(pasta: Path, raiz: Path = RAIZ_PROJETO) -> dict:
    return verificar_com_linhas(pasta, linhas_do_ajuste(pasta, raiz))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.refit_verificacao", description=__doc__)
    parser.add_argument("--pacote", type=Path, required=True)
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args(argv)
    try:
        resultado = verificar_pacote(args.pacote)
    except (rt.ErroPacote, md.ErroModelo, ds.ErroDataset, ValueError, KeyError, OSError) as erro:
        print(f"Verificação recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    texto = json.dumps(resultado, ensure_ascii=False, indent=1)
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        with open(args.saida, "x", encoding="utf-8") as arquivo:  # nunca sobrescreve
            arquivo.write(texto)
    print(texto)
    return 0 if resultado["aprovado"] else 1


if __name__ == "__main__":
    sys.exit(main())
