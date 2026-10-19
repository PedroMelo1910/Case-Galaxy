"""Treino e empacotamento dos modelos reduzidos (texto e imagem): q50 e companheiros q10/q90, configuração A fixa.

A configuração é a do modelo de vídeo servido (A), lida do metadado do artefato de referência e conferida contra a
constante abaixo: o candidato C (`min_child_samples=40`) é recusado. Não há busca, early stopping nem calibração. O
pré-processamento (PCA e códigos das categorias) é ajustado só nas linhas de treino. A faixa q10–q90 reaproveita
`faixa_previsao.executar`, de modo que a regra de coerência com o q50 é a mesma do modelo de vídeo.

Cada pacote declara a própria população de validação e a própria cobertura observada; nada é herdado do modelo de
vídeo, e os resultados não constituem ablação multimodal (populações de treino e validação diferentes).
"""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src import dataset_desenvolvimento as ds
from src import faixa_previsao as fp
from src import familias_atributos as fa
from src import modelo_desenvolvimento as md

VERSAO = "1"
TIPO_DATASET = "dataset_modalidade"
CONFIGURACAO_A = {
    "objective": "quantile", "alpha": 0.5, "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 7,
    "min_child_samples": 10, "reg_lambda": 1.0, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": md.SEMENTE, "n_jobs": 4, "deterministic": True, "force_row_wise": True, "verbose": -1,
}
LIMITACOES = [
    "Provisório: configuração A fixa, herdada do modelo de vídeo; não houve busca nem calibração para esta população.",
    "Validação de desenvolvimento, não avaliação final: não há teste final nem LIM envolvidos.",
    "A cobertura observada da faixa vale só para a população de validação deste pacote e não é garantia.",
    "Sem informação do vídeo: o modelo não vê conteúdo, transcrição nem atributos do Gemini.",
    "Retrospectivo: as curtidas lidas na coleta são acumuladas, não um valor exato aos 30 dias.",
]
NAO_E_ABLACAO = (
    "Estes resultados não constituem ablação multimodal: a população de treino (uso `treino`) e a de validação "
    "(uso `avaliacao`) são maiores que as do modelo completo, que exige anotação do Gemini. A ablação final compara "
    "todos os níveis nas mesmas linhas.")


def _iso(instante: datetime) -> str:
    return instante.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(arquivo: Path) -> str:
    return hashlib.sha256(Path(arquivo).read_bytes()).hexdigest()


def parametros_de_referencia(pasta_referencia: Path) -> dict:
    """Parâmetros do artefato de referência (o modelo de vídeo servido), que devem ser exatamente a configuração A."""
    meta = json.loads((Path(pasta_referencia) / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    parametros = meta["parametros"]
    divergentes = {k: (parametros.get(k), v) for k, v in CONFIGURACAO_A.items() if parametros.get(k) != v}
    if divergentes:
        raise md.ErroModelo(f"o artefato de referência não tem a configuração A (obtido, esperado): {divergentes}")
    return dict(parametros)


def _ausente(valor: object) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def _ausencias(linhas: Sequence[Mapping], espec: md.EspecificacaoAtributos) -> dict:
    """Ausências explícitas dos preditores (nulo ou NaN), por coluna; nada é preenchido."""
    colunas = [*espec.escalares, *espec.categoricas, *espec.embeddings]
    contagem = {c: sum(_ausente(l.get(c)) for l in linhas) for c in colunas}
    return {"n": len(linhas), "por_coluna": {c: n for c, n in contagem.items() if n}}


def _periodo(linhas: Sequence[Mapping]) -> dict:
    datas = sorted(l["published_at"] for l in linhas)
    return {"n": len(linhas), "inicio_utc": _iso(datas[0]), "fim_utc": _iso(datas[-1])}


def _dividir(linhas: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    treino = [l for l in linhas if l["uso_no_contexto"] == md.USO_TREINO]
    validacao = [l for l in linhas if l["uso_no_contexto"] == md.USO_AVALIACAO]
    if len(treino) + len(validacao) != len(linhas):
        raise md.ErroModelo("o dataset tem linhas fora de treino e avaliacao")
    return treino, validacao


def _reaproveitar(pasta_q50: Path) -> dict | None:
    """Resultado de um pacote completo já gravado; pasta existente sem resultado verificado é erro, não reuso."""
    if not pasta_q50.exists():
        return None
    arquivo = pasta_q50 / "resultado.json"
    anterior = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else None
    if not anterior or not anterior.get("q50", {}).get("recarga_identica"):
        raise md.ErroModelo(f"artefato incompleto ou sem recarga verificada em {pasta_q50}; remova-o antes de repetir")
    return anterior | {"reaproveitado": True}


def treinar_pacote(pasta_dataset: Path, saida_raiz: Path, *, pasta_referencia: Path) -> dict:
    """Treina q50, q10 e q90 no uso `treino`, avalia em `avaliacao`, grava o pacote e confere a recarga."""
    linhas, manifesto = ds.carregar_dataset(pasta_dataset)
    if manifesto.get("tipo") != TIPO_DATASET:
        raise md.ErroModelo(f"esperado um dataset do tipo {TIPO_DATASET!r}; recebido {manifesto.get('tipo')!r}")
    modalidade = manifesto["modalidade"]
    declaracao = fa.declaracao_da_modalidade(modalidade)
    espec = md.EspecificacaoAtributos(**manifesto["especificacao"])
    if fa.hash_da_especificacao(espec) != manifesto["especificacao_sha256"] or "duracao_s" in espec.escalares:
        raise md.ErroModelo("a especificação do dataset não é a operacional da modalidade")
    parametros = parametros_de_referencia(pasta_referencia)
    treino, validacao = _dividir(linhas)
    chave = hashlib.sha256(json.dumps(parametros, sort_keys=True).encode()).hexdigest()[:8]
    pasta_q50 = Path(saida_raiz) / f"provisorio=v{md.VERSAO}" / f"{manifesto['sha256_dataset'][:12]}-{chave}"
    anterior = _reaproveitar(pasta_q50)
    if anterior is not None:
        return anterior
    modelo = md.treinar(treino, espec, parametros=parametros)
    avaliacao = md.avaliar(modelo, validacao)
    ids = {"treino": [l["video_id"] for l in treino], "validacao": [l["video_id"] for l in validacao]}
    periodos = {"corte_do_treino_utc": manifesto["entradas"]["contexto"].get("treino_ate_utc"),
                "treino": _periodo(treino), "validacao": _periodo(validacao)}
    nivel = fa.nivel_do_modelo(modalidade)
    q50 = md.salvar(modelo, pasta_q50, metadados={
        "tipo": "modelo_modalidade_q50", "status": "provisorio", "modalidade": modalidade, "nivel_do_modelo": nivel,
        "dataset": {"sha256": manifesto["sha256_dataset"], "versao": manifesto["versao"], "pasta": str(pasta_dataset),
                    "entradas": manifesto["entradas"], "reconciliacao": manifesto["reconciliacao"]},
        "populacao": {"n_treino": len(treino), "n_validacao": len(validacao),
                      "sha256_ids_treino": hashlib.sha256("\n".join(ids["treino"]).encode()).hexdigest(),
                      "sha256_ids_validacao": hashlib.sha256("\n".join(ids["validacao"]).encode()).hexdigest(),
                      **manifesto["populacao"]},
        "periodos": periodos,
        "especificacao_operacional": {"sha256": manifesto["especificacao_sha256"],
                                      "experimental_sha256": manifesto["especificacao_experimental_sha256"],
                                      "adaptacao": declaracao},
        "configuracao": {"nome": "A", "origem": f"artefato servido {Path(pasta_referencia).name}",
                         "sha256_metadados_referencia": _sha256(Path(pasta_referencia) / md.ARQUIVO_METADADOS)},
        "semente": md.SEMENTE, "alvo": "y_rel = log1p(curtidas) − ref_log; previsão absoluta = ref_log + previsão, "
        "cortada em 0", "retencao": manifesto["retencao"], "limitacoes": LIMITACOES, "nao_e_ablacao": NAO_E_ABLACAO})
    (q50 / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    recarregado = md.carregar(q50)
    identico = bool(np.array_equal(modelo.prever_log(validacao)[0], recarregado.prever_log(validacao)[0]))
    if not identico:
        raise md.ErroModelo("o modelo recarregado não reproduz as previsões")
    faixa = fp.executar(pasta_dataset, q50, saida_raiz)
    pasta_faixa = Path(faixa["destino"])
    meta_q50 = json.loads((q50 / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    registro = _registro_do_pacote(modalidade, nivel, q50, pasta_faixa, manifesto, meta_q50, parametros, periodos,
                                   faixa, avaliacao, declaracao, pasta_referencia)
    (pasta_faixa / "pacote.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")
    resultado = {
        "modalidade": modalidade, "n_treino": len(treino), "n_validacao": len(validacao),
        "atributos_de_saida": len(modelo.nomes_atributos), "especificacao": manifesto["especificacao"],
        "parametros": parametros, "avaliacao": avaliacao, "reaproveitado": False,
        "ausencias_nos_preditores": {"treino": _ausencias(treino, espec), "validacao": _ausencias(validacao, espec)},
        "q50": {"pasta": str(q50), "sha256_modelo": meta_q50["sha256_modelo"], "recarga_identica": identico},
        "faixa": {"pasta": str(pasta_faixa), "recarga_identica": bool(faixa["recarga_identica"]),
                  "cobertura_observada": registro["cobertura_observada"]},
    }
    (q50 / "resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


def _registro_do_pacote(modalidade, nivel, q50, pasta_faixa, manifesto, meta_q50, parametros, periodos, faixa,
                        avaliacao, declaracao, pasta_referencia) -> dict:
    modelos = {"q50": q50, "q10": pasta_faixa / "q10", "q90": pasta_faixa / "q90"}
    cobertura = faixa["faixa"]
    return {
        "tipo": "pacote_modalidade", "versao": VERSAO, "gerado_em_utc": _iso(datetime.now(UTC)),
        "modalidade": modalidade, "nivel_do_modelo": nivel,
        **{nome: {"pasta": str(pasta), "sha256_modelo": _sha256(pasta / md.ARQUIVO_MODELO)}
           for nome, pasta in modelos.items()},
        "dataset": {"sha256": manifesto["sha256_dataset"], "versao": manifesto["versao"]},
        "especificacao": {"colunas": manifesto["especificacao"], "sha256": manifesto["especificacao_sha256"],
                          "experimental_sha256": manifesto["especificacao_experimental_sha256"],
                          "adaptacao_operacional": declaracao},
        "configuracao": meta_q50["configuracao"], "parametros": parametros,
        "populacao": meta_q50["populacao"], "periodos": periodos,
        "cobertura_observada": {
            "n": int(cobertura["n"]), "n_dentro": round(float(cobertura["cobertura"]) * int(cobertura["n"])),
            "cobertura": float(cobertura["cobertura"]), "nivel_nominal": fp.COBERTURA_NOMINAL,
            "largura_log_mediana": float(cobertura["largura_log_mediana"])},
        "validacao": {"n_comparado": avaliacao["n_comparado"], "mae_log_modelo": avaliacao["modelo"]["mae_log"],
                      "mae_log_media_aritmetica": avaliacao["baseline_media_aritmetica"]["mae_log"]},
        "dependencias": meta_q50["dependencias"], "retencao": manifesto["retencao"],
        "limitacoes": LIMITACOES, "nao_e_ablacao": NAO_E_ABLACAO,
        "referencia_de_configuracao": Path(pasta_referencia).name,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from src.config import RAIZ_PROJETO

    parser = argparse.ArgumentParser(prog="python -m src.modelo_modalidades", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="pasta do dataset reduzido da modalidade")
    parser.add_argument("--referencia", type=Path, required=True,
                        help="artefato q50 do modelo de vídeo servido, de onde vem a configuração A")
    parser.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "data" / "models")
    args = parser.parse_args(argv)
    try:
        resultado = treinar_pacote(args.dataset, args.saida, pasta_referencia=args.referencia)
    except (ValueError, KeyError, OSError) as erro:
        print(f"Treino recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in resultado.items() if k != "avaliacao"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
