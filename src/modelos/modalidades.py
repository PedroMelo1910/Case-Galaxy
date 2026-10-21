"""Módulo consolidado. Reúne:
- Treino e empacotamento dos modelos reduzidos (texto e imagem): q50 e companheiros q10/q90, configuração A fixa.
- Ajuste dos três quantis do preditor audiovisual local, com a configuração estabelecida e sem busca."""

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from src.dados import dataset as ds
from src.modelos import treino as fa, treino as fp, treino as md
from src.modelos.preprocessamento import MIN_CANAIS_CATEGORIA, PreprocessadorAudiovisual
from src.proveniencia import iso_utc as _iso, sha256_arquivo as _sha256


# ──── Treino e empacotamento dos modelos reduzidos (texto e imagem): q50 e companheiros q10/q90, configura
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
    faixa = fp.executar_faixa_previsao(pasta_dataset, q50, saida_raiz)
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

    parser = argparse.ArgumentParser(prog="python -m src.modelos.modalidades", description=__doc__)
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


# ──── Ajuste dos três quantis do preditor audiovisual local, com a configuração estabelecida e sem busca.
PARAMETROS = dict(CONFIGURACAO_A)
ALFAS = {"q10": fp.ALFA_INFERIOR, "q50": fp.ALFA_CENTRAL, "q90": fp.ALFA_SUPERIOR}


class LimiteDeAjustes(RuntimeError):
    """O número máximo de ajustes do protocolo foi atingido."""


class ErroDeParada(RuntimeError):
    """Regra de parada pré-registrada: ajuste sem nenhum split, métrica não finita, dobra abaixo do piso ou auditoria
    reprovada. O experimento para e o motivo é reportado; nada é refeito com outra configuração."""


class ContadorDeAjustes:
    """Conta ajustes de LightGBM, recusa passar do máximo congelado e mantém um livro append-only.

    Cada ajuste é registrado no livro (uma linha JSON, com `fsync`) ANTES de o ajuste começar; um processo que cai no
    meio deixa a tentativa registrada."""

    def __init__(self, maximo: int, livro: Path | None = None) -> None:
        self.maximo, self.feitos, self.livro = maximo, 0, (None if livro is None else Path(livro))

    def registrar(self, n: int = 1, descricao: str | None = None) -> None:
        if self.feitos + n > self.maximo:
            raise LimiteDeAjustes(f"o protocolo permite no máximo {self.maximo} ajustes; já feitos {self.feitos}")
        if self.livro is not None:
            self.livro.parent.mkdir(parents=True, exist_ok=True)
            with open(self.livro, "a", encoding="utf-8") as saida:
                saida.write(json.dumps({"ajuste": self.feitos + n, "maximo": self.maximo, "descricao": descricao,
                                        "hora_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
                                       ensure_ascii=False) + "\n")
                saida.flush()
                os.fsync(saida.fileno())
        self.feitos += n


def treinar_modelo(linhas_treino: Sequence[Mapping], espec: md.EspecificacaoAtributos, colunas_locais: Sequence[str],
                   parametros: Mapping, *, contador: ContadorDeAjustes | None = None,
                   descricao: str | None = None) -> md.ModeloTreinado:
    """Um LightGBM; só aceita linhas `treino` com alvo e referência. Categorias e constantes locais só do treino.

    Um ajuste sem nenhum split (o modelo seria só a constante inicial) interrompe com `ErroDeParada`."""
    from lightgbm import LGBMRegressor

    md.exigir_uso(linhas_treino, md.USO_TREINO)
    if contador is not None:
        contador.registrar(descricao=descricao)
    pre = PreprocessadorAudiovisual(
        list(espec.escalares), list(espec.categoricas), dict(espec.embeddings), semente=md.SEMENTE,
        colunas_locais=tuple(colunas_locais), min_canais_categoria=MIN_CANAIS_CATEGORIA)
    pipeline = Pipeline([("pre", pre), ("modelo", LGBMRegressor(**parametros))])
    pipeline.fit(list(linhas_treino), np.array([l["y_rel"] for l in linhas_treino], dtype=np.float64))
    if int(pipeline.named_steps["modelo"].booster_.feature_importance("split").sum()) == 0:
        raise ErroDeParada(f"ajuste sem nenhum split ({descricao or 'sem descrição'})")
    return md.ModeloTreinado(pipeline, espec, parametros)


def treinar_quantis(linhas_treino: Sequence[Mapping], espec: md.EspecificacaoAtributos, colunas_locais: Sequence[str],
                    parametros: Mapping = PARAMETROS, *, contador: ContadorDeAjustes | None = None,
                    rotulo: str = "") -> dict[str, md.ModeloTreinado]:
    """q10, q50 e q90 com os mesmos atributos e parâmetros; só o `alpha` muda."""
    return {nome: treinar_modelo(linhas_treino, espec, colunas_locais, dict(parametros) | {"alpha": alfa},
                                 contador=contador, descricao=f"{rotulo}/{nome}".strip("/"))
            for nome, alfa in ALFAS.items()}


def prever_com_faixa(modelos: Mapping[str, md.ModeloTreinado], linhas: Sequence[Mapping]) -> dict:
    """Valores exibidos em log: q50 central; limites = mínimo e máximo das três saídas, depois do corte em zero."""
    relativos = {q: np.asarray(modelos[q].prever_relativo(linhas), dtype=float) for q in ALFAS}
    faixa = fp.prever_faixa(linhas, relativos["q10"], relativos["q50"], relativos["q90"])
    return faixa | {"relativo": relativos["q50"]}


if __name__ == "__main__":
    sys.exit(main())
