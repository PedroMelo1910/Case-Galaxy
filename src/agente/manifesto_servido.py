"""Manifesto versionado do pacote ATIVO e reversão (rollback) para a versão servida anterior.

`data/models/pacote_servido=v1.json` diz, sem constantes no código, qual conjunto o agente serve: os pacotes de vídeo, imagem e
texto (SHA-256 de cada `pacote.json`, identidade dos três modelos e, no vídeo, o limiar de baixa confiança vinculado), o
dataset que é a coleção de referência dos comparáveis, a base curada do histórico, o instante de ativação e a lista das versões
anteriores mantidas para rollback. Não há retorno silencioso aos modelos de desenvolvimento: manifesto ausente, inválido ou
com pacote que não confere com os hashes é um erro claro (`ErroManifesto`).

Duas formas de versão: `final` (pacotes de `data/models/final=v1/`, com `pacote.json`) e `desenvolvimento` (pacotes antigos de
`provisorio=v1`/`faixa=v1`, só para rollback). Ativar ou reverter grava o manifesto de forma atômica, guarda uma cópia do
anterior em `pacote_servido=v1.historico/` e nunca altera nem apaga os pacotes.
"""

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from src import modelo_desenvolvimento as md
from src import refit_treino as rt
from src.avaliacao import driver_lim as dl
from src.cache_gemini import gravar_json_atomico

ARQUIVO_MANIFESTO = Path("data/models/pacote_servido=v1.json")
PASTA_HISTORICO = Path("data/models/pacote_servido=v1.historico")
VERSAO_DO_MANIFESTO = 1
MODALIDADES = ("video", "imagem", "texto")
LAYOUTS = ("final", "desenvolvimento")
FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"


class ErroManifesto(RuntimeError):
    """Manifesto de pacote ativo ausente, inválido ou com artefato que não confere."""


def _sha256(arquivo: Path) -> str:
    return hashlib.sha256(Path(arquivo).read_bytes()).hexdigest()


def _iso(instante: datetime) -> str:
    return instante.astimezone(UTC).strftime(FORMATO_UTC)


def caminho_do_manifesto(raiz: Path) -> Path:
    return Path(raiz) / ARQUIVO_MANIFESTO


def ler(raiz: Path) -> dict:
    """Manifesto da raiz, com a estrutura conferida (não os arquivos; ver `validar_versao`)."""
    arquivo = caminho_do_manifesto(raiz)
    if not arquivo.is_file():
        raise ErroManifesto(f"manifesto de pacote ativo ausente ({ARQUIVO_MANIFESTO.as_posix()}); o agente não serve "
                            "modelos sem ele e não volta aos modelos de desenvolvimento em silêncio")
    try:
        manifesto = json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError) as falha:
        raise ErroManifesto(f"manifesto de pacote ativo ilegível: {type(falha).__name__}") from falha
    if manifesto.get("tipo") != "pacote_servido" or manifesto.get("versao_do_manifesto") != VERSAO_DO_MANIFESTO:
        raise ErroManifesto("manifesto de pacote ativo de tipo ou versão desconhecidos")
    for versao in (manifesto.get("ativo"), *manifesto.get("anteriores", [])):
        _conferir_estrutura(versao)
    return manifesto


def _conferir_estrutura(versao: object) -> None:
    if not isinstance(versao, dict) or versao.get("layout") not in LAYOUTS or not versao.get("rotulo"):
        raise ErroManifesto("versão servida sem rótulo ou com layout desconhecido")
    if set(versao.get("pacotes", {})) != set(MODALIDADES):
        raise ErroManifesto(f"versão {versao['rotulo']!r} deve ter pacotes de {list(MODALIDADES)}")
    for chave in ("dataset_de_referencia", "base_historica"):
        if not versao.get(chave, {}).get("pasta"):
            raise ErroManifesto(f"versão {versao['rotulo']!r} sem {chave}")


def versao_ativa(raiz: Path) -> dict:
    return ler(raiz)["ativo"]


# --- conferência dos artefatos -------------------------------------------------------------------------------------


def validar_versao(raiz: Path, versao: Mapping) -> None:
    """Confere pastas e hashes da versão; qualquer divergência é `ErroManifesto` (nada é servido pela metade)."""
    raiz = Path(raiz)
    for chave in ("dataset_de_referencia", "base_historica"):
        if not (raiz / versao[chave]["pasta"]).is_dir():
            raise ErroManifesto(f"{chave} ausente: {versao[chave]['pasta']}")
    dataset = versao["dataset_de_referencia"]
    if dataset.get("sha256_manifesto") and _sha256(raiz / dataset["pasta"] / "manifesto.json") != dataset["sha256_manifesto"]:
        raise ErroManifesto("o dataset de referência não confere com o manifesto registrado")
    conferir = _conferir_final if versao["layout"] == "final" else _conferir_desenvolvimento
    for modalidade, entrada in versao["pacotes"].items():
        try:
            conferir(raiz, modalidade, entrada)
        except (rt.ErroPacote, md.ErroModelo, dl.LimiarIncompativel, OSError, ValueError, KeyError) as falha:
            raise ErroManifesto(f"pacote de {modalidade} da versão {versao['rotulo']!r} não confere: {falha}") from falha


def _conferir_final(raiz: Path, modalidade: str, entrada: Mapping) -> None:
    pasta = raiz / entrada["pasta"]
    if not pasta.is_dir():
        raise rt.ErroPacote(f"pasta inexistente: {entrada['pasta']}")
    if _sha256(pasta / rt.ARQUIVO_PACOTE) != entrada["sha256_pacote_json"]:
        raise rt.ErroPacote("pacote.json não confere com o hash do manifesto")
    pacote = json.loads((pasta / rt.ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    if pacote["modalidade"] != modalidade or pacote["identidade_sha256"] != entrada["identidade_sha256"]:
        raise rt.ErroPacote("modalidade ou identidade do pacote diferem do manifesto")
    rt.verificar_integridade(pasta)
    limiar = entrada.get("limiar")
    if limiar is None:
        return
    if modalidade != "video":
        raise rt.ErroPacote("só o pacote de vídeo tem limiar de baixa confiança")
    arquivo = pasta / limiar["arquivo"]
    if _sha256(arquivo) != limiar["sha256_arquivo"]:
        raise rt.ErroPacote("o arquivo do limiar não confere com o manifesto")
    lido = dl.ler_limiar(arquivo)
    if lido.papel != "servido" or lido.n != dl.N_LIM:
        raise rt.ErroPacote("o limiar do pacote servido deve ter papel 'servido' e 100 vídeos da LIM")
    dl._exigir_compativel(lido, pacote["identidade_sha256"])  # recusa limiar de outro pacote


def _conferir_desenvolvimento(raiz: Path, modalidade: str, entrada: Mapping) -> None:
    for chave in ("pasta_q50", "pasta_faixa"):
        if not (raiz / entrada[chave]).is_dir():
            raise rt.ErroPacote(f"pasta inexistente: {entrada[chave]}")
    meta = json.loads((raiz / entrada["pasta_q50"] / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    if meta["sha256_modelo"] != entrada["sha256_modelo_q50"]:
        raise rt.ErroPacote("o q50 não confere com o hash do manifesto")
    if _sha256(raiz / entrada["pasta_q50"] / md.ARQUIVO_MODELO) != meta["sha256_modelo"]:
        raise rt.ErroPacote("o arquivo do modelo q50 não confere com o hash registrado nos metadados")


# --- construção, ativação e reversão -----------------------------------------------------------------------------


def entrada_final(raiz: Path, pasta: Path, *, diagnostico_limiar: Mapping | None = None) -> dict:
    """Entrada do manifesto para um pacote final (hashes lidos do disco); o limiar, se existir, vai vinculado."""
    raiz, pasta = Path(raiz), Path(pasta)
    pacote = json.loads((pasta / rt.ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    entrada = {"pasta": pasta.relative_to(raiz).as_posix(), "sha256_pacote_json": _sha256(pasta / rt.ARQUIVO_PACOTE),
               "identidade_sha256": pacote["identidade_sha256"], "versao_servida": pacote["versao_servida"],
               "limiar": None}
    arquivo = pasta / "limiar" / dl.ARQUIVO_LIMIAR.format(papel="servido")
    if arquivo.is_file():
        lido = dl.ler_limiar(arquivo)
        entrada["limiar"] = {
            "arquivo": arquivo.relative_to(pasta).as_posix(), "sha256_arquivo": _sha256(arquivo),
            "valor_log": lido.valor, "percentil": lido.percentil, "n": lido.n, "papel": lido.papel,
            "sha256_pacote": lido.sha256_pacote, "ids_sha256": lido.ids_sha256,
            "diagnostico_de_desenvolvimento": dict(diagnostico_limiar or {})}
    return entrada


def resumo_do_diagnostico(arquivo: Path) -> dict:
    """Leitura honesta do diagnóstico de desenvolvimento do limiar, copiada do arquivo gravado (nada digitado)."""
    d = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    return {"arquivo": Path(arquivo).name, "sha256_arquivo": _sha256(arquivo), "leitura": d["leitura"],
            "limiar_de_desenvolvimento_log": d["limiar"], "pacote_de_desenvolvimento": d["pacote"]["sha256_pacote"],
            "n_sinalizados": d["n_sinalizados"], "n_nao_sinalizados": d["n_nao_sinalizados"],
            "mae_sinalizados": d["mae_sinalizados"], "mae_nao_sinalizados": d["mae_nao_sinalizados"],
            "ic95_da_diferenca": d["bootstrap"]["ic95"], "n_validacao": d["populacao"]["n_validacao"]}


def montar_versao_final(raiz: Path, *, rotulo: str, pastas: Mapping[str, Path], pasta_dataset: Path, pasta_base: Path,
                        arquivo_diagnostico_limiar: Path | None, motivo: str) -> dict:
    raiz = Path(raiz)
    diagnostico = resumo_do_diagnostico(arquivo_diagnostico_limiar) if arquivo_diagnostico_limiar else None
    dataset = json.loads((Path(pasta_dataset) / "manifesto.json").read_text(encoding="utf-8"))
    return {
        "rotulo": rotulo, "layout": "final", "motivo": motivo,
        "pacotes": {m: entrada_final(raiz, Path(pastas[m]), diagnostico_limiar=diagnostico if m == "video" else None)
                    for m in MODALIDADES},
        "dataset_de_referencia": {"pasta": Path(pasta_dataset).relative_to(raiz).as_posix(),
                                  "sha256_dataset": dataset["sha256_dataset"],
                                  "sha256_manifesto": _sha256(Path(pasta_dataset) / "manifesto.json"),
                                  "papel": "coleção de referência dos comparáveis (refit final; sem LIM)"},
        "base_historica": {"pasta": Path(pasta_base).relative_to(raiz).as_posix(),
                           "sha256_manifesto": _sha256(Path(pasta_base) / "MANIFESTO_SHA256.txt")},
    }


def _gravar(raiz: Path, manifesto: Mapping, *, agora: datetime) -> Path:
    arquivo = caminho_do_manifesto(raiz)
    if arquivo.is_file():
        copia = Path(raiz) / PASTA_HISTORICO / f"{agora.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        copia.parent.mkdir(parents=True, exist_ok=True)
        with open(copia, "x", encoding="utf-8") as saida:  # cópia do manifesto anterior; nunca sobrescreve
            saida.write(arquivo.read_text(encoding="utf-8"))
    return gravar_json_atomico(arquivo, dict(manifesto))


def ativar(raiz: Path, versao: Mapping, *, agora: datetime | None = None, motivo: str = "",
           anterior_inicial: Mapping | None = None) -> dict:
    """Ativa `versao`; a ativa atual (se houver) passa à frente da lista de anteriores. Valida antes de gravar.

    Sem manifesto prévio, `anterior_inicial` (a versão servida até então) entra como a primeira para rollback."""
    agora = agora or datetime.now(UTC)
    raiz = Path(raiz)
    _conferir_estrutura(dict(versao))
    validar_versao(raiz, versao)
    atual = ler(raiz) if caminho_do_manifesto(raiz).is_file() else None
    if atual is None and anterior_inicial is not None:
        _conferir_estrutura(dict(anterior_inicial))
        validar_versao(raiz, anterior_inicial)
        atual = {"ativo": dict(anterior_inicial), "anteriores": [], "historico": [
            {"em_utc": _iso(agora), "acao": "registrar_anterior", "de": None, "para": anterior_inicial["rotulo"],
             "motivo": "versão servida antes do manifesto, mantida para rollback"}]}
    nova = dict(versao) | {"ativado_em_utc": _iso(agora)}
    anteriores = [] if atual is None else [atual["ativo"], *[a for a in atual["anteriores"]
                                                           if a["rotulo"] != versao["rotulo"]]]
    anteriores = [a for a in anteriores if a["rotulo"] != nova["rotulo"]]
    historico = ([] if atual is None else list(atual.get("historico", []))) + [
        {"em_utc": _iso(agora), "acao": "ativar", "de": atual["ativo"]["rotulo"] if atual else None,
         "para": nova["rotulo"], "motivo": motivo}]
    manifesto = {"tipo": "pacote_servido", "versao_do_manifesto": VERSAO_DO_MANIFESTO, "ativo": nova,
                 "anteriores": anteriores, "historico": historico}
    _gravar(raiz, manifesto, agora=agora)
    return manifesto


def reverter(raiz: Path, *, rotulo: str | None = None, agora: datetime | None = None, motivo: str = "") -> dict:
    """Rollback: torna ativa a versão anterior (a mais recente, ou a de `rotulo`), depois de validá-la inteira.

    A versão que estava ativa passa a ser a primeira das anteriores, de modo que o rollback também se desfaz. Os pacotes
    em disco não são tocados."""
    agora = agora or datetime.now(UTC)
    manifesto = ler(raiz)
    anteriores = manifesto["anteriores"]
    if not anteriores:
        raise ErroManifesto("não há versão anterior mantida para rollback")
    escolhida = next((a for a in anteriores if rotulo in (None, a["rotulo"])), None)
    if escolhida is None:
        raise ErroManifesto(f"versão anterior inexistente: {rotulo!r}; disponíveis: {[a['rotulo'] for a in anteriores]}")
    validar_versao(raiz, escolhida)
    restantes = [a for a in anteriores if a["rotulo"] != escolhida["rotulo"]]
    novo = {"tipo": "pacote_servido", "versao_do_manifesto": VERSAO_DO_MANIFESTO,
            "ativo": dict(escolhida) | {"ativado_em_utc": _iso(agora)}, "anteriores": [manifesto["ativo"], *restantes],
            "historico": [*manifesto.get("historico", []),
                          {"em_utc": _iso(agora), "acao": "reverter", "de": manifesto["ativo"]["rotulo"],
                           "para": escolhida["rotulo"], "motivo": motivo}]}
    _gravar(raiz, novo, agora=agora)
    return novo


def versao_de_desenvolvimento_servida_ate_18_10(raiz: Path) -> dict:
    """A versão servida antes do refit (pacotes de desenvolvimento, dataset de 453 linhas), registrada para rollback."""
    raiz = Path(raiz)
    nomes = {"video": "6f75e1255f73", "imagem": "2a4739584168", "texto": "18be8a9f7e86"}
    pacotes = {}
    for modalidade, nome in nomes.items():
        q50 = Path("data/models/provisorio=v1") / (f"{nome}-c62b81db")
        meta = json.loads((raiz / q50 / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
        pacotes[modalidade] = {"pasta_q50": q50.as_posix(), "pasta_faixa": f"data/models/faixa=v1/{nome}-{nome}-c62b81db",
                               "sha256_modelo_q50": meta["sha256_modelo"]}
    dataset = Path("data/features/snapshot=20261014T021437Z/desenvolvimento=v1/6f75e1255f73")
    return {
        "rotulo": "desenvolvimento=v1-servido-ate-18-10", "layout": "desenvolvimento",
        "motivo": "pacotes de desenvolvimento servidos antes do refit final (vídeo 6f75e1255f73 com 346 linhas de treino; "
                  "imagem e texto com 684); mantidos só para rollback",
        "pacotes": pacotes,
        "dataset_de_referencia": {"pasta": dataset.as_posix(),
                                  "sha256_manifesto": _sha256(raiz / dataset / "manifesto.json")},
        "base_historica": {"pasta": "data/processed/snapshot=20261014T021437Z/curadoria=v1"},
    }



def main(argv: list[str] | None = None) -> int:
    """`mostrar`, `ativar-final` e `reverter`. 0 = feito; 2 = recusado (mensagem no stderr)."""
    import argparse
    import sys

    from src.config import RAIZ_PROJETO

    parser = argparse.ArgumentParser(prog="python -m src.agente.manifesto_servido", description=__doc__)
    parser.add_argument("--raiz", type=Path, default=RAIZ_PROJETO)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("mostrar", help="resume a versão ativa e as anteriores")
    ativar_p = sub.add_parser("ativar-final", help="ativa o refit final e registra a versão de desenvolvimento servida até então")
    for modalidade in MODALIDADES:
        ativar_p.add_argument(f"--{modalidade}", type=Path, required=True, help=f"pasta do pacote final de {modalidade}")
    ativar_p.add_argument("--dataset", type=Path, required=True)
    ativar_p.add_argument("--base", type=Path, required=True)
    ativar_p.add_argument("--diagnostico-limiar", type=Path, required=True)
    ativar_p.add_argument("--rotulo", default="final=v1")
    ativar_p.add_argument("--motivo", required=True)
    reverter_p = sub.add_parser("reverter", help="rollback para a versão anterior mantida")
    reverter_p.add_argument("--para", help="rótulo da versão anterior (padrão: a mais recente)")
    reverter_p.add_argument("--motivo", required=True)
    args = parser.parse_args(argv)
    raiz = Path(args.raiz).resolve()
    try:
        if args.comando == "mostrar":
            manifesto = ler(raiz)
        elif args.comando == "reverter":
            manifesto = reverter(raiz, rotulo=args.para, motivo=args.motivo)
        else:
            versao = montar_versao_final(
                raiz, rotulo=args.rotulo, pastas={m: getattr(args, m).resolve() for m in MODALIDADES},
                pasta_dataset=args.dataset.resolve(), pasta_base=args.base.resolve(),
                arquivo_diagnostico_limiar=args.diagnostico_limiar.resolve(), motivo=args.motivo)
            inicial = None if caminho_do_manifesto(raiz).is_file() else versao_de_desenvolvimento_servida_ate_18_10(raiz)
            manifesto = ativar(raiz, versao, motivo=args.motivo, anterior_inicial=inicial)
    except (ErroManifesto, rt.ErroPacote, OSError, ValueError, KeyError) as erro:
        print(f"Manifesto recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({"ativo": manifesto["ativo"]["rotulo"], "ativado_em_utc": manifesto["ativo"].get("ativado_em_utc"),
                      "anteriores": [a["rotulo"] for a in manifesto["anteriores"]],
                      "pacotes": {m: p.get("versao_servida") or p.get("pasta_q50")
                                  for m, p in manifesto["ativo"]["pacotes"].items()}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
