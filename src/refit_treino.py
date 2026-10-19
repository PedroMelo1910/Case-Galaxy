"""Ajuste e empacotamento dos modelos servidos do refit final (D-D7, ARQUITETURA §6.5).

Especificação fixa, herdada do que já foi aprovado e avaliado: LightGBM configuração A (`modelo_modalidades.CONFIGURACAO_A`,
300 árvores, sementes inalteradas), q50 central e companheiros q10/q90 de `faixa_previsao.treinar_companheiros` (só o
`alpha` muda). As listas de atributos são as dos pacotes de desenvolvimento aprovados (vídeo completo com `duracao_s`;
imagem e texto sem `duracao_s`, sem preditor novo). PCA (8 componentes por bloco), códigos de categoria e referência do
setor são ajustados só nas linhas do refit (`treino` do contexto "final"); a LIM nunca entra. Nenhuma busca, calibração,
parada antecipada ou seleção: os rótulos de teste só servem de alvo do ajuste, e nada aqui mede desempenho.

Cada pacote é uma pasta nova e imutável em `data/models/final=v1/<modalidade>-<dataset>-<parâmetros>/` com `q10`, `q50` e
`q90` (cada um com `modelo.joblib` e `metadados.json`), `pacote.json`, `ids.json`, as previsões em memória usadas na
conferência de recarga e `manifesto_arquivos.json`. Reexecutar reaproveita o pacote existente se os hashes conferirem e
nunca sobrescreve. A cobertura gravada vem do procedimento de avaliação (`refit_cobertura`), não do artefato.
"""

import argparse
import hashlib
import io
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from src import dataset_desenvolvimento as ds
from src import faixa_previsao as fp
from src import familias_atributos as fa
from src import modelo_desenvolvimento as md
from src import modelo_modalidades as mm
from src import refit_cobertura as rc
from src.refit_dados import caminho_relativo
from src.avaliacao import driver_lim as dl
from src.config import RAIZ_PROJETO

VERSAO = "1"
PASTA_RAIZ = "final=v1"
SNAPSHOT = "snapshot=20261014T021437Z"
MODALIDADES = ("video", "imagem", "texto")
REFERENCIAS_APROVADAS = {  # pacotes de desenvolvimento aprovados: listas de atributos e parâmetros que o refit repete
    "video": "provisorio=v1/1f4ef5b5eac0-c62b81db", "imagem": "provisorio=v1/2a4739584168-c62b81db",
    "texto": "provisorio=v1/18be8a9f7e86-c62b81db"}
TIPO_DO_DATASET = {"video": "dataset_desenvolvimento", "imagem": "dataset_modalidade", "texto": "dataset_modalidade"}
MODULOS_DO_AJUSTE = (
    "src/modelo_desenvolvimento.py", "src/modelo_modalidades.py", "src/faixa_previsao.py", "src/preprocessador.py",
    "src/familias_atributos.py", "src/dataset_desenvolvimento.py", "src/dataset_modalidades.py", "src/historico.py",
    "src/refit_dados.py", "src/refit_treino.py", "src/refit_cobertura.py", "src/avaliacao/contextos_finais.py",
    "src/avaliacao/declaracao_faixa.py", "src/avaliacao/driver_lim.py", "src/avaliacao/limiar_lim.py")
BIBLIOTECAS = ("lightgbm", "scikit-learn", "numpy", "scipy", "pyarrow", "joblib", "torch", "transformers")
ARQUIVO_PACOTE, ARQUIVO_IDS, ARQUIVO_PREDICOES = "pacote.json", "ids.json", "predicoes_em_memoria.npz"
ARQUIVO_MANIFESTO = "manifesto_arquivos.json"
ARQUIVO_PREPROCESSADOR = "preprocessador.joblib"
PRAZO_DAS_ANOTACOES_DOS_ALVOS = "progresso.json"
LIMITACOES = [
    "Refit final: o modelo foi ajustado com todos os vídeos anotados da amostra congelada, inclusive os 452 vídeos de "
    "teste; a LIM ficou fora. Não existe conjunto independente para este artefato.",
    "A cobertura da faixa é a do procedimento de avaliação (consulta única, 452 vídeos), não foi medida neste artefato "
    "e não é garantia para vídeos novos; ficou abaixo dos 80% pretendidos.",
    "Métricas calculadas sobre os vídeos do ajuste não são desempenho independente e nunca devem ser reportadas como tal.",
    "Avaliação retrospectiva aproximada: curtidas lidas na coleta, histórico com defasagem de 30 dias, 16 canais, 82% "
    "automotivo; a Camada 1 local do TAPI §6.3 não foi atendida (desvio declarado).",
    "Os limites da faixa são o mínimo e o máximo das três saídas não negativas, com o q50 central (sem CQR).",
]


class ErroPacote(ValueError):
    """Dataset, população, especificação ou pacote incompatível com o refit final."""


def _iso(instante: datetime) -> str:
    return instante.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def sha256_arquivo(arquivo: Path) -> str:
    return sha256_bytes(Path(arquivo).read_bytes())


def sha256_ids(ids: Sequence[str]) -> str:
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def dependencias() -> dict:
    return {"python": platform.python_version(), "plataforma": platform.platform(),
            **{nome: version(nome) for nome in BIBLIOTECAS}}


def hashes_dos_modulos(raiz: Path = RAIZ_PROJETO) -> dict:
    return {m: sha256_arquivo(Path(raiz) / m) for m in MODULOS_DO_AJUSTE}


def chave_dos_parametros(parametros: Mapping) -> str:
    return sha256_bytes(json.dumps(dict(parametros), sort_keys=True).encode())[:8]


def id_do_pacote(modalidade: str, manifesto: Mapping, parametros: Mapping) -> str:
    return f"{modalidade}-{manifesto['sha256_dataset'][:12]}-{chave_dos_parametros(parametros)}"


# --- conferências antes do ajuste ----------------------------------------------------------------------------------


def conferir_dataset(modalidade: str, linhas: Sequence[Mapping], manifesto: Mapping) -> md.EspecificacaoAtributos:
    """Dataset do refit: tipo e modalidade certos, contexto "final", todas as linhas `treino`; devolve a especificação."""
    if modalidade not in MODALIDADES:
        raise ErroPacote(f"modalidade desconhecida: {modalidade!r}")
    if manifesto.get("tipo") != TIPO_DO_DATASET[modalidade]:
        raise ErroPacote(f"tipo de dataset inesperado para {modalidade}: {manifesto.get('tipo')!r}")
    if modalidade != "video" and manifesto.get("modalidade") != modalidade:
        raise ErroPacote(f"o dataset é da modalidade {manifesto.get('modalidade')!r}, não de {modalidade!r}")
    contexto = manifesto["entradas"]["contexto"]
    if contexto["nome"] != "final" or contexto["canais_excluidos"]:
        raise ErroPacote("o refit exige o contexto 'final' sem canais excluídos")
    usos = {l["uso_no_contexto"] for l in linhas}
    if usos != {md.USO_TREINO}:
        raise ErroPacote(f"no refit final toda linha deve ser treino; encontrado {sorted(usos)}")
    espec = md.EspecificacaoAtributos(**manifesto["especificacao"])
    if modalidade != "video":
        if fa.hash_da_especificacao(espec) != manifesto["especificacao_sha256"] or "duracao_s" in espec.escalares:
            raise ErroPacote("a especificação do dataset não é a operacional da modalidade (sem duracao_s)")
    return espec


def conferir_contra_referencia(espec: md.EspecificacaoAtributos, pasta_referencia: Path) -> dict:
    """Mesmas listas de atributos, na mesma ordem, e exatamente a configuração A do pacote aprovado de referência."""
    meta = json.loads((Path(pasta_referencia) / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    ref = meta["especificacao"]
    if (list(espec.escalares) != list(ref["escalares"]) or list(espec.categoricas) != list(ref["categoricas"])
            or dict(espec.embeddings) != dict(ref["embeddings"])):
        raise ErroPacote("as listas de atributos diferem das do pacote aprovado de referência")
    parametros = mm.parametros_de_referencia(pasta_referencia)
    return {"pasta": caminho_relativo(pasta_referencia), "sha256_metadados": sha256_arquivo(
        Path(pasta_referencia) / md.ARQUIVO_METADADOS), "sha256_modelo": meta["sha256_modelo"],
        "parametros": parametros}


def aptas_ao_ajuste(linhas: Sequence[Mapping]) -> tuple[list[dict], list[str]]:
    """(linhas com `ref_log` e `y_rel`, ids excluídos por falta deles); os excluídos são contados, nunca preenchidos."""
    aptas = [dict(l) for l in linhas if l.get("ref_log") is not None and l.get("y_rel") is not None]
    ids_aptas = {l["video_id"] for l in aptas}
    return aptas, [l["video_id"] for l in linhas if l["video_id"] not in ids_aptas]


# --- identidades de entrada ----------------------------------------------------------------------------------------


def identidade_das_anotacoes(pasta_cache: Path, ids: Sequence[str]) -> dict:
    """Identidade do conjunto de anotações usadas: chave de compatibilidade e SHA-256 do conjunto de arquivos."""
    linhas, chaves = [], set()
    for video_id in sorted(ids):
        bruto = (Path(pasta_cache) / f"{video_id}.json").read_bytes()
        chaves.add(json.loads(bruto)["chave_compatibilidade"])
        linhas.append(f"{video_id}:{sha256_bytes(bruto)}")
    if len(chaves) != 1:
        raise ErroPacote(f"anotações com chaves de compatibilidade diferentes: {len(chaves)}")
    return {"n": len(linhas), "chave_compatibilidade": chaves.pop(), "sha256_conjunto": sha256_bytes("\n".join(linhas).encode())}


def fontes_de_dados(manifesto: Mapping, pasta_dataset: Path, pasta_legenda_capa: Path, pasta_base: Path,
                    pasta_amostra: Path, anotacoes: Mapping | None) -> dict:
    fontes = {
        "dataset": {"pasta": caminho_relativo(pasta_dataset), "sha256": manifesto["sha256_dataset"],
                    "sha256_manifesto": sha256_arquivo(Path(pasta_dataset) / ds.ARQUIVO_MANIFESTO),
                    "versao": manifesto["versao"], "tipo": manifesto["tipo"]},
        "historico_n1": manifesto["entradas"]["historico"],
        "base_curada": {"pasta": caminho_relativo(pasta_base), "snapshot_id": manifesto["entradas"]["base"]["snapshot_id"],
                        "sha256_manifesto": sha256_arquivo(Path(pasta_base) / "MANIFESTO_SHA256.txt")
                        if (Path(pasta_base) / "MANIFESTO_SHA256.txt").exists() else None,
                        "sha256_manifesto_registrado": manifesto["entradas"]["base"]["sha256_manifesto"]},
        "particao": manifesto["entradas"]["particao"],
        "amostra": manifesto["entradas"]["amostra"],
        "legenda_capa": {"pasta": caminho_relativo(pasta_legenda_capa),
                         "sha256_legenda": sha256_arquivo(Path(pasta_legenda_capa) / "legenda.parquet"),
                         "sha256_capa": sha256_arquivo(Path(pasta_legenda_capa) / "capa.parquet"),
                         "sha256_manifesto": sha256_arquivo(Path(pasta_legenda_capa) / "manifesto.json")},
    }
    if anotacoes is not None:
        fontes["anotacoes_do_gemini"] = anotacoes | {"progresso_sha256": sha256_arquivo(
            Path(pasta_amostra) / PRAZO_DAS_ANOTACOES_DOS_ALVOS)}
    if "codificador_transcricao" in manifesto["entradas"]:
        fontes["codificador_transcricao"] = manifesto["entradas"]["codificador_transcricao"]
    return fontes


def retencao_herdada(manifesto: Mapping, pasta_amostra: Path, pasta_lim: Path) -> dict:
    """Menor prazo do dataset e os prazos de cada origem; o pacote derivado não vive além do menor deles."""
    progresso = json.loads((Path(pasta_amostra) / PRAZO_DAS_ANOTACOES_DOS_ALVOS).read_text(encoding="utf-8"))
    lim = json.loads((Path(pasta_lim) / "amostra.json").read_text(encoding="utf-8"))
    prazos = {"dataset_menor_prazo": manifesto["retencao"]["apagar_ate_utc"],
              "anotacoes_dos_alvos": progresso["retencao"]["apagar_ate_utc"],
              "lim_legenda_capa_e_base": lim["retencao"]["apagar_ate_utc"]}
    return prazos | {"apagar_ate_utc": min(prazos.values()),
                     "regra": "dados derivados herdam o menor prazo; modelo e metadados do pacote devem ser revistos "
                              "ou apagados com os dados de origem (D-B9)"}


# --- ajuste ----------------------------------------------------------------------------------------------------------


def _serializar_preprocessador(modelo: md.ModeloTreinado) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(modelo.pipeline.named_steps["pre"], buffer)
    return buffer.getvalue()


def _salvar_modelo(modelo: md.ModeloTreinado, pasta: Path, metadados: Mapping) -> dict:
    md.salvar(modelo, pasta, metadados=metadados)
    meta = json.loads((pasta / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    (pasta / ARQUIVO_PREPROCESSADOR).write_bytes(_serializar_preprocessador(modelo))
    return {"pasta": pasta.name, "sha256_modelo": meta["sha256_modelo"],
            "sha256_metadados": sha256_arquivo(pasta / md.ARQUIVO_METADADOS),
            "sha256_preprocessador": sha256_arquivo(pasta / ARQUIVO_PREPROCESSADOR)}


def _populacao(aptas: Sequence[Mapping], excluidos: Sequence[str], manifesto: Mapping) -> dict:
    ids = [l["video_id"] for l in aptas]
    canais = sorted({l["channel_id"] for l in aptas})
    por_setor: dict[str, int] = {}
    for l in aptas:
        por_setor[str(l["setor"])] = por_setor.get(str(l["setor"]), 0) + 1
    return {"n_linhas": len(ids), "n_canais": len(canais), "por_setor": dict(sorted(por_setor.items())),
            "por_bloco": dict(sorted({b: sum(l["bloco"] == b for l in aptas) for b in {l["bloco"] for l in aptas}}.items())),
            "sha256_ids": sha256_ids(ids), "sha256_canais": sha256_ids(canais),
            "excluidos_por_falta_de_referencia_ou_alvo": {"n": len(excluidos), "sha256_ids": sha256_ids(sorted(excluidos))},
            "reconciliacao_do_dataset": manifesto.get("reconciliacao_anotacoes", manifesto.get("reconciliacao")),
            "n_treino": len(ids), "n_validacao": 0,
            "nota": "n_validacao = 0: o refit não tem conjunto de validação nem de avaliação"}


def _metadados_comuns(modalidade: str, nivel: str, manifesto: Mapping, populacao: Mapping, retencao: Mapping) -> dict:
    return {"status": "final_refit", "modalidade": modalidade, "nivel_do_modelo": nivel,
            "dataset": {"sha256": manifesto["sha256_dataset"], "versao": manifesto["versao"],
                        "entradas": manifesto["entradas"]},
            "populacao": dict(populacao), "retencao": dict(retencao), "limitacoes": LIMITACOES,
            "semente": md.SEMENTE,
            "alvo": "y_rel = log1p(curtidas) − ref_log; previsão absoluta = ref_log + saída, cortada em 0"}


def treinar_modelos(aptas: Sequence[Mapping], espec: md.EspecificacaoAtributos, parametros: Mapping) -> dict:
    q50 = md.treinar(aptas, espec, parametros=parametros)
    return {"q50": q50} | fp.treinar_companheiros(aptas, q50)


def nome_para_nivel(modalidade: str) -> str:
    return "N4: modelo completo de vídeo (refit final)" if modalidade == "video" else (
        fa.nivel_do_modelo(modalidade).replace("modelo reduzido; desenvolvimento, provisório", "modelo reduzido; refit final"))


def treinar_pacote(modalidade: str, pasta_dataset: Path, saida_raiz: Path, *, raiz_projeto: Path = RAIZ_PROJETO,
                   arquivo_principal: Path, arquivo_sha_protocolo: Path, pasta_referencia: Path | None = None) -> dict:
    """Ajusta q10/q50/q90, grava o pacote imutável e confere a recarga no mesmo processo. Idempotente."""
    raiz = Path(raiz_projeto)
    linhas, manifesto = ds.carregar_dataset(pasta_dataset)
    espec = conferir_dataset(modalidade, linhas, manifesto)
    referencia = Path(pasta_referencia) if pasta_referencia else raiz / "data/models" / REFERENCIAS_APROVADAS[modalidade]
    ref = conferir_contra_referencia(espec, referencia)
    parametros = ref["parametros"]
    destino = Path(saida_raiz) / PASTA_RAIZ / id_do_pacote(modalidade, manifesto, parametros)
    if destino.exists():
        return verificar_integridade(destino) | {"reaproveitado": True}
    aptas, excluidos = aptas_ao_ajuste(linhas)
    if not aptas:
        raise ErroPacote("nenhuma linha apta ao ajuste")
    lim = {x["video_id"] for x in json.loads((raiz / "data/annotations/gemini" / SNAPSHOT / "lim=v1" / "amostra.json")
                                              .read_text(encoding="utf-8"))["lista"]}
    if lim & {l["video_id"] for l in aptas}:
        raise ErroPacote("a LIM não pode entrar no ajuste")
    modelos = treinar_modelos(aptas, espec, parametros)
    nivel = nome_para_nivel(modalidade)
    populacao = _populacao(aptas, excluidos, manifesto)
    pasta_amostra, pasta_lim = raiz / "data/annotations/gemini" / SNAPSHOT / "amostra=v2", raiz / "data/annotations/gemini" / SNAPSHOT / "lim=v1"
    retencao = retencao_herdada(manifesto, pasta_amostra, pasta_lim)
    comuns = _metadados_comuns(modalidade, nivel, manifesto, populacao, retencao)
    parcial = Path(tempfile.mkdtemp(prefix=".pacote.", suffix=".parcial", dir=_garantir(destino.parent)))
    try:
        registro_modelos = {}
        for nome, modelo in modelos.items():
            extra = {"tipo": f"modelo_final_{nome}", "alfa": {"q10": fp.ALFA_INFERIOR, "q50": fp.ALFA_CENTRAL,
                                                               "q90": fp.ALFA_SUPERIOR}[nome]}
            registro_modelos[nome] = _salvar_modelo(modelo, parcial / nome, comuns | extra)
        previsoes = {nome: modelo.prever_relativo(aptas) for nome, modelo in modelos.items()}
        np.savez(parcial / ARQUIVO_PREDICOES, **previsoes)
        ids = [l["video_id"] for l in aptas]
        (parcial / ARQUIVO_IDS).write_text(json.dumps({"ajuste": ids, "excluidos": sorted(excluidos)}), encoding="utf-8")
        identidade = dl.identidade_do_pacote({n: r["sha256_modelo"] for n, r in registro_modelos.items()})
        pacote = _registro_do_pacote(
            modalidade, nivel, destino.name, manifesto, populacao, retencao, registro_modelos, identidade, ref, parametros,
            raiz, pasta_dataset, arquivo_principal, arquivo_sha_protocolo, aptas, espec, modelos["q50"])
        (parcial / ARQUIVO_PACOTE).write_text(json.dumps(pacote, ensure_ascii=False, indent=1), encoding="utf-8")
        _gravar_manifesto_de_arquivos(parcial)
        parcial.rename(destino)
    except BaseException:
        shutil.rmtree(parcial, ignore_errors=True)
        raise
    reconferencia = conferir_recarga(destino, aptas, previsoes)
    return {"pacote": destino.as_posix(), "id": destino.name, "identidade_sha256": identidade,
            "n_linhas": len(aptas), "recarga_identica_no_processo": reconferencia, "reaproveitado": False}


def _contexto(manifesto: Mapping) -> dict:
    contexto = manifesto["entradas"]["contexto"]
    return {"nome": contexto["nome"], "treino_ate_utc": contexto["treino_ate_utc"],
            "canais_excluidos": list(contexto["canais_excluidos"]),
            "defasagem_dias": manifesto["entradas"]["defasagem_dias"],
            "regra": "treino_ate_utc = última publicação entre os alvos da amostra; toda linha é treino; sem intervalo "
                     "excluído, sem bloco avaliado e sem canal retido"}


def _garantir(pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _registro_do_pacote(modalidade, nivel, id_, manifesto, populacao, retencao, registro_modelos, identidade, ref,
                        parametros, raiz, pasta_dataset, arquivo_principal, arquivo_sha_protocolo, aptas, espec,
                        q50) -> dict:
    pasta_lc = raiz / "data/features" / SNAPSHOT / "legenda_capa=v1"
    pasta_base = raiz / "data/processed" / SNAPSHOT / "curadoria=v1"
    pasta_amostra = raiz / "data/annotations/gemini" / SNAPSHOT / "amostra=v2"
    anotacoes = identidade_das_anotacoes(pasta_amostra / "cache", [l["video_id"] for l in aptas]) \
        if modalidade == "video" else None
    protocolo = Path(arquivo_sha_protocolo).read_text(encoding="utf-8").split()[0]
    declaracao = rc.declaracao_do_procedimento(arquivo_principal, modalidade, protocolo_sha256=protocolo)
    rc.conferir(declaracao)
    return {
        "tipo": "pacote_final", "versao": VERSAO, "id": id_, "gerado_em_utc": _iso(datetime.now(UTC)),
        "modalidade": modalidade, "nivel_do_modelo": nivel, "versao_servida": f"{PASTA_RAIZ}/{id_}",
        "modelos": registro_modelos, "identidade_sha256": identidade,
        "populacao": populacao,
        "contexto": _contexto(manifesto),
        "especificacao": {
            "escalares": list(espec.escalares), "categoricas": list(espec.categoricas),
            "embeddings": dict(espec.embeddings), "sha256": fa.hash_da_especificacao(espec),
            "atributos_de_saida": q50.nomes_atributos, "n_atributos_de_saida": len(q50.nomes_atributos),
            "duracao_s_e_preditor": "duracao_s" in espec.escalares,
            "referencia_aprovada": ref | {"parametros": "idênticos à CONFIGURACAO_A"}},
        "parametros": parametros, "sementes": {"lightgbm": parametros["random_state"], "preprocessador": md.SEMENTE,
                                              "pca_componentes_por_bloco": ds.COMPONENTES_PCA},
        "regra_da_faixa": "q50 central; limites = mínimo e máximo das três saídas reconstruídas e não negativas; sem CQR",
        "alvo": "y_rel = log1p(curtidas) − ref_log; previsão = ref_log + saída, cortada em 0",
        "cobertura_do_procedimento": declaracao,
        "fontes": fontes_de_dados(manifesto, pasta_dataset, pasta_lc, pasta_base, pasta_amostra, anotacoes),
        "codigo": {"modulos_sha256": hashes_dos_modulos(raiz)}, "dependencias": dependencias(),
        "retencao": retencao, "limitacoes": LIMITACOES,
        "lim": {"fora_do_ajuste": True, "uso": "somente a largura da faixa do vídeo (limiar de baixa confiança), sem rótulos"
                if modalidade == "video" else "não usada: imagem e texto não têm limiar (sinal sempre ligado)"},
        "verificacao_de_recarga": {"predicoes_em_memoria": ARQUIVO_PREDICOES,
                                   "hash": f"em {ARQUIVO_MANIFESTO} (gravado por último)"},
    }


def _gravar_manifesto_de_arquivos(pasta: Path) -> None:
    arquivos = {p.relative_to(pasta).as_posix(): sha256_arquivo(p)
                for p in sorted(pasta.rglob("*")) if p.is_file() and p.name != ARQUIVO_MANIFESTO}
    (pasta / ARQUIVO_MANIFESTO).write_text(json.dumps({"arquivos": arquivos}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")


# --- integridade e recarga ---------------------------------------------------------------------------------------


def verificar_integridade(pasta: Path) -> dict:
    """Todos os arquivos do pacote conferem com o manifesto de arquivos e os modelos com o `pacote.json`."""
    pasta = Path(pasta)
    manifesto = json.loads((pasta / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))["arquivos"]
    divergentes = [nome for nome, esperado in manifesto.items()
                   if not (pasta / nome).is_file() or sha256_arquivo(pasta / nome) != esperado]
    atuais = {p.relative_to(pasta).as_posix() for p in pasta.rglob("*") if p.is_file() and p.name != ARQUIVO_MANIFESTO
              and not p.relative_to(pasta).parts[0] == "limiar"}
    extras = sorted(atuais - set(manifesto))
    if divergentes or extras:
        raise ErroPacote(f"pacote {pasta.name} não confere com o manifesto: divergentes={divergentes[:3]} extras={extras[:3]}")
    pacote = json.loads((pasta / ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    for nome, registro in pacote["modelos"].items():
        if sha256_arquivo(pasta / nome / md.ARQUIVO_MODELO) != registro["sha256_modelo"]:
            raise ErroPacote(f"modelo {nome} do pacote {pasta.name} não confere com o hash registrado")
    identidade = dl.identidade_do_pacote({n: r["sha256_modelo"] for n, r in pacote["modelos"].items()})
    if identidade != pacote["identidade_sha256"]:
        raise ErroPacote("a identidade do pacote não confere com os hashes dos três modelos")
    return {"pacote": pasta.as_posix(), "id": pasta.name, "identidade_sha256": identidade,
            "n_linhas": pacote["populacao"]["n_linhas"], "integridade": "ok"}


def conferir_recarga(pasta: Path, aptas: Sequence[Mapping], previsoes_em_memoria: Mapping[str, np.ndarray]) -> bool:
    """Recarrega do disco (no mesmo processo) e exige previsões idênticas bit a bit nas mesmas linhas."""
    for nome, esperado in previsoes_em_memoria.items():
        recarregado = md.carregar(Path(pasta) / nome)
        if not np.array_equal(recarregado.prever_relativo(aptas), esperado):
            raise ErroPacote(f"o modelo {nome} recarregado não reproduz as previsões")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.refit_treino", description=__doc__)
    parser.add_argument("--modalidade", choices=MODALIDADES, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="pasta do dataset final da modalidade")
    parser.add_argument("--saida", type=Path, default=RAIZ_PROJETO / "data" / "models")
    parser.add_argument("--principal", type=Path, required=True, help="principal.json da avaliação final")
    parser.add_argument("--sha-protocolo", type=Path, required=True, help="protocolo_final.sha256")
    args = parser.parse_args(argv)
    try:
        resultado = treinar_pacote(args.modalidade, args.dataset, args.saida, arquivo_principal=args.principal,
                                   arquivo_sha_protocolo=args.sha_protocolo)
    except (ErroPacote, ds.ErroDataset, rc.ErroCobertura, md.ErroModelo, ValueError, KeyError, OSError) as erro:
        print(f"Refit recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
