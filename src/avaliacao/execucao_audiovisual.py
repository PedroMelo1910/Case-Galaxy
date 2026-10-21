"""Execução do protocolo audiovisual sobre os dados reais, em etapas separadas e verificáveis."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import pyarrow.parquet as pq

from src.avaliacao import (
    audiovisual as ae,
    experimentos as af,
    protocolo_audiovisual as ap,
    protocolo_audiovisual as at,
    protocolos as tr,
    relatorio as rl,
    validacao as ad,
    validacao as ah,
    validacao as apo,
)
from src.config import RAIZ_PROJETO
from src.dados import curadoria, historico as hs, particoes
from src.modelos import modalidades as rat, refit as ra, refit as ral, refit as rd, refit as rv
from src.proveniencia import sha256_arquivo as _sha


# ──── Execução do protocolo audiovisual sobre os dados reais, em etapas separadas e verificáveis.
ARQUIVO_DESENVOLVIMENTO = "desenvolvimento_audiovisual.parquet"
ARQUIVO_COMPLETO = "dataset_audiovisual.parquet"
ARQUIVO_MANIFESTO = "manifesto.json"
RESULTADO_EXPERIMENTO = at.NOME_RESULTADO_DESENVOLVIMENTO
LIVRO_DE_AJUSTES = "livro_de_ajustes.jsonl"
PASTA_EXPERIMENTO = "experimento_audiovisual=v1"
BLOCOS_DE_DESENVOLVIMENTO = ("treino", "validacao")
FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"
CHAVES_NAO_SERIALIZAVEIS = ("ids_elegiveis", "dobras_prontas")


@dataclass(frozen=True)
class Caminhos:
    produto: Path
    dataset: Path  # pasta audiovisual_local=v<n>/<sha12> com os dois parquets e o manifesto
    pasta_base: Path  # curadoria=v<n>
    arquivo_particao: Path
    pasta_protocolo: Path
    pasta_resultados: Path
    pasta_modelos: Path
    lock: Path
    ids_originais_teste: Path | None = None  # JSON com a lista de IDs de teste da amostra original
    substituicoes: Path | None = None  # JSON {id_original: id_substituto}
    amostra_lim: Path | None = None  # JSON da LIM (lista e retenção)
    tabela_lim: Path | None = None  # parquet com os atributos locais dos vídeos da LIM


def conferir_hashes_do_dataset(c: Caminhos) -> dict:
    """Confere os dois parquets contra o manifesto SOMENTE por bytes, sem interpretar nenhum conteúdo."""
    manifesto = json.loads((c.dataset / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))
    if _sha(c.dataset / ARQUIVO_DESENVOLVIMENTO) != manifesto["sha256_desenvolvimento"]:
        raise ae.ErroExperimento("a visão de desenvolvimento não confere com o hash do manifesto")
    if _sha(c.dataset / ARQUIVO_COMPLETO) != manifesto["sha256_completo"]:
        raise ae.ErroExperimento("a tabela completa não confere com o hash do manifesto")
    return manifesto


def carregar_dataset(c: Caminhos) -> tuple[list[dict], list[dict], dict]:
    """(visão de desenvolvimento, tabela completa, manifesto): hashes por bytes primeiro, parse depois."""
    manifesto = conferir_hashes_do_dataset(c)
    return (pq.read_table(c.dataset / ARQUIVO_DESENVOLVIMENTO).to_pylist(),
            pq.read_table(c.dataset / ARQUIVO_COMPLETO).to_pylist(), manifesto)


def conferir_entradas_com_manifesto(c: Caminhos, manifesto: Mapping) -> dict:
    """A partição informada é a registrada no handoff do dataset (nada fixo em v1) e casa com a base curada."""
    registrada = manifesto.get("entradas", {}).get("particao", {})
    if not registrada.get("sha256_arquivo"):
        raise ae.ErroExperimento("o manifesto do dataset não registra a partição usada")
    if _sha(c.arquivo_particao) != registrada["sha256_arquivo"]:
        raise ae.ErroExperimento("a partição informada difere da registrada no manifesto do dataset")
    particao = particoes.ler_particao(c.arquivo_particao, pasta_base=c.pasta_base)  # confere partição × base curada
    if registrada.get("sha256_atribuicao") and particao["sha256_atribuicao"] != registrada["sha256_atribuicao"]:
        raise ae.ErroExperimento("a atribuição da partição difere da registrada no manifesto do dataset")
    return {"sha256_particao": registrada["sha256_arquivo"], "sha256_atribuicao": particao.get("sha256_atribuicao"),
            "snapshot_id": particao.get("base", {}).get("snapshot_id")}


def atribuicao_ativa(particao: Mapping, completa: Sequence[Mapping]) -> dict[str, str]:
    """Bloco de cada ID: a partição original mais os blocos da amostra ativa (substitutos incluídos).

    Divergência entre a partição e a tabela do dataset para o mesmo ID é erro."""
    atribuicao = dict(particao["atribuicao"])
    for l in completa:
        bloco = l.get("bloco")
        if not bloco:
            continue
        if l["video_id"] in atribuicao and atribuicao[l["video_id"]] != bloco:
            raise ae.ErroExperimento(f"{l['video_id']}: bloco da tabela difere do da partição")
        atribuicao[l["video_id"]] = bloco
    return atribuicao


def limite_de_desenvolvimento(linhas_dev: Sequence[Mapping]) -> datetime:
    """Fim da última janela de desenvolvimento: a maior publicação da visão de desenvolvimento."""
    return max(l["published_at"] for l in linhas_dev)


def base_de_posts(c: Caminhos, *, mascarar_teste: bool, limite_utc: datetime | None = None,
                  atribuicao: Mapping[str, str] | None = None) -> tuple[list[hs.Post], dict, dict]:
    """Posts, canais e partição. Em desenvolvimento os rótulos de teste são mascarados por data E por bloco."""
    particao = particoes.ler_particao(c.arquivo_particao, pasta_base=c.pasta_base)
    posts, canais = hs.carregar_base(c.pasta_base)
    if mascarar_teste:
        if limite_utc is None:
            raise ae.ErroExperimento("o mascaramento por data exige o fim da última janela de desenvolvimento")
        blocos = dict(atribuicao or particao["atribuicao"])
        posts = ah.mascarar_por_data_e_bloco(posts, blocos, limite_utc, BLOCOS_DE_DESENVOLVIMENTO)
        posts = ah.mascarar_rotulos(posts, blocos, {b for b in set(blocos.values()) if b.startswith("teste")})
    return posts, canais, particao


def especificacao_do_dataset(manifesto: Mapping, especificacao_desenvolvimento: Mapping) -> dict:
    return af.especificacao_total(especificacao_desenvolvimento, manifesto["especificacao_candidata_sem_selecao"])


def preparar(c: Caminhos, especificacao_desenvolvimento: Mapping, *, pisos: Mapping = apo.PISOS,
             ids_lim: Sequence[str] = ()) -> dict:
    """Portões, dobras, auditoria do histórico por dobra e listas de IDs; nenhum ajuste de modelo.

    Se `ids_lim` não vier e houver `amostra_lim`, os 100 IDs da LIM (sem rótulos) são usados na disjunção com a visão
    de desenvolvimento e com a tabela completa."""
    if not ids_lim and c.amostra_lim is not None:
        ids_lim = _ids_da_lim(c)[0]
    linhas_dev, completa, manifesto = carregar_dataset(c)
    entradas = conferir_entradas_com_manifesto(c, manifesto)
    visoes = apo.conferir_visoes(linhas_dev, completa, ids_lim)
    if set(ids_lim) & {l["video_id"] for l in completa}:
        raise apo.ErroPortao("a LIM aparece na tabela completa da amostra ativa")
    visoes["n_ids_lim_conferidos"] = len(set(ids_lim))
    particao = particoes.ler_particao(c.arquivo_particao, pasta_base=c.pasta_base)
    blocos = atribuicao_ativa(particao, completa)
    posts, canais, _ = base_de_posts(c, mascarar_teste=True, limite_utc=limite_de_desenvolvimento(linhas_dev),
                                     atribuicao=blocos)
    elegiveis, resumo = ae.filtrar_elegiveis(linhas_dev)
    excluidas = [l for l in linhas_dev if l["video_id"] in resumo["motivo_por_id"]]
    ordem = {l["video_id"]: l.get("ordem_amostra") for l in completa}
    duplicatas = apo.duplicatas_por_cosseno(elegiveis)
    removidos = set(duplicatas["ids_removidos"])
    elegiveis = [l for l in elegiveis if l["video_id"] not in removidos]
    estrutura = ad.construir(elegiveis)
    afirmacoes = ad.afirmacoes(estrutura, elegiveis, [l["video_id"] for l in elegiveis])
    dobras = ae.preparar_dobras(estrutura, elegiveis, posts, canais)
    piso = apo.verificar_piso(elegiveis, [d.avaliacao for d in dobras], pisos)
    total = especificacao_do_dataset(manifesto, especificacao_desenvolvimento)
    portoes = {
        "visoes": visoes, "piso": piso, "teto_de_exclusoes": apo.teto_de_exclusoes(len(linhas_dev), len(excluidas)),
        "incluidos_x_excluidos": apo.comparar_incluidos_excluidos(elegiveis, excluidas, ordem),
        "duplicatas_por_cosseno": duplicatas,
        "duplicatas_entre_desenvolvimento_e_teste": apo.pares_duplicados_entre(
            elegiveis, [l for l in completa if str(l.get("bloco", "")).startswith("teste")]),
        "sonda_de_canal": apo.sonda_de_canal(elegiveis),
        "equivalencia_da_duracao": apo.equivalencia_da_duracao(elegiveis)}
    return {
        "elegiveis": {k: v for k, v in resumo.items() if k != "motivo_por_id"} | {"n_removidas_por_duplicata": len(removidos)},
        "portoes": portoes, "afirmacoes": afirmacoes, "estrutura": ad.descrever(estrutura, elegiveis),
        "ids_elegiveis": [l["video_id"] for l in elegiveis], "dobras_prontas": dobras, "especificacao_total": total,
        "dobras": [{"dobra": d.indice, "ids_treino": [l["video_id"] for l in d.treino],
                    "ids_avaliacao": [l["video_id"] for l in d.avaliacao],
                    "sha256_ids_treino": ad.sha256_ids([l["video_id"] for l in d.treino]),
                    "sha256_ids_avaliacao": ad.sha256_ids([l["video_id"] for l in d.avaliacao]),
                    "exclusoes": d.exclusoes, "auditoria_historico": d.auditoria,
                    "comparacao_com_colunas_originais": d.comparacao_com_originais} for d in dobras],
        "identidade_do_dataset": {"sha256_desenvolvimento": manifesto["sha256_desenvolvimento"],
                                  "sha256_completo": manifesto["sha256_completo"],
                                  "sha256_manifesto": _sha(c.dataset / ARQUIVO_MANIFESTO),
                                  "extracao": manifesto["extracao"]["identidade_da_configuracao"],
                                  "retencao": manifesto["retencao"], "entradas": entradas}}


def _entradas_externas(c: Caminhos) -> dict[str, dict]:
    arquivos = {"ids_originais_teste": c.ids_originais_teste, "substituicoes": c.substituicoes}
    faltam = [n for n, a in arquivos.items() if a is None or not Path(a).is_file()]
    if faltam:
        raise ae.ErroExperimento(f"entradas externas ausentes: {faltam}")
    return {n: {"arquivo": Path(a).name, "sha256": _sha(a)} for n, a in arquivos.items()}


def congelar(c: Caminhos, preparo: Mapping) -> tuple[Path, str]:
    """Protocolo com identidade do dataset, lista de linhas, dobras, entradas externas e ambiente."""
    linhas = ap.descrever_linhas(preparo["ids_elegiveis"], {
        str(d["dobra"]): {"sha256_ids_treino": d["sha256_ids_treino"], "sha256_ids_avaliacao": d["sha256_ids_avaliacao"],
                          "n_treino": len(d["ids_treino"]), "n_avaliacao": len(d["ids_avaliacao"])}
        for d in preparo["dobras"]})
    total = preparo["especificacao_total"]
    return ap.congelar(
        c.produto, c.pasta_protocolo,
        dataset=dict(preparo["identidade_do_dataset"]) | {"portoes": rl.sanear(preparo["portoes"])}, linhas=linhas,
        dobras=preparo["estrutura"],
        especificacao_total={"escalares": total["escalares"], "categoricas": total["categoricas"],
                             "embeddings": total["embeddings"], "excluidas": total["excluidas"],
                             "colunas_por_familia": af.mapa_de_familias(total),
                             "configuracoes": af.resumo_das_configuracoes(total)},
        entradas_externas=_entradas_externas(c),
        ambiente=ap.ambiente_atual(c.produto, preparo["identidade_do_dataset"]["extracao"]))


def conferir_entradas_externas(c: Caminhos, congelado: Mapping) -> dict[str, str]:
    """Os JSON de IDs originais e de substituições têm o SHA-256 congelado; devolve os hashes (vão ao marcador)."""
    atuais = _entradas_externas(c)
    for nome, esperado in congelado["entradas_externas"].items():
        if atuais[nome]["sha256"] != esperado["sha256"]:
            raise ae.ErroExperimento(f"a entrada externa {nome} difere da congelada")
    return {n: v["sha256"] for n, v in atuais.items()}


def conferir_protocolo(c: Caminhos, preparo: Mapping) -> dict:
    """Protocolo íntegro; dobras recalculadas hoje iguais às congeladas; entradas externas com o mesmo SHA-256."""
    verificacao = ap.verificar(c.produto, c.pasta_protocolo)
    if not verificacao["integro"]:
        raise ae.ErroExperimento(f"protocolo divergente do disco: {verificacao}")
    congelado = json.loads((c.pasta_protocolo / ap.ARQUIVO).read_text(encoding="utf-8"))
    if congelado["estado"] != "congelado":
        raise ae.ErroExperimento("o protocolo não está congelado")
    conferir_entradas_externas(c, congelado)
    esperado = congelado["linhas"]
    if esperado["sha256_ids_elegiveis"] != ad.sha256_ids(sorted(preparo["ids_elegiveis"])):
        raise ae.ErroExperimento("as linhas elegíveis de hoje diferem das congeladas")
    for d in preparo["dobras"]:
        d_cong = esperado["por_dobra"][str(d["dobra"])]
        if (d_cong["sha256_ids_treino"], d_cong["sha256_ids_avaliacao"]) != (d["sha256_ids_treino"], d["sha256_ids_avaliacao"]):
            raise ae.ErroExperimento(f"as listas da dobra {d['dobra']} diferem das congeladas")
    return verificacao


def experimento(c: Caminhos, especificacao_desenvolvimento: Mapping, *, comando: str,
                pisos: Mapping = apo.PISOS) -> Path:
    """Confere tudo, segura o lock e grava `resultado_experimento.json` uma única vez."""
    preparo = preparar(c, especificacao_desenvolvimento, pisos=pisos)
    verificacao = conferir_protocolo(c, preparo)
    destino = c.pasta_resultados / RESULTADO_EXPERIMENTO
    notas: list[str] = []
    c.pasta_resultados.mkdir(parents=True, exist_ok=True)
    with tr.trava_pesada(c.lock, responsavel="MODELO", comando=comando, registro=notas):
        resultado = ae.avaliar(preparo["dobras_prontas"], preparo["especificacao_total"],
                               livro=c.pasta_resultados / LIVRO_DE_AJUSTES,
                               sonda=preparo["portoes"]["sonda_de_canal"])
    corpo = {"sha256_protocolo": verificacao["sha256"], "notas_do_lock": notas,
             "identidade_do_dataset": preparo["identidade_do_dataset"], "portoes": preparo["portoes"],
             "resultado": resultado, "selecao": resultado["selecao"]}
    with open(destino, "x", encoding="utf-8") as arquivo:
        json.dump(rl.sanear(corpo), arquivo, ensure_ascii=False, indent=1, allow_nan=False)
    return destino


def teste(c: Caminhos, especificacao_desenvolvimento: Mapping, *, resultado_desenvolvimento: Path, comando: str,
          pisos: Mapping = apo.PISOS) -> Path:
    """Avaliação descritiva única no teste temporal já exposto.

    Ordem: hashes por bytes (dataset, protocolo, entradas externas) → lock e memória → marcador → só então parse da
    tabela completa e leitura de qualquer alvo de teste. Falha técnica antes de métrica deixa o marcador reabrível;
    depois de métrica, não."""
    conferir_hashes_do_dataset(c)
    verificacao = ap.verificar(c.produto, c.pasta_protocolo)
    if not verificacao["integro"]:
        raise ae.ErroExperimento(f"protocolo divergente do disco: {verificacao}")
    congelado = json.loads((c.pasta_protocolo / ap.ARQUIVO).read_text(encoding="utf-8"))
    externas = conferir_entradas_externas(c, congelado)
    with tr.trava_pesada(c.lock, responsavel="MODELO", comando=comando):
        aberto = at.abrir_teste(c.pasta_protocolo, resultado_desenvolvimento, entradas_externas=externas,
                                livro=c.pasta_resultados / LIVRO_DE_AJUSTES)
        with at.sessao(aberto):
            preparo = preparar(c, especificacao_desenvolvimento, pisos=pisos)
            conferir_protocolo(c, preparo)
            _, completa, _ = carregar_dataset(c)
            elegiveis, resumo = ae.filtrar_elegiveis(completa)
            tabela = {l["video_id"]: l for l in elegiveis}
            posts, canais, particao = base_de_posts(c, mascarar_teste=False)
            blocos = atribuicao_ativa(particao, completa)
            ids_originais = json.loads(Path(c.ids_originais_teste).read_text(encoding="utf-8"))
            substituicoes = json.loads(Path(c.substituicoes).read_text(encoding="utf-8"))
            dobras = [at.contexto_de_teste(aberto, bloco, posts, canais, blocos, particao, tabela, list(tabela))
                      for bloco in at.BLOCOS_TESTE]
            ativos = [l["video_id"] for l in completa if str(l.get("bloco", "")).startswith("teste")]
            reconciliacao = at.reconciliar_ids_de_teste(ativos, resumo["motivo_por_id"], dobras)
            avaliados = [l["video_id"] for d in dobras for l in d.avaliacao]
            populacao = at.identificar_populacao(avaliados, ids_originais, substituicoes, ativos) | {
                "reconciliacao_de_ids_ativos": reconciliacao, "n_originais_no_arquivo_congelado": len(ids_originais),
                "entradas_externas_sha256": externas}
            return at.executar(aberto, dobras, preparo["especificacao_total"], populacao, saida=c.pasta_resultados)


def linhas_do_refit(c: Caminhos, ids_lim: Sequence[str]) -> tuple[list[dict], hs.ContextoTreino, list[str], list[dict]]:
    """Linhas do refit (contexto final, LIM fora), IDs sem referência e as linhas completas elegíveis."""
    _, completa, _ = carregar_dataset(c)
    elegiveis, _ = ae.filtrar_elegiveis(completa)
    ids = [l["video_id"] for l in elegiveis]
    posts, canais, particao = base_de_posts(c, mascarar_teste=False)
    n1, ctx = rd.n1_do_refit(posts, canais, atribuicao_ativa(particao, completa), ids, ids_lim=ids_lim)
    locais = {l["video_id"]: l for l in elegiveis}
    linhas = [{**{k: v for k, v in locais[n["video_id"]].items() if k not in n}, **n} for n in n1]
    return linhas, ctx, [l["video_id"] for l in linhas if l.get("ref_log") is None or l.get("y_rel") is None], elegiveis


def _prazos(c: Caminhos, manifesto: Mapping) -> dict[str, str]:
    base = json.loads((c.pasta_base / curadoria.PROVENIENCIA).read_text(encoding="utf-8"))["retencao"]["apagar_ate_utc"]
    dataset = manifesto["retencao"]["apagar_ate_utc"]
    prazos = {"dataset_v2": dataset, "base_de_posts": base, "extracao_local": dataset}
    copiadas = manifesto["retencao"].get("herdada_de", {}).get("anotacoes_originais_copiadas_do_piloto_min")
    if copiadas:  # anotações do Gemini copiadas do piloto: o prazo mais curto entre todas as fontes do dataset
        prazos["anotacoes_gemini"] = copiadas
    return prazos


def _total_do_protocolo(c: Caminhos) -> dict:
    t = json.loads((c.pasta_protocolo / ap.ARQUIVO).read_text(encoding="utf-8"))["especificacao_total"]
    return {k: t[k] for k in ("escalares", "categoricas", "embeddings", "excluidas")}


def refit(c: Caminhos, config_id: str, *, resultado_desenvolvimento: Path, avaliacao_teste: Path | None,
          ids_lim: Sequence[str], comando: str) -> dict:
    """Ajusta a configuração escolhida (lida do resultado verificado por hash) com todas as linhas completas."""
    corpo, sha_resultado, sha_protocolo = at.ler_resultado_verificado(
        c.pasta_protocolo, resultado_desenvolvimento, livro=c.pasta_resultados / LIVRO_DE_AJUSTES)
    selecao = corpo["selecao"]
    manifesto = conferir_hashes_do_dataset(c)
    linhas, ctx, excluidos, elegiveis = linhas_do_refit(c, ids_lim)
    dimensao = {n: len(next(l[n] for l in elegiveis if l.get(n) is not None)) for n in ("avl_emb_media", "avl_emb_gancho")}
    avaliacao = cobertura = None
    if avaliacao_teste is not None:
        resultado = json.loads(Path(avaliacao_teste).read_text(encoding="utf-8"))
        if resultado.get("sha256_resultado_desenvolvimento") != sha_resultado:
            raise ae.ErroExperimento("a avaliação no teste pertence a outro resultado do desenvolvimento")
        avaliacao = {"arquivo": Path(avaliacao_teste).name, "sha256": _sha(avaliacao_teste), "rotulo": at.ROTULO}
        s = resultado["resultado"]["por_configuracao"][config_id]
        cobertura = ra.declaracao_de_cobertura(s["cobertura"], s.get("cobertura_por_setor"),
                                               resultado_arquivo=avaliacao_teste, protocolo_sha256=sha_protocolo)
    with tr.trava_pesada(c.lock, responsavel="MODELO", comando=comando):
        return ra.treinar_pacote_local(
            config_id, linhas, _total_do_protocolo(c), saida_raiz=c.pasta_modelos, selecao=selecao, ids_lim=ids_lim,
            id12=manifesto["sha256_completo"][:12],
            fontes={"dataset": {"sha256": manifesto["sha256_completo"], "pasta": c.dataset.name,
                                "sha256_manifesto": _sha(c.dataset / ARQUIVO_MANIFESTO)},
                    "entradas_do_dataset": manifesto.get("entradas"), "amostra_ativa": manifesto.get("amostra_ativa"),
                    "protocolo_sha256": sha_protocolo, "resultado_do_desenvolvimento_sha256": sha_resultado},
            prazos_de_retencao=_prazos(c, manifesto), dimensao_original=dimensao, avaliacao=avaliacao,
            cobertura=cobertura, ids_excluidos=excluidos, treino_ate_utc=ctx.treino_ate_utc.strftime(FORMATO_UTC))


def limiar(c: Caminhos, pasta_pacote: Path, ids_lim: Sequence[str], tabela_lim: Mapping[str, Mapping], *,
           prazo_lim: str, comando: str = "limiar da LIM") -> dict:
    """Limiar do artefato exato; LIM incompleta deixa o pacote com confiança indisponível e não inventa valor."""
    posts, canais, _ = base_de_posts(c, mascarar_teste=False)
    pacote = json.loads((Path(pasta_pacote) / "pacote.json").read_text(encoding="utf-8"))
    ids_atuais = [i for i in ids_lim if i in tabela_lim]
    if len(ids_atuais) != len(ids_lim):
        return {"estado": "indisponivel", "motivo": f"LIM com {len(ids_atuais)} de {len(ids_lim)} vídeos extraídos",
                "pendencia": "extração local dos vídeos da LIM", "id_artefato": pacote["identidade_sha256"]}
    corte = pacote["contexto"].get("treino_ate_utc")
    if not corte:
        raise ae.ErroExperimento("o pacote não registra o corte do contexto final")
    contexto = hs.ContextoTreino("final", datetime.strptime(corte, FORMATO_UTC).replace(tzinfo=UTC))
    linhas = ral.linhas_da_lim(ids_lim, posts, canais, contexto, tabela_lim)
    with tr.trava_pesada(c.lock, responsavel="MODELO", comando=comando):
        return ral.anexar_limiar(pasta_pacote, linhas, prazo_lim)


def verificar_pacote(c: Caminhos, pasta_pacote: Path, ids_lim: Sequence[str]) -> dict:
    """Recarga bit a bit e aditividade do SHAP nas linhas do ajuste; chamado como processo novo antes de ativar."""
    pacote = json.loads((Path(pasta_pacote) / "pacote.json").read_text(encoding="utf-8"))
    linhas, _, _, _ = linhas_do_refit(c, ids_lim)
    por_id = {l["video_id"]: l for l in linhas}
    ajuste = json.loads((Path(pasta_pacote) / "ids.json").read_text(encoding="utf-8"))["ajuste"]
    resultado = rv.verificar_com_linhas(pasta_pacote, [por_id[v] for v in ajuste])
    destino = c.pasta_resultados / f"verificacao_processo_novo_{pacote['id']}.json"
    with open(destino, "x", encoding="utf-8") as arquivo:
        json.dump(rl.sanear(resultado), arquivo, ensure_ascii=False, indent=1)
    return resultado


def _ids_da_lim(c: Caminhos) -> tuple[list[str], str]:
    amostra = json.loads(Path(c.amostra_lim).read_text(encoding="utf-8"))
    return [x["video_id"] for x in amostra["lista"]], amostra["retencao"]["apagar_ate_utc"]


def _caminhos(args: argparse.Namespace) -> Caminhos:
    produto = Path(args.produto)

    def opcional(nome: str) -> Path | None:
        return Path(getattr(args, nome)) if getattr(args, nome, None) else None

    return Caminhos(
        produto=produto, dataset=Path(args.dataset), pasta_base=Path(args.base), arquivo_particao=Path(args.particao),
        pasta_protocolo=Path(args.protocolo), pasta_resultados=produto / "data/models" / PASTA_EXPERIMENTO,
        pasta_modelos=produto / "data/models", lock=Path(args.lock), ids_originais_teste=opcional("ids_originais_teste"),
        substituicoes=opcional("substituicoes"), amostra_lim=opcional("amostra_lim"), tabela_lim=opcional("tabela_lim"))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.avaliacao.execucao_audiovisual", description=__doc__)
    p.add_argument("etapa", choices=("preparar", "congelar", "experimento", "teste", "refit", "limiar", "verificar"))
    p.add_argument("--produto", type=Path, default=RAIZ_PROJETO)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n> registrada no handoff")
    p.add_argument("--particao", type=Path, required=True, help="particao.json registrada no handoff")
    p.add_argument("--protocolo", type=Path, required=True)
    p.add_argument("--lock", type=Path, default=RAIZ_PROJETO.parent / "tmp" / "lock_pesado")
    p.add_argument("--manifesto-desenvolvimento", type=Path, required=True,
                   help="manifesto do dataset de desenvolvimento com a especificação das famílias existentes")
    p.add_argument("--ids-originais-teste", type=Path, help="JSON com a lista dos IDs de teste da amostra original")
    p.add_argument("--substituicoes", type=Path, help="JSON {id_original: id_substituto}")
    p.add_argument("--resultado-desenvolvimento", type=Path)
    p.add_argument("--avaliacao-teste", type=Path)
    p.add_argument("--amostra-lim", type=Path, help="JSON da LIM (lista e retenção)")
    p.add_argument("--tabela-lim", type=Path, help="parquet com os atributos locais dos vídeos da LIM")
    p.add_argument("--pacote", type=Path)
    p.add_argument("--configuracao", help="configuração a ajustar no refit (tem de ser a escolhida)")
    args = p.parse_args(argv)
    c = _caminhos(args)
    especificacao = json.loads(args.manifesto_desenvolvimento.read_text(encoding="utf-8"))["especificacao"]
    comando = " ".join(sys.argv)
    try:
        if args.etapa == "experimento":
            print(experimento(c, especificacao, comando=comando))
        elif args.etapa == "teste":
            print(teste(c, especificacao, resultado_desenvolvimento=args.resultado_desenvolvimento, comando=comando))
        elif args.etapa in ("refit", "limiar", "verificar"):
            conferir_protocolo(c, preparar(c, especificacao))
            ids_lim, prazo_lim = _ids_da_lim(c)
            if args.etapa == "refit":
                print(json.dumps(refit(c, args.configuracao, resultado_desenvolvimento=args.resultado_desenvolvimento,
                                       avaliacao_teste=args.avaliacao_teste, ids_lim=ids_lim, comando=comando)))
            elif args.etapa == "limiar":
                tabela = {l["video_id"]: l for l in pq.read_table(c.tabela_lim).to_pylist()} if c.tabela_lim else {}
                print(json.dumps(rl.sanear(limiar(c, args.pacote, ids_lim, tabela, prazo_lim=prazo_lim,
                                                  comando=comando)), ensure_ascii=False))
            else:
                print(json.dumps(rl.sanear(verificar_pacote(c, args.pacote, ids_lim)), ensure_ascii=False))
        else:
            preparo = preparar(c, especificacao)
            if args.etapa == "congelar":
                arquivo, sha = congelar(c, preparo)
                print(arquivo, sha)
            else:
                print(json.dumps(rl.sanear({k: v for k, v in preparo.items() if k not in CHAVES_NAO_SERIALIZAVEIS}),
                                 ensure_ascii=False, indent=1, default=str))
    except (ValueError, OSError, tr.TravaOcupada, tr.MemoriaInsuficiente, rat.ErroDeParada) as erro:
        print(f"Recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
