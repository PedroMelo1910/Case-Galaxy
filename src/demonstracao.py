"""Módulo consolidado. Reúne:
- Demonstração offline do agente com o de-para congelado, e conferência do ambiente.
- Demonstração por linha de comando da previsão de um vídeo local pela rota local (offline)."""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, UTC
from pathlib import Path

from src.agente.contratos import PedidoOrdenar, PedidoPrever
from src.config import RAIZ_PROJETO
from src.extracao.modelos import (
    AmbienteIncompleto,
    BIBLIOTECAS_DO_SISTEMA,
    CODIGO_SAIDA_AMBIENTE_INCOMPLETO,
    COMANDO_BAIXAR,
    COMANDO_VERIFICAR,
    EstadoBiblioteca,
    EstadoPeso,
    EstadoRef,
    exigir_ambiente,
    ItemPeso,
    itens_esperados,
    milhares,
    pasta_casa_padrao,
    pasta_hub_padrao,
    pasta_refs_main,
    REV_CLIP,
    REV_E5,
    texto_bibliotecas_ausentes,
    verificar_bibliotecas,
    verificar_pesos,
    verificar_refs,
)
from src.extracao.rubrica import gravar_json_atomico


# ──── Demonstração offline do agente com o de-para congelado, e conferência do ambiente.
ROTULO = "de-para retrospectivo"
PASTA_DEMONSTRACAO = Path("data/demonstracao")
PASTA_RESPOSTAS = "respostas_salvas"
RESPOSTAS_FINAIS = "respostas_final=v1.json"
FORMATO_RESPOSTAS = 1
CODIGO_SAIDA_DEMONSTRACAO_INCOMPLETA = 4
SETORES = ("beleza", "automotivo")
MINIMO_DE_POSTS_COM_REFERENCIA = 5  # abaixo disso o produto usa a referência do setor (histórico curto)
AVISO_GERAL = (
    "Demonstração retrospectiva e ilustrativa: o histórico do canal usa curtidas lidas em 18/10/2026 e os vídeos já "
    "estavam publicados. Mostra a mecânica do agente (previsão, faixa, versão e cobertura declaradas), não mede "
    "desempenho. A evidência estatística vem da avaliação final. A cobertura é a do procedimento de avaliação e não é "
    "garantida para vídeos novos.")


class ErroDemonstracao(RuntimeError):
    """Falha prevista da demonstração ou do ambiente, com mensagem própria (sem rastro de pilha)."""


# --- pesos dos codificadores congelados (tabela e conferência em src/extracao/modelos.py) -------------------------------------

def relatorio_ambiente(estados: Sequence[EstadoPeso], bibliotecas: Sequence[EstadoBiblioteca],
                       refs: Sequence[EstadoRef] = ()) -> str:
    linhas = ["Pesos dos codificadores (não vão na distribuição; licenças NÃO confirmadas, ver docs/DOCUMENTACAO.md#dados-e-retencao):"]
    for e in estados:
        extra = f" (encontrado: {milhares(e.bytes_reais)} bytes)" if e.bytes_reais is not None and e.estado != "ok" else ""
        linhas.append(f"  [{e.estado:>17}] {e.item.nome}  {milhares(e.item.bytes)} bytes  sha256 {e.item.sha256}{extra}")
        if e.estado != "ok":
            linhas.append(f"      onde: {e.item.caminho}")
    if refs:
        linhas.append("`refs/main` do Hugging Face (o runtime carrega o que ele aponta; tem de ser a revisão fixa):")
        for r in refs:
            extra = f" (encontrado: {r.encontrado})" if r.encontrado and r.estado != "ok" else ""
            linhas.append(f"  [{r.estado:>17}] {r.repo}  esperado {r.revisao}{extra}")
    ausentes = [b.nome for b in bibliotecas if b.estado != "ok"]
    if bibliotecas:
        linhas.append("Bibliotecas do sistema:")
        linhas += [f"  [{b.estado:>17}] {b.nome}" for b in bibliotecas]
        if ausentes:
            linhas += ["Faltam bibliotecas do sistema:", *texto_bibliotecas_ausentes(ausentes)]
    faltam = [e for e in estados if e.estado != "ok"]
    if faltam:
        linhas.append(f"Pendências: {len(faltam)} arquivo(s) de peso. Para obtê-los: {COMANDO_BAIXAR}")
    refs_ruins = [r for r in refs if r.estado != "ok"]
    if refs_ruins:
        linhas.append(f"Pendências: {len(refs_ruins)} `refs/main` fora da revisão fixa. Para corrigir: {COMANDO_BAIXAR}")
    if ausentes:
        linhas.append(f"Pendências: {len(ausentes)} biblioteca(s) do sistema (acima).")
    if not faltam and not ausentes and not refs_ruins:
        linhas.append("Tudo certo.")
    return "\n".join(linhas)


def _baixar_hf(repo: str, revisao: str, arquivos: Sequence[str], destino: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, revision=revisao, allow_patterns=list(arquivos), cache_dir=str(destino))


def _baixar_ocr(casa: Path) -> None:
    from src.extracao import imagem as capa

    capa.LeitorTexto(diretorio=casa)._carregar()


def _baixar_rosto(casa: Path) -> None:
    from src.extracao import imagem as capa

    detector = capa.DetectorRosto(diretorio=casa)
    detector._carregar()
    detector._detector.close()  # sem isto o MediaPipe imprime um TypeError ao encerrar o processo


def _gravar_refs_main(hub: Path, repo: str, revisao: str) -> bool:
    """O runtime offline usa `revision=None`, isto é, `refs/main`: ele tem de apontar para a revisão fixa.

    Reescreve se estiver ausente ou diferente; devolve True se mudou algo."""
    refs = pasta_refs_main(hub, repo)
    if refs.is_file() and refs.read_text(encoding="utf-8").strip() == revisao:
        return False
    refs.parent.mkdir(parents=True, exist_ok=True)
    refs.write_text(revisao, encoding="utf-8")
    return True


def _corrigir_refs(hub: Path, itens: Sequence[ItemPeso], imprimir: Callable) -> None:
    """Grava `refs/main` dos repositórios cujos arquivos estão todos corretos."""
    estados = verificar_pesos(itens, conferir_hash=False)
    for r in verificar_refs(hub, estados):
        if r.estado != "ok" and _gravar_refs_main(hub, r.repo, r.revisao):
            imprimir(f"refs/main de {r.repo} estava {r.estado}"
                     f"{'' if r.encontrado is None else f' ({r.encontrado})'}; agora aponta para {r.revisao[:8]}.")


def _separar_divergentes(pendentes: Sequence[EstadoPeso], imprimir: Callable) -> None:
    """Arquivo com tamanho ou SHA-256 errado sai do caminho (`.divergente`) para ser baixado de novo."""
    for e in pendentes:
        if e.estado == "ausente":
            continue
        destino = e.item.caminho.with_name(e.item.caminho.name + ".divergente")
        os.replace(e.item.caminho, destino)
        imprimir(f"Arquivo divergente ({e.estado.replace('_', ' ')}) movido para {destino}; será baixado de novo.")


def baixar_codificadores(hub: Path, casa: Path, *, baixar_hf: Callable = _baixar_hf,
                         baixar_ocr: Callable = _baixar_ocr, baixar_rosto: Callable = _baixar_rosto,
                         imprimir: Callable = print, conferir_hash: bool = True, apenas_ausentes: bool = True
                         ) -> list[EstadoPeso]:
    """Baixa só o que falta (revisões fixas). Imprime os tamanhos ANTES. Nunca roda sem pedido explícito."""
    itens = itens_esperados(hub, casa)
    antes = verificar_pesos(itens, conferir_hash=conferir_hash)
    pendentes = [e for e in antes if e.estado != "ok"] if apenas_ausentes else list(antes)
    if not pendentes:
        imprimir("Todos os pesos já estão no cache; nada a baixar.")
        _corrigir_refs(hub, itens, imprimir)
        return antes
    total = sum(e.item.bytes for e in pendentes)
    imprimir(f"Serão baixados {len(pendentes)} arquivo(s), cerca de {milhares(total)} bytes ({total / 1e6:.0f} MB):")
    for e in pendentes:
        imprimir(f"  {e.item.nome}  {milhares(e.item.bytes)} bytes")
    _separar_divergentes(pendentes, imprimir)
    por_repo: dict[tuple[str, str], list[str]] = {}
    for e in pendentes:
        if e.item.repo:
            por_repo.setdefault((e.item.repo, e.item.revisao), []).append(e.item.arquivo)
    for (repo, revisao), arquivos in por_repo.items():
        imprimir(f"Baixando {repo}@{revisao[:8]}: {', '.join(arquivos)}")
        baixar_hf(repo, revisao, arquivos, hub)
        _gravar_refs_main(hub, repo, revisao)
    _corrigir_refs(hub, itens, imprimir)
    if any(e.item.grupo == "easyocr" for e in pendentes):
        imprimir("Baixando os pesos do EasyOCR (CRAFT e latin_g2) pelo caminho de cache do projeto.")
        baixar_ocr(casa)
    if any(e.item.grupo == "mediapipe-face" for e in pendentes):
        imprimir("Baixando o detector de rosto do MediaPipe pelo caminho de cache do projeto.")
        baixar_rosto(casa)
    depois = verificar_pesos(itens, conferir_hash=conferir_hash)
    imprimir(relatorio_ambiente(depois, [], verificar_refs(hub, depois)))
    return depois


# --- dados do de-para -------------------------------------------------------------------------------------------------


def carregar_contextos(pasta: Path) -> list[dict]:
    arquivos = sorted((Path(pasta) / "contexto").glob("*.json"))
    contextos = [json.loads(a.read_text(encoding="utf-8")) for a in arquivos]
    if not contextos:
        raise ErroDemonstracao("Os dados do de-para não estão em data/demonstracao/contexto. Instale a distribuição "
                               "(README, seção Instalação e execução, passo 3: verificar_distribuicao.py instalar ... --destino .).")
    return sorted(contextos, key=lambda c: (SETORES.index(c["setor"]), c["ordem_no_setor"]))


def carregar_conhecido(pasta: Path) -> dict:
    arquivo = Path(pasta) / "resultado_conhecido_v1.json"
    if not arquivo.is_file():
        raise ErroDemonstracao("resultado_conhecido_v1.json ausente em data/demonstracao.")
    return json.loads(arquivo.read_text(encoding="utf-8"))


def conferir_lista_congelada(pasta: Path, contextos: Sequence[dict]) -> dict:
    """A lista congelada (SHA-256 no sidecar) tem de ser exatamente o conjunto dos contextos presentes."""
    lista = Path(pasta) / "lista_de_para_v1.json"
    sidecar = lista.with_name(lista.name + ".sha256")
    if not lista.is_file() or not sidecar.is_file():
        raise ErroDemonstracao("Lista congelada do de-para ausente (data/demonstracao/congelado).")
    soma = hashlib.sha256(lista.read_bytes()).hexdigest()
    if soma != sidecar.read_text(encoding="utf-8").split()[0]:
        raise ErroDemonstracao("A lista congelada do de-para foi alterada: o SHA-256 difere do registrado.")
    congelados = {e["video_id"] for es in json.loads(lista.read_text(encoding="utf-8"))["setores"].values() for e in es}
    presentes = {c["video_id"] for c in contextos}
    if congelados != presentes:
        raise ErroDemonstracao("Os dados do de-para não conferem com a lista congelada: "
                               f"faltam {sorted(congelados - presentes)}, sobram {sorted(presentes - congelados)}.")
    return {"sha256": soma, "n": len(congelados)}


def montar_pedido(contexto: dict, pasta_demo: Path) -> dict:
    """Pedido de `prever` do criativo: URL pública, legenda, capa salva, histórico fornecido até p - 30 dias."""
    capa = contexto["capa"].get("arquivo")
    posts = contexto["historico_do_canal"]["posts"]
    return {
        "creative": {"tipo": "video", "url": contexto["url"]},
        "caption": {"title": contexto["legenda"].get("title"), "description": contexto["legenda"].get("description")},
        "tags": list(contexto.get("tags", [])),
        "thumbnail": str(Path(pasta_demo) / capa) if capa else None,
        "planned_datetime": contexto["published_at_utc"],
        "sector": contexto["setor"], "channel_id": contexto["channel_id"], "duration_s": contexto["duracao_s"],
        "channel_stats": [{"published_at": p["published_at"], "likes": p["likes"], "collected_at": p["collected_at"],
                           "video_id": p["video_id"]} for p in posts],
        "consultar_youtube": False}


def ressalvas_do_contexto(contexto: dict) -> list[dict]:
    """O que a demonstração precisa declarar sobre este criativo (decisão do Lead de 18/10)."""
    pais = (contexto.get("canal") or {}).get("pais_api")
    r = [{"codigo": "rotulo_de_para_retrospectivo",
          "texto": "De-para retrospectivo: o histórico usa curtidas lidas em 18/10/2026 e o vídeo já estava publicado."},
         {"codigo": "tipo_canal_ausente",
          "texto": "Canal fora da base e sem tipo (marca ou criador) informado: o atributo fica ausente."},
         {"codigo": "idioma_nao_verificado",
          "texto": "O idioma do conteúdo não foi verificado; o modelo foi treinado com canais brasileiros em português."}]
    if pais == "US":
        r.append({"codigo": "pais_us", "texto": "A API informa país US: fora do domínio de treino (canais do Brasil)."})
    elif pais is None:
        r.append({"codigo": "pais_nao_confirmado", "texto": "A API não informa o país do canal: origem brasileira não confirmada."})
    elif pais != "BR":
        r.append({"codigo": "pais_fora_do_brasil", "texto": f"A API informa país {pais}: fora do domínio de treino."})
    com_curtidas = sum(1 for p in contexto["historico_do_canal"]["posts"] if p.get("likes") is not None)
    if com_curtidas < MINIMO_DE_POSTS_COM_REFERENCIA:
        r.append({"codigo": "historico_sem_curtidas_utilizaveis",
                  "texto": f"Só {com_curtidas} post(s) com curtidas até p - 30 dias: o agente usa a referência do setor "
                           "e avisa histórico curto."})
    return r


def _resultado_conhecido(contexto: dict, conhecido: dict, resposta: dict) -> dict | None:
    por_id = {v["video_id"]: v for v in conhecido.get("videos", [])}
    real = por_id.get(contexto["video_id"])
    if real is None or real.get("curtidas") is None:
        return None
    dentro = None
    if resposta.get("success"):
        d = resposta["data"]
        dentro = d["interval_low"] <= real["curtidas"] <= d["interval_high"]
    return {"curtidas": real["curtidas"], "lido_em_utc": conhecido.get("lido_em_utc"),
            "rotulo": f"resultado real conhecido ({ROTULO})", "dentro_da_faixa": dentro,
            "aviso": "Observação única e ilustrativa; as curtidas atuais de um vídeo não medem a qualidade do modelo. "
                     "A evidência estatística vem da avaliação final."}


def _cobertura(resposta: dict) -> dict | None:
    if not resposta.get("success"):
        return None
    c = resposta["data"].get("coverage_estimate") or {}
    chaves = ("nivel_do_procedimento", "cobertura_observada", "n_dentro", "n", "ic95_bootstrap_por_canal",
              "garantia_de_cobertura", "medida_no_artefato_servido", "origem")
    return {k: c[k] for k in chaves if k in c}


def _pedido_ordenar_entre_canais(contextos: Sequence[dict], pasta_demo: Path) -> dict | None:
    for setor in SETORES:
        do_setor = [c for c in contextos if c["setor"] == setor]
        if len(do_setor) >= 2:
            return {"creatives": [montar_pedido(c, pasta_demo) for c in do_setor]}
    return None


def _pedido_ordenar_ilustrativo(contextos: Sequence[dict]) -> tuple[dict, dict] | None:
    """Duas legendas planejadas do MESMO canal (modelo de texto): título e descrição × só o título."""
    candidatos = [c for c in contextos if (c["legenda"].get("description") or "").strip() and (c["legenda"].get("title") or "").strip()]
    if not candidatos:
        return None
    escolhido = max(candidatos, key=lambda c: (sum(1 for p in c["historico_do_canal"]["posts"] if p.get("likes") is not None),
                                               c["video_id"]))
    base = {k: v for k, v in montar_pedido(escolhido, Path(".")).items()
            if k in ("planned_datetime", "sector", "channel_id", "channel_stats", "consultar_youtube")}
    base["own_video_id"] = escolhido["video_id"]
    titulo, descricao = escolhido["legenda"]["title"].strip(), escolhido["legenda"]["description"].strip()
    variantes = [base | {"creative": {"tipo": "texto", "texto": f"{titulo}\n\n{descricao}"}},
                 base | {"creative": {"tipo": "texto", "texto": titulo}}]
    return {"creatives": variantes}, escolhido


def executar(contextos: Sequence[dict], conhecido: dict, servicos, pasta_demo: Path, *, prever_fn: Callable | None = None,
             ordenar_fn: Callable | None = None, agora: datetime | None = None) -> dict:
    """Roda `prever` em cada criativo e mostra `ordenar`. `servicos` e as funções são injetáveis (testes)."""
    if prever_fn is None or ordenar_fn is None:
        from src.agente import api

        prever_fn, ordenar_fn = prever_fn or api.prever, ordenar_fn or api.ordenar
    criativos = []
    for contexto in contextos:
        resposta = prever_fn(montar_pedido(contexto, pasta_demo), servicos)
        criativos.append({
            "video_id": contexto["video_id"], "setor": contexto["setor"], "ordem_no_setor": contexto["ordem_no_setor"],
            "channel_id": contexto["channel_id"], "canal": contexto.get("canal"), "url": contexto["url"],
            "published_at_utc": contexto["published_at_utc"], "ressalvas": ressalvas_do_contexto(contexto),
            "resposta": resposta, "cobertura_declarada": _cobertura(resposta),
            "resultado_conhecido": _resultado_conhecido(contexto, conhecido, resposta)})
    saida = {
        "formato": FORMATO_RESPOSTAS, "rotulo": ROTULO, "aviso": AVISO_GERAL,
        "gerado_em_utc": (agora or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"), "chamadas_online": 0,
        "versao_servida": getattr(servicos, "rotulo_da_versao", None), "criativos": criativos,
        "resumo": {"criativos": len(criativos), "com_previsao": sum(c["resposta"].get("success", False) for c in criativos),
                   "com_erro": sum(not c["resposta"].get("success", False) for c in criativos)}}
    entre = _pedido_ordenar_entre_canais(contextos, pasta_demo)
    saida["ordenar_entre_canais"] = None if entre is None else {
        "descricao": "ordenar exige criativos do MESMO canal; os criativos do de-para são de canais diferentes e a "
                     "recusa é o comportamento esperado do produto.",
        "resposta": ordenar_fn(entre, servicos)}
    ilustrativo = _pedido_ordenar_ilustrativo(contextos)
    if ilustrativo is not None:
        pedido, contexto = ilustrativo
        saida["ordenar_ilustrativo_texto"] = {
            "aviso": "Ilustração da mecânica do ranking com variantes de legenda do mesmo criativo (legenda completa × só "
                     "o título), pelo modelo de texto. São variantes de legenda: não são vídeos distintos e o resultado não prova ordenação.",
            "video_id": contexto["video_id"], "channel_id": contexto["channel_id"],
            "resposta": ordenar_fn(pedido, servicos)}
    return saida


# --- saída ----------------------------------------------------------------------------------------------------------------


def gravar_respostas(saida: dict, arquivo: Path) -> Path:
    return gravar_json_atomico(Path(arquivo), saida)


def ler_respostas(arquivo: Path) -> dict:
    arquivo = Path(arquivo)
    if not arquivo.is_file():
        raise ErroDemonstracao(f"Respostas salvas não encontradas: {arquivo.name}")
    dados = json.loads(arquivo.read_text(encoding="utf-8"))
    if dados.get("formato") != FORMATO_RESPOSTAS:
        raise ErroDemonstracao("Formato de respostas salvas desconhecido.")
    return dados


def criativos_fora_do_modelo_de_video(saida: dict) -> list[str]:
    """Criativos que não foram servidos pelo modelo de vídeo da versão servida (erro ou outro modelo).

    A distribuição leva a anotação dos 6 vídeos do de-para: todos têm de sair pelo modelo de vídeo."""
    esperado = f"{saida.get('versao_servida')}/video-"
    return [c["video_id"] for c in saida["criativos"]
            if not (c["resposta"].get("success") and str(c["resposta"]["data"].get("model_version", "")).startswith(esperado))]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + "%"


def resumo_em_texto(saida: dict) -> str:
    linhas = [f"DEMONSTRAÇÃO ({saida['rotulo']}), versão servida: {saida.get('versao_servida') or 'não informada'}",
              saida["aviso"], ""]
    for c in saida["criativos"]:
        titulo = (c.get("canal") or {}).get("titulo") or c["channel_id"]
        linhas.append(f"[{c['setor']} #{c['ordem_no_setor']}] {c['video_id']}  canal: {titulo}  publicado em {c['published_at_utc'][:10]}")
        r = c["resposta"]
        if not r.get("success"):
            e = r.get("error", {})
            linhas.append(f"   sem previsão: {e.get('code')} - {e.get('message', '')}")
        else:
            d = r["data"]
            linhas.append(f"   modelo: {d['model_level']}  versão: {d['model_version']}")
            linhas.append(f"   previsão central: {d['predicted_metric']} curtidas; faixa nominal de 80%: "
                          f"{d['interval_low']} a {d['interval_high']}; referência do canal: {d['reference_likes']} ({d['reference_source']})")
            cob = c.get("cobertura_declarada") or {}
            if "cobertura_observada" in cob:
                linhas.append(f"   cobertura declarada ({cob.get('nivel_do_procedimento')}, procedimento de avaliação): "
                              f"{_pct(cob['cobertura_observada'])} de {cob['n']} vídeos de teste contra 80% pretendidos; "
                              "sem garantia de cobertura para vídeos novos")
            codigos = [w["codigo"] for w in d.get("warnings", [])]
            linhas.append(f"   avisos do agente: {', '.join(codigos) if codigos else 'nenhum'}")
        linhas.append(f"   ressalvas da demonstração: {', '.join(x['codigo'] for x in c['ressalvas'])}")
        conhecido = c.get("resultado_conhecido")
        if conhecido:
            posicao = {True: "dentro da faixa", False: "fora da faixa", None: "sem comparação"}[conhecido["dentro_da_faixa"]]
            linhas.append(f"   {conhecido['rotulo']}: {conhecido['curtidas']} curtidas ({posicao}); ilustrativo")
        linhas.append("")
    for chave, titulo in (("ordenar_entre_canais", "ordenar entre canais diferentes"),
                          ("ordenar_ilustrativo_texto", "ordenar ilustrativo (variantes de legenda)")):
        bloco = saida.get(chave)
        if bloco:
            r = bloco["resposta"]
            resultado = ("recusado: " + r["error"]["code"]) if not r.get("success") else \
                "ranking: " + ", ".join(f"{i['posicao']}º = item {i['indice']}" for i in r["data"]["ranking"])
            linhas.append(f"{titulo}: {resultado}")
            if bloco.get("aviso"):
                linhas.append(f"   {bloco['aviso']}")
    resumo = saida["resumo"]
    por_nivel: dict[str, int] = {}
    for c in saida["criativos"]:
        if c["resposta"].get("success"):
            nivel = str(c["resposta"]["data"].get("model_level", "?")).split(":")[0]
            por_nivel[nivel] = por_nivel.get(nivel, 0) + 1
    linhas += ["", "criativos por nível do modelo: " + (", ".join(f"{n} = {q}" for n, q in sorted(por_nivel.items())) or "nenhum")]
    linhas += ["", f"{resumo['com_previsao']} de {resumo['criativos']} criativos com previsão; chamadas online: "
                   f"{saida['chamadas_online']}."]
    return "\n".join(linhas)


# --- montagem dos serviços reais ----------------------------------------------------------------------------------------


def servicos_da_demonstracao(raiz: Path, pasta_demo: Path, contextos: Sequence[dict]):
    """Serviços do manifesto ativo, com o cache de anotações SEPARADO do de-para e sem nenhum cliente de rede."""
    from src.modelos import pacote as ms
    from src.agente.atributos import FonteAnotacaoGemini
    from src.agente.servicos import MODELO_GEMINI, TEMPERATURA_EXTRACAO, Servicos
    from src.extracao.rubrica import CacheAnotacoes
    from src.extracao.rubrica import ConfiguracaoExtracao

    pasta_anotacoes = Path(pasta_demo) / "anotacoes"
    faltam = [c["video_id"] for c in contextos if not (pasta_anotacoes / f"{c['video_id']}.json").is_file()]
    if faltam:
        raise ErroDemonstracao(f"Faltam anotações do de-para em data/demonstracao/anotacoes: {faltam}")
    try:
        servicos = Servicos.padrao(raiz)
    except ms.ErroManifesto as erro:
        raise ErroDemonstracao(f"Pacote de modelos indisponível: {erro}") from erro
    config = ConfiguracaoExtracao(modelo=MODELO_GEMINI, temperatura=TEMPERATURA_EXTRACAO)
    fonte = FonteAnotacaoGemini(cache=CacheAnotacoes(pasta_anotacoes, config), cliente=None,
                                temperatura=TEMPERATURA_EXTRACAO)
    return replace(servicos, fonte_anotacao=fonte)


def _liberar(servicos) -> None:
    """Fecha o detector de rosto antes do fim do processo (evita um ruído do MediaPipe na saída) e libera os serviços."""
    detector = getattr(getattr(servicos.extratores_capa, "rosto", None), "_detector", None)
    if detector is not None:
        try:
            detector.close()
        except Exception:  # noqa: BLE001 - só limpeza; o resultado já foi gravado
            pass
    del servicos
    gc.collect()


def _pesos_pendentes(hub: Path, casa: Path) -> list[EstadoPeso]:
    return [e for e in verificar_pesos(itens_esperados(hub, casa), conferir_hash=False) if e.estado != "ok"]


def _bibliotecas_citadas(erro: BaseException) -> list[str]:
    """Bibliotecas do sistema (`libX.so[.N]`) citadas na mensagem do erro; vazio se o erro não é sobre biblioteca."""
    return list(dict.fromkeys(re.findall(r"\blib[\w+.-]*?\.so(?:\.\d+)*", str(erro))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.demonstracao", description=__doc__.split("\n\n")[0])
    parser.add_argument("--raiz", type=Path, default=RAIZ_PROJETO, help="pasta do projeto (padrão: a do código)")
    parser.add_argument("--saida", type=Path, help="arquivo JSON das respostas (padrão: data/demonstracao/respostas_salvas/)")
    parser.add_argument("--mostrar-salvas", nargs="?", const="", metavar="ARQUIVO",
                        help="mostra respostas salvas, sem carregar modelos (padrão: respostas_final=v1.json)")
    parser.add_argument("--verificar-ambiente", action="store_true", help="somente leitura: pesos e bibliotecas")
    parser.add_argument("--baixar-codificadores", action="store_true", help="OPT-IN: baixa os pesos que faltam")
    args = parser.parse_args(argv)
    raiz = Path(args.raiz).resolve()
    pasta_demo = raiz / PASTA_DEMONSTRACAO
    try:
        if args.mostrar_salvas is not None:  # antes de tudo: não importa nada do Hugging Face nem toca nos pesos
            arquivo = Path(args.mostrar_salvas) if args.mostrar_salvas else pasta_demo / PASTA_RESPOSTAS / RESPOSTAS_FINAIS
            print(resumo_em_texto(ler_respostas(arquivo)))
            return 0
        if args.verificar_ambiente:
            hub, casa = pasta_hub_padrao(), pasta_casa_padrao()
            estados, bibliotecas = verificar_pesos(itens_esperados(hub, casa), conferir_hash=True), verificar_bibliotecas()
            refs = verificar_refs(hub, estados)
            print(relatorio_ambiente(estados, bibliotecas, refs))
            pendente = (any(e.estado != "ok" for e in estados) or any(b.estado != "ok" for b in bibliotecas)
                        or any(r.estado != "ok" for r in refs))
            return CODIGO_SAIDA_AMBIENTE_INCOMPLETO if pendente else 0
        if args.baixar_codificadores:
            hub, casa = pasta_hub_padrao(), pasta_casa_padrao()
            faltam_libs = [b.nome for b in verificar_bibliotecas() if b.estado != "ok"]
            if faltam_libs:  # não baixa ~1,2 GB para depois falhar ao carregar o MediaPipe ou o LightGBM
                raise AmbienteIncompleto(_pesos_pendentes(hub, casa), faltam_libs)
            try:
                depois = baixar_codificadores(hub, casa)
            except (OSError, ImportError) as erro:
                citadas = _bibliotecas_citadas(erro)
                if not citadas:
                    raise  # erro real (disco, permissão, rede...): não é falta de biblioteca do sistema
                raise AmbienteIncompleto(_pesos_pendentes(hub, casa), citadas) from erro
            return 0 if all(e.estado == "ok" for e in depois) else CODIGO_SAIDA_AMBIENTE_INCOMPLETO
        # nenhum peso é baixado durante a demonstração. Tem de vir ANTES de qualquer importação do Hugging Face: a
        # biblioteca lê HF_HUB_OFFLINE uma única vez, ao ser importada (pasta_hub_padrao a importa).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        hub, casa = pasta_hub_padrao(), pasta_casa_padrao()
        contextos = carregar_contextos(pasta_demo)
        conferir_lista_congelada(pasta_demo / "congelado", contextos)
        exigir_ambiente(hub, casa)
        servicos = servicos_da_demonstracao(raiz, pasta_demo, contextos)
        saida = executar(contextos, carregar_conhecido(pasta_demo), servicos, pasta_demo)
        destino = args.saida or pasta_demo / PASTA_RESPOSTAS / f"execucao_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        gravar_respostas(saida, destino)
        print(resumo_em_texto(saida))
        print(f"\nRespostas salvas em {destino}")
        _liberar(servicos)
        fora = criativos_fora_do_modelo_de_video(saida)
        if fora:  # a distribuição garante os 6 vídeos com anotação: qualquer outro resultado é instalação incompleta
            print(f"Demonstração incompleta: {len(fora)} de {len(saida['criativos'])} criativos não foram servidos pelo "
                  f"modelo de vídeo {saida.get('versao_servida')} (ou deram erro): {', '.join(fora)}. Confira data/demonstracao/"
                  "anotacoes e a distribuição instalada.", file=sys.stderr)
            return CODIGO_SAIDA_DEMONSTRACAO_INCOMPLETA
        return 0
    except AmbienteIncompleto as erro:
        print(erro, file=sys.stderr)
        return CODIGO_SAIDA_AMBIENTE_INCOMPLETO
    except ErroDemonstracao as erro:
        print(f"Demonstração interrompida: {erro}", file=sys.stderr)
        return 2


# ──── Demonstração por linha de comando da previsão de um vídeo local pela rota local (offline).
ROTULO_demonstracao_local = "vídeo local (rota local)"
AVISO = (
    "Previsão de um vídeo local com o contexto de canal informado na chamada. O resultado mostra o fluxo do agente "
    "(extração local, previsão, faixa, confiança e explicação); não mede o desempenho do modelo. No vídeo de exemplo "
    "distribuído (banco de imagens) o canal é hipotético. A cobertura da faixa descreve o procedimento de avaliação e "
    "não é garantida para vídeos novos.")
CODIGO_SAIDA_LOCAL_INDISPONIVEL = 4  # ambiente ou pacote indisponível na demonstração local (a demo principal usa 3)
CODIGO_SAIDA_SEM_PREVISAO = 1


def montar_pedido_demonstracao_local(arquivo: Path, *, canal: str, setor: str, planejada: datetime) -> dict:
    """Pedido de previsão pela rota local: o arquivo, o canal da base, o setor e a data planejada."""
    return {"creative": {"tipo": "video", "arquivo": str(arquivo)}, "planned_datetime": planejada.isoformat(),
            "sector": setor, "channel_id": canal, "video_route": "local"}


def montar_saida(pedido: Mapping, resposta: Mapping, *, versao_servida: str | None, agora: datetime) -> dict:
    """Envolve a resposta no formato de respostas salvas (um criativo), legível por `src.demonstracao`."""
    sucesso = bool(resposta.get("success"))
    instante = agora.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "formato": FORMATO_RESPOSTAS, "rotulo": ROTULO_demonstracao_local, "aviso": AVISO,
        "gerado_em_utc": instante, "chamadas_online": 0, "versao_servida": versao_servida,
        "criativos": [{
            "video_id": "arquivo_local", "setor": pedido["sector"], "ordem_no_setor": 1,
            "channel_id": pedido["channel_id"], "canal": None, "url": "arquivo local de exemplo",
            "published_at_utc": instante,
            "ressalvas": [{"codigo": "exemplo_funcional", "texto": "Vídeo de banco de imagens e canal hipotético: exemplo "
                                                                  "funcional, sem medida de desempenho."}],
            "resposta": dict(resposta), "cobertura_declarada": None, "resultado_conhecido": None}],
        "resumo": {"criativos": 1, "com_previsao": int(sucesso), "com_erro": int(not sucesso)}}


def detalhes_da_rota(resposta: Mapping) -> list[str]:
    """Linhas sobre rota, confiança e explicação, ausentes do resumo geral das respostas salvas."""
    if not resposta.get("success"):
        return []
    dados = resposta["data"]
    rota = dados.get("route") or {}
    confianca = dados.get("statistical_confidence") or {}
    linhas = [f"   rota: {rota.get('rotulo', 'não informada')} (offline: {rota.get('offline')}, usa Gemini: {rota.get('usa_gemini')})",
              f"   confiança: {dados.get('confidence_flag', 'não informada')}"
              + (f" ({confianca['estado']})" if confianca.get("estado") else ""),
              f"   explicação ({dados.get('explanation_source', '?')}): {dados.get('explanation', '')}"]
    fatores = [f"{f['atributo']} ({f['sentido']})" for f in dados.get("top_drivers", ())[:3]]
    if fatores:
        linhas.append("   principais fatores (SHAP descreve o modelo, não é causal): " + "; ".join(fatores))
    return linhas


def executar_demonstracao_local(pedido: dict, servicos, *, prever_fn: Callable | None = None, agora: datetime | None = None) -> dict:
    """Roda a previsão e devolve a saída no formato de respostas salvas. `servicos` e `prever_fn` são injetáveis."""
    if prever_fn is None:
        from src.agente import api

        prever_fn = api.prever
    resposta = prever_fn(pedido, servicos)
    return montar_saida(pedido, resposta, versao_servida=getattr(servicos, "rotulo_da_versao", None),
                        agora=agora or datetime.now(UTC))


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", type=Path, required=True, help="arquivo de vídeo vertical ou quadrado, até 180 s e 100 MB")
    parser.add_argument("--canal", required=True, help="identificador do canal, com histórico próprio ou contexto local")
    parser.add_argument("--setor", required=True, choices=SETORES)
    parser.add_argument("--planejada", default=None, help="data planejada ISO 8601 com fuso (padrão: agora)")
    parser.add_argument("--salvar", type=Path, default=None, help="grava a resposta no formato das respostas salvas")
    parser.add_argument("--historico", type=Path, help="CSV de posts próprios: published_at, likes, collected_at, video_id opcional")
    parser.add_argument("--tipo-canal", choices=("marca", "criador"))
    parser.add_argument("--raiz", type=Path, default=RAIZ_PROJETO)
    return parser.parse_args(argv)


def main_demonstracao_local(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    try:
        planejada = datetime.fromisoformat(args.planejada) if args.planejada else datetime.now(UTC)
    except ValueError:
        sys.stderr.write("Data planejada inválida; use ISO 8601 com fuso (por exemplo 2026-10-21T12:00:00-03:00).\n")
        return 2
    if planejada.tzinfo is None:
        sys.stderr.write("A data planejada precisa de fuso horário (por exemplo 2026-10-21T12:00:00-03:00).\n")
        return 2
    from src.agente.servicos import Servicos
    from src.extracao.modelos import AmbienteIncompleto

    try:
        servicos = Servicos.padrao(args.raiz)
    except AmbienteIncompleto as erro:
        sys.stderr.write(f"Ambiente incompleto: {erro}\n")
        return CODIGO_SAIDA_LOCAL_INDISPONIVEL
    except Exception as erro:  # noqa: BLE001 - mensagem própria no lugar de rastro de pilha
        sys.stderr.write(f"Pacote de modelos indisponível ({type(erro).__name__}).\n")
        return CODIGO_SAIDA_LOCAL_INDISPONIVEL
    pedido = montar_pedido_demonstracao_local(args.video, canal=args.canal, setor=args.setor, planejada=planejada)
    if args.historico:
        from src.interface_web.validacao import interpretar_historico
        try:
            posts, erros = interpretar_historico(args.historico.read_text(encoding="utf-8-sig"))
        except OSError:
            sys.stderr.write("Arquivo de histórico não encontrado ou ilegível.\n")
            return 2
        if erros:
            sys.stderr.write("Histórico inválido: " + "; ".join(erros) + "\n")
            return 2
        pedido["channel_stats"] = posts
    if args.tipo_canal:
        pedido["channel_type"] = args.tipo_canal
    saida = executar_demonstracao_local(pedido, servicos)
    resposta = saida["criativos"][0]["resposta"]
    if args.salvar:
        gravar_respostas(saida, args.salvar)
    if not resposta.get("success"):
        erro = resposta.get("error", {})
        sys.stderr.write(f"Sem previsão: {erro.get('code')} - {erro.get('message', '')}\n")
        return CODIGO_SAIDA_SEM_PREVISAO
    sys.stdout.write(resumo_em_texto(saida) + "\n" + "\n".join(detalhes_da_rota(resposta)) + "\n")
    return 0


def _script_demonstracao() -> None:
    sys.exit(main())


def _script_demonstracao_local() -> None:
    try:
        sys.exit(main_demonstracao_local())
    except ErroDemonstracao as erro:
        sys.stderr.write(f"{erro}\n")
        sys.exit(CODIGO_SAIDA_LOCAL_INDISPONIVEL)


_COMANDOS = {
    "demonstracao": _script_demonstracao,
    "demonstracao_local": _script_demonstracao_local,
}


def _despachar() -> None:
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in _COMANDOS:
        raise SystemExit("uso: python -m src.demonstracao {" + "|".join(_COMANDOS) + "} [argumentos]")
    comando = sys.argv.pop(1)
    sys.argv[0] = f"{sys.argv[0]} {comando}"
    _COMANDOS[comando]()


if __name__ == "__main__":
    _despachar()
