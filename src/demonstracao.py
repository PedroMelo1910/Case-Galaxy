"""Demonstração offline do agente com o de-para congelado (D-F6), e conferência do ambiente.

    python -m src.demonstracao                         # demonstração: 6 criativos, sem rede
    python -m src.demonstracao --mostrar-salvas        # só relê as respostas salvas (não carrega modelos)
    python -m src.demonstracao --verificar-ambiente    # somente leitura: lista o que falta, com tamanho e SHA-256
    python -m src.demonstracao --baixar-codificadores  # opt-in: baixa os pesos que faltam (nunca é padrão)

A demonstração roda `prever` e `ordenar` pelos serviços reais do manifesto de pacote ativo, com as anotações do Gemini
dos 6 criativos lidas de um cache SEPARADO (`data/demonstracao/anotacoes`), o histórico e as capas já salvos em
`data/demonstracao/`. Não cria cliente do Gemini nem do YouTube: nenhuma chamada online. As respostas vão para um JSON.

O de-para é retrospectivo (o histórico usa curtidas atuais) e só ilustra a mecânica: não é evidência de desempenho.
A evidência estatística vem da avaliação final.
"""

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from src.agente.contratos import PedidoOrdenar, PedidoPrever
from src.config import RAIZ_PROJETO

ROTULO = "de-para retrospectivo"
PASTA_DEMONSTRACAO = Path("data/demonstracao")
PASTA_RESPOSTAS = "respostas_salvas"
RESPOSTAS_FINAIS = "respostas_final=v1.json"
FORMATO_RESPOSTAS = 1
SETORES = ("beleza", "automotivo")
MINIMO_DE_POSTS_COM_REFERENCIA = 5  # abaixo disso o produto usa a referência do setor (histórico curto)
REV_E5 = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
REV_CLIP = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
BIBLIOTECAS_DO_SISTEMA = ("libgomp.so.1", "libGLESv2.so.2", "libEGL.so.1")
AVISO_GERAL = (
    "Demonstração retrospectiva e ilustrativa: o histórico do canal usa curtidas lidas em 18/10/2026 e os vídeos já "
    "estavam publicados. Mostra a mecânica do agente (previsão, faixa, versão e cobertura declaradas), não mede "
    "desempenho. A evidência estatística vem da avaliação final. A cobertura é a do procedimento de avaliação e não é "
    "garantida para vídeos novos.")


class ErroDemonstracao(RuntimeError):
    """Falha prevista da demonstração ou do ambiente, com mensagem própria (sem rastro de pilha)."""


# --- pesos dos codificadores congelados ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemPeso:
    grupo: str
    arquivo: str
    caminho: Path
    bytes: int
    sha256: str
    repo: str | None = None
    revisao: str | None = None

    @property
    def nome(self) -> str:
        return f"{self.grupo}/{self.arquivo}"


@dataclass(frozen=True)
class EstadoPeso:
    item: ItemPeso
    estado: str  # ok | ausente | tamanho_diferente | hash_diferente
    bytes_reais: int | None = None


@dataclass(frozen=True)
class EstadoBiblioteca:
    nome: str
    estado: str  # ok | ausente


_E5 = (("config.json", 655, "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
       ("model.safetensors", 470_641_600, "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
       ("sentencepiece.bpe.model", 5_069_051, "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
       ("special_tokens_map.json", 167, "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"),
       ("tokenizer.json", 17_082_730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
       ("tokenizer_config.json", 443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"))
_CLIP = (("config.json", 4_186, "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770"),
         ("preprocessor_config.json", 316, "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"),
         ("pytorch_model.bin", 605_247_071, "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"))
_OCR = (("craft_mlt_25k.pth", 83_152_330, "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"),
        ("latin_g2.pth", 15_406_141, "aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55"))
_ROSTO = (("blaze_face_short_range.tflite", 229_746,
           "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"),)


def pasta_hub_padrao() -> Path:
    """Cache de modelos do Hugging Face, na mesma ordem de precedência da biblioteca."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    base = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return base / "hub"


def pasta_casa_padrao() -> Path:
    return Path.home() / ".cache" / "casegalaxy"


def itens_esperados(hub: Path, casa: Path) -> list[ItemPeso]:
    """Pesos que o runtime carrega, com tamanho e SHA-256 conferidos na máquina de desenvolvimento (18/10/2026)."""
    itens = []
    for grupo, repo, revisao, arquivos in (("e5-small", "intfloat/multilingual-e5-small", REV_E5, _E5),
                                           ("clip-vit-b32", "openai/clip-vit-base-patch32", REV_CLIP, _CLIP)):
        base = Path(hub) / f"models--{repo.replace('/', '--')}" / "snapshots" / revisao
        itens += [ItemPeso(grupo, nome, base / nome, tamanho, sha, repo, revisao) for nome, tamanho, sha in arquivos]
    itens += [ItemPeso("easyocr", nome, Path(casa) / "easyocr" / nome, tamanho, sha) for nome, tamanho, sha in _OCR]
    itens += [ItemPeso("mediapipe-face", nome, Path(casa) / "mediapipe" / nome, tamanho, sha) for nome, tamanho, sha in _ROSTO]
    return itens


def _sha256(arquivo: Path) -> str:
    h = hashlib.sha256()
    with Path(arquivo).open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_pesos(itens: Sequence[ItemPeso], *, conferir_hash: bool) -> list[EstadoPeso]:
    estados = []
    for item in itens:
        if not item.caminho.is_file():
            estados.append(EstadoPeso(item, "ausente"))
            continue
        tamanho = item.caminho.stat().st_size
        if tamanho != item.bytes:
            estados.append(EstadoPeso(item, "tamanho_diferente", tamanho))
        elif conferir_hash and _sha256(item.caminho) != item.sha256:
            estados.append(EstadoPeso(item, "hash_diferente", tamanho))
        else:
            estados.append(EstadoPeso(item, "ok", tamanho))
    return estados


def _milhares(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def verificar_bibliotecas(nomes: Sequence[str] = BIBLIOTECAS_DO_SISTEMA) -> list[EstadoBiblioteca]:
    achados = []
    for nome in nomes:
        try:
            ctypes.CDLL(nome)
            achados.append(EstadoBiblioteca(nome, "ok"))
        except OSError:
            achados.append(EstadoBiblioteca(nome, "ausente"))
    return achados


def relatorio_ambiente(estados: Sequence[EstadoPeso], bibliotecas: Sequence[EstadoBiblioteca]) -> str:
    linhas = ["Pesos dos codificadores (não vão na distribuição; licenças NÃO confirmadas, ver docs/dados_e_retencao.md):"]
    for e in estados:
        extra = f" (encontrado: {_milhares(e.bytes_reais)} bytes)" if e.bytes_reais is not None and e.estado != "ok" else ""
        linhas.append(f"  [{e.estado:>17}] {e.item.nome}  {_milhares(e.item.bytes)} bytes  sha256 {e.item.sha256}{extra}")
        if e.estado != "ok":
            linhas.append(f"      onde: {e.item.caminho}")
    if bibliotecas:
        linhas.append("Bibliotecas do sistema:")
        for b in bibliotecas:
            linhas.append(f"  [{b.estado:>17}] {b.nome}")
        if any(b.estado != "ok" for b in bibliotecas):
            linhas.append("  Faltam bibliotecas do sistema: instale-as (por exemplo libgomp1, libgles2 e libegl1 no "
                          "Ubuntu/Debian; não verificado em máquina limpa) ou aponte LD_LIBRARY_PATH para uma pasta "
                          "que as contenha.")
    faltam = [e for e in estados if e.estado != "ok"]
    linhas.append("Tudo certo." if not faltam and all(b.estado == "ok" for b in bibliotecas) else
                  f"Pendências: {len(faltam)} arquivo(s) de peso. Para obtê-los: python -m src.demonstracao --baixar-codificadores")
    return "\n".join(linhas)


def exigir_pesos(estados: Sequence[EstadoPeso]) -> None:
    faltam = [e for e in estados if e.estado != "ok"]
    if not faltam:
        return
    itens = "\n".join(f"  - {e.item.nome} ({_milhares(e.item.bytes)} bytes), {e.estado.replace('_', ' ')}: {e.item.caminho}"
                      for e in faltam)
    raise ErroDemonstracao(
        "Faltam pesos dos codificadores locais (o produto não os inclui nem os baixa sozinho nesta demonstração):\n"
        f"{itens}\nPara baixá-los, com sua confirmação explícita: python -m src.demonstracao --baixar-codificadores\n"
        "Para só conferir: python -m src.demonstracao --verificar-ambiente")


def _baixar_hf(repo: str, revisao: str, arquivos: Sequence[str], destino: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, revision=revisao, allow_patterns=list(arquivos), cache_dir=str(destino))


def _baixar_ocr(casa: Path) -> None:
    from src import extracao_capa as capa

    capa.LeitorTexto(diretorio=casa)._carregar()


def _baixar_rosto(casa: Path) -> None:
    from src import extracao_capa as capa

    capa.DetectorRosto(diretorio=casa)._carregar()


def _gravar_refs_main(hub: Path, repo: str, revisao: str) -> None:
    """Sem `refs/main` o runtime offline (que usa `revision=None`) não acha o snapshot baixado por hash."""
    refs = Path(hub) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    if not refs.is_file():
        refs.parent.mkdir(parents=True, exist_ok=True)
        refs.write_text(revisao, encoding="utf-8")


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
        return antes
    total = sum(e.item.bytes for e in pendentes)
    imprimir(f"Serão baixados {len(pendentes)} arquivo(s), cerca de {_milhares(total)} bytes ({total / 1e6:.0f} MB):")
    for e in pendentes:
        imprimir(f"  {e.item.nome}  {_milhares(e.item.bytes)} bytes")
    por_repo: dict[tuple[str, str], list[str]] = {}
    for e in pendentes:
        if e.item.repo:
            por_repo.setdefault((e.item.repo, e.item.revisao), []).append(e.item.arquivo)
    for (repo, revisao), arquivos in por_repo.items():
        imprimir(f"Baixando {repo}@{revisao[:8]}: {', '.join(arquivos)}")
        baixar_hf(repo, revisao, arquivos, hub)
        _gravar_refs_main(hub, repo, revisao)
    if any(e.item.grupo == "easyocr" for e in pendentes):
        imprimir("Baixando os pesos do EasyOCR (CRAFT e latin_g2) pelo caminho de cache do projeto.")
        baixar_ocr(casa)
    if any(e.item.grupo == "mediapipe-face" for e in pendentes):
        imprimir("Baixando o detector de rosto do MediaPipe pelo caminho de cache do projeto.")
        baixar_rosto(casa)
    depois = verificar_pesos(itens, conferir_hash=conferir_hash)
    imprimir(relatorio_ambiente(depois, []))
    return depois


# --- dados do de-para -------------------------------------------------------------------------------------------------


def carregar_contextos(pasta: Path) -> list[dict]:
    arquivos = sorted((Path(pasta) / "contexto").glob("*.json"))
    contextos = [json.loads(a.read_text(encoding="utf-8")) for a in arquivos]
    if not contextos:
        raise ErroDemonstracao("Os dados do de-para não estão em data/demonstracao/contexto. Descompacte a distribuição "
                               "sobre a pasta do projeto (ver README, seção Instalação e execução).")
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
                     "recusa é o comportamento esperado do produto (D-C5).",
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
    arquivo = Path(arquivo)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    arquivo.write_text(json.dumps(saida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return arquivo


def ler_respostas(arquivo: Path) -> dict:
    arquivo = Path(arquivo)
    if not arquivo.is_file():
        raise ErroDemonstracao(f"Respostas salvas não encontradas: {arquivo.name}")
    dados = json.loads(arquivo.read_text(encoding="utf-8"))
    if dados.get("formato") != FORMATO_RESPOSTAS:
        raise ErroDemonstracao("Formato de respostas salvas desconhecido.")
    return dados


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
    linhas += ["", f"{resumo['com_previsao']} de {resumo['criativos']} criativos com previsão; chamadas online: "
                   f"{saida['chamadas_online']}."]
    return "\n".join(linhas)


# --- montagem dos serviços reais ----------------------------------------------------------------------------------------


def servicos_da_demonstracao(raiz: Path, pasta_demo: Path, contextos: Sequence[dict]):
    """Serviços do manifesto ativo, com o cache de anotações SEPARADO do de-para e sem nenhum cliente de rede."""
    from src.agente import manifesto_servido as ms
    from src.agente.atributos import FonteAnotacaoGemini
    from src.agente.servicos import MODELO_GEMINI, TEMPERATURA_EXTRACAO, Servicos
    from src.cache_gemini import CacheAnotacoes
    from src.extracao_gemini import ConfiguracaoExtracao

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
    hub, casa = pasta_hub_padrao(), pasta_casa_padrao()
    try:
        if args.verificar_ambiente:
            print(relatorio_ambiente(verificar_pesos(itens_esperados(hub, casa), conferir_hash=True), verificar_bibliotecas()))
            return 0
        if args.baixar_codificadores:
            depois = baixar_codificadores(hub, casa)
            return 0 if all(e.estado == "ok" for e in depois) else 3
        if args.mostrar_salvas is not None:
            arquivo = Path(args.mostrar_salvas) if args.mostrar_salvas else pasta_demo / PASTA_RESPOSTAS / RESPOSTAS_FINAIS
            print(resumo_em_texto(ler_respostas(arquivo)))
            return 0
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nenhum peso é baixado durante a demonstração
        contextos = carregar_contextos(pasta_demo)
        conferir_lista_congelada(pasta_demo / "congelado", contextos)
        exigir_pesos(verificar_pesos(itens_esperados(hub, casa), conferir_hash=False))
        servicos = servicos_da_demonstracao(raiz, pasta_demo, contextos)
        saida = executar(contextos, carregar_conhecido(pasta_demo), servicos, pasta_demo)
        destino = args.saida or pasta_demo / PASTA_RESPOSTAS / f"execucao_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        gravar_respostas(saida, destino)
        print(resumo_em_texto(saida))
        print(f"\nRespostas salvas em {destino}")
        _liberar(servicos)
        return 0
    except ErroDemonstracao as erro:
        print(f"Demonstração interrompida: {erro}", file=sys.stderr)
        return 3 if "pesos" in str(erro) else 2


if __name__ == "__main__":
    sys.exit(main())
