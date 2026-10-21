"""Executor retomável da anotação pelo Gemini sobre a amostra congelada."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, UTC
from pathlib import Path

from src.extracao.rubrica import CacheAnotacoes, CacheCorrompido, CacheIncompativel, gravar_json_atomico
from src.config import ChaveAusente
from src.dados import amostra as am, curadoria, particoes
from src.dados.coleta import ItemExtracao, Papel, url_do_video
from src.extracao.rubrica import anotar, ConfiguracaoExtracao, ExecucaoInterrompida, Politica, POLITICA_ANOTACAO
from src.integracoes import gemini as ga
from src.proveniencia import iso_utc as _iso


# ──── Executor retomável da anotação pelo Gemini sobre a amostra congelada.
MODELO = "gemini-3.5-flash-lite"
TEMPERATURA = 1.0
INTERVALO_PADRAO_S = 10.0
FALHAS_SEM_CACHE = frozenset({"requisicao_recusada", "tentativas_esgotadas"})
MAX_FALHAS_CONSECUTIVAS = 3
CAMPOS_DO_LOG = ("fase", "numero", "instante_utc", "resultado", "http_status", "duracao_s", "uso", "versao_modelo")
ARQUIVO_REUSO = "reuso.json"
ARQUIVO_REUSO_ANTIGO = "reuso_piloto.json"  # nome usado pela amostra v1; só lido, nunca regravado
MAX_SEGUNDOS_PADRAO = 28_800
REGRA_REUSO = ("só anotações válidas, com a mesma chave de compatibilidade, de vídeos desta amostra; piloto só da "
               "1ª passada; cópia byte a byte; a data-limite de retenção da origem não é renovada; o registro de "
               "requisições da origem não é copiado")
ARQUIVO_PROGRESSO = "progresso.json"
PARADAS_NORMAIS = frozenset({"lista_concluida", ga.LimiteDaExecucao.codigo})
COLUNAS_DO_ITEM = ["video_id", "channel_id", "snapshot_id", "papel", "thumbnail_url", "duracao_s", "published_at",
                   "collected_at"]


class ErroAnotacao(ValueError):
    """Entrada da anotação inconsistente (amostra, base ou piloto)."""


def configuracao() -> ConfiguracaoExtracao:
    """Configuração fixa: modelo, temperatura e o pedido e o esquema atuais da rubrica."""
    return ConfiguracaoExtracao(modelo=MODELO, temperatura=TEMPERATURA)


def itens_da_amostra(pasta_base: Path, amostra: dict) -> list[ItemExtracao]:
    """Itens de extração na ordem congelada, só com colunas sem resultado de desempenho; o papel vem da amostra."""
    papel = Papel(amostra.get("papel", Papel.ALVO))
    tabelas = curadoria.ler_base(pasta_base)
    setor = {c["channel_id"]: c["setor"] for c in tabelas["canais.parquet"].select(["channel_id", "setor"]).to_pylist()}
    videos = {v["video_id"]: v for v in tabelas["videos.parquet"].select(COLUNAS_DO_ITEM).to_pylist()}
    itens = []
    for entrada in amostra["lista"]:
        video = videos.get(entrada["video_id"])
        if video is None or video["papel"] != papel or video["channel_id"] != entrada["channel_id"]:
            raise ErroAnotacao(f"{entrada['video_id']}: não tem o papel {papel} no canal registrado na base")
        itens.append(
            ItemExtracao(
                video_id=video["video_id"],
                channel_id=video["channel_id"],
                snapshot_id=video["snapshot_id"],
                papel=papel,
                setor=setor[video["channel_id"]],
                url_video=url_do_video(video["video_id"]),
                thumbnail_url=video["thumbnail_url"],
                duracao_s=video["duracao_s"],
                published_at=video["published_at"],
                collected_at=video["collected_at"],
            )
        )
    return itens


def _copiar_atomico(origem: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".parcial", dir=destino.parent)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(origem.read_bytes())
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, destino)
    except BaseException:
        Path(temporario).unlink(missing_ok=True)
        raise


def _ler_reuso(pasta: Path) -> dict:
    for nome in (ARQUIVO_REUSO, ARQUIVO_REUSO_ANTIGO):
        arquivo = Path(pasta) / nome
        if arquivo.is_file():
            try:
                return json.loads(arquivo.read_text(encoding="utf-8"))
            except json.JSONDecodeError as erro:
                raise ErroAnotacao(f"{nome} ilegível; não é seguro reaproveitar anotações") from erro
    return {"copias": [], "recusados": []}


def _copiar_validas(
    fonte: CacheAnotacoes,
    rotulo: str,
    pasta: Path,
    itens: Sequence[ItemExtracao],
    config: ConfiguracaoExtracao,
    *,
    agora: Callable[[], datetime],
    dados_da_copia: Callable[[str, dict], dict],
) -> dict:
    """Copia, byte a byte, os resultados válidos e compatíveis de `fonte` para os vídeos de `itens`.

    O registro é gravado logo depois de cada cópia. Se uma queda deixar uma cópia sem registro, a próxima chamada a
    reconhece pelo conteúdo idêntico ao da fonte e a registra, para que a retenção da origem não se perca.
    """
    destino = CacheAnotacoes(Path(pasta) / "cache", config)
    registro = _ler_reuso(pasta)
    registro["regra"] = REGRA_REUSO
    conhecidos = {r["video_id"] for r in registro["copias"]} | {
        r["video_id"] for r in registro["recusados"] if r.get("fonte", rotulo) == rotulo}
    for item in itens:
        caminho = destino.caminho(item.video_id)
        if item.video_id in conhecidos:
            continue
        try:
            resultado = fonte.ler(item.video_id)
        except CacheIncompativel:
            if not caminho.exists():
                registro["recusados"].append({"video_id": item.video_id, "fonte": rotulo,
                                              "motivo": "chave de compatibilidade diferente"})
            continue
        if resultado is None or resultado.estado != "ok":
            continue
        origem = fonte.caminho(item.video_id)
        dados = origem.read_bytes()
        if caminho.exists() and caminho.read_bytes() != dados:
            continue  # anotação produzida aqui, não cópia da fonte
        if not caminho.exists():
            _copiar_atomico(origem, caminho)
        registro["copias"].append(
            {"video_id": item.video_id, "fonte": rotulo, "origem": f"{rotulo}/{origem.name}",
             "sha256": hashlib.sha256(dados).hexdigest(), **dados_da_copia(item.video_id, json.loads(dados)),
             "copiado_em_utc": _iso(agora())}
        )
        gravar_json_atomico(Path(pasta) / ARQUIVO_REUSO, registro)
    gravar_json_atomico(Path(pasta) / ARQUIVO_REUSO, registro)
    return registro


def reaproveitar_piloto(
    pasta_cache_piloto: Path,
    pasta: Path,
    itens: Sequence[ItemExtracao],
    config: ConfiguracaoExtracao,
    *,
    agora: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict:
    """Reaproveita as anotações válidas e compatíveis da 1ª passada do piloto para os vídeos sorteados."""
    pasta_cache_piloto = Path(pasta_cache_piloto)
    if pasta_cache_piloto.name != "cache":
        raise ErroAnotacao("só a primeira passada do piloto (pasta cache) pode ser reaproveitada")
    prov_piloto = json.loads((pasta_cache_piloto.parent / "proveniencia.json").read_text(encoding="utf-8"))
    apagar_ate = prov_piloto["retencao"]["apagar_ate_utc"]

    def dados(_: str, original: dict) -> dict:
        return {"snapshot_id_original": original["item"]["snapshot_id"],
                "collected_at_original": original["item"]["collected_at"],
                "gravado_no_piloto_utc": original.get("gravado_em_utc"), "apagar_ate_utc": apagar_ate}

    return _copiar_validas(CacheAnotacoes(pasta_cache_piloto, config), f"{pasta_cache_piloto.parent.name}/cache",
                           pasta, itens, config, agora=agora, dados_da_copia=dados)


def reaproveitar_amostra(
    pasta_origem: Path,
    pasta: Path,
    itens: Sequence[ItemExtracao],
    config: ConfiguracaoExtracao,
    *,
    agora: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict:
    """Reaproveita as anotações válidas e compatíveis de outra amostra para os vídeos desta que também estão nela.

    Cada cópia herda a data-limite da origem: a da amostra de origem ou, se lá ela já era cópia do piloto, a do
    piloto. As requisições da origem não são copiadas e não contam como requisições desta amostra.
    """
    pasta_origem = Path(pasta_origem)
    if pasta_origem.resolve() == Path(pasta).resolve():
        raise ErroAnotacao("a amostra de origem e a de destino são a mesma pasta")
    retencao_origem = am.ler_amostra(pasta_origem / am.ARQUIVO)["retencao"]["apagar_ate_utc"]
    anteriores = {c["video_id"]: c for c in _ler_reuso(pasta_origem)["copias"]}

    def dados(video_id: str, original: dict) -> dict:
        anterior = anteriores.get(video_id)
        return {"snapshot_id_original": original["item"]["snapshot_id"],
                "apagar_ate_utc": anterior["apagar_ate_utc"] if anterior else retencao_origem,
                "origem_anterior": anterior}

    return _copiar_validas(CacheAnotacoes(pasta_origem / "cache", config), f"{pasta_origem.name}/cache", pasta,
                           itens, config, agora=agora, dados_da_copia=dados)


def _registrador(log: Path, item: ItemExtracao) -> Callable:
    def gravar(tentativa) -> None:
        dados = tentativa.model_dump(mode="json", include=set(CAMPOS_DO_LOG))
        linha = {"video_id": item.video_id, "duracao_video_s": item.duracao_s, **dados}
        with log.open("a", encoding="utf-8") as arquivo:
            arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")

    return gravar


def _anotar_lista(
    itens: Sequence[ItemExtracao],
    cache: CacheAnotacoes,
    cliente: ga.ClienteGemini,
    log: Path,
    *,
    politica: Politica,
    espera: Callable[[float], None],
    agora: Callable[[], datetime],
) -> str:
    """Anota os vídeos sem resultado no cache; devolve o motivo da parada."""
    falhas_seguidas = 0
    try:
        for item in itens:
            if cache.ler(item.video_id) is not None:
                continue
            resultado = anotar(item, cliente, cache.config, politica=politica, espera=espera, agora=agora,
                               ao_tentar=_registrador(log, item))
            if resultado.motivo_falha not in FALHAS_SEM_CACHE:
                cache.gravar(item, resultado, agora=agora())
            falhas_seguidas = 0 if resultado.estado == "ok" else falhas_seguidas + 1
            if falhas_seguidas >= MAX_FALHAS_CONSECUTIVAS:
                return "FALHAS_CONSECUTIVAS"
    except ExecucaoInterrompida as erro:
        return erro.codigo
    except KeyboardInterrupt:
        return "INTERROMPIDA_PELO_USUARIO"
    return "lista_concluida"


def executar(
    pasta: Path,
    itens: Sequence[ItemExtracao],
    cliente: ga.ClienteGemini,
    config: ConfiguracaoExtracao,
    *,
    politica: Politica = POLITICA_ANOTACAO,
    espera: Callable[[float], None] = time.sleep,
    agora: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict:
    """Uma execução sobre a lista congelada: registra a execução e devolve o progresso acumulado."""
    pasta = Path(pasta)
    cache = CacheAnotacoes(pasta / "cache", config)
    inicio = agora()
    pasta_execucao = pasta / "execucoes" / f"execucao={inicio.astimezone(UTC):%Y%m%dT%H%M%S%fZ}"
    pasta_execucao.mkdir(parents=True, exist_ok=False)
    log = pasta_execucao / "requisicoes.jsonl"
    log.touch()
    motivo = "ERRO_INESPERADO"
    try:
        motivo = _anotar_lista(itens, cache, cliente, log, politica=politica, espera=espera, agora=agora)
    finally:
        gravar_json_atomico(
            pasta_execucao / "execucao.json",
            {
                "inicio_utc": _iso(inicio),
                "fim_utc": _iso(agora()),
                "modelo": config.modelo,
                "chave_compatibilidade": config.chave_compatibilidade(),
                "limites": {
                    "max_requisicoes": cliente.limites.max_requisicoes,
                    "max_segundos_video": cliente.limites.max_segundos_video,
                    "intervalo_minimo_s": cliente.limites.intervalo_minimo_s,
                },
                "politica": {"max_tentativas_por_video": politica.max_tentativas_por_video,
                             "max_novas_tentativas_invalida": politica.max_novas_tentativas_invalida},
                "requisicoes_enviadas": cliente.requisicoes_enviadas,
                "segundos_video_enviados": cliente.segundos_video_enviados,
                "motivo_parada": motivo,
            },
        )
    return resumir(pasta, config)


def _linhas_do_registro(pasta: Path) -> list[dict]:
    """Requisições de todas as execuções. Só a última linha de um arquivo pode estar truncada (queda)."""
    linhas = []
    for arquivo in sorted(pasta.glob("execucoes/*/requisicoes.jsonl")):
        textos = [t for t in arquivo.read_text(encoding="utf-8").splitlines() if t]
        for posicao, texto in enumerate(textos):
            try:
                linhas.append(json.loads(texto))
            except json.JSONDecodeError:
                if posicao != len(textos) - 1:
                    raise
    return linhas


def _situacao(resultado, reaproveitado: bool, tentado: bool) -> str:
    if resultado is None:
        return "tentado_sem_resultado" if tentado else "nunca_tentado"
    if reaproveitado:
        return "reaproveitado"
    return "valido" if resultado.estado == "ok" else "falha"


def _contar(situacoes: list[tuple[str, bool]]) -> dict:
    contagem = Counter(s for s, _ in situacoes)
    return {
        "selecionados": len(situacoes),
        "tentados": sum(tentado for _, tentado in situacoes),
        "validos": contagem["valido"],
        "reaproveitados": contagem["reaproveitado"],
        "falhas": contagem["falha"],
        "tentados_sem_resultado": contagem["tentado_sem_resultado"],
        "pendentes": contagem["tentado_sem_resultado"] + contagem["nunca_tentado"],
    }


def resumir(pasta: Path, config: ConfiguracaoExtracao) -> dict:
    """Progresso acumulado por estrato e por lote, recursos usados em todas as execuções e retenção."""
    pasta = Path(pasta)
    amostra = am.ler_amostra(pasta / am.ARQUIVO)
    cache = CacheAnotacoes(pasta / "cache", config)
    reuso = {c["video_id"]: c for c in _ler_reuso(pasta)["copias"]}
    linhas = _linhas_do_registro(pasta)
    tentados = Counter(linha["video_id"] for linha in linhas)
    situacoes: dict[str, tuple[str, bool]] = {}
    por_estrato: dict[tuple, list[str]] = {}
    transcricao = Counter()
    for entrada in amostra["lista"]:
        video_id = entrada["video_id"]
        resultado = cache.ler(video_id)
        tentado = tentados[video_id] > 0
        situacoes[video_id] = (_situacao(resultado, video_id in reuso, tentado), tentado)
        por_estrato.setdefault((entrada["bloco"], entrada["setor"], entrada["tipo_canal"]), []).append(video_id)
        if resultado is not None:
            transcricao[resultado.transcricao_status] += 1
    execucoes = [json.loads(a.read_text(encoding="utf-8")) for a in sorted(pasta.glob("execucoes/*/execucao.json"))]
    pelo_cliente = sum(e["requisicoes_enviadas"] for e in execucoes)
    segundos_cliente = sum(e["segundos_video_enviados"] for e in execucoes)
    segundos_registro = sum(linha["duracao_video_s"] for linha in linhas)
    prazos = [amostra["retencao"]["apagar_ate_utc"], *(c["apagar_ate_utc"] for c in reuso.values())]
    lotes = am.ler_lotes(pasta, amostra)
    progresso = {
        "amostra": {"papel": amostra.get("papel", Papel.ALVO), "n_planejado": amostra["orcamento"]["n_planejado"],
                    "selecionados": amostra["selecionados"], "sha256_lista": amostra["sha256_lista"]},
        "configuracao": {"modelo": config.modelo, "temperatura": config.temperatura,
                         "chave_compatibilidade": config.chave_compatibilidade()},
        "totais": _contar(list(situacoes.values())),
        "estratos": [
            {"bloco": e[0], "setor": e[1], "tipo_canal": e[2], **_contar([situacoes[v] for v in por_estrato[e]])}
            for e in sorted(por_estrato, key=am._ordem_do_estrato)
        ],
        "lotes": [
            {"lote": lote["lote"], **_contar([situacoes[v] for v in lote["video_ids"]])}
            for lote in (lotes["lotes"] if lotes else [])
        ],
        "transcricao_status": dict(transcricao),
        "recursos": {
            "requisicoes": max(len(linhas), pelo_cliente),
            "segundos_video": max(segundos_registro, segundos_cliente),
            "novas_tentativas": sum(linha["numero"] > 1 for linha in linhas),
            "requisicoes_no_registro": len(linhas),
            "requisicoes_contadas_pelo_cliente": pelo_cliente,
            "regra": "o maior entre o registro por requisição e o contador do cliente: uma requisição cortada por "
            "interrupção é contada pelo cliente mesmo sem linha no registro; cópias não contam",
            "execucoes": len(execucoes),
        },
        "ultima_execucao": execucoes[-1] if execucoes else None,
        "retencao": {
            "apagar_ate_utc": min(prazos),
            "origem": "a data-limite mais próxima entre a da base e a de cada anotação copiada",
        },
    }
    gravar_json_atomico(pasta / ARQUIVO_PROGRESSO, progresso)
    return progresso


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.dados.anotacao", description="Anota uma amostra congelada.")
    parser.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n> da base curada")
    parser.add_argument("--amostra", type=Path, required=True, help="pasta da amostra, com amostra.json")
    parser.add_argument("--max-requisicoes", type=int, help="teto desta execução, com repetições")
    parser.add_argument("--max-segundos", type=int, help="teto de segundos de vídeo enviados, com reenvios")
    parser.add_argument("--intervalo", type=float, default=INTERVALO_PADRAO_S, help="segundos entre requisições")
    parser.add_argument("--piloto", type=Path, help="pasta cache da 1ª passada do piloto, para reuso restrito")
    parser.add_argument("--reaproveitar-amostra", type=Path, help="pasta de amostra anterior, para reuso restrito")
    parser.add_argument("--preparar", action="store_true", help="só reaproveita e congela lotes; nunca chama o Gemini")
    parser.add_argument("--tamanho-lote", type=int, help="com --preparar: congela lotes deste tamanho")
    parser.add_argument("--lote", type=int, help="número do lote congelado a anotar")
    return parser.parse_args(argv)


def _carregar(args: argparse.Namespace) -> tuple[dict, list[ItemExtracao]]:
    registrada = am.ler_amostra(args.amostra / am.ARQUIVO)
    particao = args.base.parent / registrada["particao"]["arquivo"] if registrada.get("particao") else None
    amostra = am.ler_amostra(args.amostra / am.ARQUIVO, pasta_base=args.base, arquivo_particao=particao)
    return amostra, itens_da_amostra(args.base, amostra)


def _preparar(args: argparse.Namespace, amostra: dict, config: ConfiguracaoExtracao) -> dict:
    """Congela os lotes (só na primeira vez) dos vídeos ainda sem resultado no cache e devolve o progresso.

    Com lotes já congelados, apenas os lê: repetir a preparação depois de anotar um lote não os remonta.
    Nenhuma requisição.
    """
    if args.tamanho_lote is not None and args.tamanho_lote < 1:
        raise ErroAnotacao("--tamanho-lote deve ser pelo menos 1")
    if args.tamanho_lote and am.ler_lotes(args.amostra, amostra) is None:
        cache = CacheAnotacoes(args.amostra / "cache", config)
        anotados = {e["video_id"] for e in amostra["lista"] if cache.ler(e["video_id"]) is not None}
        limites = {"max_requisicoes": am.requisicoes_para(args.tamanho_lote),
                   "max_segundos_video": args.max_segundos or MAX_SEGUNDOS_PADRAO,
                   "intervalo_minimo_s": args.intervalo}
        am.congelar_lotes(args.amostra, am.montar_lotes(args.base, amostra, anotados, tamanho=args.tamanho_lote,
                                                        limites=limites))
    return resumir(args.amostra, config)


def _itens_do_lote(args: argparse.Namespace, amostra: dict, itens: list[ItemExtracao]) -> list[ItemExtracao]:
    """Com lotes congelados, só os vídeos do lote pedido; os tetos não podem passar dos congelados."""
    lotes = am.ler_lotes(args.amostra, amostra)
    if lotes is None:
        if args.lote is not None:
            raise ErroAnotacao("esta amostra não tem lotes congelados")
        return itens
    if args.lote is None:
        raise ErroAnotacao("esta amostra tem lotes congelados: informe --lote")
    lote = next((l for l in lotes["lotes"] if l["lote"] == args.lote), None)
    if lote is None:
        raise ErroAnotacao(f"lote {args.lote} não existe")
    teto = lotes["limites_por_lote"]
    if (args.max_requisicoes > teto["max_requisicoes"] or args.max_segundos > teto["max_segundos_video"]
            or args.intervalo < teto["intervalo_minimo_s"]):
        raise ErroAnotacao(f"tetos acima dos congelados para o lote: {teto}")
    ids = set(lote["video_ids"])
    return [item for item in itens if item.video_id in ids]


def main(
    argv: Sequence[str] | None = None, *, cliente: ga.ClienteGemini | None = None, espera: Callable = time.sleep
) -> int:
    """0 = preparação, lista ou lote concluídos ou teto atingido; 2 = entrada inválida; 3 = erro fatal ou 3 falhas
    seguidas; 130 = interrompida pelo usuário."""
    args = _argumentos(argv)
    config = configuracao()
    try:
        amostra, itens = _carregar(args)
        if args.piloto:
            reaproveitar_piloto(args.piloto, args.amostra, itens, config)
        if args.reaproveitar_amostra:
            reaproveitar_amostra(args.reaproveitar_amostra, args.amostra, itens, config)
        if args.preparar:
            progresso = _preparar(args, amostra, config)
            print(json.dumps({"modo": "preparacao", "totais": progresso["totais"], "lotes": progresso["lotes"],
                              "recursos": progresso["recursos"], "retencao": progresso["retencao"]},
                             ensure_ascii=False, indent=1))
            return 0
        if args.max_requisicoes is None or args.max_segundos is None:
            raise ErroAnotacao("informe --max-requisicoes e --max-segundos para anotar")
        itens = _itens_do_lote(args, amostra, itens)
        limites = ga.Limites(args.max_requisicoes, args.max_segundos, args.intervalo)
        cliente = cliente or ga.ClienteGemini.a_partir_da_configuracao(MODELO, limites)
        with cliente:
            progresso = executar(args.amostra, itens, cliente, config, espera=espera)
    except (OSError, ValueError, KeyError, ChaveAusente, CacheIncompativel, CacheCorrompido) as erro:
        print(f"Anotação recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    motivo = progresso["ultima_execucao"]["motivo_parada"]
    resumo = {"motivo_parada": motivo, "lote": args.lote, "totais": progresso["totais"], "lotes": progresso["lotes"],
              "recursos": progresso["recursos"], "retencao": progresso["retencao"]}
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    if motivo == "INTERROMPIDA_PELO_USUARIO":
        return 130
    return 0 if motivo in PARADAS_NORMAIS else 3


if __name__ == "__main__":
    sys.exit(main())
