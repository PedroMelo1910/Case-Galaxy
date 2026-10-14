"""Executor retomável da anotação pelo Gemini sobre a amostra congelada (D-E2, D-E3).

Percorre a lista congelada na ordem registrada. Cada vídeo com resultado no cache (anotação válida ou falha de
extração já registrada) é pulado; os demais passam pela extração existente (`extracao_gemini.anotar`), com a mesma
política de tentativas. O pedido leva só a URL pública e o setor (contrato `ItemExtracao`): nunca curtidas,
visualizações, comentários, inscritos ou histórico de desempenho.

Registro e retomada:
- toda requisição enviada vai, assim que termina, para `execucoes/execucao=<UTC>/requisicoes.jsonl`, com a duração
  do vídeo enviado; repetições e falhas contam;
- anotações válidas e falhas de conteúdo (`resposta_invalida`) vão para o cache, uma por arquivo, com gravação
  atômica; falhas que nada dizem sobre o vídeo (pedido recusado, tentativas esgotadas) não vão, e o vídeo volta a
  ser tentado numa próxima execução;
- tetos de requisições e de segundos de vídeo, cota diária, chave inválida e 3 falhas seguidas param a execução sem
  perder o que já foi gravado.

Reuso do piloto: só a primeira passada (`.../cache`), só vídeos sorteados nesta amostra, só anotações válidas com a
mesma chave de compatibilidade. O arquivo é copiado byte a byte, com a origem e a data-limite de retenção do piloto.

Exemplo:
    uv run python -m src.anotacao --base data/processed/snapshot=<id>/curadoria=v1 \\
        --amostra data/annotations/gemini/snapshot=<id>/amostra=v1 --max-requisicoes 100 --max-segundos 7200
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from src import amostra_anotacao as am
from src import curadoria, particoes
from src import gemini_api as ga
from src.cache_gemini import CacheAnotacoes, CacheCorrompido, CacheIncompativel, gravar_json_atomico
from src.config import ChaveAusente
from src.contratos import ItemExtracao, Papel, url_do_video
from src.extracao_gemini import POLITICA_ANOTACAO, ConfiguracaoExtracao, ExecucaoInterrompida, Politica, anotar

MODELO = "gemini-3.5-flash-lite"
TEMPERATURA = 1.0
INTERVALO_PADRAO_S = 5.0
FALHAS_SEM_CACHE = frozenset({"requisicao_recusada", "tentativas_esgotadas"})
MAX_FALHAS_CONSECUTIVAS = 3
CAMPOS_DO_LOG = ("fase", "numero", "instante_utc", "resultado", "http_status", "duracao_s", "uso", "versao_modelo")
ARQUIVO_REUSO = "reuso_piloto.json"
ARQUIVO_PROGRESSO = "progresso.json"
PARADAS_NORMAIS = frozenset({"lista_concluida", ga.LimiteDaExecucao.codigo})
COLUNAS_DO_ITEM = ["video_id", "channel_id", "snapshot_id", "papel", "thumbnail_url", "duracao_s", "published_at",
                   "collected_at"]


class ErroAnotacao(ValueError):
    """Entrada da anotação inconsistente (amostra, base ou piloto)."""


def configuracao() -> ConfiguracaoExtracao:
    """Configuração aprovada (D-E3): modelo, temperatura e o pedido e o esquema atuais da rubrica."""
    return ConfiguracaoExtracao(modelo=MODELO, temperatura=TEMPERATURA)


def _iso(instante: datetime) -> str:
    return instante.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def itens_da_amostra(pasta_base: Path, amostra: dict) -> list[ItemExtracao]:
    """Itens de extração na ordem congelada, só com colunas sem resultado de desempenho."""
    tabelas = curadoria.ler_base(pasta_base)
    setor = {c["channel_id"]: c["setor"] for c in tabelas["canais.parquet"].select(["channel_id", "setor"]).to_pylist()}
    videos = {v["video_id"]: v for v in tabelas["videos.parquet"].select(COLUNAS_DO_ITEM).to_pylist()}
    itens = []
    for entrada in amostra["lista"]:
        video = videos.get(entrada["video_id"])
        if video is None or video["papel"] != Papel.ALVO or video["channel_id"] != entrada["channel_id"]:
            raise ErroAnotacao(f"{entrada['video_id']}: não é alvo do canal registrado na base")
        itens.append(
            ItemExtracao(
                video_id=video["video_id"],
                channel_id=video["channel_id"],
                snapshot_id=video["snapshot_id"],
                papel=Papel.ALVO,
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
    arquivo = pasta / ARQUIVO_REUSO
    if not arquivo.is_file():
        return {"copias": [], "recusados": []}
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))
    except json.JSONDecodeError as erro:
        raise ErroAnotacao(f"{ARQUIVO_REUSO} ilegível; não é seguro reaproveitar o piloto") from erro


def reaproveitar_piloto(
    pasta_cache_piloto: Path,
    pasta: Path,
    itens: Sequence[ItemExtracao],
    config: ConfiguracaoExtracao,
    *,
    agora: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict:
    """Copia, byte a byte, as anotações válidas e compatíveis da 1ª passada do piloto para os vídeos sorteados.

    O registro é gravado logo depois de cada cópia. Se uma queda deixar uma cópia sem registro, a próxima chamada a
    reconhece pelo conteúdo idêntico ao do piloto e a registra, para que a retenção do piloto não se perca.
    """
    pasta_cache_piloto = Path(pasta_cache_piloto)
    if pasta_cache_piloto.name != "cache":
        raise ErroAnotacao("só a primeira passada do piloto (pasta cache) pode ser reaproveitada")
    prov_piloto = json.loads((pasta_cache_piloto.parent / "proveniencia.json").read_text(encoding="utf-8"))
    apagar_ate = prov_piloto["retencao"]["apagar_ate_utc"]
    piloto, destino = CacheAnotacoes(pasta_cache_piloto, config), CacheAnotacoes(pasta / "cache", config)
    registro = _ler_reuso(pasta)
    registro["regra"] = ("só a 1ª passada, só vídeos sorteados, só anotações válidas com a mesma chave; cópia "
                         "byte a byte; a retenção do piloto não é renovada")
    conhecidos = {r["video_id"] for r in registro["copias"] + registro["recusados"]}
    for item in itens:
        caminho = destino.caminho(item.video_id)
        if item.video_id in conhecidos:
            continue
        try:
            resultado = piloto.ler(item.video_id)
        except CacheIncompativel:
            if not caminho.exists():
                registro["recusados"].append({"video_id": item.video_id,
                                              "motivo": "chave de compatibilidade diferente"})
            continue
        if resultado is None or resultado.estado != "ok":
            continue
        origem = piloto.caminho(item.video_id)
        dados = origem.read_bytes()
        if caminho.exists() and caminho.read_bytes() != dados:
            continue  # anotação produzida por este executor, não cópia do piloto
        original = json.loads(dados)
        if not caminho.exists():
            _copiar_atomico(origem, caminho)
        registro["copias"].append(
            {
                "video_id": item.video_id,
                "origem": f"{pasta_cache_piloto.parent.name}/cache/{origem.name}",
                "sha256": hashlib.sha256(dados).hexdigest(),
                "snapshot_id_original": original["item"]["snapshot_id"],
                "collected_at_original": original["item"]["collected_at"],
                "gravado_no_piloto_utc": original.get("gravado_em_utc"),
                "apagar_ate_utc": apagar_ate,
                "copiado_em_utc": _iso(agora()),
            }
        )
        gravar_json_atomico(pasta / ARQUIVO_REUSO, registro)
    gravar_json_atomico(pasta / ARQUIVO_REUSO, registro)
    return registro


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
    """Progresso acumulado por estrato, recursos usados em todas as execuções e retenção; grava progresso.json."""
    pasta = Path(pasta)
    amostra = am.ler_amostra(pasta / am.ARQUIVO)
    cache = CacheAnotacoes(pasta / "cache", config)
    reuso = {c["video_id"]: c for c in _ler_reuso(pasta)["copias"]}
    linhas = _linhas_do_registro(pasta)
    tentados = Counter(linha["video_id"] for linha in linhas)
    por_estrato: dict[tuple, list[tuple[str, bool]]] = {}
    transcricao = Counter()
    for entrada in amostra["lista"]:
        resultado = cache.ler(entrada["video_id"])
        tentado = tentados[entrada["video_id"]] > 0
        estrato = (entrada["bloco"], entrada["setor"], entrada["tipo_canal"])
        por_estrato.setdefault(estrato, []).append((_situacao(resultado, entrada["video_id"] in reuso, tentado),
                                                    tentado))
        if resultado is not None:
            transcricao[resultado.transcricao_status] += 1
    execucoes = [json.loads(a.read_text(encoding="utf-8")) for a in sorted(pasta.glob("execucoes/*/execucao.json"))]
    pelo_cliente = sum(e["requisicoes_enviadas"] for e in execucoes)
    segundos_cliente = sum(e["segundos_video_enviados"] for e in execucoes)
    segundos_registro = sum(linha["duracao_video_s"] for linha in linhas)
    prazos = [amostra["retencao"]["apagar_ate_utc"], *(c["apagar_ate_utc"] for c in reuso.values())]
    progresso = {
        "amostra": {"n_planejado": amostra["orcamento"]["n_planejado"], "selecionados": amostra["selecionados"],
                    "sha256_lista": amostra["sha256_lista"]},
        "configuracao": {"modelo": config.modelo, "temperatura": config.temperatura,
                         "chave_compatibilidade": config.chave_compatibilidade()},
        "totais": _contar([s for lista in por_estrato.values() for s in lista]),
        "estratos": [
            {"bloco": e[0], "setor": e[1], "tipo_canal": e[2], **_contar(por_estrato[e])}
            for e in sorted(por_estrato, key=lambda e: (particoes.BLOCOS.index(e[0]), e[1], e[2]))
        ],
        "transcricao_status": dict(transcricao),
        "recursos": {
            "requisicoes": max(len(linhas), pelo_cliente),
            "segundos_video": max(segundos_registro, segundos_cliente),
            "requisicoes_no_registro": len(linhas),
            "requisicoes_contadas_pelo_cliente": pelo_cliente,
            "regra": "o maior entre o registro por requisição e o contador do cliente: uma requisição cortada por "
            "interrupção é contada pelo cliente mesmo sem linha no registro",
            "execucoes": len(execucoes),
        },
        "ultima_execucao": execucoes[-1] if execucoes else None,
        "retencao": {
            "apagar_ate_utc": min(prazos),
            "origem": "a data-limite mais próxima entre a da base e a de cada anotação copiada do piloto",
        },
    }
    gravar_json_atomico(pasta / ARQUIVO_PROGRESSO, progresso)
    return progresso


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.anotacao", description="Anota a amostra congelada.")
    parser.add_argument("--base", type=Path, required=True, help="pasta curadoria=v<n> da base curada")
    parser.add_argument("--amostra", type=Path, required=True, help="pasta amostra=v<n> com amostra.json")
    parser.add_argument("--max-requisicoes", type=int, required=True, help="teto desta execução, com repetições")
    parser.add_argument("--max-segundos", type=int, required=True, help="teto de segundos de vídeo enviados")
    parser.add_argument("--intervalo", type=float, default=INTERVALO_PADRAO_S, help="segundos entre requisições")
    parser.add_argument("--piloto", type=Path, help="pasta cache da 1ª passada do piloto, para reuso restrito")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, *, cliente: ga.ClienteGemini | None = None, espera: Callable = time.sleep
) -> int:
    """0 = lista concluída ou teto atingido; 2 = entrada inválida; 3 = erro fatal ou 3 falhas seguidas;
    130 = interrompida pelo usuário."""
    args = _argumentos(argv)
    config = configuracao()
    try:
        arquivo_amostra = args.amostra / am.ARQUIVO
        registrada = am.ler_amostra(arquivo_amostra)
        arquivo_particao = args.base.parent / registrada["particao"]["arquivo"]
        amostra = am.ler_amostra(arquivo_amostra, pasta_base=args.base, arquivo_particao=arquivo_particao)
        itens = itens_da_amostra(args.base, amostra)
        if args.piloto:
            reaproveitar_piloto(args.piloto, args.amostra, itens, config)
        limites = ga.Limites(args.max_requisicoes, args.max_segundos, args.intervalo)
        cliente = cliente or ga.ClienteGemini.a_partir_da_configuracao(MODELO, limites)
        with cliente:
            progresso = executar(args.amostra, itens, cliente, config, espera=espera)
    except (OSError, ValueError, KeyError, ChaveAusente, CacheIncompativel, CacheCorrompido) as erro:
        print(f"Anotação recusada ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    motivo = progresso["ultima_execucao"]["motivo_parada"]
    resumo = {"motivo_parada": motivo, "totais": progresso["totais"], "recursos": progresso["recursos"],
              "retencao": progresso["retencao"]}
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    if motivo == "INTERROMPIDA_PELO_USUARIO":
        return 130
    return 0 if motivo in PARADAS_NORMAIS else 3


if __name__ == "__main__":
    sys.exit(main())
