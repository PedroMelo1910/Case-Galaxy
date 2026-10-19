"""Protocolo final da avaliação e portões de falha fechada (DECISOES D-D6, V15; protocolo final §10).

O teste só é consultado depois de: (1) protocolo congelado em JSON com SHA-256 (definições, hashes dos módulos, do
dataset, da amostra, da partição e da LIM, sorteios de canais, versões das bibliotecas); (2) passagem registrada do
`ecc:mle-reviewer` ligada a esse protocolo (APPROVE, ou WARN com aceite por escrito; BLOCK nunca abre); (3) nenhuma
consulta anterior. O marcador de consulta é criado ANTES da primeira leitura de rótulo de teste e não pode ser criado
de novo: a consulta é única. Qualquer divergência de hash recusa antes de marcar.
"""

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src import modelo_modalidades as mm
from src.avaliacao import baseline_inscritos as bi
from src.avaliacao import bootstrap as bs
from src.avaliacao import canais_retidos as cr
from src.avaliacao import ganho as gn
from src.avaliacao import ordenacao as od

VERSAO = "1"
ARQUIVO_PROTOCOLO = "protocolo_final.json"
ARQUIVO_SHA = "protocolo_final.sha256"
ARQUIVO_REVISAO = "revisao_mle_reviewer.json"
ARQUIVO_CONSULTA_INICIADA = "consulta_teste.iniciada.json"  # fase 1: portão aberto, nenhum rótulo lido
ARQUIVO_ROTULOS_LIDOS = "consulta_teste.rotulos_lidos.json"  # fase 2: o primeiro rótulo de teste vai ser lido
DECISOES_VALIDAS = ("APPROVE", "WARN", "BLOCK")
ESTADO_CONGELADO = "congelado_para_consulta_unica"

DEFINICOES = {
    "versao_do_protocolo": VERSAO,
    "intervalo_treino_teste_dias": 30,
    "defasagem_historico_dias": 30,
    "bootstrap": {"n_repeticoes": bs.N_REPETICOES, "semente": bs.SEMENTE, "nivel": bs.NIVEL},
    "iteracoes": {"n_estimators": 300, "parada_antecipada": False},
    "modelo": {"algoritmo": "LightGBM configuracao A (salvo regra pre-registrada de CatBoost, ver selecao)",
               "parametros": dict(mm.CONFIGURACAO_A), "ignorados_na_comparacao": ["n_jobs", "verbose"]},
    "blocos_anteriores": ["treino", "validacao"],
    "delta_relevancia_pratica": gn.DELTA_RELEVANCIA_PRATICA,
    "dobras": {"blocos": ["teste_1", "teste_2", "teste_3"], "janela": "expansiva",
               "treino": "linhas anotadas publicadas até o corte congelado da partição, com o intervalo de 30 dias"},
    "ordenacao": {"min_videos_por_celula": od.MIN_VIDEOS_CANAL, "fator_distinguivel": od.FATOR_DISTINGUIVEL,
                  "empate_previsto": 0.5, "unidade": "celula canal x sub-bloco; bootstrap sorteia canais"},
    "faixas_f1": {"n_faixas": 3, "bordas": "tercis do alvo relativo no treino de cada dobra",
                  "nota": "o TAPI cita quatro faixas sem pontos de corte; mantidas as três decididas (D-D3)"},
    "canais_nao_vistos": {"fracao": cr.FRACAO, "estratificacao": "setor x tipo_canal", "sementes": list(cr.SEMENTES),
                          "historico_na_inferencia": "proprio do canal (posts >= 30 dias antes), sem rotulos no treino"},
    "sensibilidade_exploratoria": {
        "S1": {"defasagem_historico_dias": 90, "intervalo_treino_teste_dias": 30},
        "S2": {"defasagem_historico_dias": 30, "intervalo_treino_teste_dias": 90}},
    "ablacao": {"niveis": ["E1", "E2", "E3", "E4"], "corte": "teste_1", "avalia": "teste_1 + teste_2 + teste_3",
                "contraste_principal": ["E4", "E3"], "contrastes_secundarios": [["E3", "E2"], ["E2", "E1"]],
                "mesmas_linhas_e_canais": True},
    "faixa": {"central": "q50", "limites": "minimo_e_maximo_das_tres_saidas", "rearranjo_total_dos_quantis": False,
              "largura": "limite_superior_menos_limite_inferior_da_faixa_corrigida",
              "nota": "regra aprovada em 15/10: o q50 nunca muda; sem cruzamento a faixa e q10-q90 brutos"},
    "cobertura": {"nivel_nominal": 0.8, "por": ["modelo", "setor"],
                  "nota": "medida no procedimento, sobre a faixa corrigida; sem alegar 80% sem o numero medido"},
    "lim": {"uso": "limiar de largura (p80 da largura da faixa corrigida do modelo de video) apos o congelamento; "
                   "sem rotulos", "percentil": 80,
            "pacotes": {"desenvolvimento": "modelo ajustado so com o treino da validacao; diagnostico validado com o "
                                           "limiar deste pacote",
                        "servido": "refit final com todos os dados; limiar operacional das larguras dele na LIM"},
            "transferencia": "cada limiar guarda o hash do pacote e recusa outro pacote: sem transferir limiar entre "
                             "pacotes (driver_lim)"},
    "selecao": {"algoritmo": "LightGBM configuracao A, salvo regra pre-registrada de CatBoost no desenvolvimento",
                "teste_nao_alimenta_selecao": True},
    "baseline_inscritos": {"definicao": bi.DEFINICAO, "tabela": bi.TABELA_CANAIS_RESTRITA,
                           "ajuste": "so no treino de cada dobra (e so nos canais mantidos nos sorteios de canais nao vistos)",
                           "nunca_atributo": True, "min_linhas_ajuste": bi.MIN_LINHAS_AJUSTE,
                           "min_canais_ajuste": bi.MIN_CANAIS_AJUSTE, "limitacoes": list(bi.LIMITACOES)},
}


# Módulos que definem o ajuste, o contexto e a avaliação: o hash do protocolo precisa cobri-los (revisão Python, HIGH 1).
MODULOS_EXIGIDOS = (
    "src/preprocessador.py", "src/modelo_desenvolvimento.py", "src/modelo_modalidades.py", "src/familias_atributos.py",
    "src/faixa_previsao.py", "src/historico.py", "src/dataset_desenvolvimento.py", "src/particoes.py",
    "src/avaliacao/protocolo_final.py", "src/avaliacao/contextos_finais.py", "src/avaliacao/fornecedor_real.py",
    "src/avaliacao/dobras.py", "src/avaliacao/final.py", "src/avaliacao/desenvolvimento.py",
    "src/avaliacao/relatorio.py", "src/avaliacao/ordenacao.py", "src/avaliacao/bootstrap.py",
    "src/avaliacao/faixas.py", "src/avaliacao/ganho.py", "src/avaliacao/canais_retidos.py",
    "src/avaliacao/limiar_lim.py", "src/avaliacao/driver_lim.py", "src/avaliacao/diagnosticos.py",
    "src/avaliacao/baseline_inscritos.py", "src/avaliacao/cobertura_setor.py", "src/avaliacao/declaracao_faixa.py", "src/avaliacao/execucao_final.py", "src/avaliacao/entradas_finais.py",
    "src/metricas.py", "src/curadoria.py",
    "src/contratos.py", "src/extracao_legenda.py", "src/rubrica.py", "uv.lock")


ARQUIVO_RETOMADA = "retomada_tecnica_{n:02d}.json"
# Numa retomada técnica depois da consulta, estes módulos NÃO podem mudar: definem método, alvo, seleção, população,
# métricas, o próprio portão ou a dependência. Os demais (orquestração, gravação, relatório, entradas, LIM) podem ser
# reparados com registro ligado à tentativa original; uma correção que exija mexer em um destes muda a metodologia.
MODULOS_DE_METODO = frozenset({
    "src/preprocessador.py", "src/modelo_desenvolvimento.py", "src/modelo_modalidades.py",
    "src/familias_atributos.py", "src/faixa_previsao.py", "src/historico.py", "src/dataset_desenvolvimento.py",
    "src/particoes.py", "src/metricas.py", "src/curadoria.py", "src/contratos.py", "src/extracao_legenda.py",
    "src/rubrica.py", "src/avaliacao/protocolo_final.py", "src/avaliacao/dobras.py", "src/avaliacao/ordenacao.py",
    "src/avaliacao/bootstrap.py", "src/avaliacao/faixas.py", "src/avaliacao/ganho.py",
    "src/avaliacao/canais_retidos.py", "src/avaliacao/limiar_lim.py", "uv.lock"})


class PortaoFechado(RuntimeError):
    """O teste não pode ser aberto: falta um pré-requisito do protocolo ou algo mudou depois do congelamento."""


_CHAVE = object()  # só `abrir_consulta_ao_teste` a conhece: ConsultaAberta não se constrói por fora


@dataclass(frozen=True)
class ConsultaAberta:
    """Prova de que o portão foi aberto: nasce só de `abrir_consulta_ao_teste` e carrega o protocolo congelado.

    Duas fases: a abertura (fase 1) não queima a consulta; ela só é consumida quando o primeiro rótulo de teste vai ser
    lido (`marcar_leitura`, fase 2), registrada no arquivo da pasta e no registro global por partição."""

    chave: object = field(repr=False)
    sha256_protocolo: str
    marcador: Path
    definicoes: Mapping
    revisao: str
    hash_particao: str
    arquivo_leitura: Path
    registro_global: Path

    def __post_init__(self) -> None:
        if self.chave is not _CHAVE:
            raise TypeError("ConsultaAberta só é criada por abrir_consulta_ao_teste")

    @property
    def aberta(self) -> bool:
        return True

    def get(self, chave: str, padrao=None):
        return {"aberta": True, "sha256_protocolo": self.sha256_protocolo, "revisao": self.revisao}.get(chave, padrao)

    def verificar(self) -> None:
        """O marcador criado ANTES da primeira leitura de rótulo ainda existe e é deste protocolo."""
        try:
            gravado = json.loads(Path(self.marcador).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise PortaoFechado("o marcador da consulta ao teste não existe ou está ilegível") from None
        if gravado.get("sha256_protocolo") != self.sha256_protocolo:
            raise PortaoFechado("o marcador da consulta pertence a outro protocolo")

    def marcar_leitura(self) -> None:
        """Fase 2: chamado ANTES de qualquer leitura de rótulo de teste. Idempotente para esta consulta."""
        try:
            _gravar_novo(self.arquivo_leitura, json.dumps(
                {"sha256_protocolo": self.sha256_protocolo, "hash_particao": self.hash_particao,
                 "rotulos_lidos_a_partir_de_utc": _agora()}, indent=1))
        except FileExistsError:
            return  # já marcada por esta consulta
        Path(self.registro_global).parent.mkdir(parents=True, exist_ok=True)
        with open(self.registro_global, "a", encoding="utf-8") as saida:
            saida.write(json.dumps({"fase": "rotulos_lidos", "hash_particao": self.hash_particao,
                                    "sha256_protocolo": self.sha256_protocolo, "em_utc": _agora()}) + "\n")


def _sha(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def _agora() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_dos_modulos(raiz: Path, modulos: Sequence[str]) -> dict[str, str]:
    saida = {}
    for relativo in modulos:
        arquivo = Path(raiz) / relativo
        if not arquivo.is_file():
            raise FileNotFoundError(f"módulo do protocolo ausente: {relativo}")
        saida[relativo] = _sha(arquivo.read_bytes())
    return dict(sorted(saida.items()))


def montar_protocolo(*, raiz: Path, modulos: Sequence[str], hash_dataset: str, hash_amostra: str, hash_particao: str,
                     hash_lim: str, especificacoes: Mapping[str, str], entradas: Mapping[str, str],
                     sorteios: Sequence[Mapping],
                     bibliotecas: Mapping[str, str], extras: Mapping | None = None,
                     modulos_exigidos: Sequence[str] = MODULOS_EXIGIDOS, coleta_utc: str | None = None) -> dict:
    """Protocolo completo, ainda não gravado. Não lê nenhum dado de teste: só definições, hashes e sorteios."""
    faltam = sorted(set(modulos_exigidos) - set(modulos))
    if faltam:
        raise ValueError(f"o hash do protocolo precisa cobrir também: {faltam}")
    definicoes = copy.deepcopy(DEFINICOES)
    if coleta_utc is not None:
        definicoes["coleta_dos_rotulos_utc"] = coleta_utc  # a configuração que roda tem de usar a mesma coleta
    return {
        "estado": ESTADO_CONGELADO, "teste_consultado": False, "montado_em_utc": _agora(),
        "definicoes": definicoes,
        "hashes": {"modulos": hash_dos_modulos(raiz, modulos), "dataset": hash_dataset, "amostra": hash_amostra,
                   "particao": hash_particao, "lim": hash_lim, "especificacoes": dict(especificacoes),
                   "entradas": dict(entradas)},
        "sorteios_de_canais": [dict(s) for s in sorteios], "bibliotecas": dict(bibliotecas),
        "extras": dict(extras or {}),
    }


def _gravar_novo(arquivo: Path, conteudo: str) -> None:
    with open(arquivo, "x", encoding="utf-8") as saida:  # "x": nunca sobrescreve
        saida.write(conteudo)


def congelar(protocolo: Mapping, pasta: Path) -> Path:
    """Grava `protocolo_final.json` e o seu SHA-256; recusa sobrescrever."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(protocolo, ensure_ascii=False, indent=1, sort_keys=True)
    arquivo = pasta / ARQUIVO_PROTOCOLO
    _gravar_novo(arquivo, texto)
    _gravar_novo(pasta / ARQUIVO_SHA, _sha(texto.encode("utf-8")) + "\n")
    return arquivo


def _ler_protocolo(pasta: Path) -> tuple[dict, str]:
    arquivo, sha_arquivo = Path(pasta) / ARQUIVO_PROTOCOLO, Path(pasta) / ARQUIVO_SHA
    if not (arquivo.is_file() and sha_arquivo.is_file()):
        raise PortaoFechado("protocolo congelado não encontrado: o teste não pode ser aberto")
    dados = arquivo.read_bytes()
    sha = _sha(dados)
    if sha != sha_arquivo.read_text(encoding="utf-8").strip():
        raise PortaoFechado("o SHA-256 do protocolo não confere com o registrado no congelamento")
    return json.loads(dados.decode("utf-8")), sha


def registrar_revisao_mle(pasta: Path, *, decisao: str, resumo: str, aceite_por_escrito: str | None = None) -> Path:
    """Registra a passagem do `ecc:mle-reviewer` ligada ao protocolo congelado."""
    if decisao not in DECISOES_VALIDAS:
        raise ValueError(f"decisão inválida: {decisao!r}; use {DECISOES_VALIDAS}")
    if decisao == "WARN" and not (aceite_por_escrito or "").strip():
        raise PortaoFechado("a decisão WARN só vale com aceite por escrito de Pedro")
    _, sha = _ler_protocolo(pasta)
    arquivo = Path(pasta) / ARQUIVO_REVISAO
    conteudo = {"sha256_protocolo": sha, "decisao": decisao, "resumo": resumo, "aceite_por_escrito": aceite_por_escrito,
                "registrado_em_utc": _agora(), "revisor": "ecc:mle-reviewer"}
    try:
        _gravar_novo(arquivo, json.dumps(conteudo, ensure_ascii=False, indent=1))  # BLOCK também fica registrado
    except FileExistsError:
        raise PortaoFechado("a revisão do mle-reviewer já registrada para este protocolo: passagem única") from None
    if decisao == "BLOCK":
        raise PortaoFechado("a decisão BLOCK do mle-reviewer não abre o portão: corrija e congele de novo")
    return arquivo


def _conferir_revisao(pasta: Path, sha_protocolo: str) -> dict:
    arquivo = Path(pasta) / ARQUIVO_REVISAO
    if not arquivo.is_file():
        raise PortaoFechado("falta o registro da passagem do mle-reviewer sobre este protocolo")
    revisao = json.loads(arquivo.read_text(encoding="utf-8"))
    if revisao.get("sha256_protocolo") != sha_protocolo:
        raise PortaoFechado("a revisão do mle-reviewer pertence a outro protocolo")
    if revisao.get("decisao") == "BLOCK" or revisao.get("decisao") not in ("APPROVE", "WARN"):
        raise PortaoFechado("decisão BLOCK ou inválida do mle-reviewer")
    if revisao["decisao"] == "WARN" and not (revisao.get("aceite_por_escrito") or "").strip():
        raise PortaoFechado("a decisão WARN do mle-reviewer não tem aceite por escrito")
    return revisao


def _exigir_igual(nome: str, atual, congelado) -> None:
    if atual != congelado:
        raise PortaoFechado(f"{nome} atual não é o do protocolo congelado")


def _exigir_dados_congelados(protocolo: Mapping, hash_dataset_atual: str, hash_amostra_atual: str,
                             hash_particao_atual: str, hash_lim_atual: str, bibliotecas_atuais: Mapping[str, str],
                             hashes_especificacoes_atuais: Mapping[str, str],
                             hashes_entradas_atuais: Mapping[str, str]) -> None:
    """Dados, partição, LIM, especificações, entradas e bibliotecas iguais aos do protocolo congelado."""
    hashes = protocolo["hashes"]
    _exigir_igual("o dataset", hash_dataset_atual, hashes["dataset"])
    _exigir_igual("a amostra", hash_amostra_atual, hashes["amostra"])
    _exigir_igual("a partição", hash_particao_atual, hashes["particao"])
    _exigir_igual("a LIM", hash_lim_atual, hashes["lim"])
    _exigir_igual("as especificações", dict(hashes_especificacoes_atuais), hashes["especificacoes"])
    _exigir_igual("as entradas do teste (anotações, legenda e capa, embeddings, uv.lock)",
                  dict(hashes_entradas_atuais), hashes["entradas"])
    _exigir_igual("as bibliotecas", dict(bibliotecas_atuais), protocolo["bibliotecas"])


def _consulta_ja_consumida(registro_global: Path, hash_particao: str) -> bool:
    if not Path(registro_global).is_file():
        return False
    for linha in Path(registro_global).read_text(encoding="utf-8").splitlines():
        if linha.strip() and (evento := json.loads(linha)).get("hash_particao") == hash_particao \
                and evento.get("fase") == "rotulos_lidos":
            return True
    return False


def abrir_consulta_ao_teste(pasta: Path, *, raiz: Path, registro_global: Path, hash_dataset_atual: str,
                            hash_amostra_atual: str, hash_particao_atual: str, hash_lim_atual: str,
                            bibliotecas_atuais: Mapping[str, str], hashes_especificacoes_atuais: Mapping[str, str],
                            hashes_entradas_atuais: Mapping[str, str],
                            validar_configuracao: Callable[[Mapping], None]) -> ConsultaAberta:
    """Confere todos os portões e a configuração e só então cria o marcador da fase 1. Recusa antes de marcar.

    A consulta é única por PARTIÇÃO (registro global), não por pasta: congelar de novo outro protocolo depois de ler
    rótulos de teste não abre uma segunda consulta."""
    pasta = Path(pasta)
    protocolo, sha = _ler_protocolo(pasta)
    if protocolo.get("teste_consultado") is not False:
        raise PortaoFechado("o protocolo não declara o teste como ainda não consultado")
    revisao = _conferir_revisao(pasta, sha)
    atuais = hash_dos_modulos(raiz, list(protocolo["hashes"]["modulos"]))
    mudaram = sorted(m for m, h in protocolo["hashes"]["modulos"].items() if atuais[m] != h)
    if mudaram:
        raise PortaoFechado(f"módulos alterados depois do congelamento: {mudaram}")
    hashes = protocolo["hashes"]
    _exigir_dados_congelados(protocolo, hash_dataset_atual, hash_amostra_atual, hash_particao_atual, hash_lim_atual,
                             bibliotecas_atuais, hashes_especificacoes_atuais, hashes_entradas_atuais)
    if _consulta_ja_consumida(registro_global, hashes["particao"]):
        raise PortaoFechado("a consulta ao teste é uma única por partição e já foi consumida: há rótulos de teste "
                            "lidos registrados para esta partição")
    validar_configuracao(protocolo["definicoes"])  # antes de marcar: configuração errada não queima a consulta
    marcador = pasta / ARQUIVO_CONSULTA_INICIADA
    if (pasta / ARQUIVO_ROTULOS_LIDOS).is_file():
        raise PortaoFechado("a consulta ao teste é uma única e já foi iniciada")
    try:
        _gravar_novo(marcador, json.dumps({"sha256_protocolo": sha, "iniciada_em_utc": _agora()}, indent=1))
    except FileExistsError:
        if json.loads(marcador.read_text(encoding="utf-8")).get("sha256_protocolo") != sha:
            raise PortaoFechado("o marcador da consulta pertence a outro protocolo") from None
    return ConsultaAberta(_CHAVE, sha256_protocolo=sha, marcador=marcador, definicoes=protocolo["definicoes"],
                          revisao=revisao["decisao"], hash_particao=hashes["particao"],
                          arquivo_leitura=pasta / ARQUIVO_ROTULOS_LIDOS, registro_global=Path(registro_global))


def _resultados_parciais(pasta: Path) -> dict[str, str]:
    """SHA-256 de cada arquivo já gravado nas pastas `resultados*` (tentativa original e retomadas anteriores)."""
    return {arquivo.relative_to(pasta).as_posix(): _sha(arquivo.read_bytes())
            for sub in sorted(pasta.iterdir()) if sub.is_dir() and sub.name.startswith("resultados")
            for arquivo in sorted(sub.rglob("*")) if arquivo.is_file()}


def _exigir_texto(nome: str, texto: str | None) -> None:
    if not (texto or "").strip():
        raise PortaoFechado(f"a retomada técnica exige {nome}")


def abrir_retomada_tecnica(pasta: Path, *, raiz: Path, registro_global: Path, modulos_reparados: Sequence[str],
                           motivo: str, autorizacao: str, declaracao_metodo_inalterado: str, hash_dataset_atual: str,
                           hash_amostra_atual: str, hash_particao_atual: str, hash_lim_atual: str,
                           bibliotecas_atuais: Mapping[str, str], hashes_especificacoes_atuais: Mapping[str, str],
                           hashes_entradas_atuais: Mapping[str, str],
                           validar_configuracao: Callable[[Mapping], None]) -> ConsultaAberta:
    """Retomada técnica depois de a consulta ao teste ter sido iniciada: método, alvo, seleção e população inalterados.

    Nada do que existe é apagado ou reescrito: o marcador, a leitura de rótulos, os resultados parciais e o registro
    global ficam como estão, e a retomada acrescenta `retomada_tecnica_NN.json` ligado à tentativa anterior, com o
    hash dos resultados parciais e o hash antigo e o novo de cada módulo reparado. Só módulos fora de
    `MODULOS_DE_METODO` podem mudar e a lista declarada tem de ser exatamente a dos que mudaram. Dados, partição,
    LIM, especificações, entradas e bibliotecas continuam iguais aos congelados e a configuração é confrontada com as
    definições. O teste nunca volta a ser descrito como não consultado."""
    pasta = Path(pasta)
    for nome, texto in (("motivo", motivo), ("autorizacao", autorizacao),
                        ("declaracao_metodo_inalterado", declaracao_metodo_inalterado)):
        _exigir_texto(nome, texto)
    protocolo, sha = _ler_protocolo(pasta)
    revisao = _conferir_revisao(pasta, sha)
    marcador = pasta / ARQUIVO_CONSULTA_INICIADA
    if not marcador.is_file():
        raise PortaoFechado("não há consulta iniciada para retomar: use a abertura normal do portão")
    if json.loads(marcador.read_text(encoding="utf-8")).get("sha256_protocolo") != sha:
        raise PortaoFechado("o marcador da consulta pertence a outro protocolo")
    congelados = protocolo["hashes"]["modulos"]
    atuais = hash_dos_modulos(raiz, list(congelados))
    mudaram = sorted(m for m, h in congelados.items() if atuais[m] != h)
    declarados = sorted(set(modulos_reparados))
    if mudaram != declarados:
        raise PortaoFechado(f"os módulos reparados declarados {declarados} não são exatamente os que mudaram "
                            f"{mudaram}")
    de_metodo = sorted(set(mudaram) & MODULOS_DE_METODO)
    if de_metodo:
        raise PortaoFechado(f"a correção altera módulos de método ({de_metodo}); isso não é retomada técnica: "
                            "pare o trabalho dependente e decida com Pedro")
    _exigir_dados_congelados(protocolo, hash_dataset_atual, hash_amostra_atual, hash_particao_atual, hash_lim_atual,
                             bibliotecas_atuais, hashes_especificacoes_atuais, hashes_entradas_atuais)
    validar_configuracao(protocolo["definicoes"])  # antes de gravar: configuração errada não deixa registro
    n = len(list(pasta.glob("retomada_tecnica_*.json"))) + 1
    leitura = pasta / ARQUIVO_ROTULOS_LIDOS
    registro = {
        "n": n, "tentativa_anterior": "original" if n == 1 else n - 1, "sha256_protocolo": sha,
        "marcador_original_sha256": _sha(marcador.read_bytes()),
        "rotulos_lidos_sha256": _sha(leitura.read_bytes()) if leitura.is_file() else None,
        "modulos_reparados": {m: {"congelado": congelados[m], "atual": atuais[m]} for m in mudaram},
        "resultados_parciais": _resultados_parciais(pasta), "motivo": motivo.strip(), "autorizacao": autorizacao.strip(),
        "declaracao_metodo_inalterado": declaracao_metodo_inalterado.strip(), "registrado_em_utc": _agora(),
    }
    _gravar_novo(pasta / ARQUIVO_RETOMADA.format(n=n), json.dumps(registro, ensure_ascii=False, indent=1))
    Path(registro_global).parent.mkdir(parents=True, exist_ok=True)
    with open(registro_global, "a", encoding="utf-8") as saida:
        saida.write(json.dumps({"fase": "retomada_tecnica", "n": n, "hash_particao": protocolo["hashes"]["particao"],
                                "sha256_protocolo": sha, "em_utc": _agora()}) + "\n")
    return ConsultaAberta(_CHAVE, sha256_protocolo=sha, marcador=marcador, definicoes=protocolo["definicoes"],
                          revisao=revisao["decisao"], hash_particao=protocolo["hashes"]["particao"],
                          arquivo_leitura=leitura, registro_global=Path(registro_global))
