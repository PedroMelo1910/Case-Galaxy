"""Módulo consolidado. Reúne:
- Cache de resultados em memória: a mesma análise não repete extração, inferência nem chamada de API.
- Recursos compartilhados entre execuções e sessões do Streamlit (carregados uma vez por processo).
- Orquestra uma análise: validação, cache por conteúdo, arquivo temporário, adaptador. Sem dependência do Streamlit."""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.agente.contratos import Codigo, envelope_erro, ErroAgente
from src.config import RAIZ_PROJETO
from src.interface_web import adaptador as ad, arquivos as uploads
from src.interface_web.adaptador import Adaptador
from src.interface_web.arquivos import arquivo_temporario
from src.interface_web.validacao import (
    CanalBase,
    EntradaFormulario,
    ErroCampo,
    listar_canais_da_base,
    MODO_ONLINE,
    montar_pedido,
    pedido_de_resposta_salva,
    validar,
)


# ──── Cache de resultados em memória: a mesma análise não repete extração, inferência nem chamada de API.
CAPACIDADE_PADRAO = 32
VALIDADE_CONSULTA_AO_VIVO_S = 900.0


def chave_de_cache(sha256_do_arquivo: str, pedido_relevante: dict, identidade: str, origem: str) -> str:
    """Chave estável: muda se mudar o arquivo, qualquer entrada relevante, o modelo/extrator ou a origem."""
    corpo = json.dumps({"arquivo": sha256_do_arquivo, "pedido": pedido_relevante, "identidade": identidade,
                        "origem": origem}, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(corpo.encode("utf-8")).hexdigest()


class CacheResultados:
    def __init__(self, capacidade: int = CAPACIDADE_PADRAO, relogio: Callable[[], float] = time.monotonic) -> None:
        self._capacidade, self._relogio = capacidade, relogio
        self._itens: OrderedDict[str, tuple[dict, float | None]] = OrderedDict()
        self._trava = threading.Lock()

    def obter(self, chave: str) -> dict | None:
        with self._trava:
            item = self._itens.get(chave)
            if item is None:
                return None
            envelope, expira = item
            if expira is not None and self._relogio() >= expira:
                del self._itens[chave]
                return None
            self._itens.move_to_end(chave)
            return envelope

    def guardar(self, chave: str, envelope: dict, *, validade_s: float | None = None) -> bool:
        """Guarda só envelopes de sucesso; devolve se guardou."""
        if not envelope.get("success"):
            return False
        expira = None if validade_s is None else self._relogio() + validade_s
        with self._trava:
            self._itens[chave] = (envelope, expira)
            self._itens.move_to_end(chave)
            while len(self._itens) > self._capacidade:
                self._itens.popitem(last=False)
        return True

    def __len__(self) -> int:
        return len(self._itens)


# ──── Recursos compartilhados entre execuções e sessões do Streamlit (carregados uma vez por processo).
@st.cache_resource(show_spinner=False)
def servicos_reais() -> object:
    """`Servicos.padrao` (ou a versão online, se habilitada no ambiente) montado uma única vez. Falha não fica em cache."""
    return ad.fabrica_de_servicos(RAIZ_PROJETO)


@st.cache_resource(show_spinner=False)
def adaptador_real() -> ad.AdaptadorReal:
    return ad.AdaptadorReal(servicos_reais, [uploads.pasta_base()], raiz=RAIZ_PROJETO, online=ad.modo_online_ligado())


@st.cache_resource(show_spinner=False)
def adaptador_salvo() -> ad.AdaptadorSalvo | None:
    try:
        return ad.AdaptadorSalvo.padrao(RAIZ_PROJETO)
    except (OSError, ValueError, KeyError):
        return None


@st.cache_resource(show_spinner=False)
def cache_de_resultados() -> CacheResultados:
    return CacheResultados()


@st.cache_resource(show_spinner=False)
def preparar_diretorio_temporario() -> int:
    """Uma vez por processo: remove restos de execuções anteriores interrompidas."""
    return uploads.limpar_restos()


@st.cache_data(show_spinner=False)
def canais_da_base() -> tuple[tuple[CanalBase, ...], str | None]:
    """(canais da base curada, motivo da indisponibilidade ou None)."""
    try:
        return listar_canais_da_base(RAIZ_PROJETO), None
    except Exception:
        return (), "A base de canais não pôde ser lida neste computador."


@st.cache_data(ttl=60, show_spinner=False)
def estado_do_preditor() -> tuple[bool, str]:
    return ad.preditor_disponivel(RAIZ_PROJETO)


# ──── Orquestra uma análise: validação, cache por conteúdo, arquivo temporário, adaptador. Sem dependência
@dataclass(frozen=True)
class ResultadoAnalise:
    envelope: dict
    do_cache: bool
    origem: str


def envelope_de_formulario_invalido(problemas: list[ErroCampo]) -> dict:
    """Envelope de erro de entrada com um item por problema encontrado no formulário."""
    erro = ErroAgente(Codigo.ENTRADA_INVALIDA, "Complete as informações abaixo e clique em Analisar de novo.",
                      {"erros": [{"campo": p.campo, "problema": p.mensagem} for p in problemas]})
    return envelope_erro(erro)


def _sem_progresso(_: str) -> None:
    return None


def analisar(entrada: EntradaFormulario, canais: dict[str, CanalBase], conteudo: bytes, adaptador: Adaptador,
             cache: CacheResultados, *, pasta_base: Path | None = None,
             ao_progresso: Callable[[str], None] = _sem_progresso) -> ResultadoAnalise:
    """Nova previsão de um vídeo enviado. O envelope de erro volta como resultado; nada levanta para o chamador."""
    problemas = validar(entrada, canais)
    if problemas:
        return ResultadoAnalise(envelope_de_formulario_invalido(problemas), False, adaptador.origem)
    ao_progresso("Conferindo o arquivo e as informações")
    sha256 = hashlib.sha256(conteudo).hexdigest()
    chave = chave_de_cache(sha256, montar_pedido(entrada, canais, None), adaptador.identidade(), adaptador.origem)
    guardado = cache.obter(chave)
    if guardado is not None:
        return ResultadoAnalise(guardado, True, adaptador.origem)
    extensao = (entrada.arquivo_nome or "").rsplit(".", 1)[-1]
    ao_progresso("Extraindo atributos do vídeo e calculando a previsão (pode levar alguns minutos)")
    with arquivo_temporario(conteudo, extensao, pasta_base) as caminho:
        envelope = adaptador.prever(montar_pedido(entrada, canais, str(caminho)))
    validade = VALIDADE_CONSULTA_AO_VIVO_S if entrada.modo_canal == MODO_ONLINE else None
    cache.guardar(chave, envelope, validade_s=validade)
    return ResultadoAnalise(envelope, False, adaptador.origem)


def reproduzir(url: str, adaptador: Adaptador) -> ResultadoAnalise:
    """Reprodução de uma resposta salva (nunca conta como nova previsão)."""
    return ResultadoAnalise(adaptador.prever(pedido_de_resposta_salva(url)), False, adaptador.origem)
