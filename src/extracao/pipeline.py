"""Módulo consolidado. Reúne:
- Linearização da extração em colunas fixas (`avl_*`) e checagem de compatibilidade treino × inferência.
- Orquestrador da extração audiovisual local (TAPI §6.3, camada 1): vídeo real -> estados por etapa -> colunas `avl_*`.
- CLI da extração audiovisual local: `casegalaxy-extrair-av VIDEO [--saida JSON]`.
- Preparação dos pesos da camada audiovisual local: o ÚNICO passo desta camada que usa rede."""

import argparse
import gc
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.extracao import modelos as esp
from src.extracao.audio import analisar_fala, AudioDecodificado, decodificar_audio, medidas_energia, Transcritor
from src.extracao.cache import CacheExtracao, RAIZ_PADRAO
from src.extracao.embeddings import resumir_embeddings
from src.extracao.imagem import aplicar_extrator, resumir_rostos, resumir_texto
from src.extracao.modelos import (
    DadosInsuficientes,
    ErroExtracao,
    localizar,
    ModeloAusente,
    ModeloCorrompido,
    sem_caminhos,
    verificar_integridade,
)
from src.extracao.video import (
    contar_cortes,
    Deteccao,
    extrair_quadros,
    ler_pares_fluxo,
    ler_png,
    planejar_instantes,
    PlanoAmostragem,
    QuadroAmostrado,
    resumir_cor,
    resumir_movimento,
    resumir_objetos,
    SondaMidia,
    sondar_midia,
)


# ──── Linearização da extração em colunas fixas (`avl_*`) e checagem de compatibilidade treino × inferênci
PREFIXO = "avl_"
FAMILIA = "audiovisual_local"

# etapa -> {chave no dicionário da etapa: sufixo da coluna}
ATRIBUTOS: dict[str, dict[str, str]] = {
    "sondagem": {k: k for k in ("duracao_s", "largura", "altura", "fps", "orientacao", "tem_audio", "quadros_n",
                                "quadros_gancho_n")},
    "medidas": {k: k for k in ("brilho_medio", "brilho_std", "contraste_medio", "contraste_std", "saturacao_medio",
                               "saturacao_std", "cor_r", "cor_g", "cor_b", "cor_fracao_dominante")},
    "movimento": {k: k for k in ("fluxo_medio", "fluxo_std", "fluxo_p90", "diff_quadros_medio", "pares_n")},
    "cortes": {k: k for k in ("cortes_n", "cortes_por_10s", "cena_media_s", "cena_max_frac", "cortes_gancho_n")},
    "rosto": {k: k for k in ("rosto_frac_quadros", "rosto_n_max", "rosto_area_max", "rosto_gancho_frac")},
    "ocr": {k: k for k in ("ocr_frac_quadros", "ocr_caracteres_medio", "ocr_area_media", "ocr_gancho_frac")},
    "objetos": {k: k for k in ("obj_pessoa_frac_quadros", "obj_pessoa_area_max", "obj_veiculo_frac_quadros",
                               "obj_n_max", "obj_classes_distintas")},
    "embedding": {"emb_media": "emb_media", "emb_gancho": "emb_gancho"},
    "audio": {k: f"aud_{k}" for k in ("rms_db_medio", "rms_db_std", "pico_db", "fracao_ativa", "faixa_dinamica_db",
                                       "centroide_hz", "planura")},
    "fala": {"palavras": "fala_palavras", "palavras_por_s": "fala_palavras_por_s",
             "palavras_por_s_fala": "fala_palavras_por_s_fala", "fracao_tempo_fala": "fala_fracao_tempo",
             "idioma": "fala_idioma", "prob_idioma": "fala_prob_idioma"},
}
SUFIXOS: dict[str, tuple[str, ...]] = {etapa: tuple(chaves) for etapa, chaves in ATRIBUTOS.items()}
COLUNAS_VETORIAIS = (f"{PREFIXO}emb_media", f"{PREFIXO}emb_gancho")

_ESTADOS_PADRAO = frozenset({"ok", "falha_extrator", "nao_executada", "tempo_esgotado", "dependencia_falhou",
                             "dados_insuficientes"})  # insuficientes: entrada curta demais para medir (não é falha)
ESTADOS_VALIDOS: dict[str, frozenset[str]] = {etapa: _ESTADOS_PADRAO for etapa in ATRIBUTOS} | {
    "audio": _ESTADOS_PADRAO | {"sem_stream_audio", "falha_decodificacao"},
    "fala": (_ESTADOS_PADRAO - {"ok"}) | {"fala_transcrita", "audio_sem_fala_detectada", "transcricao_incerta",
                                          "sem_audio"},
}
ESTADO_OK: dict[str, str] = {etapa: "ok" for etapa in ATRIBUTOS} | {"fala": "fala_transcrita"}
_ESTADOS_COM_VALORES: dict[str, frozenset[str]] = {etapa: frozenset({"ok"}) for etapa in ATRIBUTOS} | {
    "fala": frozenset({"fala_transcrita", "audio_sem_fala_detectada"})}

COLUNAS_DE_ATRIBUTO: tuple[str, ...] = tuple(PREFIXO + sufixo for chaves in ATRIBUTOS.values()
                                             for sufixo in chaves.values())
COLUNAS_DE_ESTADO: tuple[str, ...] = tuple(f"{PREFIXO}estado_{etapa}" for etapa in ATRIBUTOS)
COLUNAS: tuple[str, ...] = COLUNAS_DE_ATRIBUTO + COLUNAS_DE_ESTADO


# Colunas que DESCREVEM o arquivo ou a política de amostragem e não o conteúdo: resolução, fps e contagens de quadros e
# de pares identificam a origem do arquivo (canal, perfil de aquisição) e virariam atalho de canal. Ficam na extração
# (rastreabilidade) e FORA dos preditores. `avl_duracao_s` continua preditor: a duração do Short é conhecida antes de publicar.
COLUNAS_DE_IMPRESSAO_DIGITAL: tuple[str, ...] = tuple(
    PREFIXO + s for s in ("largura", "altura", "fps", "quadros_n", "quadros_gancho_n", "pares_n"))
COLUNAS_PREDITORAS: tuple[str, ...] = tuple(c for c in COLUNAS if c not in COLUNAS_DE_IMPRESSAO_DIGITAL)

# Política única, igual no treino e na inferência: falha, falta de tempo, dependência que falhou ou etapa não executada
# recusam a previsão (ou, no treino, excluem a linha). `dados_insuficientes`, `sem_stream_audio` e `sem_audio` são
# estados legítimos do vídeo, não falhas, e não recusam.
ESTADOS_DE_RECUSA = frozenset({"falha_extrator", "falha_decodificacao", "tempo_esgotado", "dependencia_falhou",
                               "nao_executada"})


class ColunasIncompativeis(ValueError):
    """As colunas do inferidor não são exatamente as do treino."""


def motivos_de_recusa(linha: Mapping) -> list[str]:
    """Etapas cujo estado impede usar a linha como preditor; vazio = a linha pode seguir."""
    return [f"{etapa}: {estado}" for etapa in ATRIBUTOS
            if (estado := linha.get(f"{PREFIXO}estado_{etapa}", "nao_executada")) in ESTADOS_DE_RECUSA]


def exigir_mesma_identidade(identidade_do_treino: str, identidade_atual: str) -> None:
    """O extrator de hoje precisa ter a mesma identidade de configuração (parâmetros, modelos, versões) do treino."""
    if identidade_do_treino != identidade_atual:
        raise ColunasIncompativeis("identidade da configuração do extrator diferente da usada no treino: a extração de "
                                   "hoje não é comparável")


def _limpar(valor, coluna: str):
    """Mantém texto, None e vetores; recusa número não finito (um extrator em 'ok' não pode entregar NaN)."""
    if valor is None or isinstance(valor, str):
        return valor
    if isinstance(valor, (list, tuple)):
        return [_limpar(v, coluna) for v in valor]
    if isinstance(valor, bool):
        return int(valor)
    if isinstance(valor, int):
        return valor
    numero = float(valor)
    if not math.isfinite(numero):
        raise ValueError(f"valor não finito em {coluna}: um extrator saiu de 'ok' com número inválido")
    return numero


def linearizar(extracao: Mapping) -> dict:
    """Linha de colunas `avl_*` a partir de `extracao["etapas"]`; ausência e falha ficam None, nunca zero."""
    etapas = extracao.get("etapas", {})
    linha: dict = {}
    estados: dict[str, str] = {}
    for etapa, chaves in ATRIBUTOS.items():
        dados = etapas.get(etapa)
        estado = dados["estado"] if dados else "nao_executada"
        if estado not in ESTADOS_VALIDOS[etapa]:
            raise ValueError(f"estado desconhecido na etapa {etapa}: {estado!r}")
        estados[etapa] = estado
        com_valores = estado in _ESTADOS_COM_VALORES[etapa]
        for chave, sufixo in chaves.items():
            coluna = PREFIXO + sufixo
            linha[coluna] = _limpar(dados.get(chave), coluna) if com_valores else None
    for etapa, estado in estados.items():
        linha[f"{PREFIXO}estado_{etapa}"] = estado
    return linha


def alinhar_colunas(linha: Mapping, colunas_treino: Sequence[str], *, dimensao_vetores: int | None = None) -> dict:
    """A linha na ordem do treino; recusa coluna a mais ou a menos (nunca cria nem descarta uma em silêncio).

    Com `dimensao_vetores`, também recusa um embedding com tamanho diferente do usado no treino.
    """
    treino = list(colunas_treino)
    a_mais = sorted(set(linha) - set(treino))
    a_menos = sorted(set(treino) - set(linha))
    if a_mais or a_menos:
        raise ColunasIncompativeis(f"colunas incompatíveis com o treino; a mais: {a_mais}; a menos: {a_menos}")
    if dimensao_vetores is not None:
        for coluna in COLUNAS_VETORIAIS:
            vetor = linha.get(coluna)
            if vetor is not None and len(vetor) != dimensao_vetores:
                raise ColunasIncompativeis(f"{coluna} tem {len(vetor)} valores; o treino usou {dimensao_vetores}")
    return {coluna: linha[coluna] for coluna in treino}


# ──── Orquestrador da extração audiovisual local (TAPI §6.3, camada 1): vídeo real -> estados por etapa ->
MODELOS_NECESSARIOS = ("siglip2", "whisper", "yolox", "rosto", "ocr")
ESTADOS_CACHEAVEIS = frozenset({"ok", "sem_stream_audio", "sem_audio", "audio_sem_fala_detectada",
                                "transcricao_incerta", "fala_transcrita", "dados_insuficientes"})
ESTADOS_SEM_RESULTADO = frozenset({"falha_extrator", "tempo_esgotado", "dependencia_falhou"})


@dataclass
class Componentes:
    """Componentes pesados, injetáveis (os testes de fluxo usam dublês). Cada um carrega o modelo no primeiro uso."""

    transcritor: Transcritor
    detector_objetos: Callable[[np.ndarray], list[Deteccao]]
    detector_rosto: Callable[[np.ndarray], dict]
    leitor_texto: Callable[[np.ndarray], dict]
    codificador: Callable[[Sequence[np.ndarray]], np.ndarray]

    @classmethod
    def padrao(cls) -> "Componentes":
        from src.extracao.imagem import RostoPorQuadro, TextoPorQuadro
        from src.extracao.embeddings import CodificadorQuadros
        from src.extracao.audio import TranscritorWhisper
        from src.extracao.video import DetectorObjetos

        return cls(transcritor=TranscritorWhisper(), detector_objetos=DetectorObjetos(),
                   detector_rosto=RostoPorQuadro(), leitor_texto=TextoPorQuadro(), codificador=CodificadorQuadros())


def garantir_pesos(nomes: Sequence[str] = MODELOS_NECESSARIOS) -> None:
    """Falha cedo, antes de qualquer etapa, se faltar ou estiver corrompido algum peso (a inferência não baixa nada)."""
    for nome in nomes:
        localizar(nome)


def _finito(valor) -> bool:
    if isinstance(valor, float):
        return math.isfinite(valor)
    if isinstance(valor, (list, tuple)):
        return all(_finito(v) for v in valor)
    if isinstance(valor, dict):
        return all(_finito(v) for v in valor.values())
    return True


def _liberar(componente) -> None:
    liberar = getattr(componente, "liberar", None)
    if callable(liberar):
        liberar()
    gc.collect()


def proveniencia() -> dict:
    """Versões, identidade e licenças dos modelos e a configuração completa desta extração."""
    modelos = {nome: {"fonte": m.get("repo", m.get("url")), "revisao": m["revisao"], "papel": m["papel"],
                      "licenca_codigo": m["licenca_codigo"], "licenca_pesos": m["licenca_pesos"],
                      "arquivos": {a: d.get("sha256", d.get("git_blob")) for a, d in m["arquivos"].items()}}
               for nome, m in esp.MODELOS.items()}
    return {"python": platform.python_version(), "sistema": platform.system(),
            "bibliotecas": esp.versoes_das_bibliotecas(), "modelos": modelos, "configuracao": esp.configuracao()}


class _Execucao:
    """Estado de uma extração: cache, quadros em memória sob demanda, áudio decodificado uma vez e resultados."""

    def __init__(self, dono: "ExtratorAudiovisual", caminho: Path, sonda: SondaMidia, cache: CacheExtracao,
                 inicio: float) -> None:
        self.dono, self.caminho, self.sonda, self.cache, self.inicio = dono, caminho, sonda, cache, inicio
        self.plano: PlanoAmostragem = planejar_instantes(sonda.duracao_s, sonda.fps)
        self.etapas: dict[str, dict] = {}
        self.executadas: list[str] = []
        self.do_cache: list[str] = []
        self.cache_gravado = True  # vira False se alguma gravação falhar: o cache é opcional, a extração segue
        self._imagens: list[np.ndarray] | None = None
        self._audio: AudioDecodificado | None = None
        self._assinatura = self._assinatura_do_arquivo()

    def _assinatura_do_arquivo(self) -> tuple[int, int] | None:
        try:
            estado = self.caminho.stat()
        except OSError:
            return None
        return estado.st_size, estado.st_mtime_ns

    def _arquivo_mudou(self) -> bool:
        """O arquivo lido pelas etapas é o que a sonda mediu e o SHA-256 identificou? Se mudou, nada é gravado."""
        return self._assinatura is None or self._assinatura_do_arquivo() != self._assinatura

    def _gravar(self, nome: str, dados: dict) -> None:
        try:
            self.cache.gravar(nome, dados)
        except OSError:  # disco cheio, permissão, antivírus no Windows: o resultado vale mesmo sem cache
            self.cache_gravado = False

    # ---- execução de uma etapa -------------------------------------------------------------------------------
    def rodar(self, nome: str, calcular: Callable[[], dict], *, depende: Sequence[str] = (),
              liberar: object | None = None) -> None:
        faltando = [d for d in depende if self.etapas[d]["estado"] in ESTADOS_SEM_RESULTADO]
        if faltando:
            self.etapas[nome] = {"estado": "dependencia_falhou", "depende_de": faltando}
            return
        guardado = self.cache.ler(nome)
        if guardado is not None:
            self.etapas[nome] = {**guardado, "origem": "cache"}
            self.do_cache.append(nome)
            return
        if self.dono.relogio() - self.inicio > self.dono.orcamento_s:
            self.etapas[nome] = {"estado": "tempo_esgotado", "orcamento_s": self.dono.orcamento_s}
            return
        comeco = self.dono.relogio()
        dados = self._executar(calcular) if not self._arquivo_mudou() else self._alterado()
        if liberar is not None and self.dono.liberar_modelos:
            _liberar(liberar)
        if dados["estado"] != "falha_extrator" and self._arquivo_mudou():
            dados = self._alterado()  # mudou durante a etapa: o resultado pode misturar dois conteúdos
        dados["tempo_s"] = round(self.dono.relogio() - comeco, 3)
        if dados["estado"] in ESTADOS_CACHEAVEIS:
            self._gravar(nome, dados)
        self.etapas[nome] = {**dados, "origem": "executada"}
        self.executadas.append(nome)

    @staticmethod
    def _alterado() -> dict:
        return {"estado": "falha_extrator", "erro": "arquivo_alterado",
                "mensagem": "O arquivo mudou durante a extração; o resultado desta etapa não foi aproveitado nem gravado."}

    @staticmethod
    def _executar(calcular: Callable[[], dict]) -> dict:
        try:
            dados = calcular()
        except DadosInsuficientes as erro:
            return {"estado": "dados_insuficientes", "motivo": sem_caminhos(str(erro))}
        except ErroExtracao as erro:
            return {"estado": "falha_extrator", "erro": erro.codigo, "mensagem": erro.mensagem}
        except Exception as erro:  # noqa: BLE001 - toda falha de etapa é registrada, nunca vira ausência nem zero
            return {"estado": "falha_extrator", "erro": type(erro).__name__,
                    "mensagem": sem_caminhos(str(erro))[:200]}
        dados = {"estado": "ok", **dados} if "estado" not in dados else dict(dados)
        if not _finito(dados):
            return {"estado": "falha_extrator", "erro": "valor_nao_finito",
                    "mensagem": "um extrator devolveu número não finito"}
        return dados

    # ---- insumos compartilhados ------------------------------------------------------------------------------
    def quadros(self) -> list[QuadroAmostrado]:
        meta = self.etapas["quadros"]
        pasta = self.cache.diretorio_quadros()
        return [QuadroAmostrado(q["ordem"], q["indice_planejado"], q["instante_planejado_s"], q["instante_real_s"],
                                pasta / q["arquivo"], q["largura"], q["altura"]) for q in meta["quadros"]]

    def imagens(self) -> list[np.ndarray]:
        """Quadros RGB; se os PNGs já foram limpos (cache de uma execução anterior), são lidos de novo do vídeo."""
        if self._imagens is None:
            if not all(q.caminho.is_file() for q in self.quadros()):
                self._extrair_quadros()
            self._imagens = [ler_png(q.caminho) for q in self.quadros()]
        return self._imagens

    def n_gancho(self) -> int:
        return sum(1 for q in self.quadros() if q.instante_planejado_s < esp.JANELA_GANCHO_S)

    def audio(self) -> AudioDecodificado:
        if self._audio is None:
            self._audio = decodificar_audio(self.caminho, max_duracao_s=self.dono.max_duracao_s)
        return self._audio

    # ---- etapas ----------------------------------------------------------------------------------------------
    def _extrair_quadros(self) -> list[QuadroAmostrado]:
        return extrair_quadros(self.caminho, self.plano, self.cache.diretorio_quadros(),
                               rotacao_graus=self.sonda.rotacao_graus)

    def c_quadros(self) -> dict:
        quadros = self._extrair_quadros()
        return {"n": len(quadros), "poucos_quadros": len(quadros) < esp.MIN_QUADROS_TAPI,
                "quadros": [{"ordem": q.ordem, "indice_planejado": q.indice_planejado,
                             "instante_planejado_s": q.instante_planejado_s, "instante_real_s": q.instante_real_s,
                             "arquivo": q.caminho.name, "largura": q.largura, "altura": q.altura} for q in quadros]}

    def c_medidas(self) -> dict:
        return resumir_cor(self.imagens())

    def c_movimento(self) -> dict:
        s = self.sonda
        fim = max(s.duracao_s - (esp.FLUXO_SALTO_QUADROS + 1) / (s.fps or esp.FPS_PADRAO), 0.0)
        instantes = sorted({round(fim * (k + 0.5) / esp.FLUXO_PARES, 6) for k in range(esp.FLUXO_PARES)})  # sem repetir
        pares = ler_pares_fluxo(self.caminho, instantes, salto=esp.FLUXO_SALTO_QUADROS, lado_px=esp.FLUXO_LADO_PX,
                                rotacao_graus=s.rotacao_graus, fps=s.fps)
        cinzas = []
        for imagem in self.imagens():
            escala = min(1.0, esp.FLUXO_LADO_PX / max(imagem.shape[:2]))
            cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
            cinzas.append(cv2.resize(cinza, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA))
        return resumir_movimento(pares, cinzas)

    def c_cortes(self) -> dict:
        return contar_cortes(self.caminho, self.sonda.duracao_s, self.sonda.fps)

    def c_audio(self) -> dict:
        audio = self.audio()
        if audio.estado == "sem_stream_audio":
            return {"estado": "sem_stream_audio"}
        if audio.estado != "ok":
            return {"estado": "falha_decodificacao", "erro": audio.erro or "", "mensagem": audio.erro or ""}
        if audio.sinal.size < round(esp.AUDIO_JANELA_S * audio.taxa_hz):
            raise DadosInsuficientes("o áudio é mais curto que uma janela de análise")
        return medidas_energia(audio.sinal, audio.taxa_hz)  # sinal não finito levanta ValueError: é falha, não "curto"

    def c_fala(self) -> dict:
        return analisar_fala(self.audio(), self.etapas["audio"], self.sonda.duracao_s, self.dono.componentes.transcritor)

    def _por_quadro(self, funcao: Callable[[np.ndarray], dict]) -> list[dict]:
        return [aplicar_extrator(funcao, imagem) for imagem in self.imagens()]

    def c_rosto(self) -> dict:
        return resumir_rostos(self._por_quadro(self.dono.componentes.detector_rosto), self.n_gancho())

    def c_ocr(self) -> dict:
        return resumir_texto(self._por_quadro(self.dono.componentes.leitor_texto), self.n_gancho())

    def c_objetos(self) -> dict:
        return resumir_objetos([self.dono.componentes.detector_objetos(i) for i in self.imagens()])

    def c_embedding(self) -> dict:
        return resumir_embeddings(self.dono.componentes.codificador(self.imagens()), self.n_gancho())


class ExtratorAudiovisual:
    """Extrai atributos audiovisuais de um arquivo de vídeo local, sem nenhum serviço hospedado."""

    def __init__(self, *, cache_raiz: Path = RAIZ_PADRAO, componentes: Componentes | None = None,
                 verificar_pesos: bool | None = None, relogio: Callable[[], float] = time.monotonic,
                 orcamento_s: float = esp.ORCAMENTO_TOTAL_S, max_bytes: int = esp.MAX_BYTES,
                 max_duracao_s: float = esp.MAX_DURACAO_S, manter_quadros: bool = False,
                 liberar_modelos: bool = True) -> None:
        """`liberar_modelos=False` mantém os modelos carregados entre vídeos (uso em lote): mais memória, menos
        recargas; não altera nenhum resultado nem a identidade do cache."""
        self.liberar_modelos = liberar_modelos
        self.cache_raiz, self.relogio, self.orcamento_s = Path(cache_raiz), relogio, orcamento_s
        self.max_bytes, self.max_duracao_s, self.manter_quadros = max_bytes, max_duracao_s, manter_quadros
        self.verificar_pesos = componentes is None if verificar_pesos is None else verificar_pesos
        self.componentes = componentes or Componentes.padrao()
        cv2.setNumThreads(esp.THREADS)

    def extrair(self, caminho: Path) -> dict:
        """Resultado com `etapas` (estado de cada uma), `colunas` (`avl_*`), `proveniencia` e `resumo`.

        Levanta `ErroExtracao` se o arquivo não for um vídeo legível dentro dos limites, e `ModeloAusente` ou
        `ModeloCorrompido` se faltarem pesos, ambos ANTES de qualquer etapa.
        """
        inicio = self.relogio()
        if self.verificar_pesos:
            garantir_pesos()
        caminho = Path(caminho)
        sonda = sondar_midia(caminho, max_bytes=self.max_bytes, max_duracao_s=self.max_duracao_s)
        identidade = esp.identidade_da_configuracao()
        execucao = _Execucao(self, caminho, sonda, CacheExtracao(self.cache_raiz, sonda.sha256, identidade), inicio)
        c = self.componentes
        execucao.rodar("quadros", execucao.c_quadros)
        execucao.rodar("medidas", execucao.c_medidas, depende=("quadros",))
        execucao.rodar("movimento", execucao.c_movimento, depende=("quadros",))
        execucao.rodar("cortes", execucao.c_cortes)
        execucao.rodar("audio", execucao.c_audio)
        execucao.rodar("fala", execucao.c_fala, depende=("audio",), liberar=c.transcritor)
        execucao.rodar("rosto", execucao.c_rosto, depende=("quadros",), liberar=c.detector_rosto)
        execucao.rodar("ocr", execucao.c_ocr, depende=("quadros",), liberar=c.leitor_texto)
        execucao.rodar("objetos", execucao.c_objetos, depende=("quadros",), liberar=c.detector_objetos)
        execucao.rodar("embedding", execucao.c_embedding, depende=("quadros",), liberar=c.codificador)
        if not self.manter_quadros:
            execucao.cache.apagar_quadros()
        return self._resultado(execucao, identidade, inicio)

    def _resultado(self, x: _Execucao, identidade: str, inicio: float) -> dict:
        quadros = x.etapas["quadros"]
        sonda = {"estado": "ok", "duracao_s": x.sonda.duracao_s, "largura": x.sonda.largura, "altura": x.sonda.altura,
                 "fps": x.sonda.fps, "orientacao": x.sonda.orientacao, "tem_audio": int(x.sonda.tem_audio),
                 "rotacao_graus": x.sonda.rotacao_graus, "codec_video": x.sonda.codec_video,
                 "tamanho_bytes": x.sonda.tamanho_bytes,
                 "quadros_n": quadros.get("n") if quadros["estado"] == "ok" else None,
                 "quadros_gancho_n": x.n_gancho() if quadros["estado"] == "ok" else None}
        etapas = {"sondagem": sonda, **x.etapas}
        return {"versao": esp.VERSAO, "identidade": identidade, "sha256": x.sonda.sha256, "etapas": etapas,
                "colunas": linearizar({"etapas": etapas}), "proveniencia": proveniencia(),
                "limites": {"max_duracao_s": self.max_duracao_s, "max_bytes": self.max_bytes,
                            "orcamento_s": self.orcamento_s, "threads": esp.THREADS,
                            "lado_max_quadro_px": esp.LADO_MAX_QUADRO_PX},
                "resumo": {"estados": {nome: e["estado"] for nome, e in etapas.items()},
                           "executadas": x.executadas, "do_cache": x.do_cache, "cache_gravado": x.cache_gravado,
                           "tempo_total_s": round(self.relogio() - inicio, 3)}}


# ──── CLI da extração audiovisual local: `casegalaxy-extrair-av VIDEO [--saida JSON]`.
ESTADOS_BONS = frozenset({"ok", "sem_stream_audio", "sem_audio", "audio_sem_fala_detectada", "fala_transcrita",
                          "transcricao_incerta", "dados_insuficientes"})  # estados legítimos: nenhum é falha


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="casegalaxy-extrair-av",
                                     description="Extrai atributos audiovisuais de um vídeo local, sem rede.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--saida", type=Path, help="grava o resultado completo (etapas, colunas, proveniência) em JSON")
    parser.add_argument("--cache", type=Path, default=RAIZ_PADRAO, help="raiz do cache de extração")
    parser.add_argument("--manter-quadros", action="store_true", help="não apaga os quadros PNG ao terminar")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # os pesos nunca são baixados aqui; o preparo é outro comando
    try:
        resultado = ExtratorAudiovisual(cache_raiz=args.cache, manter_quadros=args.manter_quadros).extrair(args.video)
    except (ModeloAusente, ModeloCorrompido) as erro:
        sys.stderr.write(f"{erro}\n")
        return 3
    except ErroExtracao as erro:
        sys.stderr.write(f"{erro.codigo}: {erro.mensagem}\n")
        return 4
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    for nome, estado in resultado["resumo"]["estados"].items():
        etapa = resultado["etapas"][nome]
        sys.stdout.write(f"{nome:10s} {estado:26s} {etapa.get('tempo_s', '-')} s ({etapa.get('origem', '-')})\n")
    sys.stdout.write(f"total {resultado['resumo']['tempo_total_s']} s\n")
    return 0 if all(e in ESTADOS_BONS for e in resultado["resumo"]["estados"].values()) else 2


# ──── Preparação dos pesos da camada audiovisual local: o ÚNICO passo desta camada que usa rede.
FOLGA_DE_DISCO = 1.3


def bytes_a_baixar(nome: str, problemas: Sequence[str]) -> int:
    """Soma dos tamanhos fixados dos arquivos que faltam ou estão errados."""
    return sum(f["bytes"] for arquivo, f in esp.MODELOS[nome]["arquivos"].items()
               if any(arquivo in p for p in problemas))


def exigir_espaco(necessario: int, pasta: Path) -> None:
    """`OSError` se o disco não comporta o download com folga; confere antes de baixar, não depois."""
    pasta.mkdir(parents=True, exist_ok=True)
    livre = shutil.disk_usage(pasta).free
    if livre < necessario * FOLGA_DE_DISCO:
        raise OSError(f"Espaço insuficiente: {livre / 1e9:.1f} GB livres, {necessario * FOLGA_DE_DISCO / 1e9:.1f} GB "
                      "necessários (com folga).")


def _baixar_arquivo(url: str, destino: Path, sha256_esperado: str) -> None:
    """Baixa para um temporário no mesmo diretório, confere o SHA-256 e só então troca (nunca deixa arquivo parcial)."""
    import hashlib

    import httpx

    destino.parent.mkdir(parents=True, exist_ok=True)
    descritor, nome = tempfile.mkstemp(dir=destino.parent, prefix=destino.name + ".", suffix=".parcial")
    resumo = hashlib.sha256()
    try:
        with os.fdopen(descritor, "wb") as arquivo, httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            for bloco in r.iter_bytes(1_048_576):
                resumo.update(bloco)
                arquivo.write(bloco)
        if resumo.hexdigest() != sha256_esperado:
            raise ValueError(f"sha256 do download de {destino.name} difere do fixado")
        os.replace(nome, destino)
    finally:
        Path(nome).unlink(missing_ok=True)


def _baixar(nome: str, diretorio_modelos: Path, cache_hf: Path | None, forcar: bool) -> None:
    modelo = esp.MODELOS[nome]
    if nome in ("siglip2", "whisper"):
        from huggingface_hub import snapshot_download

        snapshot_download(modelo["repo"], revision=modelo["revisao"], allow_patterns=list(modelo["arquivos"]),
                          cache_dir=cache_hf, force_download=forcar)
    elif nome == "yolox":
        arquivo, fixado = next(iter(modelo["arquivos"].items()))
        _baixar_arquivo(modelo["url"], diretorio_modelos / "yolox" / arquivo, fixado["sha256"])
    elif nome == "rosto":
        from src.extracao.imagem import DetectorRosto

        # O extrator de capa só baixa se o arquivo não existir: um .tflite corrompido seria mantido para sempre.
        (diretorio_modelos / "mediapipe" / "blaze_face_short_range.tflite").unlink(missing_ok=True)
        DetectorRosto(diretorio=diretorio_modelos)._carregar()
    elif nome == "ocr":
        from src.extracao.imagem import LeitorTexto

        LeitorTexto(diretorio=diretorio_modelos)._carregar()


def preparar(nomes: Sequence[str] = MODELOS_NECESSARIOS, *, diretorio_modelos: Path | None = None,
             cache_hf: Path | None = None, baixar: bool = True) -> dict[str, dict]:
    """Estado de cada modelo depois da preparação: `ok` (já estava), `baixado` ou `falha`, com os problemas."""
    base = Path(diretorio_modelos or esp.DIRETORIO_MODELOS)
    relatorio: dict[str, dict] = {}
    for nome in nomes:
        modelo = esp.MODELOS[nome]
        problemas = verificar_integridade(nome, diretorio_modelos=base, cache_hf=cache_hf, completo=True)
        estado = "ok"
        if problemas and baixar:
            try:
                exigir_espaco(bytes_a_baixar(nome, problemas), base)
                _baixar(nome, base, cache_hf, forcar=any(not p.startswith("ausente") for p in problemas))
                problemas = verificar_integridade(nome, diretorio_modelos=base, cache_hf=cache_hf, completo=True)
                estado = "baixado" if not problemas else "falha"
            except Exception as erro:  # noqa: BLE001 - rede, disco ou hash: o relatório diz, o código de saída reflete
                problemas, estado = [*problemas, f"{type(erro).__name__}: {str(erro)[:160]}"], "falha"
        elif problemas:
            estado = "falha"
        relatorio[nome] = {"estado": estado, "problemas": problemas, "revisao": modelo["revisao"],
                           "licenca_codigo": modelo["licenca_codigo"], "licenca_pesos": modelo["licenca_pesos"],
                           "bytes": sum(f["bytes"] for f in modelo["arquivos"].values())}
    return relatorio


def main_preparar(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara (com rede) ou verifica (sem rede) os pesos locais.")
    parser.add_argument("--verificar", action="store_true", help="só verifica, sem baixar nada")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS_NECESSARIOS, default=list(MODELOS_NECESSARIOS))
    parser.add_argument("--registro", type=Path, help="grava o relatório em JSON neste arquivo")
    args = parser.parse_args(argv)
    relatorio = preparar(args.modelos, baixar=not args.verificar)
    for nome, r in relatorio.items():
        sys.stdout.write(f"{nome:8s} {r['estado']:8s} {r['bytes'] / 1e6:9.1f} MB  código: {r['licenca_codigo']} | "
                         f"pesos: {r['licenca_pesos']}\n")
        for problema in r["problemas"]:
            sys.stdout.write(f"    ! {problema}\n")
    if args.registro:
        args.registro.write_text(json.dumps(relatorio, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all(r["estado"] in ("ok", "baixado") for r in relatorio.values()) else 1


def _script_cli() -> None:
    sys.exit(main())


def _script_preparar() -> None:
    sys.exit(main_preparar())


_COMANDOS = {
    "cli": _script_cli,
    "preparar": _script_preparar,
}


def _despachar() -> None:
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in _COMANDOS:
        raise SystemExit("uso: python -m src.extracao.pipeline {" + "|".join(_COMANDOS) + "} [argumentos]")
    comando = sys.argv.pop(1)
    sys.argv[0] = f"{sys.argv[0]} {comando}"
    _COMANDOS[comando]()


if __name__ == "__main__":
    _despachar()
