"""Anotação de um vídeo LOCAL pela Files API do Gemini, com o mesmo modelo, rubrica, esquema e temperatura da rota por URL.

Fluxo (nada sai antes de a sonda local ter aprovado o arquivo e de o orçamento da execução caber):
cache por SHA-256 do arquivo → envio retomável → espera por ACTIVE (limitada) → conferência de tamanho, hash e duração →
`generateContent` com a URI do arquivo → exclusão do arquivo em `finally` (falha ao apagar vira aviso, nunca exceção).

Nunca há repetição: uma tentativa de geração por arquivo (cada repetição de um vídeo de 180 s custaria 180 s de cota).
Erro de cota diária interrompe o fluxo na hora. Só anotações válidas entram no cache, que guarda o SHA-256 e a anotação
por `RETENCAO_UPLOAD_DIAS` dias, nunca o nome, o caminho ou os bytes do vídeo. O arquivo no Gemini expira sozinho em 48 h
se a exclusão falhar; isso é dito no aviso.
"""

import logging
from dataclasses import dataclass, field, replace

from src import cache_gemini as cg
from src import extracao_gemini as eg
from src import gemini_api as ga
from src import rubrica as rb
from src.agente.video_local import MAX_DURACAO_S, SondaVideo
from src.contratos import Setor

logger = logging.getLogger(__name__)
REQUISICOES_MINIMAS = 4  # iniciar, enviar, uma consulta de estado e a geração (a exclusão ignora os tetos)
TOLERANCIA_DURACAO_S = 1.0


@dataclass(frozen=True)
class ResultadoUpload:
    """`estado`: ok | cache | indisponivel | falha_...; `anotacao` só existe em ok e cache."""

    anotacao: dict | None
    estado: str
    avisos: list[dict] = field(default_factory=list)


def _aviso_nao_apagado() -> dict:
    return {"classe": "atencao", "codigo": "arquivo_gemini_nao_apagado",
            "descricao": "O arquivo temporário enviado ao Gemini não pôde ser apagado agora; ele expira sozinho em 48 horas."}


@dataclass
class FonteAnotacaoArquivo:
    """Anotação de arquivo local. Sem `cliente` só lê o cache; sem nenhum dos dois, o estado é `indisponivel`."""

    config: eg.ConfiguracaoExtracao
    cache: cg.CacheUploads | None = None
    cliente: ga.ClienteGemini | None = None
    intervalo_estado_s: float = 5.0
    max_consultas_estado: int = 12

    def anotar(self, sonda: SondaVideo, setor: Setor) -> ResultadoUpload:
        guardada = self._do_cache(sonda, setor)
        if guardada is not None:
            return ResultadoUpload(guardada, "cache")
        if self.cliente is None:
            return ResultadoUpload(None, "indisponivel")
        try:
            self.cliente.verificar_orcamento(sonda.duracao_inteira_s, REQUISICOES_MINIMAS)
        except ga.ErroGemini as erro:
            return ResultadoUpload(None, _estado_de_erro(erro))
        return self._enviar_e_anotar(sonda, setor)

    def _do_cache(self, sonda: SondaVideo, setor: Setor) -> dict | None:
        if self.cache is None:
            return None
        try:
            return self.cache.ler(sonda.sha256, str(setor))
        except (cg.CacheIncompativel, cg.CacheCorrompido) as erro:
            logger.warning("cache de upload ignorado (%s)", type(erro).__name__)
            return None

    def _enviar_e_anotar(self, sonda: SondaVideo, setor: Setor) -> ResultadoUpload:
        try:
            enviado = self.cliente.enviar_arquivo(sonda.caminho, sonda.mime,
                                                  nome_exibicao=f"casegalaxy-{sonda.sha256[:12]}")
        except ga.ErroGemini as erro:  # sem nome devolvido não há o que apagar
            return ResultadoUpload(None, _estado_de_erro(erro))
        avisos: list[dict] = []
        try:
            resultado = self._anotar_arquivo_enviado(sonda, setor, enviado)
        except ga.ErroGemini as erro:
            resultado = ResultadoUpload(None, _estado_de_erro(erro))
        finally:
            self._apagar(enviado.nome, avisos)
        return replace(resultado, avisos=avisos)

    def _anotar_arquivo_enviado(self, sonda: SondaVideo, setor: Setor, enviado: ga.ArquivoGemini) -> ResultadoUpload:
        divergencia = _divergencia(sonda, enviado)
        if divergencia is not None:
            return ResultadoUpload(None, divergencia)
        ativo = self.cliente.aguardar_ativo(enviado.nome, intervalo_s=self.intervalo_estado_s,
                                            max_consultas=self.max_consultas_estado)
        divergencia = _divergencia_de_duracao(ativo)
        if divergencia is not None:
            return ResultadoUpload(None, divergencia)
        anotacao, estado = self._gerar(sonda, setor, ativo.uri)
        if anotacao is not None and self.cache is not None:
            self._guardar(sonda, setor, anotacao)
        return ResultadoUpload(anotacao, estado)

    def _guardar(self, sonda: SondaVideo, setor: Setor, anotacao: dict) -> None:
        """A anotação já foi paga em cota: falha de disco no cache não pode jogá-la fora."""
        try:
            self.cache.gravar(sonda.sha256, str(setor), anotacao)
        except OSError:
            logger.warning("o cache de uploads não pôde ser gravado; a anotação segue sem cache")

    def _gerar(self, sonda: SondaVideo, setor: Setor, uri: str) -> tuple[dict | None, str]:
        corpo = eg.corpo_para_arquivo(uri, sonda.mime, setor, self.config)
        resposta = self.cliente.gerar(corpo, duracao_video_s=sonda.duracao_inteira_s)
        if resposta.situacao != "ok" or resposta.texto is None:
            return None, f"falha_resposta_{resposta.situacao}"
        try:
            return rb.validar_resposta(resposta.texto, setor).model_dump(mode="json"), "ok"
        except rb.RespostaInvalida as erro:
            return None, f"falha_invalida_{erro.categoria}"

    def _apagar(self, nome: str, avisos: list[dict]) -> None:
        try:
            self.cliente.apagar_arquivo(nome)
        except Exception as falha:
            logger.warning("arquivo temporário do Gemini não apagado (%s)", type(falha).__name__)
            avisos.append(_aviso_nao_apagado())


def _estado_de_erro(erro: ga.ErroGemini) -> str:
    return "falha_cota_esgotada" if isinstance(erro, ga.CotaDiariaEsgotada) else f"falha_{type(erro).__name__}"


def _divergencia(sonda: SondaVideo, enviado: ga.ArquivoGemini) -> str | None:
    if enviado.sha256_enviado != sonda.sha256:
        return "falha_arquivo_alterado"
    if enviado.tamanho_bytes is not None and enviado.tamanho_bytes != sonda.tamanho_bytes:
        return "falha_tamanho_divergente"
    return None


def _divergencia_de_duracao(ativo: ga.ArquivoGemini) -> str | None:
    if ativo.duracao_s is not None and ativo.duracao_s > MAX_DURACAO_S + TOLERANCIA_DURACAO_S:
        return "falha_duracao_do_arquivo"
    return None
