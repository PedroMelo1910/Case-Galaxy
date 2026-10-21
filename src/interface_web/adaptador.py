"""Adaptadores de previsão: a interface chama `prever(pedido) -> envelope`, o mesmo contrato de `src.agente.api.prever`."""

import copy
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from src.agente.contratos import Codigo, envelope_erro, ErroAgente, video_id_da_url
from src.config import RAIZ_PROJETO
from src.demonstracao import ErroDemonstracao, ler_respostas, gravar_respostas, PASTA_DEMONSTRACAO, PASTA_RESPOSTAS
from src.interface_web.textos import ocultar_caminhos


# ──── Adaptadores de previsão: a interface chama `prever(pedido) -> envelope`, o mesmo contrato de `src.ag
logger = logging.getLogger(__name__)

# Duas sessões do navegador compartilham os mesmos serviços e o mesmo extrator: uma previsão por vez evita carregar os
# modelos duas vezes com pouca memória.
_TRAVA_DE_PREVISAO = threading.Lock()

ORIGEM_NOVA = "nova_previsao"
ORIGEM_SALVA = "resposta_salva"
ROTULO_NOVA = "Nova previsão"
ROTULO_SALVA = "Resposta salva (reprodução, não é uma nova previsão)"
VARIAVEL_ONLINE = "CASEGALAXY_MCP_ONLINE"
_ARQUIVO_FINAL = re.compile(r"^respostas_final=v(\d+)\.json$")


class Adaptador(Protocol):
    """Contrato que a interface usa. Implementações são trocáveis (real, salvo, falso nos testes)."""

    origem: str
    rotulo: str

    def identidade(self) -> str:
        """Texto estável que muda quando muda o modelo ou o extrator servidos (entra na chave do cache)."""

    def prever(self, pedido: dict) -> dict:
        """Envelope `{success, data | error}` igual ao de `src.agente.api.prever`. Nunca levanta exceção."""


def _erro(codigo: Codigo, mensagem: str, **detalhes: object) -> dict:
    return envelope_erro(ErroAgente(codigo, mensagem, detalhes))


# --- adaptador real ---------------------------------------------------------------------------------------------------

def modo_online_ligado(ambiente: dict | None = None) -> bool:
    """O modo online do agente (histórico ao vivo do YouTube, anotação e explicação pelo Gemini) é opt-in por ambiente."""
    return ((ambiente if ambiente is not None else os.environ).get(VARIAVEL_ONLINE) or "").strip() == "1"


def fabrica_de_servicos(raiz: Path = RAIZ_PROJETO) -> object:
    """Serviços reais: offline por padrão; online só com `CASEGALAXY_MCP_ONLINE=1` (mesma regra do servidor MCP)."""
    from src.integracoes import mcp as on
    from src.agente.servicos import Servicos

    configuracao = on.ConfiguracaoOnline.do_ambiente(os.environ)
    if configuracao is not None:
        return on.criar_servicos_online(configuracao)
    return Servicos.padrao(Path(raiz))


def _falha_de_carga(falha: Exception) -> dict:
    """Envelope para uma falha ao montar os serviços reais, sem caminhos nem pilha."""
    nome = type(falha).__name__
    if nome == "AmbienteIncompleto":
        return _erro(Codigo.MODELO_INDISPONIVEL,
                     "Faltam pesos dos codificadores ou bibliotecas do sistema neste computador.",
                     **falha.detalhes())
    if nome in {"ErroManifesto", "ErroModelo", "ErroPacote", "ModeloAusente", "ModeloCorrompido", "LimiarIncompativel"}:
        return _erro(Codigo.MODELO_INDISPONIVEL,
                     "O pacote de modelos ativo não pôde ser carregado ou não confere com o manifesto.", tipo=nome)
    if nome == "ErroConfiguracao":
        return _erro(Codigo.MODELO_INDISPONIVEL, f"Configuração do modo online inválida: {ocultar_caminhos(str(falha))}")
    logger.exception("falha ao montar os serviços do agente")
    return _erro(Codigo.ERRO_INTERNO, "Não foi possível preparar o preditor local.", tipo=nome)


def identidade_do_modelo_ativo(raiz: Path = RAIZ_PROJETO, *, online: bool = False) -> str:
    """Versão do pacote de vídeo ativo, hash de identidade e, se houver, o extrator declarado (só lê JSON)."""
    from src.modelos import pacote as ms

    versao = ms.versao_ativa(Path(raiz))
    video = versao["pacotes"]["video"]
    partes = [versao["rotulo"], str(video.get("versao_servida") or video.get("pasta_q50")),
              str(video.get("identidade_sha256")), f"online={int(online)}"]
    for chave in ("video", "video_local"):
        pacote = versao["pacotes"].get(chave)
        if not pacote:
            continue
        if chave == "video_local":  # o preditor local tem identidade e versão próprias, distintas do pacote Gemini
            partes += [str(pacote.get("versao_servida")), str(pacote.get("identidade_sha256"))]
        pasta = pacote.get("pasta")
        arquivo = Path(raiz) / pasta / "pacote.json" if pasta else None
        if arquivo is not None and arquivo.is_file():
            extrator = json.loads(arquivo.read_text(encoding="utf-8")).get("extrator")
            if extrator is not None:
                partes.append("extrator=" + json.dumps(extrator, sort_keys=True, ensure_ascii=False))
    return "|".join(partes)


def preditor_disponivel(raiz: Path = RAIZ_PROJETO) -> tuple[bool, str]:
    """Conferência leve (sem carregar modelo): manifesto legível e pesos dos codificadores presentes."""
    try:
        from src.extracao import modelos as amb
        from src.modelos import pacote as ms

        ms.versao_ativa(Path(raiz))
        amb.exigir_ambiente()
    except Exception as falha:
        nome = type(falha).__name__
        if nome == "AmbienteIncompleto":
            return False, "Faltam pesos dos codificadores ou bibliotecas do sistema."
        return False, "O pacote de modelos ativo não está disponível."
    return True, ""


class AdaptadorReal:
    origem = ORIGEM_NOVA
    rotulo = ROTULO_NOVA

    def __init__(self, obter_servicos: Callable[[], object], raizes_de_arquivos: Sequence[Path], *,
                 raiz: Path = RAIZ_PROJETO, online: bool = False) -> None:
        self._obter_servicos = obter_servicos
        self._raizes = tuple(Path(r).resolve() for r in raizes_de_arquivos)
        self._raiz, self._online = Path(raiz), online

    def identidade(self) -> str:
        try:
            return identidade_do_modelo_ativo(self._raiz, online=self._online)
        except Exception:
            return "modelo-indisponivel"

    def prever(self, pedido: dict) -> dict:
        from src.agente import api
        from src.integracoes import mcp as fm

        try:
            servicos = self._obter_servicos()
        except Exception as falha:
            return _falha_de_carga(falha)
        try:
            verificado = fm.verificar_pedido(pedido, fm.ConfiguracaoFronteira(raizes=self._raizes))
            with _TRAVA_DE_PREVISAO:  # os serviços e o extrator são compartilhados entre sessões do navegador
                return api.prever(verificado, servicos)
        except ErroAgente as erro:
            return envelope_erro(erro)
        except Exception as falha:
            logger.exception("falha inesperada ao prever")
            return _erro(Codigo.ERRO_INTERNO, "Falha interna ao analisar o vídeo.", tipo=type(falha).__name__)


# --- adaptador de respostas salvas ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RespostaSalva:
    video_id: str
    setor: str
    canal_titulo: str
    url: str
    resposta: dict
    ressalvas: tuple[dict, ...]
    resultado_conhecido: dict | None
    gravacao: dict | None = None


def salvar_previsao(envelope: dict, *, nome: str, setor: str, contexto: str,
                    sha_video: str, raiz: Path | None = None) -> Path | None:
    """Guarda apenas resultados de sucesso, sem vídeo ou CSV, na pasta privada local.

    A chave inclui conteúdo, contexto e resposta; repetir o mesmo resultado não cria cópias.
    """
    if not envelope.get("success") or not isinstance(envelope.get("data"), dict):
        return None
    dados = envelope["data"]
    if dados.get("predicted_metric") is None:
        return None
    chave = hashlib.sha256(json.dumps([sha_video, setor, contexto, envelope], sort_keys=True,
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    destino = (Path(raiz) if raiz is not None else RAIZ_PROJETO) / "data/privado/respostas_salvas" / f"{chave}.json"
    if destino.exists():
        return destino
    registro = {
        "formato": 1, "rotulo": "Previsão salva neste computador",
        "aviso": "Reprodução do resultado anterior; nenhum modelo é executado ao reproduzir.",
        "gerado_em_utc": datetime.now(UTC).isoformat(),
        "versao_servida": dados.get("model_version"),
        "criativos": [{"video_id": chave, "setor": setor,
                       "canal": {"titulo": nome}, "url": "", "resposta": envelope}],
    }
    return gravar_respostas(registro, destino)


def arquivo_de_respostas_padrao(raiz: Path = RAIZ_PROJETO) -> Path:
    """Arquivo `respostas_final=v<N>.json` de maior versão em `data/demonstracao/respostas_salvas` (ignora `execucao_*`)."""
    pasta = Path(raiz) / PASTA_DEMONSTRACAO / PASTA_RESPOSTAS
    candidatos = [(int(m.group(1)), a) for a in pasta.glob("respostas_final=v*.json")
                  if (m := _ARQUIVO_FINAL.match(a.name))]
    if not candidatos:
        raise FileNotFoundError("Nenhum arquivo de respostas salvas encontrado.")
    return max(candidatos, key=lambda c: c[0])[1]


class AdaptadorSalvo:
    origem = ORIGEM_SALVA
    rotulo = ROTULO_SALVA

    def __init__(self, arquivo: Path, adicionais: Sequence[Path] = ()) -> None:
        arquivos = [Path(arquivo), *adicionais]
        self._sha256 = hashlib.sha256(b"".join(a.read_bytes() for a in arquivos)).hexdigest()
        self._dados = ler_respostas(arquivos[0])
        self._por_id = {}
        self._gravacoes = {}
        for a in arquivos:
            dados = ler_respostas(a)
            for c in dados.get("criativos", ()):
                self._por_id[c["video_id"]] = c
                self._gravacoes[c["video_id"]] = {k: dados.get(k) for k in ("gerado_em_utc", "rotulo", "aviso")}

    @classmethod
    def padrao(cls, raiz: Path = RAIZ_PROJETO) -> "AdaptadorSalvo":
        locais = sorted((Path(raiz) / "data/privado/respostas_salvas").glob("*.json"))
        # Um arquivo local incompleto não impede a leitura dos demais resultados.
        validos = []
        for a in locais:
            try:
                dados = ler_respostas(a)
                if all(isinstance(c, dict) and all(k in c for k in ("video_id", "setor", "url", "resposta"))
                       for c in dados.get("criativos", ())):
                    validos.append(a)
            except (OSError, ValueError, KeyError, AttributeError, TypeError, ErroDemonstracao):
                logger.warning("Não foi possível ler uma resposta salva local.")
        try:
            validos.insert(0, arquivo_de_respostas_padrao(raiz))
        except FileNotFoundError:
            pass
        if not validos:
            raise FileNotFoundError("Nenhum arquivo de respostas salvas encontrado.")
        return cls(validos[0], validos[1:])

    def identidade(self) -> str:
        return f"salvo:{self._sha256[:16]}"

    def metadados(self) -> dict:
        d = self._dados
        return {"rotulo": d.get("rotulo"), "aviso": d.get("aviso"), "gerado_em_utc": d.get("gerado_em_utc"),
                "versao_servida": d.get("versao_servida"), "chamadas_online": d.get("chamadas_online")}

    def itens(self) -> tuple[RespostaSalva, ...]:
        return tuple(RespostaSalva(
            video_id=c["video_id"], setor=c["setor"], canal_titulo=(c.get("canal") or {}).get("titulo") or "canal sem título",
            url=c["url"], resposta=c["resposta"], ressalvas=tuple(c.get("ressalvas", ())),
            resultado_conhecido=c.get("resultado_conhecido"), gravacao=self._gravacoes[c["video_id"]])
            for c in sorted(self._por_id.values(),
                            key=lambda c: self._gravacoes[c["video_id"]].get("gerado_em_utc") or "", reverse=True))

    def prever(self, pedido: dict) -> dict:
        video_id = None
        if isinstance(pedido, dict):
            criativo = pedido.get("creative") or {}
            video_id = pedido.get("own_video_id") or video_id_da_url(criativo.get("url"))
        salvo = self._por_id.get(video_id) if video_id else None
        if salvo is None:
            return _erro(Codigo.ENTRADA_INVALIDA, "Não há resposta salva para este vídeo.")
        return copy.deepcopy(salvo["resposta"])
