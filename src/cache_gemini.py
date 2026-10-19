"""Cache da extração pelo Gemini: um JSON por vídeo, com a configuração que o produziu.

A chave do arquivo é o `video_id` (contrato da coleta). Dentro dele fica a chave de compatibilidade
(modelo, versões da rubrica e do pedido, parâmetros, hash do pedido e do esquema). Ler um arquivo
gerado por outra configuração é um erro explícito, nunca reutilização silenciosa. A gravação é
atômica: uma falha no meio não deixa arquivo final parcial.
"""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from src import rubrica as rb
from src.contratos import ItemExtracao
from src.extracao_gemini import ConfiguracaoExtracao, ResultadoAnotacao

FORMATO = 1
FORMATO_UPLOAD = 1
RETENCAO_UPLOAD_DIAS = 30
ROTA_UPLOAD = "upload_files_api"
_PADRAO_VIDEO = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PADRAO_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMATO_INSTANTE = "%Y-%m-%dT%H:%M:%SZ"


class CacheIncompativel(RuntimeError):
    """Arquivo produzido por outra configuração (modelo, pedido, esquema ou parâmetros)."""


class CacheCorrompido(RuntimeError):
    """Arquivo ilegível, de formato desconhecido ou com anotação que não passa na validação."""


class CacheAnotacoes:
    def __init__(self, raiz: Path, config: ConfiguracaoExtracao):
        self.raiz = Path(raiz)
        self.config = config
        self.chave = config.chave_compatibilidade()

    def caminho(self, video_id: str) -> Path:
        if not _PADRAO_VIDEO.fullmatch(video_id):
            raise ValueError(f"video_id inválido: {video_id!r}")
        return self.raiz / f"{video_id}.json"

    def ler(self, video_id: str) -> ResultadoAnotacao | None:
        caminho = self.caminho(video_id)
        if not caminho.is_file():
            return None
        try:
            conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError) as erro:
            raise CacheCorrompido(f"{video_id}: arquivo de cache ilegível ({type(erro).__name__})") from erro
        if not isinstance(conteudo, dict) or conteudo.get("formato") != FORMATO:
            raise CacheCorrompido(f"{video_id}: formato de cache desconhecido")
        if conteudo.get("chave_compatibilidade") != self.chave:
            raise CacheIncompativel(
                f"{video_id}: o cache foi gerado com outra configuração; use outra pasta ou nova versão, sem reaproveitar"
            )
        return self._resultado(video_id, conteudo.get("resultado"))

    def _resultado(self, video_id: str, dados: object) -> ResultadoAnotacao:
        try:
            resultado = ResultadoAnotacao.model_validate(dados)
            if resultado.anotacao is not None:
                rb.modelo_do_setor(resultado.setor).model_validate_json(json.dumps(resultado.anotacao))
        except (ValidationError, ValueError) as erro:
            raise CacheCorrompido(f"{video_id}: conteúdo do cache não passa na validação") from erro
        if resultado.video_id != video_id:
            raise CacheCorrompido(f"{video_id}: o arquivo contém outro vídeo")
        return resultado

    def gravar(self, item: ItemExtracao, resultado: ResultadoAnotacao, *, agora: datetime | None = None) -> Path:
        if resultado.video_id != item.video_id:
            raise ValueError("o resultado e o item têm video_id diferentes")
        instante = (agora or datetime.now(UTC)).astimezone(UTC)
        conteudo = {
            "formato": FORMATO,
            "chave_compatibilidade": self.chave,
            "configuracao": self.config.descricao(),
            "item": item.model_dump(mode="json"),
            "resultado": resultado.model_dump(mode="json"),
            "gravado_em_utc": instante.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return gravar_json_atomico(self.caminho(item.video_id), conteudo)


class CacheUploads:
    """Anotações de vídeos enviados pelo usuário, chaveadas pelo SHA-256 do ARQUIVO (nunca por um `video_id` falso).

    Guarda só o que o cache de URL guarda (a anotação validada e a configuração que a produziu) mais o hash do arquivo,
    o setor e a data de expiração. Nenhum nome, caminho ou byte do vídeo. Retenção: `RETENCAO_UPLOAD_DIAS` dias a partir
    da gravação (mesma regra de 30 dias dos dados derivados do YouTube, por prudência); depois disso a entrada é
    apagada na leitura. Só anotações válidas são guardadas. Chave de compatibilidade própria da rota, derivada da
    configuração de extração, para que uma mudança de modelo, rubrica, pedido ou parâmetros invalide a entrada.
    """

    def __init__(self, raiz: Path, config: ConfiguracaoExtracao, *, agora: Callable[[], datetime] | None = None):
        self.raiz = Path(raiz)
        self.config = config
        self._agora = agora or (lambda: datetime.now(UTC))
        self.chave = hashlib.sha256(json.dumps({"base": config.chave_compatibilidade(), "rota": ROTA_UPLOAD},
                                               sort_keys=True).encode("utf-8")).hexdigest()

    def caminho(self, sha256: str) -> Path:
        if not _PADRAO_SHA256.fullmatch(sha256):
            raise ValueError("sha256 inválido")
        return self.raiz / f"{sha256}.json"

    def ler(self, sha256: str, setor: str) -> dict | None:
        """Anotação guardada para o arquivo, ou None (ausente, expirada ou de outro setor)."""
        caminho = self.caminho(sha256)
        if not caminho.is_file():
            return None
        try:
            conteudo = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError) as erro:
            raise CacheCorrompido(f"upload {sha256[:12]}: arquivo de cache ilegível ({type(erro).__name__})") from erro
        if not isinstance(conteudo, dict) or conteudo.get("formato") != FORMATO_UPLOAD:
            raise CacheCorrompido(f"upload {sha256[:12]}: formato de cache desconhecido")
        if conteudo.get("chave_compatibilidade") != self.chave:
            raise CacheIncompativel(f"upload {sha256[:12]}: o cache foi gerado com outra configuração")
        try:
            expira = datetime.strptime(conteudo["expira_em_utc"], _FORMATO_INSTANTE).replace(tzinfo=UTC)
        except (KeyError, ValueError, TypeError) as erro:
            raise CacheCorrompido(f"upload {sha256[:12]}: data de expiração ilegível") from erro
        if expira <= self._agora():
            caminho.unlink(missing_ok=True)
            return None
        if conteudo.get("sha256_arquivo") != sha256 or conteudo.get("setor") != setor:
            return None
        try:
            rb.modelo_do_setor(setor).model_validate_json(json.dumps(conteudo.get("anotacao")))
        except (ValidationError, ValueError) as erro:
            raise CacheCorrompido(f"upload {sha256[:12]}: anotação não passa na validação") from erro
        return dict(conteudo["anotacao"])

    def gravar(self, sha256: str, setor: str, anotacao: dict) -> Path:
        agora = self._agora().astimezone(UTC)
        conteudo = {
            "formato": FORMATO_UPLOAD, "chave_compatibilidade": self.chave, "rota": ROTA_UPLOAD,
            "configuracao": self.config.descricao(), "origem": "arquivo_enviado_pelo_usuario",
            "sha256_arquivo": sha256, "setor": setor, "anotacao": anotacao,
            "gravado_em_utc": agora.strftime(_FORMATO_INSTANTE),
            "expira_em_utc": (agora + timedelta(days=RETENCAO_UPLOAD_DIAS)).strftime(_FORMATO_INSTANTE),
        }
        return gravar_json_atomico(self.caminho(sha256), conteudo)


def gravar_json_atomico(destino: Path, conteudo: object) -> Path:
    """Grava num temporário da mesma pasta e renomeia; em caso de falha, remove o temporário."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    dados = json.dumps(conteudo, ensure_ascii=False, indent=1).encode("utf-8")
    descritor, temporario = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".parcial", dir=destino.parent)
    try:
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(dados)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, destino)
    except BaseException:
        Path(temporario).unlink(missing_ok=True)
        raise
    return destino
