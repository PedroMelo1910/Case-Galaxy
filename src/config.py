"""Configuração local: chaves de API lidas do arquivo .env na raiz do projeto."""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_ENV = RAIZ_PROJETO / ".env"


class ChaveAusente(RuntimeError):
    """A chave exigida não está definida no .env nem no ambiente."""


class Configuracao(BaseSettings):
    """Chaves como `SecretStr`: o valor não aparece em `repr`, `str` nem em logs."""

    model_config = SettingsConfigDict(env_file=ARQUIVO_ENV, env_file_encoding="utf-8", extra="ignore")

    youtube_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    @field_validator("youtube_api_key", "gemini_api_key", mode="before")
    @classmethod
    def _vazio_conta_como_ausente(cls, valor: object) -> object:
        if isinstance(valor, str):
            valor = valor.strip()
            return valor or None
        return valor

    def chave_youtube(self) -> SecretStr:
        if self.youtube_api_key is None:
            raise ChaveAusente("YOUTUBE_API_KEY não está definida no .env (ver README, seção 'Chaves de API').")
        return self.youtube_api_key

    def chave_gemini(self) -> SecretStr:
        if self.gemini_api_key is None:
            raise ChaveAusente("GEMINI_API_KEY não está definida no .env (ver README, seção 'Chaves de API').")
        return self.gemini_api_key
