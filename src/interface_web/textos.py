"""Formatação em português do Brasil e saneamento de textos exibidos."""

import re


# ──── Formatação em português do Brasil e saneamento de textos exibidos.
_CAMINHO_WINDOWS = re.compile(r"[A-Za-z]:\\[^\s'\"<>|]*")
_CAMINHO_UNIX = re.compile(r"(?<![\w.])/(?:home|mnt|tmp|Users|var|usr|opt|etc|root)/[^\s'\"<>|]*")
_ESTAGIOS = {"N1": "histórico", "N2": "histórico e legenda", "N3": "histórico, legenda e capa", "N4": "modelo completo"}
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+!|<>~$])")


def inteiro(valor: float | int | None) -> str:
    """12345 -> '12.345'."""
    return "n/d" if valor is None else f"{round(valor):,}".replace(",", ".")


def decimal(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def percentual(fracao: float, casas: int = 1) -> str:
    return decimal(fracao * 100, casas) + "%"


def ocultar_caminhos(texto: str) -> str:
    """Troca caminhos de arquivo por um marcador: mensagens exibidas nunca revelam a estrutura de pastas."""
    return _CAMINHO_UNIX.sub("[caminho omitido]", _CAMINHO_WINDOWS.sub("[caminho omitido]", texto))


def sem_estagios(texto: str) -> str:
    """Troca a sigla N1..N4 por nome comum: a interface não apresenta níveis internos como etapas."""
    texto = re.sub(r"\bmodelo N4 \(vídeo, modelo completo\)", "modelo completo de vídeo", texto)
    texto = re.sub(r"^N[1-4]:\s*", "", texto)
    return re.sub(r"\bN([1-4])\b", lambda m: _ESTAGIOS[f"N{m.group(1)}"], texto)


_ROTAS = (("`video_route = local`", "a rota local"), ("`video_route = gemini`", "a rota Gemini"),
          ("`video_route = auto`", "a rota automática"), ("`video_route`", "a rota do vídeo"))


def nomes_de_usuario(texto: str) -> str:
    """Troca nomes técnicos de campos por termos que o usuário vê na tela."""
    for tecnico, comum in _ROTAS:
        texto = texto.replace(tecnico, comum)
    return texto


def exibivel(texto: object) -> str:
    """Texto vindo da resposta ou do agente, pronto para exibir: sem caminhos, siglas internas nem nomes de campo."""
    return nomes_de_usuario(sem_estagios(ocultar_caminhos(str(texto))))


def escapar_markdown(texto: str) -> str:
    """Texto digitado pelo usuário exibido literalmente (sem links, imagens ou formatação)."""
    return _MARKDOWN.sub(r"\\\1", texto)
