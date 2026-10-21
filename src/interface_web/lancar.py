"""Inicia a interface Streamlit em `127.0.0.1`, sem telemetria e sem abrir páginas externas.

    uv run casegalaxy-interface [--porta 8501] [--abrir-navegador]

Os mesmos valores de `.streamlit/config.toml` são passados como opções, então a interface sobe igual de qualquer pasta.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.interface_web.tema import opcoes_streamlit

PORTA_PADRAO = 8501
PAGINA = Path(__file__).resolve().parent / "app.py"


def argumentos_do_streamlit(porta: int, abrir_navegador: bool, pagina: Path = PAGINA) -> list[str]:
    """Linha de comando do `streamlit run`: só localhost, sem telemetria, upload até 100 MB, sem erros detalhados."""
    opcoes = {
        "server.address": "127.0.0.1", "server.port": str(porta), "server.headless": str(not abrir_navegador).lower(),
        "server.maxUploadSize": "100", "server.fileWatcherType": "none", "server.runOnSave": "false",
        "server.enableXsrfProtection": "true", "browser.gatherUsageStats": "false", "browser.serverAddress": "127.0.0.1",
        "client.showErrorDetails": "none", "client.toolbarMode": "minimal", **opcoes_streamlit(),
    }
    return ["streamlit", "run", str(pagina), *(f"--{chave}={valor}" for chave, valor in opcoes.items())]


def main(argv: Sequence[str] | None = None) -> int:
    analisador = argparse.ArgumentParser(prog="casegalaxy-interface", description="Interface web local do agente.")
    analisador.add_argument("--porta", type=int, default=PORTA_PADRAO, help="porta local (padrão 8501)")
    analisador.add_argument("--abrir-navegador", action="store_true", help="abre o navegador padrão ao iniciar")
    args = analisador.parse_args(argv)
    if not 1024 <= args.porta <= 65535:
        analisador.error("a porta deve estar entre 1024 e 65535")
    from streamlit.web import cli

    sys.argv = argumentos_do_streamlit(args.porta, args.abrir_navegador)
    return cli.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
