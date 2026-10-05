"""Coleta metadados dos canais da lista na YouTube Data API e grava um snapshot.

Exemplo de piloto com canais candidatos e limites explícitos:
    uv run python -m src.coletar_snapshot --canais UC...,UC... --incluir-candidatos --piloto \\
        --max-paginas 2 --limite-unidades 20
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from src.coleta import coletar
from src.config import RAIZ_PROJETO, ChaveAusente
from src.lista_canais import StatusConferencia, ler_lista
from src.snapshot import DIRETORIO_PADRAO, salvar_snapshot
from src.youtube_api import ClienteYouTube, ErroYouTube

LISTA_PADRAO = RAIZ_PROJETO / "configs" / "canais.csv"


def _inteiro_positivo(texto: str) -> int:
    try:
        valor = int(texto)
    except ValueError:
        raise argparse.ArgumentTypeError(f"esperado inteiro, recebido {texto!r}") from None
    if valor < 1:
        raise argparse.ArgumentTypeError(f"deve ser pelo menos 1, recebido {valor}")
    return valor


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.coletar_snapshot", description=__doc__.splitlines()[0])
    parser.add_argument("--lista", type=Path, default=LISTA_PADRAO, help="CSV da lista de canais")
    parser.add_argument("--canais", help="IDs separados por vírgula; sem isso, todos os canais aprovados")
    parser.add_argument(
        "--incluir-candidatos", action="store_true", help="aceita canais ainda não aprovados (exige --piloto)"
    )
    parser.add_argument("--max-paginas", type=_inteiro_positivo, help="limite de páginas de uploads por canal")
    parser.add_argument("--limite-unidades", type=_inteiro_positivo, help="teto de unidades de cota desta execução")
    parser.add_argument("--piloto", action="store_true", help="marca o snapshot como piloto")
    parser.add_argument("--destino", type=Path, default=DIRETORIO_PADRAO, help="pasta dos snapshots")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, *, cliente: ClienteYouTube | None = None, coletado_em: datetime | None = None
) -> int:
    """0 = snapshot gravado; 2 = erro antes da coleta; 3 = coleta interrompida; 4 = falha ao gravar."""
    args = _argumentos(argv)
    status = {StatusConferencia.APROVADO}
    if args.incluir_candidatos:
        status.add(StatusConferencia.CANDIDATO)
    try:
        if args.incluir_candidatos and not args.piloto:
            raise ValueError("--incluir-candidatos só é aceito com --piloto: candidatos não entram em coleta")
        lista = ler_lista(args.lista)
        ids = [i.strip() for i in args.canais.split(",") if i.strip()] if args.canais else None
        alvos = lista.selecionar(ids, status=status)
        if not alvos:
            raise ValueError("nenhum canal selecionado")
        cliente = cliente or ClienteYouTube.a_partir_da_configuracao(limite_unidades=args.limite_unidades)
    except (OSError, ValueError, ChaveAusente) as erro:
        print(f"Erro antes da coleta: {erro}", file=sys.stderr)
        return 2
    tipo = "piloto" if args.piloto else "coleta"
    try:
        with cliente:
            resultado = coletar(cliente, alvos, coletado_em=coletado_em, max_paginas_por_canal=args.max_paginas)
    except ErroYouTube as erro:
        print(f"Coleta interrompida [{erro.codigo}]: {erro}. Nenhum snapshot foi gravado.", file=sys.stderr)
        return 3
    except (KeyError, TypeError, ValueError) as erro:
        print(f"Coleta interrompida por dado inesperado ({type(erro).__name__}). Nenhum snapshot foi gravado.",
              file=sys.stderr)
        return 3
    if resultado.videos_invalidos:
        print(f"Aviso: {len(resultado.videos_invalidos)} vídeo(s) inválido(s); motivos em proveniencia.json.",
              file=sys.stderr)
    try:
        pasta = salvar_snapshot(
            resultado, lista=lista, destino=args.destino, tipo=tipo, status_permitidos=[str(s) for s in status]
        )
    except OSError as erro:
        print(f"Falha ao gravar o snapshot ({type(erro).__name__}: {erro}).", file=sys.stderr)
        return 4
    print(json.dumps({"tipo": tipo, "pasta": str(pasta), **resultado.resumo()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
