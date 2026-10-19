"""Cobertura do procedimento de avaliação final, lida do resultado gravado e gravada nos pacotes servidos (D-D7, D-F1).

Os números (N4 empilhado, N3, N2) vêm de `principal.json` da consulta única de 18/10/2026, lidos do arquivo e nunca
digitados. Eles descrevem o PROCEDIMENTO de avaliação nos 452 vídeos de teste. O artefato servido é retreinado com todos
os vídeos anotados (inclusive esses 452): a cobertura NÃO foi medida nele, nenhuma métrica do conjunto de ajuste vale como
desempenho independente e não há cobertura garantida para vídeos novos. O texto-base é o da declaração `teste_final` de
`avaliacao.declaracao_faixa`.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from src.avaliacao import declaracao_faixa as dfx

NIVEL_DO_PROCEDIMENTO = {"video": "N4", "imagem": "N3", "texto": "N2"}
ROTULO_DO_MODELO = {
    "video": "N4 (vídeo, modelo completo)", "imagem": "N3 (imagem: histórico, legenda e capa)",
    "texto": "N2 (texto: histórico e legenda)"}
AVISO_DO_REFIT = (
    "O modelo servido foi retreinado com todos os vídeos anotados da amostra, inclusive os de teste: esta cobertura "
    "foi medida pelo procedimento de avaliação congelado em 452 vídeos de teste e NÃO foi medida neste artefato. "
    "Métricas calculadas sobre os vídeos do ajuste não são desempenho independente. Não há cobertura garantida para "
    "vídeos novos.")


class ErroCobertura(ValueError):
    """Resultado da avaliação ausente, sem o bloco esperado ou fora do contrato."""


def sha256_arquivo(arquivo: Path) -> str:
    return hashlib.sha256(Path(arquivo).read_bytes()).hexdigest()


def _bloco(principal: Mapping, nivel: str) -> tuple[Mapping, Mapping]:
    try:
        empilhado = principal[nivel]["empilhado"]
        return empilhado["cobertura"], empilhado["cobertura_por_setor"]
    except (KeyError, TypeError) as falha:
        raise ErroCobertura(f"principal.json sem {nivel}.empilhado.cobertura / cobertura_por_setor") from falha


def declaracao_do_procedimento(arquivo_principal: Path, modalidade: str, *, protocolo_sha256: str | None) -> dict:
    """Declaração (texto e números) do modelo da modalidade, com a origem explícita no procedimento de avaliação."""
    if modalidade not in NIVEL_DO_PROCEDIMENTO:
        raise ErroCobertura(f"modalidade sem cobertura de procedimento: {modalidade!r}")
    nivel = NIVEL_DO_PROCEDIMENTO[modalidade]
    principal = json.loads(Path(arquivo_principal).read_text(encoding="utf-8"))
    total, por_setor = _bloco(principal, nivel)
    declaracao = dfx.declarar("teste_final", modelo=ROTULO_DO_MODELO[modalidade], total=total, por_setor=por_setor)
    return declaracao | {
        "aviso": f"{declaracao['aviso']} {AVISO_DO_REFIT}",
        "origem": "procedimento_de_avaliacao_congelado", "nivel_do_procedimento": nivel,
        "medida_no_artefato_servido": False, "garantia_de_cobertura": False,
        "n_canais": total.get("n_canais"), "n_videos_de_teste": total["n"],
        "procedimento": {"consulta": "única, 18/10/2026", "arquivo": Path(arquivo_principal).name,
                         "sha256_arquivo": sha256_arquivo(arquivo_principal), "protocolo_sha256": protocolo_sha256,
                         "metodo_do_intervalo": total.get("metodo_do_intervalo")},
    }


def conferir(declaracao: Mapping) -> None:
    """Falha se a declaração gravada deixar de afirmar a origem no procedimento ou prometer garantia."""
    if (declaracao.get("origem") != "procedimento_de_avaliacao_congelado"
            or declaracao.get("medida_no_artefato_servido") is not False
            or declaracao.get("garantia_de_cobertura") is not False
            or declaracao.get("nivel_nominal") != dfx.NIVEL_NOMINAL):
        raise ErroCobertura("a declaração de cobertura não descreve o procedimento de avaliação sem garantia")
