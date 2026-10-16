"""Dependências do agente: modelos, base, extratores, caches, fontes e clientes, todos injetáveis.

`Servicos.padrao` monta o conjunto real do projeto (artefatos de desenvolvimento, base curada, codificadores locais e
cache do Gemini). Clientes de rede (Gemini, YouTube) só entram quando passados: sem eles nenhuma chamada de rede
acontece. Os modelos de linguagem e de imagem são carregados no primeiro uso.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src import dataset_desenvolvimento as ds
from src import extracao_capa as capa
from src import extracao_legenda as leg
from src import gemini_api as ga
from src import historico as hist
from src.agente.atributos import (
    CacheConteudo,
    CodificadorEmCache,
    ExtratoresCapa,
    FonteAnotacao,
    FonteAnotacaoGemini,
)
from src.agente.explicacao import Subagente, SubagenteGemini
from src.agente.historico_canal import BaseHistorica, Consultor
from src.agente.pacote import Pacote, carregar_pacote
from src.cache_gemini import CacheAnotacoes
from src.config import RAIZ_PROJETO
from src.extracao_gemini import ConfiguracaoExtracao

SNAPSHOT = "snapshot=20261014T021437Z"
PASTA_Q50 = Path("data/models/provisorio=v1/6f75e1255f73-c62b81db")
PASTA_FAIXA = Path("data/models/faixa=v1/6f75e1255f73-6f75e1255f73-c62b81db")
PASTA_DATASET = Path(f"data/features/{SNAPSHOT}/desenvolvimento=v1/6f75e1255f73")
PASTA_BASE = Path(f"data/processed/{SNAPSHOT}/curadoria=v1")
PASTA_GEMINI = Path(f"data/annotations/gemini/{SNAPSHOT}/amostra=v2/cache")
PASTA_TRABALHO_CAPA = Path(f"data/features/{SNAPSHOT}/legenda_capa=v1_trabalho")
PASTA_CACHE_AGENTE = Path("data/agente/cache")
MODELO_GEMINI = "gemini-3.5-flash-lite"
TEMPERATURA_EXTRACAO = 1.0


@dataclass
class Servicos:
    pacote: Pacote
    base: BaseHistorica
    linhas_desenvolvimento: Sequence[Mapping]
    codificador_texto: CodificadorEmCache
    codificador_transcricao: leg.Codificador
    pasta_cache_transcricao: Path
    extratores_capa: ExtratoresCapa
    cache_capa: CacheConteudo
    fonte_anotacao: FonteAnotacao
    subagente: Subagente | None = None
    consultor_historico: Consultor | None = None
    imagens_conferidas: Callable[[str], bytes | None] | None = None
    baixar_imagem: Callable[[str], bytes | None] | None = None
    relogio: Callable[[], datetime] = lambda: datetime.now(UTC)

    @classmethod
    def padrao(cls, raiz: Path = RAIZ_PROJETO, *, cliente_gemini: ga.ClienteGemini | None = None,
               consultor_historico: Consultor | None = None, explicar_com_gemini: bool = False) -> "Servicos":
        """Conjunto real. `cliente_gemini` habilita anotação de vídeo público e, com `explicar_com_gemini`, o subagente."""
        raiz = Path(raiz)
        pacote = carregar_pacote(raiz / PASTA_Q50, raiz / PASTA_FAIXA)
        linhas, manifesto = ds.carregar_dataset(raiz / PASTA_DATASET)
        posts, canais = hist.carregar_base(raiz / PASTA_BASE)
        dados_contexto = manifesto["entradas"]["contexto"]
        contexto = hist.ContextoTreino(
            dados_contexto["nome"],
            datetime.strptime(dados_contexto["treino_ate_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC),
            frozenset(dados_contexto["canais_excluidos"]))
        snapshot_id = manifesto["entradas"]["base"]["snapshot_id"]
        coletado = datetime.strptime(snapshot_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        cache_agente = raiz / PASTA_CACHE_AGENTE
        texto = leg.CodificadorTexto()
        import cv2

        extratores = ExtratoresCapa(cor=capa.medidas_cor, rosto=capa.DetectorRosto(), ocr=capa.LeitorTexto(),
                                    clip=capa.CodificadorImagem(),
                                    identificacao_cor={"modelo": "OpenCV", "opencv": cv2.__version__})
        config = ConfiguracaoExtracao(modelo=MODELO_GEMINI, temperatura=TEMPERATURA_EXTRACAO)
        fonte = FonteAnotacaoGemini(cache=CacheAnotacoes(raiz / PASTA_GEMINI, config), cliente=cliente_gemini,
                                    temperatura=TEMPERATURA_EXTRACAO)
        subagente = SubagenteGemini(cliente_gemini) if (cliente_gemini and explicar_com_gemini) else None
        pasta_imagens = raiz / PASTA_TRABALHO_CAPA
        return cls(pacote=pacote, base=BaseHistorica(posts, canais, contexto, coletado),
                   linhas_desenvolvimento=linhas,
                   codificador_texto=CodificadorEmCache(texto, CacheConteudo(cache_agente / "legenda")),
                   codificador_transcricao=texto, pasta_cache_transcricao=cache_agente / "transcricao",
                   extratores_capa=extratores, cache_capa=CacheConteudo(cache_agente / "capa"),
                   fonte_anotacao=fonte, subagente=subagente, consultor_historico=consultor_historico,
                   imagens_conferidas=lambda video_id: imagem_conferida(pasta_imagens, video_id))


def imagem_conferida(pasta_trabalho: Path, video_id: str) -> bytes | None:
    """Bytes da thumbnail já baixada, só se o SHA-256 conferir com o registrado no download."""
    registro = pasta_trabalho / "baixar" / f"{video_id}.json"
    imagem = pasta_trabalho / "imagens" / f"{video_id}.img"
    if not (registro.is_file() and imagem.is_file()):
        return None
    esperado = json.loads(registro.read_text(encoding="utf-8")).get("resultado", {}).get("sha256")
    dados = imagem.read_bytes()
    return dados if esperado and hashlib.sha256(dados).hexdigest() == esperado else None
