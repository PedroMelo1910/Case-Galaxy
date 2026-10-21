"""Dependências do agente: modelos, base, extratores, caches, fontes e clientes, todos injetáveis.

`Servicos.padrao` monta o conjunto real do projeto a partir do MANIFESTO DE PACOTE ATIVO (`manifesto_servido`): pacotes de
vídeo, imagem e texto, coleção de referência dos comparáveis, base curada do histórico, codificadores locais e cache do
Gemini. Sem manifesto válido há erro claro; não existe volta silenciosa aos modelos de desenvolvimento. Sem biblioteca do sistema ou
peso de codificador, `AmbienteIncompleto` (src/extracao/modelos.py) lista o que falta. As anotações do de-para da distribuição
(`data/demonstracao/anotacoes`), se existirem, completam o cache da amostra. Clientes de rede (Gemini, YouTube) só entram quando passados: sem eles nenhuma chamada de rede
acontece. Os modelos de linguagem e de imagem são carregados no primeiro uso.

Se o manifesto ativo traz `pacotes.video_local`, `pacote_local` e `extrator_local` habilitam a rota local de vídeo
(arquivo de vídeo previsto sem Gemini e sem rede; ver `rota_local`). O extrator local só é criado na primeira extração.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.extracao import modelos as amb
from src.dados import dataset as ds
from src.extracao import imagem as capa
from src.extracao import texto as leg
from src.integracoes import gemini as ga
from src.dados import historico as hist
from src.modelos import treino as md
from src.modelos import pacote as ms
from src.extracao.rubrica import FonteAnotacaoArquivo
from src.agente.atributos import (
    CacheConteudo,
    CodificadorEmCache,
    ExtratoresCapa,
    FonteAnotacao,
    FonteAnotacaoGemini,
)
from src.agente.explicacao import Subagente, SubagenteGemini
from src.dados.historico import BaseHistorica, Consultor
from src.modelos.pacote import (
    MODALIDADE_COM_MODELO,
    ROTA_VIDEO_LOCAL,
    Pacote,
    carregar_pacote,
    carregar_pacote_final,
    carregar_pacote_video_local,
)
from src.agente.atributos import ExtratorLocalSobDemanda
from src.extracao.rubrica import CacheAnotacoes, CacheUploads
from src.config import RAIZ_PROJETO
from src.extracao.rubrica import ConfiguracaoExtracao

logger = logging.getLogger(__name__)

SNAPSHOT = "snapshot=20261014T021437Z"
PASTA_GEMINI = Path(f"data/annotations/gemini/{SNAPSHOT}/amostra=v2/cache")
PASTA_TRABALHO_CAPA = Path(f"data/features/{SNAPSHOT}/legenda_capa=v1_trabalho")
PASTA_ANOTACOES_DEMONSTRACAO = Path("data/demonstracao/anotacoes")  # anotações dos 6 vídeos do de-para (distribuição)
PASTA_CACHE_AGENTE = Path("data/agente/cache")
PASTA_CACHE_UPLOAD = Path("data/agente/upload_gemini")  # anotações de vídeos locais, por SHA-256, 30 dias
MODELO_GEMINI = "gemini-3.5-flash-lite"
TEMPERATURA_EXTRACAO = 1.0


@dataclass
class Servicos:
    pacote: Pacote
    base: BaseHistorica
    colecao_de_referencia: Sequence[Mapping]
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
    fonte_arquivo: FonteAnotacaoArquivo | None = None
    pacotes_por_modalidade: Mapping[str, Pacote] = field(default_factory=dict)
    pacote_local: Pacote | None = None  # preditor local de vídeo ativo no manifesto (None = não há)
    extrator_local: object | None = None  # `extrair(caminho) -> dict`: o `ExtratorAudiovisual` ou um dublê
    relogio: Callable[[], datetime] = lambda: datetime.now(UTC)
    origem_da_colecao: str = "desenvolvimento_real"  # rótulo `origem` dos comparáveis
    rotulo_da_versao: str | None = None  # versão do manifesto de pacote ativo que montou estes serviços

    @property
    def linhas_desenvolvimento(self) -> Sequence[Mapping]:
        """Nome antigo da coleção de referência (nas versões finais ela é o dataset do refit, sem LIM)."""
        return self.colecao_de_referencia

    @property
    def rota_local_ativa(self) -> bool:
        return self.pacote_local is not None and self.extrator_local is not None

    def pacote_para(self, modalidade: str) -> Pacote:
        """Pacote da modalidade: o vídeo é sempre o servido em `pacote`; imagem e texto, o reduzido configurado.

        Sem pacote para a modalidade o erro é `MODELO_INDISPONIVEL`, explícito, nunca um pacote de outra modalidade.
        """
        if modalidade == MODALIDADE_COM_MODELO:
            return self.pacote
        if modalidade in self.pacotes_por_modalidade:
            return self.pacotes_por_modalidade[modalidade]
        return self.pacote.modelo_para(modalidade)

    @classmethod
    def padrao(cls, raiz: Path = RAIZ_PROJETO, *, cliente_gemini: ga.ClienteGemini | None = None,
               consultor_historico: Consultor | None = None, explicar_com_gemini: bool = False,
               baixar_imagem: Callable[[str], bytes | None] | None = None) -> "Servicos":
        """Conjunto real. `cliente_gemini` habilita anotação de vídeo público, o envio de vídeo local pela Files API
        e, com `explicar_com_gemini`, o subagente. Sem ele, só os caches (anotações da amostra e de uploads) são lidos."""
        raiz = Path(raiz)
        versao = ms.versao_ativa(raiz)
        ms.validar_versao(raiz, versao)
        amb.exigir_ambiente()  # AmbienteIncompleto com a lista do que falta, antes de importar LightGBM ou carregar pesos
        pacotes = carregar_pacotes_da_versao(raiz, versao)
        if versao.get("contexto_execucao"):
            from src.dados.execucao import carregar
            linhas_publicas, base_publica = carregar(raiz, versao)
        referencia = versao["dataset_de_referencia"]
        if versao.get("contexto_execucao"):
            linhas = linhas_publicas
            posts, canais = base_publica.posts, base_publica.canais
            contexto, coletado = base_publica.contexto, base_publica.coletado_em
        else:
            linhas, manifesto = ds.carregar_dataset(raiz / referencia["pasta"])
            if referencia.get("sha256_dataset") and manifesto["sha256_dataset"] != referencia["sha256_dataset"]:
                raise ms.ErroManifesto("o dataset de referência não é o registrado no manifesto de pacote ativo")
            posts, canais = hist.carregar_base(raiz / versao["base_historica"]["pasta"])
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
        fonte = montar_fonte_anotacao(raiz, config, cliente_gemini)
        subagente = SubagenteGemini(cliente_gemini) if (cliente_gemini and explicar_com_gemini) else None
        fonte_arquivo = FonteAnotacaoArquivo(config=config, cache=CacheUploads(raiz / PASTA_CACHE_UPLOAD, config),
                                             cliente=cliente_gemini)
        pasta_imagens = raiz / PASTA_TRABALHO_CAPA
        local = pacotes.get(ROTA_VIDEO_LOCAL)
        return cls(pacote=pacotes[MODALIDADE_COM_MODELO], pacote_local=local,
                   extrator_local=ExtratorLocalSobDemanda() if local is not None else None, base=BaseHistorica(posts, canais, contexto, coletado),
                   colecao_de_referencia=linhas,
                   origem_da_colecao=("historico_autorizado_local" if versao.get("contexto_execucao") else
                                      "refit_final_real" if versao["layout"] == "final" else "desenvolvimento_real"),
                   rotulo_da_versao=versao["rotulo"],
                   codificador_texto=CodificadorEmCache(texto, CacheConteudo(cache_agente / "legenda")),
                   codificador_transcricao=texto, pasta_cache_transcricao=cache_agente / "transcricao",
                   extratores_capa=extratores, cache_capa=CacheConteudo(cache_agente / "capa"),
                   fonte_anotacao=fonte, subagente=subagente, consultor_historico=consultor_historico,
                   imagens_conferidas=lambda video_id: imagem_conferida(pasta_imagens, video_id),
                   baixar_imagem=baixar_imagem, fonte_arquivo=fonte_arquivo,
                   pacotes_por_modalidade={m: p for m, p in pacotes.items()
                                           if m not in (MODALIDADE_COM_MODELO, ROTA_VIDEO_LOCAL)})


def montar_fonte_anotacao(raiz: Path, config: ConfiguracaoExtracao, cliente_gemini: ga.ClienteGemini | None) -> FonteAnotacaoGemini:
    """Cache da amostra e, se a distribuição o instalou, o das anotações do de-para (lido só se o principal não tem o vídeo)."""
    pasta_demo = Path(raiz) / PASTA_ANOTACOES_DEMONSTRACAO
    return FonteAnotacaoGemini(cache=CacheAnotacoes(Path(raiz) / PASTA_GEMINI, config), cliente=cliente_gemini,
                               temperatura=TEMPERATURA_EXTRACAO,
                               cache_complementar=CacheAnotacoes(pasta_demo, config) if pasta_demo.is_dir() else None)


def carregar_pacotes_da_versao(raiz: Path, versao: Mapping) -> dict[str, Pacote]:
    """Pacotes de vídeo, imagem e texto da versão, cada um conferido contra a modalidade que o manifesto declara.

    Pacote ausente, corrompido ou de outra modalidade interrompe a montagem: nunca é usado em silêncio."""
    pacotes = {}
    for modalidade, entrada in versao["pacotes"].items():
        if versao["layout"] == "final" and modalidade == ROTA_VIDEO_LOCAL:
            pacote = carregar_pacote_video_local(Path(raiz) / entrada["pasta"], entrada_do_manifesto=entrada)
            if not pacote.eh_local:
                raise md.ErroModelo("o pacote video_local do manifesto não é um preditor local de vídeo")
            pacotes[modalidade] = pacote
            continue
        if versao["layout"] == "final":
            pacote = carregar_pacote_final(Path(raiz) / entrada["pasta"], entrada_do_manifesto=entrada)
        else:
            pacote = carregar_pacote(Path(raiz) / entrada["pasta_q50"], Path(raiz) / entrada["pasta_faixa"])
        if pacote.modalidade != modalidade:
            raise md.ErroModelo(f"o pacote de {modalidade!r} do manifesto é de {pacote.modalidade!r}")
        pacotes[modalidade] = pacote
    return pacotes


def imagem_conferida(pasta_trabalho: Path, video_id: str) -> bytes | None:
    """Bytes da thumbnail já baixada, só se o SHA-256 conferir com o registrado no download."""
    registro = pasta_trabalho / "baixar" / f"{video_id}.json"
    imagem = pasta_trabalho / "imagens" / f"{video_id}.img"
    if not (registro.is_file() and imagem.is_file()):
        return None
    esperado = json.loads(registro.read_text(encoding="utf-8")).get("resultado", {}).get("sha256")
    dados = imagem.read_bytes()
    return dados if esperado and hashlib.sha256(dados).hexdigest() == esperado else None
