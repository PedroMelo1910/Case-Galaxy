"""Módulo consolidado. Reúne:
- Erros da camada audiovisual local. Mensagens e detalhes nunca trazem caminho de arquivo.
- Especificação congelada da camada audiovisual local: versão, amostragem, limites de recurso e identidade dos modelos.
- Localização e verificação dos pesos locais, sem rede.
- Conferência do ambiente local: bibliotecas do sistema e pesos dos codificadores congelados."""

import ctypes
import functools
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


# ──── Erros da camada audiovisual local. Mensagens e detalhes nunca trazem caminho de arquivo.
_CAMINHO_WINDOWS = re.compile(r"[A-Za-z]:\\[^\s'\"<>|]*")
_CAMINHO_POSIX = re.compile(r"(?<![\w:])/(?:[^\s'\"/]+/)+[^\s'\"]*")


def sem_caminhos(texto: str) -> str:
    """Troca caminhos absolutos (Windows e POSIX) por `<caminho>`: erros do FFmpeg e do sistema carregam o nome do arquivo."""
    return _CAMINHO_POSIX.sub("<caminho>", _CAMINHO_WINDOWS.sub("<caminho>", str(texto)))


class ErroExtracao(Exception):
    """Falha prevista na leitura ou no processamento da mídia.

    `codigo`: `arquivo_ilegivel` (não existe, não é arquivo, vazio, sem permissão), `midia_invalida` (não é um vídeo
    legível: formato, truncamento, sem quadros), `limite_excedido` (duração ou tamanho acima do limite de recurso).
    """

    def __init__(self, codigo: str, mensagem: str, detalhes: Mapping[str, object] | None = None) -> None:
        mensagem = sem_caminhos(mensagem)
        super().__init__(mensagem)
        self.codigo, self.mensagem, self.detalhes = codigo, mensagem, dict(detalhes or {})


class DadosInsuficientes(Exception):
    """A entrada é curta demais para a medida (por exemplo, movimento com um único quadro): não é falha nem zero."""


# ──── Especificação congelada da camada audiovisual local: versão, amostragem, limites de recurso e identi
VERSAO = "audiovisual_local=v1"

# Amostragem (TAPI §6.2: 5 a 8 quadros, incluindo 2 ou 3 dos primeiros 3 s).
N_QUADROS = 8
N_QUADROS_GANCHO = 3
JANELA_GANCHO_S = 3.0
MIN_QUADROS_TAPI = 5
FPS_PADRAO = 25.0

# Limites de recurso: toda tarefa é finita e de memória limitada, e os modelos são carregados um de cada vez.
MAX_DURACAO_S = 180.0
MAX_BYTES = 500_000_000
LADO_MAX_QUADRO_PX = 960
THREADS = 4
ORCAMENTO_TOTAL_S = 1200.0

# Opções do FFmpeg/PyAV: só o protocolo `file`. Um "vídeo" que na verdade é uma playlist (HLS, concat) não pode abrir
# URLs nem outros arquivos: o arquivo aceito na fronteira é o único que o FFmpeg pode ler.
OPCOES_FFMPEG = {"protocol_whitelist": "file"}
OPENCV_FFMPEG_OPCOES = "protocol_whitelist;file"

# Movimento (fluxo óptico de Farnebäck entre quadros próximos, em imagem reduzida).
# FARNEBACK: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags.
FARNEBACK = (0.5, 3, 15, 3, 5, 1.2, 0)
FLUXO_PARES = 6
FLUXO_SALTO_QUADROS = 2
FLUXO_LADO_PX = 192

# Cortes de cena (PySceneDetect ContentDetector). O limiar é o padrão da biblioteca; não foi calibrado.
CORTES_LIMIAR = 27.0
CORTES_DURACAO_MIN_S = 0.5

# Áudio: 16 kHz mono (o que o Whisper espera), janelas de 25 ms com passo de 10 ms.
AUDIO_TAXA_HZ = 16_000
AUDIO_JANELA_S = 0.025
AUDIO_PASSO_S = 0.010
AUDIO_SILENCIO_DIGITAL_DB = -80.0
AUDIO_ATIVO_ACIMA_DB = -45.0
AUDIO_PISO_DB = -120.0  # nível atribuído a RMS nulo: evita -inf e mantém a média finita

# Fala (faster-whisper). Os limiares de incerteza são os padrão do próprio Whisper; não foram validados aqui.
WHISPER_TAMANHO = "small"
WHISPER_COMPUTE = "int8"
WHISPER_BEAM = 5
WHISPER_VAD = True
WHISPER_TEMPERATURA = 0.0
WHISPER_CONDICIONAR_NO_TEXTO = False
WHISPER_LOGPROB_MIN = -1.0
WHISPER_NO_SPEECH_MAX = 0.6
WHISPER_COMPRESSAO_MAX = 2.4
WHISPER_PROB_IDIOMA_MIN = 0.5

# Detectores.
LOTE_EMBEDDING = 4
YOLOX_ENTRADA = 416
YOLOX_CONFIANCA = 0.30
YOLOX_NMS = 0.45
CONFIANCA_ROSTO = 0.5
CONFIANCA_OCR = 0.3

DIRETORIO_MODELOS = Path.home() / ".cache" / "casegalaxy"

MODELOS: dict[str, dict] = {
    "siglip2": {
        "papel": "embedding dos quadros (encoder congelado, torre de visão)",
        "repo": "google/siglip2-base-patch16-224",
        "revisao": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "dimensao": 768,
        "licenca_codigo": "Apache-2.0 (transformers)",
        "licenca_pesos": "Apache-2.0 (cartão do modelo no Hugging Face, consultado em 19/10/2026)",
        "arquivos": {
            "model.safetensors": {"bytes": 1_500_800_904,
                                  "sha256": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"},
            "config.json": {"bytes": 253, "git_blob": "c8cd2a20e58a738f44f267ae19f9568ff1095698"},
            "preprocessor_config.json": {"bytes": 394, "git_blob": "2e52d8e8492b5c496ae04c37bfa09760469fb18b"},
        },
    },
    "whisper": {
        "papel": "transcrição local (faster-whisper sobre a conversão CTranslate2 do Whisper small)",
        "repo": "Systran/faster-whisper-small",
        "revisao": "536b0662742c02347bc0e980a01041f333bce120",
        "licenca_codigo": "MIT (faster-whisper)",
        "licenca_pesos": "MIT (cartão do modelo no Hugging Face, consultado em 19/10/2026)",
        "arquivos": {
            "model.bin": {"bytes": 483_546_902,
                          "sha256": "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"},
            "config.json": {"bytes": 2370, "git_blob": "e5047537059bd8f182d9ca64c470201585015187"},
            "tokenizer.json": {"bytes": 2_203_239, "git_blob": "7818adb6de9fa3064d3ff81226fdd675be1f6344"},
            "vocabulary.txt": {"bytes": 459_861, "git_blob": "c9074644d9d1205686f16d411564729461324b75"},
        },
    },
    "yolox": {
        "papel": "objetos (YOLOX-tiny, 80 classes COCO) em ONNX Runtime",
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "revisao": "e1052df71842031413f6030723c3607b839c80ce",
        "licenca_codigo": "Apache-2.0 (repositório YOLOX) e MIT (onnxruntime)",
        "licenca_pesos": "Apache-2.0 do repositório; sem licença separada declarada para o ONNX publicado",
        "arquivos": {
            "yolox_tiny.onnx": {"bytes": 20_219_662,
                                "sha256": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"},
        },
    },
    # Reutilizados do extrator de capa (`src/extracao/imagem.py`); a URL não tem versão, então valem o tamanho e o SHA-256.
    "rosto": {
        "papel": "rosto por quadro (MediaPipe BlazeFace de curto alcance)",
        "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/"
               "blaze_face_short_range.tflite",
        "revisao": "sem versão na URL; identificado por SHA-256",
        "licenca_codigo": "Apache-2.0 (MediaPipe)",
        "licenca_pesos": "Apache-2.0 (ficha do modelo, conforme o registro de 19/10/2026)",
        "arquivos": {
            "blaze_face_short_range.tflite": {
                "bytes": 229_746, "sha256": "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"},
        },
    },
    "ocr": {
        "papel": "texto na tela por quadro (EasyOCR, português e inglês)",
        "url": "baixado pelo próprio EasyOCR na preparação",
        "revisao": "sem versão fixa; identificado por SHA-256",
        "licenca_codigo": "Apache-2.0 (EasyOCR)",
        "licenca_pesos": "NÃO CONFIRMADA: as fontes oficiais não declaram licença para os pesos (registro de 19/10/2026)",
        "arquivos": {
            "craft_mlt_25k.pth": {"bytes": 83_152_330,
                                  "sha256": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"},
            "latin_g2.pth": {"bytes": 15_406_141,
                             "sha256": "aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55"},
        },
    },
}


PACOTES_NA_IDENTIDADE = ("av", "opencv-python-headless", "numpy", "scenedetect", "faster-whisper", "ctranslate2",
                         "onnxruntime", "torch", "transformers", "easyocr", "mediapipe")


@functools.cache
def _versoes_instaladas() -> tuple[tuple[str, str | None], ...]:
    """Consulta única, por processo, aos metadados dos pacotes (identidade e proveniência a pedem várias vezes)."""
    from importlib import metadata

    versoes = []
    for pacote in PACOTES_NA_IDENTIDADE:
        try:
            versoes.append((pacote, metadata.version(pacote)))
        except metadata.PackageNotFoundError:
            versoes.append((pacote, None))
    return tuple(versoes)


def versoes_das_bibliotecas() -> dict[str, str | None]:
    """Versões instaladas das bibliotecas que produzem os números; entram na identidade do cache."""
    return dict(_versoes_instaladas())


def configuracao() -> dict:
    """Tudo que determina o resultado de uma extração, em forma serializável (inclui as versões das bibliotecas)."""
    return {
        "versao": VERSAO,
        "amostragem": {"n_quadros": N_QUADROS, "n_gancho": N_QUADROS_GANCHO, "janela_gancho_s": JANELA_GANCHO_S,
                       "lado_max_px": LADO_MAX_QUADRO_PX, "fps_padrao": FPS_PADRAO},
        "fluxo": {"pares": FLUXO_PARES, "salto_quadros": FLUXO_SALTO_QUADROS, "lado_px": FLUXO_LADO_PX,
                  "farneback": list(FARNEBACK)},
        "cortes": {"limiar": CORTES_LIMIAR, "duracao_min_s": CORTES_DURACAO_MIN_S},
        "audio": {"taxa_hz": AUDIO_TAXA_HZ, "janela_s": AUDIO_JANELA_S, "passo_s": AUDIO_PASSO_S,
                  "silencio_digital_db": AUDIO_SILENCIO_DIGITAL_DB, "ativo_acima_db": AUDIO_ATIVO_ACIMA_DB,
                  "piso_db": AUDIO_PISO_DB},
        "fala": {"tamanho": WHISPER_TAMANHO, "compute": WHISPER_COMPUTE, "beam": WHISPER_BEAM, "vad": WHISPER_VAD,
                 "temperatura": WHISPER_TEMPERATURA, "condicionar_no_texto": WHISPER_CONDICIONAR_NO_TEXTO,
                 "logprob_min": WHISPER_LOGPROB_MIN, "no_speech_max": WHISPER_NO_SPEECH_MAX,
                 "compressao_max": WHISPER_COMPRESSAO_MAX, "prob_idioma_min": WHISPER_PROB_IDIOMA_MIN},
        "embedding": {"lote": LOTE_EMBEDDING},
        "bibliotecas": versoes_das_bibliotecas(),
        "detectores": {"yolox_entrada": YOLOX_ENTRADA, "yolox_confianca": YOLOX_CONFIANCA, "yolox_nms": YOLOX_NMS,
                       "rosto_confianca": CONFIANCA_ROSTO, "ocr_confianca": CONFIANCA_OCR},
        "modelos": {nome: {"fonte": m.get("repo", m.get("url")), "revisao": m["revisao"],
                           "arquivos": {a: d.get("sha256", d.get("git_blob")) for a, d in m["arquivos"].items()}}
                    for nome, m in MODELOS.items()},
    }


def identidade_da_configuracao(config: Mapping | None = None) -> str:
    """SHA-256 da configuração em JSON canônico: a ordem das chaves não importa, qualquer valor diferente muda o hash."""
    corpo = json.dumps(configuracao() if config is None else config, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"))
    return hashlib.sha256(corpo.encode("utf-8")).hexdigest()


# ──── Localização e verificação dos pesos locais, sem rede.
COMANDO_PREPARAR = "python -m src.extracao.pipeline preparar"
_SUBDIRETORIO = {"yolox": "yolox", "rosto": "mediapipe", "ocr": "easyocr"}  # layout em DIRETORIO_MODELOS
_BLOCO = 1_048_576


class ModeloAusente(Exception):
    def __init__(self, nome: str, faltando: list[str], local: Path) -> None:
        super().__init__(f"Pesos ausentes de '{nome}': {', '.join(faltando)}. A inferência não baixa nada. Prepare uma "
                         f"vez, com rede, por: {COMANDO_PREPARAR}")
        self.nome, self.faltando, self.local = nome, faltando, local


class ModeloCorrompido(Exception):
    def __init__(self, nome: str, problemas: list[str], local: Path) -> None:
        super().__init__(f"Pesos de '{nome}' com problema: {'; '.join(problemas)}. Refaça a preparação: "
                         f"{COMANDO_PREPARAR}")
        self.nome, self.problemas, self.local = nome, problemas, local


def _pasta_hf(nome: str, cache_hf: Path | None) -> Path | None:
    """Pasta do snapshot fixado no cache do Hugging Face, ou None se ela não estiver em disco."""
    from huggingface_hub import snapshot_download

    modelo = MODELOS[nome]
    try:
        return Path(snapshot_download(modelo["repo"], revision=modelo["revisao"], local_files_only=True,
                                      cache_dir=cache_hf, allow_patterns=list(modelo["arquivos"])))
    except Exception:  # noqa: BLE001 - LocalEntryNotFoundError e correlatos: não está em disco
        return None


def _pasta(nome: str, diretorio_modelos: Path | None, cache_hf: Path | None) -> Path | None:
    MODELOS[nome]  # KeyError para nome desconhecido: é erro de programação, não de ambiente
    if nome in _SUBDIRETORIO:
        return Path(diretorio_modelos or DIRETORIO_MODELOS) / _SUBDIRETORIO[nome]
    return _pasta_hf(nome, cache_hf)


def _sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while bloco := arquivo.read(_BLOCO):
            resumo.update(bloco)
    return resumo.hexdigest()


def _git_blob(caminho: Path) -> str:
    dados = caminho.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(dados) + dados).hexdigest()  # noqa: S324 - identificador git, não segurança


def _problemas(nome: str, pasta: Path | None, completo: bool) -> tuple[list[str], list[str]]:
    """(arquivos ausentes, outros problemas) dos pesos fixados de `nome`."""
    ausentes: list[str] = []
    problemas: list[str] = []
    for arquivo, fixado in MODELOS[nome]["arquivos"].items():
        caminho = None if pasta is None else pasta / arquivo
        if caminho is None or not caminho.is_file():
            ausentes.append(arquivo)
        elif caminho.stat().st_size != fixado["bytes"]:
            problemas.append(f"{arquivo}: tamanho {caminho.stat().st_size} em vez de {fixado['bytes']}")
        elif completo and "sha256" in fixado and _sha256(caminho) != fixado["sha256"]:
            problemas.append(f"{arquivo}: sha256 diferente do fixado")
        elif completo and "git_blob" in fixado and _git_blob(caminho) != fixado["git_blob"]:
            problemas.append(f"{arquivo}: git_blob diferente do fixado")
    return ausentes, problemas


def verificar_integridade(nome: str, *, diretorio_modelos: Path | None = None, cache_hf: Path | None = None,
                          completo: bool = False) -> list[str]:
    """Lista de problemas dos pesos de `nome` (vazia = íntegros); `completo` confere SHA-256 e git blob."""
    pasta = _pasta(nome, diretorio_modelos, cache_hf)
    ausentes, problemas = _problemas(nome, pasta, completo)
    return [f"ausente: {a}" for a in ausentes] + problemas


def localizar(nome: str, *, diretorio_modelos: Path | None = None, cache_hf: Path | None = None) -> Path:
    """Pasta dos pesos (ou o próprio arquivo, quando o modelo é um arquivo só); nunca baixa nada."""
    pasta = _pasta(nome, diretorio_modelos, cache_hf)
    ausentes, problemas = _problemas(nome, pasta, completo=False)
    local = pasta or Path(".")
    if ausentes:
        raise ModeloAusente(nome, ausentes, local)
    if problemas:
        raise ModeloCorrompido(nome, problemas, local)
    arquivos = list(MODELOS[nome]["arquivos"])
    return pasta / arquivos[0] if len(arquivos) == 1 else pasta


# ──── Conferência do ambiente local: bibliotecas do sistema e pesos dos codificadores congelados.
REV_E5 = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
REV_CLIP = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
BIBLIOTECAS_DO_SISTEMA = ("libgomp.so.1", "libGLESv2.so.2", "libEGL.so.1")
PACOTES_DEBIAN = {"libgomp.so.1": ("libgomp1", "LightGBM"), "libGLESv2.so.2": ("libgles2", "MediaPipe"),
                  "libEGL.so.1": ("libegl1", "MediaPipe")}
CODIGO_SAIDA_AMBIENTE_INCOMPLETO = 3
COMANDO_BAIXAR = "uv run casegalaxy-demo --baixar-codificadores"
COMANDO_VERIFICAR = "uv run casegalaxy-demo --verificar-ambiente"


@dataclass(frozen=True)
class ItemPeso:
    grupo: str
    arquivo: str
    caminho: Path
    bytes: int
    sha256: str
    repo: str | None = None
    revisao: str | None = None

    @property
    def nome(self) -> str:
        return f"{self.grupo}/{self.arquivo}"


@dataclass(frozen=True)
class EstadoPeso:
    item: ItemPeso
    estado: str  # ok | ausente | tamanho_diferente | hash_diferente
    bytes_reais: int | None = None


@dataclass(frozen=True)
class EstadoBiblioteca:
    nome: str
    estado: str  # ok | ausente


@dataclass(frozen=True)
class EstadoRef:
    """`refs/main` de um repositório do Hugging Face: tem de apontar para a revisão fixa."""

    repo: str
    revisao: str
    caminho: Path
    estado: str  # ok | ausente | divergente
    encontrado: str | None = None


_E5 = (("config.json", 655, "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
       ("model.safetensors", 470_641_600, "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
       ("sentencepiece.bpe.model", 5_069_051, "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
       ("special_tokens_map.json", 167, "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"),
       ("tokenizer.json", 17_082_730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
       ("tokenizer_config.json", 443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"))
_CLIP = (("config.json", 4_186, "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770"),
         ("preprocessor_config.json", 316, "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"),
         ("pytorch_model.bin", 605_247_071, "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"))
_OCR = (("craft_mlt_25k.pth", 83_152_330, "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17"),
        ("latin_g2.pth", 15_406_141, "aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55"))
_ROSTO = (("blaze_face_short_range.tflite", 229_746,
           "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"),)


def pasta_hub_padrao() -> Path:
    """Cache do Hugging Face exatamente como o runtime o resolve (`HF_HUB_CACHE`, `HF_HOME`, `XDG_CACHE_HOME`, `~`)."""
    from huggingface_hub import constants  # leve: não importa torch nem transformers

    return Path(constants.HF_HUB_CACHE).expanduser()


def pasta_casa_padrao() -> Path:
    return Path.home() / ".cache" / "casegalaxy"


def itens_esperados(hub: Path, casa: Path) -> list[ItemPeso]:
    """Pesos que o runtime carrega, com tamanho e SHA-256 esperados."""
    itens = []
    for grupo, repo, revisao, arquivos in (("e5-small", "intfloat/multilingual-e5-small", REV_E5, _E5),
                                           ("clip-vit-b32", "openai/clip-vit-base-patch32", REV_CLIP, _CLIP)):
        base = Path(hub) / f"models--{repo.replace('/', '--')}" / "snapshots" / revisao
        itens += [ItemPeso(grupo, nome, base / nome, tamanho, sha, repo, revisao) for nome, tamanho, sha in arquivos]
    itens += [ItemPeso("easyocr", nome, Path(casa) / "easyocr" / nome, tamanho, sha) for nome, tamanho, sha in _OCR]
    itens += [ItemPeso("mediapipe-face", nome, Path(casa) / "mediapipe" / nome, tamanho, sha) for nome, tamanho, sha in _ROSTO]
    return itens


def pasta_refs_main(hub: Path, repo: str) -> Path:
    return Path(hub) / f"models--{repo.replace('/', '--')}" / "refs" / "main"


def verificar_refs(hub: Path, estados: Sequence[EstadoPeso]) -> list[EstadoRef]:
    """`refs/main` de cada repositório cujos arquivos estão completos (senão o download grava junto com o snapshot)."""
    refs = []
    for repo, revisao in (("intfloat/multilingual-e5-small", REV_E5), ("openai/clip-vit-base-patch32", REV_CLIP)):
        do_repo = [e for e in estados if e.item.repo == repo]
        if not do_repo or any(e.estado != "ok" for e in do_repo):
            continue
        caminho = pasta_refs_main(hub, repo)
        encontrado = caminho.read_text(encoding="utf-8").strip() if caminho.is_file() else None
        estado = "ausente" if encontrado is None else "ok" if encontrado == revisao else "divergente"
        refs.append(EstadoRef(repo, revisao, caminho, estado, encontrado))
    return refs


def _sha256_ambiente(arquivo: Path) -> str:
    h = hashlib.sha256()
    with Path(arquivo).open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_pesos(itens: Sequence[ItemPeso], *, conferir_hash: bool) -> list[EstadoPeso]:
    estados = []
    for item in itens:
        if not item.caminho.is_file():
            estados.append(EstadoPeso(item, "ausente"))
            continue
        tamanho = item.caminho.stat().st_size
        if tamanho != item.bytes:
            estados.append(EstadoPeso(item, "tamanho_diferente", tamanho))
        elif conferir_hash and _sha256_ambiente(item.caminho) != item.sha256:
            estados.append(EstadoPeso(item, "hash_diferente", tamanho))
        else:
            estados.append(EstadoPeso(item, "ok", tamanho))
    return estados


def milhares(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def verificar_bibliotecas(nomes: Sequence[str] = BIBLIOTECAS_DO_SISTEMA) -> list[EstadoBiblioteca]:
    if not sys.platform.startswith("linux"):
        return []  # nomes de biblioteca do Linux: fora dele só os pesos são conferidos
    achados = []
    for nome in nomes:
        try:
            ctypes.CDLL(nome)
            achados.append(EstadoBiblioteca(nome, "ok"))
        except OSError:
            achados.append(EstadoBiblioteca(nome, "ausente"))
    return achados


def texto_bibliotecas_ausentes(ausentes: Sequence[str]) -> list[str]:
    """Linhas de orientação para as bibliotecas do sistema que faltam (nomes de pacote Debian/Ubuntu)."""
    linhas = []
    for nome in ausentes:
        pacote, usada_por = PACOTES_DEBIAN.get(nome, (None, None))
        linhas.append(f"  - {nome}" + (f" (pacote {pacote}; usada pelo {usada_por})" if pacote else ""))
    pacotes = [PACOTES_DEBIAN[n][0] for n in ausentes if n in PACOTES_DEBIAN]
    if pacotes:
        linhas.append(f"  Em Ubuntu/Debian: sudo apt install {' '.join(pacotes)}")
    linhas.append("  Sem permissão de administrador: coloque as .so numa pasta e aponte LD_LIBRARY_PATH para ela.")
    return linhas


class AmbienteIncompleto(RuntimeError):
    """Faltam bibliotecas do sistema, pesos dos codificadores e/ou `refs/main` certo. A mensagem diz o que fazer.

    `str(erro)` traz os caminhos completos (stderr e linha de comando); `mensagem_sem_caminhos()` é a versão para clientes
    remotos (envelope do MCP), sem caminhos absolutos."""

    def __init__(self, pesos: Sequence[EstadoPeso] = (), bibliotecas: Sequence[str] = (),
                 refs: Sequence[EstadoRef] = ()) -> None:
        self.pesos_ausentes = tuple(pesos)
        self.bibliotecas_ausentes = tuple(bibliotecas)
        self.refs_divergentes = tuple(refs)
        super().__init__(self._texto(com_caminhos=True))

    def mensagem_sem_caminhos(self) -> str:
        return self._texto(com_caminhos=False)

    def _texto(self, *, com_caminhos: bool) -> str:
        onde = (lambda caminho: f": {caminho}") if com_caminhos else (lambda caminho: "")
        linhas = ["Ambiente incompleto: faltam itens que o projeto não inclui nem instala sozinho."]
        if self.bibliotecas_ausentes:
            linhas += ["Bibliotecas do sistema ausentes:", *texto_bibliotecas_ausentes(self.bibliotecas_ausentes)]
        if self.pesos_ausentes:
            total = sum(e.item.bytes for e in self.pesos_ausentes)
            linhas.append(f"Pesos dos codificadores ausentes ou divergentes ({len(self.pesos_ausentes)} arquivo(s), "
                          f"cerca de {total / 1e6:.0f} MB no total):")
            linhas += [f"  - {e.item.nome} ({milhares(e.item.bytes)} bytes), {e.estado.replace('_', ' ')}"
                       f"{'' if e.bytes_reais is None else f' (encontrado: {milhares(e.bytes_reais)} bytes)'}"
                       f"{onde(e.item.caminho)}" for e in self.pesos_ausentes]
        if self.refs_divergentes:
            linhas.append("`refs/main` do Hugging Face fora da revisão fixa (o runtime carrega o que ele aponta):")
            linhas += [f"  - {r.repo}: {r.estado}"
                       f"{'' if r.encontrado is None else f' (encontrado: {r.encontrado})'}; esperado {r.revisao}"
                       f"{onde(r.caminho)}" for r in self.refs_divergentes]
        if self.pesos_ausentes or self.refs_divergentes:
            linhas.append(f"Para baixá-los ou corrigir (precisa de rede para baixar; licenças dos pesos não confirmadas): "
                          f"{COMANDO_BAIXAR}")
            linhas.append(f"Para só conferir: {COMANDO_VERIFICAR}")
        return "\n".join(linhas)

    def detalhes(self) -> dict:
        """Resumo curto para envelopes de erro (nomes apenas, sem caminhos)."""
        return {"tipo": type(self).__name__, "pesos_ausentes": [e.item.nome for e in self.pesos_ausentes],
                "bibliotecas_ausentes": list(self.bibliotecas_ausentes),
                "refs_divergentes": [r.repo for r in self.refs_divergentes]}


def exigir_ambiente(hub: Path | None = None, casa: Path | None = None, *, conferir_hash: bool = False,
                    bibliotecas: Sequence[str] = BIBLIOTECAS_DO_SISTEMA) -> None:
    """Levanta `AmbienteIncompleto` se faltar biblioteca, peso ou `refs/main` certo (bibliotecas só em Linux).

    Sem `conferir_hash` compara só o tamanho dos pesos (rápido)."""
    hub = pasta_hub_padrao() if hub is None else hub
    casa = pasta_casa_padrao() if casa is None else casa
    estados = verificar_pesos(itens_esperados(hub, casa), conferir_hash=conferir_hash)
    pesos = [e for e in estados if e.estado != "ok"]
    refs = [r for r in verificar_refs(hub, estados) if r.estado != "ok"]
    libs = [b.nome for b in verificar_bibliotecas(bibliotecas) if b.estado != "ok"]
    if pesos or libs or refs:
        raise AmbienteIncompleto(pesos, libs, refs)
