"""Atributos locais da legenda (C7; família N2): embedding pré-treinado bruto e contagens simples."""

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol


# ──── Atributos locais da legenda (C7; família N2): embedding pré-treinado bruto e contagens simples.
VERSAO = "1"
MODELO_TEXTO = "intfloat/multilingual-e5-small"
PREFIXO = "query: "
MAX_TOKENS = 512
DIMENSAO_TEXTO = 384
THREADS = 4
CONTAGENS_LEGENDA = (
    "leg_n_caracteres_titulo", "leg_n_caracteres_descricao", "leg_n_palavras", "leg_n_hashtags", "leg_n_tags",
    "leg_n_emojis", "leg_n_links", "leg_tem_descricao",
)
REGRAS = {
    "composicao": "título, descrição e 'tags: t1, t2', nessa ordem, separados por uma linha em branco; partes vazias "
    "ficam fora; nada é inventado para uma parte ausente",
    "normalizacao": "Unicode NFC; quebras de linha unificadas; espaços repetidos viram um; espaços nas pontas das "
    "linhas removidos; no máximo uma linha em branco seguida. Caixa, acentos, hashtags, emojis e links preservados",
    "contagens": {
        "leg_n_caracteres_titulo": "caracteres do título normalizado",
        "leg_n_caracteres_descricao": "caracteres da descrição normalizada",
        "leg_n_palavras": "fragmentos separados por espaço, com ao menos uma letra ou dígito, em título e descrição",
        "leg_n_hashtags": "ocorrências de #palavra em título e descrição",
        "leg_n_tags": "tags não vazias",
        "leg_n_emojis": "caracteres nas faixas Unicode de emoji e símbolos pictográficos (aproximação; bandeiras "
        "contam dois)",
        "leg_n_links": "trechos iniciados por http://, https:// ou www.",
        "leg_tem_descricao": "1 se a descrição normalizada não é vazia",
    },
    "codificador": f"{MODELO_TEXTO}; entrada = '{PREFIXO}' + texto composto; corte em {MAX_TOKENS} tokens; média com "
    "máscara; normalização L2",
}

_HASHTAG = re.compile(r"#\w+")
_LINK = re.compile(r"(?:https?://|www\.)\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿]")
_ESPACOS = re.compile(r"[^\S\n]+")
_LINHAS_VAZIAS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class VetorTexto:
    vetor: tuple[float, ...]
    n_tokens: int
    truncado: bool


class Codificador(Protocol):
    def identificacao(self) -> dict: ...

    def codificar(self, textos: Sequence[str]) -> list[VetorTexto]: ...


def normalizar_texto(texto: str | None) -> str:
    """Normalização determinística que preserva o conteúdo do texto."""
    if not texto:
        return ""
    texto = unicodedata.normalize("NFC", texto).replace("\r\n", "\n").replace("\r", "\n")
    linhas = [_ESPACOS.sub(" ", linha).strip() for linha in texto.split("\n")]
    return _LINHAS_VAZIAS.sub("\n\n", "\n".join(linhas)).strip()


def _tags(tags: Iterable[str] | None) -> list[str]:
    return [t for t in (normalizar_texto(tag) for tag in (tags or ())) if t]


def compor_legenda(titulo: str | None, descricao: str | None, tags: Iterable[str] | None) -> str:
    """Texto único enviado ao codificador, no treino e na previsão."""
    lista = _tags(tags)
    partes = [normalizar_texto(titulo), normalizar_texto(descricao), "tags: " + ", ".join(lista) if lista else ""]
    return "\n\n".join(p for p in partes if p)


def contagens_legenda(titulo: str | None, descricao: str | None, tags: Iterable[str] | None) -> dict[str, int]:
    """Contagens documentadas em REGRAS["contagens"], na ordem de CONTAGENS_LEGENDA."""
    titulo, descricao = normalizar_texto(titulo), normalizar_texto(descricao)
    corpo = f"{titulo}\n{descricao}"
    return {
        "leg_n_caracteres_titulo": len(titulo),
        "leg_n_caracteres_descricao": len(descricao),
        "leg_n_palavras": sum(any(c.isalnum() for c in parte) for parte in corpo.split()),
        "leg_n_hashtags": len(_HASHTAG.findall(corpo)),
        "leg_n_tags": len(_tags(tags)),
        "leg_n_emojis": len(_EMOJI.findall(corpo)),
        "leg_n_links": len(_LINK.findall(corpo)),
        "leg_tem_descricao": int(bool(descricao)),
    }


def validar_vetor(vetor: Iterable[float], dimensao: int) -> tuple[float, ...]:
    """Vetor com a dimensão esperada e só valores finitos; caso contrário, ValueError."""
    valores = tuple(float(v) for v in vetor)
    if len(valores) != dimensao:
        raise ValueError(f"vetor com {len(valores)} valores; esperado {dimensao}")
    if not all(math.isfinite(v) for v in valores):
        raise ValueError("vetor com valor não finito")
    return valores


def _falha(erro: Exception) -> dict:
    return {"estado": "falha_extrator", "vetor": None, "n_tokens": None, "truncado": None,
            "erro": type(erro).__name__, "mensagem": str(erro)[:200]}


def extrair_legendas(legendas: Sequence[tuple[str | None, str | None, Iterable[str] | None]],
                     codificador: Codificador) -> list[dict]:
    """Uma entrada por legenda: estado, texto composto, vetor, tokens, corte e contagens. Falha nunca vira zero."""
    compostos = [compor_legenda(*legenda) for legenda in legendas]
    resultados: list[dict] = [{"estado": "texto_vazio", "vetor": None, "n_tokens": 0, "truncado": False}
                              for _ in legendas]
    com_texto = [i for i, texto in enumerate(compostos) if texto]
    if com_texto:
        try:
            vetores = codificador.codificar([compostos[i] for i in com_texto])
        except Exception:  # noqa: BLE001 - o lote falhou; cada legenda é tentada sozinha para isolar a falha
            vetores = None
        for posicao, i in enumerate(com_texto):
            try:
                v = vetores[posicao] if vetores is not None else codificador.codificar([compostos[i]])[0]
                resultados[i] = {"estado": "ok", "vetor": validar_vetor(v.vetor, DIMENSAO_TEXTO),
                                 "n_tokens": int(v.n_tokens), "truncado": bool(v.truncado)}
            except Exception as erro:  # noqa: BLE001 - falha registrada com o tipo, sem valor inventado
                resultados[i] = _falha(erro)
    return [r | {"texto_composto": texto} | contagens_legenda(*legenda)
            for r, texto, legenda in zip(resultados, compostos, legendas, strict=True)]


def extrair_legenda(titulo: str | None, descricao: str | None, tags: Iterable[str] | None,
                    codificador: Codificador) -> dict:
    """Mesmo caminho para a legenda publicada (treino) e para a legenda planejada (previsão)."""
    return extrair_legendas([(titulo, descricao, tags)], codificador)[0]


class CodificadorTexto:
    """e5-small em CPU, carregado só no primeiro uso."""

    def __init__(self, modelo: str = MODELO_TEXTO, revisao: str | None = None, threads: int = THREADS,
                 lote: int = 16) -> None:
        self.modelo, self.revisao, self.threads, self.lote = modelo, revisao, threads, lote
        self._tokenizador = self._rede = None

    def _carregar(self) -> None:
        if self._rede is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(self.threads)
        self._tokenizador = AutoTokenizer.from_pretrained(self.modelo, revision=self.revisao)
        self._rede = AutoModel.from_pretrained(self.modelo, revision=self.revisao).eval()

    def identificacao(self) -> dict:
        self._carregar()
        import torch
        import transformers

        return {"modelo": self.modelo, "revisao": getattr(self._rede.config, "_commit_hash", None),
                "dimensao": DIMENSAO_TEXTO, "prefixo": PREFIXO, "max_tokens": MAX_TOKENS,
                "pooling": "média com máscara", "normalizacao": "L2", "torch": torch.__version__,
                "transformers": transformers.__version__}

    def codificar(self, textos: Sequence[str]) -> list[VetorTexto]:
        if any(not t for t in textos):
            raise ValueError("texto vazio não é codificado")
        self._carregar()
        import torch

        entradas = [PREFIXO + t for t in textos]
        n_tokens = [len(ids) for ids in self._tokenizador(entradas, truncation=False)["input_ids"]]
        vetores: list[list[float]] = []
        for inicio in range(0, len(entradas), self.lote):
            lote = self._tokenizador(entradas[inicio:inicio + self.lote], max_length=MAX_TOKENS, truncation=True,
                                     padding=True, return_tensors="pt")
            with torch.inference_mode():
                estados = self._rede(**lote).last_hidden_state
            mascara = lote["attention_mask"].unsqueeze(-1).to(estados.dtype)
            media = (estados * mascara).sum(dim=1) / mascara.sum(dim=1)
            vetores.extend(torch.nn.functional.normalize(media, p=2, dim=1).tolist())
        return [VetorTexto(tuple(v), n, n > MAX_TOKENS) for v, n in zip(vetores, n_tokens, strict=True)]
