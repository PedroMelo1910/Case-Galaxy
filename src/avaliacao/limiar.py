"""Módulo consolidado. Reúne:
- Limiar de largura de baixa confiança (D-E5, D-F2b; protocolo final §8).
- Declaração da faixa de previsão por contexto: validação de desenvolvimento ou teste final (protocolo §8, §14.9).
- Driver da LIM: larguras de faixa dos 100 vídeos reservados e limiar de baixa confiança por pacote (protocolo §8)."""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.avaliacao import metricas as bs
from src.dados import dataset as ds, historico as hs
from src.modelos import treino as fp, treino as md


# ──── Limiar de largura de baixa confiança (D-E5, D-F2b; protocolo final §8).
PERCENTIL = 80
LEITURAS = {"favoravel_ao_modelo": "sinalizados_erram_mais_com_evidencia",
            "desfavoravel_ao_modelo": "sinalizados_erram_menos_com_evidencia", "inconclusivo": "inconclusivo",
            "indefinido": "indefinido"}


def larguras(inferior_log: Sequence[float], superior_log: Sequence[float]) -> np.ndarray:
    return np.asarray(superior_log, dtype=float) - np.asarray(inferior_log, dtype=float)


def limiar_de_largura(larguras: Sequence[float], percentil: float = PERCENTIL) -> float:
    valores = np.asarray(larguras, dtype=float)
    if valores.size == 0 or not np.all(np.isfinite(valores)):
        raise ValueError("larguras vazias ou não finitas: o limiar não é definido")
    return float(np.percentile(valores, percentil))


def sinalizar(larguras: Sequence[float], limiar: float) -> np.ndarray:
    """Baixa confiança estatística: largura estritamente acima do limiar."""
    return np.asarray(larguras, dtype=float) > limiar


def validar_regra(larguras: Sequence[float], erros_absolutos: Sequence[float], canais: Sequence[str], *, limiar: float,
                  n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE) -> dict:
    """Erro absoluto médio dos sinalizados e dos não sinalizados, e a diferença com IC por reamostragem de canais."""
    marca = sinalizar(larguras, limiar)
    erro, canais = np.asarray(erros_absolutos, dtype=float), np.asarray([str(c) for c in canais])
    saida = {"limiar": float(limiar), "n_sinalizados": int(marca.sum()), "n_nao_sinalizados": int((~marca).sum())}
    if not marca.any() or marca.all():
        return saida | {"mae_sinalizados": None, "mae_nao_sinalizados": None, "bootstrap": None,
                        "leitura": "sem_sinalizados" if not marca.any() else "todos_sinalizados"}
    nomes = sorted(set(canais))
    por_canal = {c: canais == c for c in nomes}
    soma = {k: np.array([erro[m & sel].sum() for m in por_canal.values()]) for k, sel in
            (("sinal", marca), ("nao", ~marca))}
    n = {k: np.array([(m & sel).sum() for m in por_canal.values()], dtype=float) for k, sel in
         (("sinal", marca), ("nao", ~marca))}
    contagens = bs.sorteio_de_canais(len(nomes), n_repeticoes=n_repeticoes, semente=semente)

    def razao(chave: str) -> np.ndarray:
        num, den = contagens @ soma[chave], contagens @ n[chave]
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    resumo = bs.resumir_diferenca(razao("sinal"), razao("nao"), menor_e_melhor=False)
    return saida | {"mae_sinalizados": float(erro[marca].mean()), "mae_nao_sinalizados": float(erro[~marca].mean()),
                    "bootstrap": resumo, "leitura": LEITURAS[resumo["leitura"]],
                    "nota": "no modelo servido a validade da regra é presumida e declarada"}


# ──── Declaração da faixa de previsão por contexto: validação de desenvolvimento ou teste final (protocolo
NIVEL_NOMINAL = 0.8  # igual a `faixa_previsao.COBERTURA_NOMINAL`; repetido só para não importar o módulo sob hash
CONTEXTOS = {
    "desenvolvimento": "na validação de desenvolvimento",
    "teste_final": "no teste final retrospectivo aproximado (consulta única)",
}
RESSALVA = {
    "desenvolvimento": "É um resultado da validação de desenvolvimento, não é o teste final, e é sem garantia para novos "
                       "vídeos; foi medido sobre a faixa corrigida (q50 central; limites = mínimo e máximo das três saídas).",
    "teste_final": "É a estimativa do teste final retrospectivo aproximado (consulta única), medida sobre a faixa corrigida "
                   "(q50 central; limites = mínimo e máximo das três saídas); descreve o procedimento, não o artefato "
                   "servido, e é sem garantia para novos vídeos.",
}


def _pct(valor: float) -> str:
    return f"{100 * valor:.1f}%".replace(".", ",")


def _intervalo(ic: list[float] | None) -> str:
    return f"IC 95% por canal {_pct(ic[0])} a {_pct(ic[1])}" if ic else "IC 95% por canal indefinido"


def _linha_do_setor(setor: str, r: Mapping) -> dict:
    cobertura = f"{setor}: {r['n_dentro']} de {r['n']} vídeos ({_pct(r['cobertura'])}), {r['n_canais']} canais"
    if r.get("ic95_bootstrap_por_canal"):
        extra = f"{_intervalo(r['ic95_bootstrap_por_canal'])}"
        if r.get("ic_estado") == "estimado_aproximado":
            extra += " (aproximado: poucos canais)"
    else:
        extra = f"IC 95% indefinido: {r.get('motivo_ic_indefinido') or 'não estimável'}"
    return {"setor": setor, "texto": f"{cobertura}; {extra}", "ic_estado": r.get("ic_estado")}


def declarar(contexto: str, *, modelo: str, total: Mapping, por_setor: Mapping[str, Mapping]) -> dict:
    """Declaração (texto e números) da faixa de um modelo servido, no contexto dado."""
    if contexto not in CONTEXTOS:
        raise ValueError(f"contexto desconhecido: {contexto!r}; use {sorted(CONTEXTOS)}")
    observada = total["n_dentro"] / total["n"]
    abaixo = observada < NIVEL_NOMINAL
    comparacao = "abaixo do pretendido" if abaixo else "no nível pretendido ou acima"
    aviso = (f"Faixa de previsão do modelo {modelo}, com nível pretendido de {NIVEL_NOMINAL:.0%}. {CONTEXTOS[contexto].capitalize()}, "
             f"{total['n_dentro']} de {total['n']} vídeos ({_pct(observada)}) ficaram dentro dela ({comparacao}; "
             f"{_intervalo(total.get('ic95_bootstrap_por_canal'))}). {RESSALVA[contexto]}")
    return {"nome": "faixa de previsão", "contexto": contexto, "modelo": modelo, "nivel_nominal": NIVEL_NOMINAL,
            "cobertura_observada": observada, "n_dentro": total["n_dentro"], "n": total["n"],
            "ic95_bootstrap_por_canal": total.get("ic95_bootstrap_por_canal"), "abaixo_do_pretendido": abaixo,
            "por_setor": [_linha_do_setor(setor, r) for setor, r in sorted(por_setor.items())], "aviso": aviso}


# ──── Driver da LIM: larguras de faixa dos 100 vídeos reservados e limiar de baixa confiança por pacote (p
PAPEIS = ("desenvolvimento", "servido")
N_LIM = 100
COLUNAS_ROTULO = tuple(hs.COLUNAS_ALVO)
QUANTIS = ("q10", "q50", "q90")
ARQUIVO_LIMIAR = "limiar_largura_{papel}.json"
FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"
NOTA_DA_LIM = ("Limiar calculado nas larguras das previsões da LIM, sem rótulos; a LIM não calibra nem seleciona "
               "nada. Sinaliza previsões mais largas que as demais e não garante cobertura da faixa.")


class LimiarIncompativel(RuntimeError):
    """O limiar pertence a outro pacote (ou a outro papel) que o usado: não se transfere em silêncio."""


class ErroLim(ValueError):
    """Entrada da LIM incompleta ou inconsistente."""


@dataclass(frozen=True)
class LimiarDeLargura:
    papel: str
    sha256_pacote: str
    ids_sha256: str
    n: int
    percentil: float
    valor: float


# --- linhas sem rótulo ----------------------------------------------------------------------------------------------


def sem_rotulos(linhas: Sequence[Mapping]) -> list[dict]:
    """Cópias das linhas sem as colunas-alvo; `ref_log` fica, pois a faixa em log depende dela."""
    return [{c: v for c, v in linha.items() if c not in COLUNAS_ROTULO} for linha in linhas]


def exigir_sem_rotulos(linhas: Sequence[Mapping]) -> None:
    com_rotulo = sorted({c for linha in linhas for c in COLUNAS_ROTULO if c in linha})
    if com_rotulo:
        raise ValueError(f"a LIM não tem rótulos: as linhas trazem {com_rotulo}")


def montar_linhas(ids: Sequence[str], *, n1: Mapping[str, Mapping], legenda: Mapping[str, Mapping],
                  capa: Mapping[str, Mapping], gemini: Mapping[str, Mapping]) -> list[dict]:
    """Mesma junção do desenvolvimento (`dataset_desenvolvimento.juntar`), com o N1 já sem as colunas-alvo."""
    n1_limpo = {v: sem_rotulos([linha])[0] for v, linha in n1.items()}
    return ds.juntar(list(ids), n1=n1_limpo, legenda=legenda, capa=capa, gemini=gemini)


# --- larguras e limiar ----------------------------------------------------------------------------------------------


def larguras_do_pacote(q10, q50, q90, linhas: Sequence[Mapping]) -> np.ndarray:
    """Largura em log da faixa corrigida (mínimo e máximo das três saídas, depois do corte em zero)."""
    exigir_sem_rotulos(linhas)
    faixa = fp.prever_faixa(linhas, q10.prever_relativo(linhas), q50.prever_relativo(linhas),
                            q90.prever_relativo(linhas))
    return np.asarray(faixa["superior_log"] - faixa["inferior_log"], dtype=float)


def identidade_do_pacote(hashes: Mapping[str, str]) -> str:
    """SHA-256 que identifica o pacote: os hashes dos três modelos (q10, q50, q90)."""
    faltam = sorted(set(QUANTIS) - set(hashes))
    if faltam:
        raise ValueError(f"a identidade do pacote exige os três modelos; faltam {faltam}")
    return hashlib.sha256(json.dumps({q: hashes[q] for q in QUANTIS}, sort_keys=True).encode("utf-8")).hexdigest()


def calcular_limiar(papel: str, sha256_pacote: str, ids: Sequence[str], larguras: Sequence[float]) -> LimiarDeLargura:
    if papel not in PAPEIS:
        raise ValueError(f"papel desconhecido: {papel!r}; use um de {PAPEIS}")
    ids = list(ids)
    if len(ids) != len(larguras):
        raise ValueError("ids e larguras têm tamanhos diferentes")
    if len(set(ids)) != len(ids):
        raise ValueError("ids repetidos na LIM")
    if len(ids) != N_LIM:
        raise ValueError(f"a LIM tem {N_LIM} vídeos e o limiar só é definido com todos; recebi {len(ids)}")
    valor = limiar_de_largura(larguras, PERCENTIL)  # recusa vazio e valores não finitos
    return LimiarDeLargura(papel, sha256_pacote, hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
                           len(ids), float(PERCENTIL), valor)


def _exigir_compativel(limiar: LimiarDeLargura, sha256_pacote: str) -> None:
    if limiar.sha256_pacote != sha256_pacote:
        raise LimiarIncompativel(
            f"o limiar do pacote {limiar.papel} ({limiar.sha256_pacote[:12]}…) não vale para o pacote "
            f"{sha256_pacote[:12]}…; calcule o limiar com as larguras da LIM deste pacote")


def sinalizar_com(limiar: LimiarDeLargura, sha256_pacote: str, larguras: Sequence[float]) -> np.ndarray:
    _exigir_compativel(limiar, sha256_pacote)
    return sinalizar(larguras, limiar.valor)


def validar_no_pacote(limiar: LimiarDeLargura, sha256_pacote: str, larguras: Sequence[float],
                      erros_absolutos: Sequence[float], canais: Sequence[str], **opcoes) -> dict:
    """Diagnóstico de desenvolvimento: só com o limiar do pacote de desenvolvimento que gerou estas larguras."""
    if limiar.papel != "desenvolvimento":
        raise LimiarIncompativel("a regra só é validada com o limiar do pacote de desenvolvimento; o limiar "
                                 f"{limiar.papel} é operacional e sua validade é presumida e declarada")
    _exigir_compativel(limiar, sha256_pacote)
    resultado = validar_regra(larguras, erros_absolutos, canais, limiar=limiar.valor, **opcoes)
    return resultado | {"pacote": {"papel": limiar.papel, "sha256_pacote": limiar.sha256_pacote}}


def diagnostico_de_desenvolvimento(q10, q50, q90, linhas_avaliacao: Sequence[Mapping], limiar: LimiarDeLargura,
                                   sha256_pacote: str, **opcoes) -> dict:
    """Larguras e erro absoluto em log das linhas de VALIDAÇÃO de desenvolvimento (têm rótulo; a LIM não)."""
    faixa = fp.prever_faixa(linhas_avaliacao, q10.prever_relativo(linhas_avaliacao),
                            q50.prever_relativo(linhas_avaliacao), q90.prever_relativo(linhas_avaliacao))
    y = np.array([l["y_log"] for l in linhas_avaliacao], dtype=float)
    return validar_no_pacote(limiar, sha256_pacote, faixa["superior_log"] - faixa["inferior_log"],
                             np.abs(y - faixa["central_log"]), [l["channel_id"] for l in linhas_avaliacao], **opcoes)


# --- gravação -------------------------------------------------------------------------------------------------------


def _sha_do_conteudo(conteudo: Mapping) -> str:
    return hashlib.sha256(json.dumps(conteudo, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def gravar_limiar(limiar: LimiarDeLargura, pasta: Path) -> Path:
    """Grava `limiar_largura_<papel>.json` com o próprio SHA-256; nunca sobrescreve."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    conteudo = asdict(limiar) | {"nota": NOTA_DA_LIM, "gravado_em_utc": datetime.now(UTC).strftime(FORMATO_UTC)}
    arquivo = pasta / ARQUIVO_LIMIAR.format(papel=limiar.papel)
    with open(arquivo, "x", encoding="utf-8") as saida:
        json.dump(conteudo | {"sha256": _sha_do_conteudo(conteudo)}, saida, ensure_ascii=False, indent=1)
    return arquivo


def ler_limiar(arquivo: Path) -> LimiarDeLargura:
    conteudo = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    gravado = conteudo.pop("sha256", None)
    if gravado != _sha_do_conteudo(conteudo):
        raise ValueError(f"o hash do limiar não confere com o conteúdo de {arquivo}")
    return LimiarDeLargura(**{c: conteudo[c] for c in LimiarDeLargura.__dataclass_fields__})


def declaracao(limiar: LimiarDeLargura) -> str:
    return (f"Limiar de baixa confiança do pacote {limiar.papel} ({limiar.sha256_pacote[:12]}…): {limiar.valor:.4f} "
            f"em log (percentil {limiar.percentil:g} das larguras de {limiar.n} vídeos da LIM, sem rótulos). "
            "Sinaliza faixas mais largas que as demais e não garante cobertura nem desempenho.")


# --- leitura do disco -----------------------------------------------------------------------------------------------


def _por_id(arquivo: Path) -> dict[str, dict]:
    return {linha["video_id"]: linha for linha in pq.read_table(arquivo).to_pylist()}


def linhas_da_lim(ids: Sequence[str], *, pasta_base: Path, contexto: hs.ContextoTreino, defasagem_dias: int,
                  pasta_legenda_capa_lim: Path, carregar_anotacoes, codificador, cache_embeddings: Path) -> list[dict]:
    """Linhas de inferência da LIM com o mesmo código do desenvolvimento e sem colunas-alvo.

    `contexto` é o do pacote em questão (o fallback do setor só pode vir do treino dele). Anotação, legenda ou capa
    ausente de qualquer um dos 100 vídeos é erro: não existe linha descartada nem preenchida."""
    ids = list(ids)
    posts, canais = hs.carregar_base(pasta_base)
    alvos = [p for p in posts if p.video_id in set(ids)]
    if len(alvos) != len(ids):
        raise ErroLim(f"{len(ids) - len(alvos)} vídeos da LIM não estão na base curada")
    n1 = {l["video_id"]: l for l in hs.calcular(posts, canais, alvos, contexto, defasagem_dias=defasagem_dias)}
    anotacoes = carregar_anotacoes(ids)
    ausentes = [v for v in ids if v not in anotacoes]
    if ausentes:
        raise ErroLim(f"{len(ausentes)} vídeos da LIM sem anotação válida (ex.: {ausentes[:3]})")
    transcricoes = ds.embeddings_transcricao(
        {v: ds.texto_para_embedding(anotacoes[v]["anotacao"]) for v in ids}, codificador, Path(cache_embeddings))
    if any(r["estado"] == "falha_extrator" for r in transcricoes.values()):
        raise ErroLim("o codificador falhou em transcrições da LIM")
    gemini = {}
    for v in ids:
        linha = ds.linha_gemini(anotacoes[v]["anotacao"], anotacoes[v]["setor"])
        if linha[ds.COLUNA_ESTADO_EMBEDDING] is None:
            linha[ds.COLUNA_EMBEDDING_TRANSCRICAO] = transcricoes[v]["vetor"]
            linha[ds.COLUNA_ESTADO_EMBEDDING] = transcricoes[v]["estado"]
        gemini[v] = linha
    pasta = Path(pasta_legenda_capa_lim)
    return montar_linhas(ids, n1=n1, legenda=_por_id(pasta / "legenda.parquet"), capa=_por_id(pasta / "capa.parquet"),
                         gemini=gemini)


def _carregar_pacote(pastas: Mapping[str, Path]) -> tuple[dict, str]:
    modelos = {q: md.carregar(Path(pastas[q])) for q in QUANTIS}
    hashes = {q: json.loads((Path(pastas[q]) / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))["sha256_modelo"]
              for q in QUANTIS}
    return modelos, identidade_do_pacote(hashes)


def calcular_para_o_pacote(papel: str, pastas: Mapping[str, Path],
                           linhas_lim: Sequence[Mapping]) -> tuple[LimiarDeLargura, np.ndarray]:
    modelos, sha = _carregar_pacote(pastas)
    larguras = larguras_do_pacote(modelos["q10"], modelos["q50"], modelos["q90"], linhas_lim)
    return calcular_limiar(papel, sha, [l["video_id"] for l in linhas_lim], larguras), larguras


def main(argv: Sequence[str] | None = None) -> int:
    from src.extracao import texto as leg
    from src.config import RAIZ_PROJETO

    p = argparse.ArgumentParser(prog="python -m src.avaliacao.limiar", description=__doc__)
    p.add_argument("--papel", choices=PAPEIS, required=True)
    for q in QUANTIS:
        p.add_argument(f"--{q}", type=Path, required=True, help=f"pasta do modelo {q} do pacote")
    p.add_argument("--treino-ate-utc", required=True, help="corte de treino do pacote (ex.: 2026-07-27T18:49:55Z)")
    p.add_argument("--defasagem-dias", type=int, default=hs.DEFASAGEM_PADRAO)
    p.add_argument("--legenda-capa-lim", type=Path, required=True)
    p.add_argument("--saida", type=Path, required=True)
    a = p.parse_args(argv)
    snap = "snapshot=20261014T021437Z"
    lim_pasta = RAIZ_PROJETO / "data/annotations/gemini" / snap / "lim=v1"
    ids = sorted(x["video_id"] for x in json.loads((lim_pasta / "amostra.json").read_text(encoding="utf-8"))["lista"])
    corte = datetime.strptime(a.treino_ate_utc, FORMATO_UTC).replace(tzinfo=UTC)
    try:
        linhas = linhas_da_lim(
            ids, pasta_base=RAIZ_PROJETO / "data/processed" / snap / "curadoria=v1",
            contexto=hs.ContextoTreino(f"lim_{a.papel}", corte, frozenset()), defasagem_dias=a.defasagem_dias,
            pasta_legenda_capa_lim=a.legenda_capa_lim, carregar_anotacoes=ds._anotacoes_do_cache(lim_pasta),
            codificador=leg.CodificadorTexto(),
            cache_embeddings=RAIZ_PROJETO / "data/features" / snap / "cache_embeddings_transcricao")
        limiar, _ = calcular_para_o_pacote(a.papel, {q: getattr(a, q) for q in QUANTIS}, linhas)
        print(gravar_limiar(limiar, a.saida))
        print(declaracao(limiar))
    except (ErroLim, ValueError, FileNotFoundError, FileExistsError, md.ErroModelo) as erro:
        print(f"Recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
