"""Declaração da faixa de previsão por contexto: validação de desenvolvimento ou teste final (protocolo §8, §14.9).

`faixa_previsao.declaracao_da_faixa` fala sempre em "validação de desenvolvimento" e não muda (módulo sob hash). Para
as avaliações novas, esta declaração diz de qual população vem a cobertura: a do desenvolvimento nunca se apresenta como
a do teste final e vice-versa. A cobertura descreve o procedimento, não o artefato servido (D-D7), e vem por modelo e por
setor, com o IC por canal quando estimável ou o motivo de não existir.
"""

from collections.abc import Mapping

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
