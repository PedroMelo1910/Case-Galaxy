"""Módulo consolidado. Reúne:
- Modelo de exibição da resposta: converte `data` do envelope em textos para a tela, usando só o que a resposta traz.
- Desenho do chat: resumo do pedido e resposta em linguagem natural, só com o que a resposta traz."""

import html
from dataclasses import dataclass, field

import streamlit as st

from src.interface_web.adaptador import ORIGEM_SALVA
from src.interface_web.textos import decimal, escapar_markdown, exibivel, inteiro, percentual
from src.interface_web.validacao import mensagem_do_envelope, MensagemErro


# ──── Modelo de exibição da resposta: converte `data` do envelope em textos para a tela, usando só o que a
AVISO_SHAP = ("Os fatores abaixo descrevem como o modelo chegou ao número (contribuições SHAP, em escala logarítmica). "
              "Não são causas: mudar um deles não garante mudar as curtidas.")
AVISO_RETROSPECTIVO = ("A data planejada já passou: o vídeo já podia estar publicado. Isto não é uma previsão feita "
                       "antes da publicação.")
ROTA_FALLBACK_IMAGEM = "video_gemini_fallback_imagem"
ROTA_LOCAL = "video_local"
CODIGO_AVISO_COBERTURA = "faixa_sem_cobertura_suficiente"
CODIGO_AVISO_MATERIAL_PUBLICADO = "material_ja_publicado"
FONTES_DA_EXPLICACAO = {
    "template": "Texto de modelo fixo (determinístico, sem LLM), escrito a partir dos números do modelo.",
    "llm": "Escrita por um LLM (Gemini) a partir dos números do modelo; os números vêm do modelo, não do LLM.",
}


@dataclass(frozen=True)
class Fator:
    nome: str
    sentido: str  # "aumenta" | "reduz"
    contribuicao: float
    valor: str | None = None

    @property
    def frase(self) -> str:
        direcao = "puxa a previsão para cima" if self.sentido == "aumenta" else "puxa a previsão para baixo"
        valor = f" (valor {self.valor})" if self.valor else ""
        sinal = "+" if self.contribuicao >= 0 else "−"
        return f"{self.nome}{valor}: {direcao} ({sinal}{decimal(abs(self.contribuicao), 2)} em escala log)"


@dataclass(frozen=True)
class Confianca:
    estado: str  # "indisponivel" | "baixa" | "dentro_do_limiar"
    titulo: str
    texto: str


@dataclass(frozen=True)
class Aviso:
    classe: str  # "atencao" | "info"
    texto: str


@dataclass(frozen=True)
class ModeloResposta:
    origem: str
    curtidas_central: str
    faixa: str
    referencia: str
    comparacao: str
    fatores: tuple[Fator, ...]
    familias: tuple[Fator, ...]
    confianca: Confianca
    cobertura: str
    avisos: tuple[Aviso, ...]
    modelo: str
    rota: str
    explicacao: str
    fonte_explicacao: str
    retrospectivo: str | None
    historico: str | None
    extracao: tuple[str, ...] = field(default_factory=tuple)
    detalhe_cobertura: str | None = None
    alerta_rota: str | None = None
    extracao_resumo: str | None = None
    mensagem_simples: str = ""


def _comparacao(razao: float | None, fonte: str | None) -> str:
    if razao is None:
        return "Sem razão em relação à referência nesta resposta."
    alvo = "do canal" if fonte != "setor" else "do setor"
    diferenca = (razao - 1) * 100
    if abs(diferenca) < 0.5:
        return f"A previsão central é praticamente igual à referência {alvo}."
    sentido = "acima" if diferenca > 0 else "abaixo"
    return (f"A previsão central é {decimal(razao)} vez a referência {alvo} "
            f"({decimal(abs(diferenca), 0)}% {sentido}).")


def _referencia(d: dict) -> str:
    curtidas, fonte = d.get("reference_likes"), d.get("reference_source")
    if curtidas is None:
        return "Referência: não informada nesta resposta."
    quem = "do setor (o canal tem histórico curto)" if fonte == "setor" else "do canal"
    definicao = exibivel(d.get("reference_definition") or "").rstrip(". ")
    return f"Referência {quem}: {inteiro(curtidas)} curtidas" + (f" ({definicao})." if definicao else ".")


def _texto_do_valor(valor: object) -> str | None:
    if valor is None:
        return None
    return decimal(valor, 2) if isinstance(valor, float) else str(valor)


def _fatores(itens: list | None, chave_nome: str, chave_valor: str | None) -> tuple[Fator, ...]:
    resultado = []
    for i in itens or ():
        phi = i.get("phi", i.get("soma_phi"))
        if phi is None:
            continue
        valor = _texto_do_valor(i.get(chave_valor)) if chave_valor else None
        sentido = "aumenta" if i.get("sentido") == "aumenta" else "reduz"
        resultado.append(Fator(exibivel(i.get(chave_nome, "fator")), sentido, float(phi), valor))
    return tuple(resultado)


def _confianca(d: dict) -> Confianca:
    c = d.get("statistical_confidence")
    if not isinstance(c, dict) or not c.get("aplicavel") or c.get("estado") == "indisponivel":
        motivo = exibivel((c or {}).get("motivo") or "não há limiar de largura medido para este modelo")
        return Confianca("indisponivel", "Confiança insuficiente ou indisponível",
                         f"Nenhum sinal de confiança é afirmado: {motivo}.")
    largura, limiar = c.get("largura_log"), c.get("limiar_log")
    medida = (f" (largura {decimal(largura, 2)} contra limiar {decimal(limiar, 2)}, em escala log)"
              if largura is not None and limiar is not None else "")
    if c.get("sinalizada"):
        return Confianca("baixa", "Baixa confiança estatística",
                         f"A faixa é mais larga que o limiar de referência{medida}. Indica incerteza maior, não prova "
                         "erro maior. A regra não garante cobertura nem desempenho.")
    return Confianca("dentro_do_limiar", "Largura da faixa dentro do limiar",
                     f"A faixa não passa do limiar de referência{medida}. Isso não garante cobertura nem desempenho.")


def _cobertura(d: dict) -> tuple[str, str | None]:
    c = d.get("coverage_estimate")
    nominal = d.get("interval_nominal_level")
    if not isinstance(c, dict) or c.get("cobertura_observada") is None:
        return "Cobertura da faixa: não informada para este modelo.", None
    alvo = f"{percentual(c.get('nivel_nominal', nominal), 0)} pretendidos" if c.get("nivel_nominal", nominal) else "pretendido"
    onde = ("Medida no procedimento de avaliação, não neste modelo servido; " if not c.get("medida_no_artefato_servido")
            else "Medida neste modelo; ")
    abaixo = " (abaixo do pretendido)" if c.get("abaixo_do_pretendido") else ""
    texto = (f"Na avaliação, {percentual(c['cobertura_observada'])} dos vídeos ({inteiro(c.get('n_dentro'))} de "
             f"{inteiro(c.get('n'))}) ficaram dentro da faixa, contra {alvo}{abaixo}. {onde}sem garantia para vídeos novos.")
    return texto, exibivel(c["aviso"]) if c.get("aviso") else None


def _rota(d: dict) -> str:
    r = d.get("route")
    if not isinstance(r, dict):
        return "Rota: não informada nesta resposta (resposta anterior à indicação de rota)."
    partes = [exibivel(r.get("rotulo") or r.get("id") or "rota sem nome")]
    if r.get("offline") is not None:
        partes.append("offline" if r["offline"] else "usa rede")
    if r.get("usa_gemini") is not None:
        partes.append("usa Gemini" if r["usa_gemini"] else "sem Gemini")
    return "Rota: " + ", ".join(partes) + "."


def _modelo(d: dict) -> str:
    versao = d.get("model_version") or (d.get("route") or {}).get("versao_do_modelo")
    nivel = exibivel(d["model_level"]) if d.get("model_level") else None
    if nivel and versao:
        return f"Modelo: {nivel}. Versão {exibivel(versao)}."
    if nivel or versao:
        return f"Modelo: {nivel or 'versão ' + exibivel(versao)}."
    return "Modelo: não informado nesta resposta."


def _avisos(d: dict) -> tuple[Aviso, ...]:
    """Avisos da resposta. A cobertura tem texto próprio e o aviso de material já publicado vira a faixa de destaque."""
    omitidos = {CODIGO_AVISO_COBERTURA} | ({CODIGO_AVISO_MATERIAL_PUBLICADO} if d.get("retrospective") else set())
    return tuple(Aviso("atencao" if a.get("classe") == "atencao" else "info", exibivel(a.get("descricao", "")))
                 for a in d.get("warnings") or () if a.get("codigo") not in omitidos)


def _historico(d: dict) -> str | None:
    h = d.get("history")
    if not isinstance(h, dict):
        return None
    origens = {"historico_do_snapshot": "histórico da base curada", "historico_fornecido": "histórico enviado por você",
               "consulta_atual": "histórico consultado ao vivo no YouTube"}
    return f"Histórico usado: {origens.get(h.get('origem'), exibivel(h.get('origem', 'origem não informada')))}, " \
           f"{inteiro(h.get('n_posts'))} posts elegíveis."


def _extracao_local(d: dict) -> dict | None:
    e = d.get("extraction")
    if isinstance(e, dict) and (e.get("rota") == ROTA_LOCAL or (e.get("gemini") == "nao_aplicavel" and "etapas" in e)):
        return e
    return None


def _extracao_resumo(d: dict) -> str | None:
    """Frase discreta: a extração foi local (sem Gemini) e se veio do cache de extração."""
    e = _extracao_local(d)
    if e is None:
        return None
    do_cache, executadas = list(e.get("do_cache") or ()), list(e.get("executadas") or ())
    if do_cache and not executadas:
        origem = "resultado reaproveitado do cache de extração"
    elif do_cache:
        origem = f"{len(do_cache)} etapa(s) reaproveitadas do cache e {len(executadas)} executadas agora"
    else:
        origem = "extração executada agora"
    return f"Atributos do vídeo extraídos localmente neste computador, sem Gemini ({origem})."


def _extracao(d: dict) -> tuple[str, ...]:
    e = d.get("extraction")
    if not isinstance(e, dict):
        return ()
    linhas = []
    for chave, valor in e.items():
        if chave == "etapas" and isinstance(valor, dict):
            linhas += [f"etapa {exibivel(n)}: {exibivel(estado)}" for n, estado in valor.items()]
        elif chave == "sha256_video" and valor:
            linhas.append(f"SHA-256 do vídeo: {str(valor)[:16]}…")
        elif isinstance(valor, (str, int, float, bool)):
            linhas.append(f"{exibivel(chave)}: {exibivel(valor)}")
    return tuple(linhas)


def _alerta_da_rota(d: dict) -> str | None:
    """Vídeo que caiu no modelo de imagem NÃO é processamento completo de vídeo: destaque obrigatório."""
    rota = d.get("route")
    if not ((isinstance(rota, dict) and rota.get("id") == ROTA_FALLBACK_IMAGEM) or d.get("fallback")):
        return None
    motivo = (d.get("fallback") or {}).get("motivo") if isinstance(d.get("fallback"), dict) else None
    detalhe = f" Motivo: {exibivel(motivo)}." if motivo else ""
    return ("Atenção: esta resposta NÃO é processamento completo de vídeo. A previsão veio do modelo de imagem, "
            f"sem usar o conteúdo do vídeo, e tem confiança baixa.{detalhe}")


def _fonte_da_explicacao(d: dict) -> str:
    if d.get("explanation_label"):
        return exibivel(d["explanation_label"])
    fonte = d.get("explanation_source")
    if fonte in FONTES_DA_EXPLICACAO:
        return FONTES_DA_EXPLICACAO[fonte]
    return f"Fonte não identificada ({exibivel(fonte)})." if fonte else "Fonte da explicação não informada."


def _mensagem_simples(d: dict) -> str:
    """Resumo de apresentação a partir dos números calculados, sem modificar a resposta da API."""
    texto = f"Para este criativo, a estimativa é de **{inteiro(d['predicted_metric'])} curtidas**."
    referencia, razao = d.get("reference_likes"), d.get("relative_to_channel")
    if referencia is not None:
        quem = "setor" if d.get("reference_source") == "setor" else "canal"
        if isinstance(razao, (int, float)):
            posicao = "próximo da" if abs(razao - 1) < 0.005 else "acima da" if razao > 1 else "abaixo da"
            texto += f" Isso fica {posicao} referência do {quem}, de **{inteiro(referencia)} curtidas**."
        else:
            texto += f" A referência do {quem} é de **{inteiro(referencia)} curtidas**."
    if d.get("interval_low") is not None and d.get("interval_high") is not None:
        texto += f" A faixa estimada vai de **{inteiro(d['interval_low'])} a {inteiro(d['interval_high'])} curtidas**."
    estado = _confianca(d).estado
    if estado == "baixa":
        texto += " Como a faixa é ampla, há bastante incerteza: use o resultado como uma referência, não como garantia."
    elif estado == "indisponivel":
        texto += " A confiança desta estimativa está indisponível; o resultado não é uma garantia."
    else:
        texto += " O resultado é uma estimativa e pode variar."
    return texto


def _sem_cobertura(texto: str) -> str:
    """Cobertura estatística permanece na API e na documentação, fora da apresentação web."""
    return ". ".join(p for p in texto.split(". ") if "cobertura" not in p.casefold()).strip()


def construir_modelo(envelope: dict, origem: str) -> ModeloResposta | None:
    """Modelo de exibição de uma resposta de sucesso; `None` se o envelope não for de sucesso."""
    d = envelope.get("data") if isinstance(envelope, dict) and envelope.get("success") else None
    if not isinstance(d, dict) or d.get("predicted_metric") is None:
        return None
    cobertura, detalhe = _cobertura(d)
    nominal = d.get("interval_nominal_level")
    pretendido = f" (faixa de {percentual(nominal, 0)} pretendidos)" if nominal else ""
    return ModeloResposta(
        origem=origem, curtidas_central=inteiro(d["predicted_metric"]),
        faixa=f"de {inteiro(d.get('interval_low'))} a {inteiro(d.get('interval_high'))} curtidas{pretendido}",
        referencia=_referencia(d), comparacao=_comparacao(d.get("relative_to_channel"), d.get("reference_source")),
        fatores=_fatores(d.get("top_drivers"), "atributo", "valor"),
        familias=_fatores(d.get("family_contributions"), "nome", None),
        confianca=_confianca(d), cobertura=cobertura, avisos=_avisos(d), modelo=_modelo(d), rota=_rota(d),
        explicacao=exibivel(d.get("explanation") or "Sem texto de explicação nesta resposta."),
        fonte_explicacao=_fonte_da_explicacao(d),
        retrospectivo=AVISO_RETROSPECTIVO if d.get("retrospective") else None, historico=_historico(d),
        extracao=_extracao(d), detalhe_cobertura=detalhe, alerta_rota=_alerta_da_rota(d),
        extracao_resumo=_extracao_resumo(d), mensagem_simples=_mensagem_simples(d))


def selo_de_origem(origem: str) -> str:
    return "Resposta salva" if origem == ORIGEM_SALVA else "Nova previsão"


# ──── Desenho do chat: resumo do pedido e resposta em linguagem natural, só com o que a resposta traz.
@dataclass(frozen=True)
class Turno:
    """Uma pergunta e a resposta, guardadas na sessão."""

    resumo: tuple[str, ...]
    origem: str
    envelope: dict
    do_cache: bool = False
    nota: str | None = None


def _faixa(classe: str, texto: str, titulo: str | None = None) -> None:
    cabeca = f"<strong>{html.escape(titulo)}</strong><br>" if titulo else ""
    st.markdown(f'<div class="gx-faixa {classe}">{cabeca}{html.escape(texto)}</div>', unsafe_allow_html=True)


def desenhar_erro(erro: MensagemErro) -> None:
    _faixa("erro", erro.texto, erro.titulo)
    for item in erro.itens:
        st.markdown(f"- {escapar_markdown(item)}")
    for dica in erro.dicas:
        st.caption(dica)


def _selo(origem: str) -> None:
    classe = "salva" if origem == ORIGEM_SALVA else ""
    st.markdown(f'<span class="gx-selo {classe}">{html.escape(selo_de_origem(origem))}</span>', unsafe_allow_html=True)


def _lista(fatores: tuple[Fator, ...]) -> None:
    for f in fatores:
        st.markdown(f"- {escapar_markdown(f.frase)}")


def desenhar_resposta(m: ModeloResposta, nota: str | None = None) -> None:
    _selo(m.origem)
    if m.origem == ORIGEM_SALVA:
        st.caption("Reprodução de uma resposta gravada antes: nenhum modelo foi executado agora e isto não conta como "
                   "nova previsão.")
    if m.alerta_rota:
        _faixa("erro", m.alerta_rota)
    st.markdown(m.mensagem_simples or escapar_markdown(m.explicacao))
    if m.retrospectivo:
        _faixa("atencao", "A data escolhida já passou. Se quiser simular uma publicação futura, ajuste a data e a hora.")
    _bloco_fatores(m)
    if nota:
        st.caption(nota)
    with st.expander("Detalhes técnicos"):
        st.caption("Mensagem principal formatada a partir dos valores calculados pelo modelo.")
        st.markdown(escapar_markdown(f"{m.modelo} {m.rota}"))
        if m.historico:
            st.markdown(escapar_markdown(m.historico))
        confianca = _sem_cobertura(m.confianca.texto)
        if confianca:
            st.markdown(f"**{escapar_markdown(m.confianca.titulo)}:** {escapar_markdown(confianca)}")
        for aviso in m.avisos:
            if "cobertura" not in aviso.texto.casefold():
                st.caption(aviso.texto)
        if m.extracao_resumo:
            st.caption(m.extracao_resumo)
        for linha in m.extracao:
            st.markdown(f"- {escapar_markdown(linha)}")


def _bloco_fatores(m: ModeloResposta) -> None:
    if not (m.fatores or m.familias):
        return
    with st.expander("Principais fatores", expanded=False):
        st.caption(AVISO_SHAP)
        _lista(m.fatores)
        if m.familias:
            st.markdown("**Por grupo de informações**")
            _lista(m.familias)



def desenhar_turno(turno: Turno) -> None:
    with st.chat_message("user"):
        for linha in turno.resumo:
            st.markdown(escapar_markdown(linha))
    with st.chat_message("assistant"):
        modelo = construir_modelo(turno.envelope, turno.origem)
        if modelo is None:
            desenhar_erro(mensagem_do_envelope(turno.envelope))
            return
        if turno.do_cache:
            st.caption("Resultado reaproveitado: mesmo arquivo, mesmas entradas e mesmo modelo já analisados.")
        desenhar_resposta(modelo, turno.nota)
