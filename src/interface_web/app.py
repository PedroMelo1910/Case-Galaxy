"""Página única em estilo de chat. Só desenha; as regras ficam nos módulos vizinhos.

Nada é executado antes do clique explícito em "Analisar". O botão e os campos ficam desabilitados durante a execução.
`executar` recebe as fábricas de adaptador para que os testes e verificações troquem o preditor sem mudar a tela.
"""

import sys
from collections.abc import Callable
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:  # Streamlit só põe a pasta do script no caminho
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.agente.contratos import Codigo, ErroAgente, envelope_erro  # noqa: E402
from src.interface_web import analise, formulario, analise as recursos, tema  # noqa: E402
from src.interface_web.adaptador import ORIGEM_NOVA, ORIGEM_SALVA, Adaptador, modo_online_ligado  # noqa: E402
from src.interface_web.estado import ControleEnvio  # noqa: E402
from src.interface_web.validacao import EntradaFormulario, validar  # noqa: E402
from src.interface_web.resultados import Turno, desenhar_turno  # noqa: E402

MODO_NOVA = "nova"
MODO_SALVA = "salva"
MAX_TURNOS = 5
_ROTULOS_DE_MODO = {MODO_NOVA: "Nova previsão", MODO_SALVA: "Reprodução de resposta salva"}


def _cabecalho() -> None:
    st.markdown('''<div class="gx-nav"><div class="gx-brand"><span class="gx-orbit" aria-hidden="true">✳</span>
        <span>Galaxies<span class="gx-brand-sub"> / Creative Lab</span></span></div>
        <span class="gx-academic">PROJETO ACADÊMICO</span></div>
        <div class="gx-hero"><div class="gx-eyebrow">INTELIGÊNCIA PARA O SEU PRÓXIMO SHORT</div>
        <h1>Antes de publicar,<br>explore o potencial.</h1>
        <p>Transforme um vídeo em uma previsão de curtidas, com contexto,<br class="gx-desktop-break">
        uma faixa de resultados e os fatores que pesaram na análise.</p></div>
        <div class="gx-steps"><span><b>01</b> Envie seu vídeo</span><span><b>02</b> Informe o contexto</span>
        <span><b>03</b> Explore a previsão</span></div>''', unsafe_allow_html=True)


def _resumo_da_entrada(e: EntradaFormulario, canais: dict) -> tuple[str, ...]:
    tamanho = f"{(e.arquivo_bytes or 0) / 1e6:.1f} MB".replace(".", ",")
    canal = canais.get(e.canal_base_id or "")
    contexto = {"base": f"canal da base: {canal.titulo}" if canal else "canal da base",
                "historico": "histórico enviado por você", "online": "histórico ao vivo no YouTube"}.get(e.modo_canal, "")
    quando = e.data.strftime("%d/%m/%Y") + " " + e.hora.strftime("%H:%M") if e.data and e.hora else ""
    return (f"Vídeo: {e.arquivo_nome} ({tamanho})", f"Setor: {e.setor}. Publicação planejada: {quando} (Brasília).",
            f"Contexto do canal: {contexto}.")


def _adicionar(turno: Turno) -> None:
    turnos = [turno, *st.session_state.get("turnos", [])]
    st.session_state["turnos"] = turnos[:MAX_TURNOS]


def _executar_nova(entrada: EntradaFormulario, video, canais: dict, adaptador: Adaptador | None) -> None:
    problemas = validar(entrada, canais)
    if problemas:
        _adicionar(Turno(("Tentativa de análise com o formulário incompleto.",), ORIGEM_NOVA,
                         analise.envelope_de_formulario_invalido(problemas)))
        return
    if adaptador is None:
        _adicionar(Turno(_resumo_da_entrada(entrada, canais), ORIGEM_NOVA, envelope_erro(ErroAgente(
            Codigo.MODELO_INDISPONIVEL, "O preditor local não está disponível neste computador."))))
        return
    with st.status("Analisando o vídeo…", expanded=True) as andamento:
        resultado = analise.analisar(entrada, canais, video.getvalue(), adaptador, recursos.cache_de_resultados(),
                                     ao_progresso=andamento.write)
        sucesso = bool(resultado.envelope.get("success"))
        andamento.update(label="Análise concluída" if sucesso else "A análise não foi concluída",
                         state="complete" if sucesso else "error", expanded=False)
    _adicionar(Turno(_resumo_da_entrada(entrada, canais), resultado.origem, resultado.envelope, resultado.do_cache))


def _executar_salva(item, adaptador) -> None:
    resultado = analise.reproduzir(item.url, adaptador)
    meta = adaptador.metadados()
    nota = f"Resposta gravada em {meta.get('gerado_em_utc')} ({meta.get('rotulo')}). {meta.get('aviso') or ''}".strip()
    _adicionar(Turno((f"Reproduzir resposta salva: {item.canal_titulo} ({item.setor}), vídeo {item.video_id}",),
                     ORIGEM_SALVA, resultado.envelope, False, nota))


def _escolher_salva(salvo, desabilitado: bool):
    itens = salvo.itens() if salvo else ()
    if not itens:
        st.info("Não há respostas salvas instaladas neste computador.")
        return None
    return st.selectbox("Resposta salva", itens, index=None, key="f_salva", disabled=desabilitado,
                        format_func=lambda i: f"{i.canal_titulo} · {i.setor} · {i.video_id}",
                        placeholder="Escolha uma resposta gravada")


def executar(fabrica_nova: Callable[[], Adaptador | None] | None = None,
             fabrica_salva: Callable[[], object | None] | None = None, online: bool | None = None) -> None:
    st.set_page_config(page_title="Creative Lab · Previsão de curtidas", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(tema.css(), unsafe_allow_html=True)
    recursos.preparar_diretorio_temporario()
    controle = ControleEnvio(st.session_state)
    _cabecalho()
    canais_base, motivo_canais = recursos.canais_da_base()
    canais = {c.channel_id: c for c in canais_base}
    online_ligado = modo_online_ligado() if online is None else online
    disponivel, motivo = (True, "") if fabrica_nova else recursos.estado_do_preditor()
    salvo = (fabrica_salva or recursos.adaptador_salvo)()
    with st.container(border=True, key="gx_workspace"):
        padrao = 0 if disponivel else 1
        modo = st.radio("O que você quer fazer?", [MODO_NOVA, MODO_SALVA], index=padrao, horizontal=True,
                        format_func=_ROTULOS_DE_MODO.get, key="f_modo_uso", disabled=controle.executando)
        if not disponivel and modo == MODO_NOVA:
            st.warning(f"{motivo} Rode 'uv run casegalaxy-demo --verificar-ambiente' para ver o que falta, ou use a "
                       "reprodução de resposta salva.")
        if motivo_canais and modo == MODO_NOVA:
            st.caption(motivo_canais)
        entrada = video = item = None
        if modo == MODO_NOVA:
            entrada, video = formulario.ler_formulario(canais_base, online_ligado, controle.executando)
        else:
            item = _escolher_salva(salvo, controle.executando)
        rotulo = "Analisar" if modo == MODO_NOVA else "Reproduzir resposta salva"
        pode = modo == MODO_NOVA or item is not None
        st.button(rotulo, type="primary", key="f_botao", on_click=controle.solicitar,
                  disabled=controle.executando or not pode, use_container_width=True)
        st.caption("A análise começa só quando você clicar. A previsão é uma estimativa, não uma garantia de desempenho.")
    if controle.executando and controle.consumir_solicitacao():
        try:
            if modo == MODO_NOVA:
                adaptador = (fabrica_nova or recursos.adaptador_real)() if disponivel else None
                _executar_nova(entrada, video, canais, adaptador)
            else:
                _executar_salva(item, salvo)
        finally:
            controle.concluir()
        st.rerun()
    elif controle.executando:
        controle.concluir()
    if st.session_state.get("turnos"):
        st.markdown('<div class="gx-results-heading">SUAS ANÁLISES</div>', unsafe_allow_html=True)
    for turno in st.session_state.get("turnos", []):
        desenhar_turno(turno)
    st.markdown(f'<footer class="gx-footer">{tema.AVISO_PROTOTIPO}</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    executar()
