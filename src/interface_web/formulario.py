"""Campos do formulário (widgets) e leitura para `EntradaFormulario`."""

import streamlit as st

from src.interface_web import validacao as ht
from src.interface_web.textos import escapar_markdown
from src.interface_web.validacao import (
    CanalBase,
    EntradaFormulario,
    EXTENSOES_VIDEO,
    MODO_BASE,
    MODO_HISTORICO,
    MODO_ONLINE,
    ROTA_AUTO,
    ROTA_GEMINI,
    ROTAS_DE_VIDEO,
    SETORES,
    TIPO_NAO_INFORMADO,
)


# ──── Campos do formulário (widgets) e leitura para `EntradaFormulario`.
ARQUIVO_CSV_MAX_BYTES = 1_000_000
_MODOS = {MODO_BASE: "Escolher um canal que já está na base",
          MODO_HISTORICO: "Informar os posts anteriores do canal",
          MODO_ONLINE: "Consultar o histórico ao vivo no YouTube (online, opcional)"}
_TIPOS = {"marca": "Marca", "criador": "Criador", TIPO_NAO_INFORMADO: "Não informar (o atributo fica ausente)"}


def _tipo_do_canal(desabilitado: bool, chave: str) -> str | None:
    return st.selectbox("Tipo do canal", list(_TIPOS), index=None, format_func=_TIPOS.get, key=chave,
                        placeholder="Marca, criador ou não informar", disabled=desabilitado,
                        help="Canais fora da base não têm tipo registrado. Escolha explicitamente.")


def _canal_da_base(canais: tuple[CanalBase, ...], setor: str | None, desabilitado: bool) -> str | None:
    opcoes = [c for c in canais if c.setor == setor]
    if setor is None:
        st.caption("Escolha o setor acima para ver os canais da base.")
    elif not opcoes:
        st.caption("Não há canais da base neste setor.")
    escolhido = st.selectbox("Canal da base", [c.channel_id for c in opcoes], index=None, key="f_canal_base",
                             format_func=lambda i: next(c.rotulo for c in opcoes if c.channel_id == i),
                             placeholder="Escolha o canal", disabled=desabilitado or not opcoes)
    if escolhido:
        canal = next(c for c in opcoes if c.channel_id == escolhido)
        st.caption(f"{escapar_markdown(canal.titulo)}: {canal.tipo_canal}, setor {canal.setor}. O histórico vem da base "
                   "curada coletada em 14/10/2026, não de uma consulta atual.")
    return escolhido


def _historico_enviado(desabilitado: bool) -> tuple[str, bool]:
    texto = st.text_area("Posts anteriores (CSV colado)", key="f_historico_texto", height=140, disabled=desabilitado,
                         placeholder=ht.MODELO_CSV, help="Colunas: published_at, likes, collected_at e, opcional, video_id.")
    arquivo = st.file_uploader("ou envie um arquivo CSV", type=["csv", "txt"], key="f_historico_arquivo",
                               disabled=desabilitado)
    st.caption("Datas em ISO 8601 (sem fuso, vale UTC). 'likes' vazio conta só para a cadência de publicação. "
               f"Até {ht.MAX_POSTS} posts.")
    if arquivo is None:
        return texto, False
    if arquivo.size > ARQUIVO_CSV_MAX_BYTES:
        st.error("O arquivo CSV passa de 1 MB; envie só os posts do canal.")
        return "", bool(texto.strip())
    return arquivo.getvalue().decode("utf-8-sig", errors="replace"), bool(texto.strip())


def _canal_ao_vivo(online_disponivel: bool, desabilitado: bool) -> str:
    if not online_disponivel:
        st.info("Desligado neste computador. É um caminho online opcional: para habilitar, defina CASEGALAXY_MCP_ONLINE=1 "
                "e CASEGALAXY_MCP_LIVRO e reinicie (ver docs/DOCUMENTACAO.md#integracao). Ele usa a cota da YouTube Data API.")
    else:
        st.warning("Caminho online: consulta o canal no YouTube agora e consome cota da YouTube Data API.")
    return st.text_input("ID do canal no YouTube (começa com UC)", key="f_canal_online", max_chars=24,
                         disabled=desabilitado or not online_disponivel)


def _contexto_do_canal(canais: tuple[CanalBase, ...], setor: str | None, online: bool, desabilitado: bool) -> dict:
    modos = list(_MODOS) if canais else [MODO_HISTORICO, MODO_ONLINE]
    modo = st.radio("Como informar o canal?", modos, index=None, format_func=_MODOS.get, key="f_modo_canal",
                    disabled=desabilitado, help="Nenhum canal é assumido: escolha uma das opções.")
    dados: dict = {"modo_canal": modo}
    if modo == MODO_BASE:
        dados["canal_base_id"] = _canal_da_base(canais, setor, desabilitado)
    elif modo == MODO_HISTORICO:
        dados["tipo_canal"] = _tipo_do_canal(desabilitado, "f_tipo_historico")
        dados["historico_texto"], dados["historico_conflito"] = _historico_enviado(desabilitado)
    elif modo == MODO_ONLINE:
        dados["tipo_canal"] = _tipo_do_canal(desabilitado, "f_tipo_online")
        dados["canal_online_id"] = _canal_ao_vivo(online, desabilitado)
    return dados


def _rota_do_video(online_disponivel: bool, desabilitado: bool) -> str:
    """Seletor da rota de previsão. A opção Gemini só existe com o modo online configurado (mesma regra do histórico ao vivo)."""
    opcoes = [r for r in ROTAS_DE_VIDEO if r != ROTA_GEMINI or online_disponivel]
    escolhida = st.radio("Rota do vídeo", opcoes, index=0, format_func=ROTAS_DE_VIDEO.get, key="f_rota_video",
                         disabled=desabilitado,
                         help="Automática usa o preditor local quando o vídeo é um arquivo. 'Preditor local' não cai "
                              "para o modelo de imagem: sem o preditor ativo, a análise é recusada.")
    if not online_disponivel:
        st.caption(f"{ROTAS_DE_VIDEO[ROTA_GEMINI]}: desligada neste computador (exige CASEGALAXY_MCP_ONLINE=1).")
    return escolhida or ROTA_AUTO


def ler_formulario(canais: tuple[CanalBase, ...], online_disponivel: bool, desabilitado: bool):
    """Desenha os campos e devolve `(EntradaFormulario, arquivo enviado ou None)`."""
    criativo, contexto = st.columns([1.05, 1], gap="large")
    with criativo:
        st.markdown('<div class="gx-section"><span>01</span><h3>Seu criativo</h3></div>', unsafe_allow_html=True)
        st.caption("O próximo vídeo da sua marca começa aqui.")
        st.markdown('''<div class="gx-video-note"><span class="gx-play" aria-hidden="true">▶</span>
            <strong>Um novo olhar para o seu vídeo</strong>
            <p>Envie um Short vertical ou quadrado.<br>Até 3 minutos · arquivo de até 100 MB</p></div>''',
                    unsafe_allow_html=True)
        video = st.file_uploader("Vídeo", type=list(EXTENSOES_VIDEO), key="f_video", disabled=desabilitado,
                                 help="Short de até 180 s, vertical ou quadrado, até 100 MB.")
        with st.expander("Título e legenda (opcionais)"):
            titulo = st.text_input("Título (opcional)", key="f_titulo", disabled=desabilitado)
            descricao = st.text_area("Legenda ou contexto (opcional)", key="f_descricao", height=90,
                                     disabled=desabilitado)
        st.markdown('''<div class="gx-expect"><strong>O que você vai receber</strong>
            <p>Estimativa de curtidas e faixa de previsão</p><p>Comparação com o histórico do canal</p>
            <p>Explicação dos principais fatores e limitações</p></div>''', unsafe_allow_html=True)
    with contexto:
        st.markdown('<div class="gx-section"><span>02</span><h3>Contexto da publicação</h3></div>',
                    unsafe_allow_html=True)
        st.caption("O mesmo vídeo pode ter resultados diferentes em cada canal.")
        setor = st.selectbox("Setor", SETORES, index=None, key="f_setor", placeholder="Escolha o setor",
                             disabled=desabilitado)
        col_data, col_hora = st.columns(2)
        data = col_data.date_input("Data planejada", value=None, format="DD/MM/YYYY", key="f_data",
                                   disabled=desabilitado)
        hora = col_hora.time_input("Hora (Brasília)", value=None, key="f_hora", disabled=desabilitado, step=900)
        st.markdown("**Contexto do canal**")
        canal = _contexto_do_canal(canais, setor, online_disponivel, desabilitado)
    with st.expander("Opções avançadas"):
        rota = _rota_do_video(online_disponivel, desabilitado)
        proprio = st.text_input("ID do próprio vídeo no YouTube, se já publicado", key="f_own_video", max_chars=11,
                                disabled=desabilitado,
                                help="Permite excluir o próprio vídeo do histórico, da referência e da cadência.")
    entrada = EntradaFormulario(
        arquivo_nome=video.name if video else None, arquivo_bytes=video.size if video else None, titulo=titulo,
        descricao=descricao, setor=setor, data=data, hora=hora, own_video_id=proprio, rota_video=rota,
        online_disponivel=online_disponivel, **canal)
    return entrada, video
