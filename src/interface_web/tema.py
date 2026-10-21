"""Tema visual da interface.

Cores e fontes vêm do site público da Galaxies (https://www.galaxieslabs.com/pt/), lidas dos estilos computados da
página inicial: fundo da página `#f9fafb`, superfícies brancas e `#ebecf0`, azul de ação `#0065ff`, botão escuro e
títulos `#111827`, texto `#374151`, texto secundário `#6b7280`, bordas `#d1d5db`/`#e5e7eb`, famílias DM Sans e Inter.
Os tons de estado (`#eb3131`, `#ffdc42`, `#4ef263`) são tokens de cor declarados na mesma página; aqui só marcam
bordas e faixas, nunca texto. O logotipo da empresa não é usado.

As fontes não são baixadas de servidores externos: a pilha usa DM Sans e Inter se estiverem instaladas e, senão, a
fonte padrão do sistema.
"""

FUNDO = "#f9fafb"
SUPERFICIE = "#ffffff"
SUPERFICIE_SUAVE = "#ebecf0"
PRIMARIA = "#0065ff"
TINTA = "#111827"
TEXTO = "#374151"
TEXTO_SUAVE = "#6b7280"
BORDA = "#d1d5db"
BORDA_SUAVE = "#e5e7eb"
ESTADO_ATENCAO = "#ffdc42"
ESTADO_ERRO = "#eb3131"
ESTADO_OK = "#4ef263"
FONTE = '"DM Sans", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif'
FONTE_STREAMLIT = "DM Sans, Inter, system-ui, Segoe UI, sans-serif"

AVISO_PROTOTIPO = ("Protótipo acadêmico do Inteli Academy. Não é um serviço da Galaxies nem tem endosso da empresa; "
                   "as cores apenas seguem a identidade visual pública dela.")


def opcoes_streamlit() -> dict[str, str]:
    """Opções `--chave valor` do Streamlit iguais às de `.streamlit/config.toml` (valem de qualquer pasta)."""
    return {
        "theme.base": "light", "theme.primaryColor": PRIMARIA, "theme.backgroundColor": FUNDO,
        "theme.secondaryBackgroundColor": SUPERFICIE_SUAVE, "theme.textColor": TINTA,
        "theme.font": FONTE_STREAMLIT, "theme.baseRadius": "12px", "theme.borderColor": BORDA,
    }


def css() -> str:
    """Estilo estático da página (sem dados do usuário). Foco visível, alvos de toque grandes, largura de leitura."""
    return f"""
<style>
:root {{ --gx-primaria: {PRIMARIA}; --gx-tinta: {TINTA}; --gx-texto: {TEXTO}; --gx-suave: {TEXTO_SUAVE};
  --gx-borda: {BORDA_SUAVE}; --gx-superficie: {SUPERFICIE}; --gx-fundo-suave: {SUPERFICIE_SUAVE}; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{ font-family: {FONTE}; background: {FUNDO}; color: {TEXTO}; }}
.stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea {{ font-family: {FONTE}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1160px; padding: 1.3rem 2rem 3rem; }}
h1, h2, h3 {{ color: {TINTA}; font-weight: 650; letter-spacing: -0.035em; }}
.gx-nav {{ display:flex; align-items:center; justify-content:space-between; padding: 0.35rem 0 1.35rem;
  border-bottom:1px solid {BORDA_SUAVE}; gap:1rem; }}
.gx-brand {{ display:flex; align-items:center; gap:0.65rem; font-size:1.16rem; font-weight:750; color:{TINTA}; }}
.gx-orbit {{ width:34px; height:34px; display:grid; place-items:center; color:{PRIMARIA}; font-size:2rem; }}
.gx-brand-sub {{ font-size:0.95rem; font-weight:400; color:{TEXTO_SUAVE}; margin-left:0.3rem; }}
.gx-academic {{ font-size:0.65rem; letter-spacing:0.12em; font-weight:650; color:{TEXTO};
  padding:0.45rem 0.65rem; border:1px solid {BORDA}; border-radius:6px; white-space:nowrap; }}
.gx-hero {{ padding:2.4rem 0 1.5rem; }}
.gx-eyebrow {{ color:{PRIMARIA}; font-size:0.68rem; font-weight:750; letter-spacing:0.14em; margin-bottom:0.8rem; }}
.gx-hero h1 {{ font-size:clamp(2.15rem, 4.1vw, 3.4rem); line-height:1.07; padding:0; margin:0 0 1rem;
  font-weight:700; letter-spacing:-0.055em; max-width:850px; }}
.gx-hero p {{ font-size:1rem; line-height:1.7; color:{TEXTO_SUAVE}; margin:0; }}
.gx-steps {{ display:flex; gap:2.2rem; margin:0 0 1.1rem; color:{TEXTO}; font-size:0.79rem; flex-wrap:wrap; }}
.gx-steps b {{ color:{PRIMARIA}; margin-right:0.45rem; font-size:0.7rem; }}
.st-key-gx_workspace {{ background:{SUPERFICIE}; border-radius:20px;
  border-color:{BORDA_SUAVE} !important; padding:1rem !important;
  border-radius:20px !important; box-shadow:0 12px 40px -28px #11182745; }}
.gx-section {{ display:flex; gap:0.65rem; align-items:center; margin:0.8rem 0 0.15rem; }}
.gx-section > span {{ color:{PRIMARIA}; background:#edf3ff; border-radius:8px; font-size:0.7rem; font-weight:700;
  padding:0.42rem 0.5rem; }}
.gx-section h3 {{ font-size:1.22rem; padding:0; margin:0; letter-spacing:-0.025em; }}
.gx-video-note {{ background:#f3f7ff; border:1px solid #dce7fc; border-radius:14px; text-align:center;
  padding:1.7rem 1rem 1.2rem; margin-top:0.3rem; color:{TINTA}; }}
.gx-play {{ display:grid; place-items:center; background:{SUPERFICIE}; width:48px; height:48px; border-radius:14px;
  margin:0 auto 0.8rem; color:{PRIMARIA}; box-shadow:0 4px 14px #0065ff10; font-size:1rem; }}
.gx-video-note strong {{ font-size:0.95rem; font-weight:650; }}
.gx-video-note p {{ font-size:0.79rem; line-height:1.7; color:{TEXTO_SUAVE}; margin:0.45rem 0 0; }}
.gx-expect {{ margin:0.5rem 0; padding:0.6rem 0; color:{TEXTO}; }}
.gx-expect strong {{ font-size:0.82rem; font-weight:650; }}
.gx-expect p {{ font-size:0.78rem; margin:0.5rem 0; line-height:1.5; }}
.gx-expect p::before {{ content:'↗'; color:{PRIMARIA}; margin-right:0.65rem; }}
[data-testid="stWidgetLabel"] p {{ font-size:0.8rem; font-weight:550; color:{TEXTO}; }}
[data-testid="stCaptionContainer"] {{ opacity:1; }}
[data-testid="stCaptionContainer"] p {{ font-size:0.75rem; line-height:1.55; color:{TEXTO_SUAVE}; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{ background:#fafbfd;
  border-color:{BORDA_SUAVE}; border-radius:9px; }}
[data-testid="stFileUploaderDropzone"] {{ background:{SUPERFICIE}; border:1px dashed #bccce8;
  border-radius:11px; padding:0.85rem; }}
[data-testid="stFileUploaderDropzone"] small {{ font-size:0.68rem; }}
[data-testid="stExpander"] {{ border-color:{BORDA_SUAVE}; border-radius:11px; box-shadow:none; }}
[data-testid="stExpander"] summary p {{ font-size:0.8rem; }}
.st-key-f_modo_uso [role="radiogroup"] {{ padding-bottom:0.8rem; border-bottom:1px solid {BORDA_SUAVE}; }}
.st-key-f_modo_uso [data-testid="stWidgetLabel"] {{ margin-bottom:0.4rem; }}
.st-key-f_modo_canal [role="radiogroup"] {{ gap:0.45rem; }}
.st-key-f_modo_canal label p {{ font-size:0.79rem; line-height:1.5; }}
.gx-footer {{ color:{TEXTO_SUAVE}; font-size:0.7rem; line-height:1.65; max-width:800px;
  margin:1.2rem auto 0; padding-top:1rem; text-align:center; }}
.gx-results-heading {{ color:{TEXTO_SUAVE}; font-size:0.7rem; font-weight:650; letter-spacing:0.12em;
  margin-top:1.5rem; padding-top:1.4rem; border-top:1px solid {BORDA_SUAVE}; }}
.gx-metric {{ padding:1.2rem 0 0.9rem; }}
.gx-aviso {{ font-size: 0.82rem; color: {TEXTO_SUAVE}; border-left: 3px solid {BORDA}; padding: 0.2rem 0 0.2rem 0.75rem;
  margin: 0.25rem 0 1.25rem 0; }}
.gx-selo {{ display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  color: #fff; background: {PRIMARIA}; border-radius: 999px; padding: 0.15rem 0.65rem; }}
.gx-selo.salva {{ background: {TINTA}; }}
.gx-numero {{ font-size: 3.6rem; font-weight: 650; color: {TINTA}; line-height: 1.1; letter-spacing:-0.055em; }}
.gx-legenda {{ font-size: 0.85rem; color: {TEXTO_SUAVE}; }}
.gx-faixa {{ border-left: 4px solid {PRIMARIA}; background: {SUPERFICIE}; padding: 0.6rem 0.9rem; border-radius: 0 10px 10px 0;
  margin: 0.4rem 0; color: {TINTA}; }}
.gx-faixa.atencao {{ border-left-color: {ESTADO_ATENCAO}; }}
.gx-faixa.erro {{ border-left-color: {ESTADO_ERRO}; }}
.gx-faixa.ok {{ border-left-color: {ESTADO_OK}; }}
[data-testid="stChatMessageAvatarAssistant"] {{ background: {PRIMARIA}; }}
[data-testid="stChatMessageAvatarUser"] {{ background: {TINTA}; }}
[data-testid="stChatMessage"] {{ background: {SUPERFICIE}; border: 1px solid {BORDA_SUAVE}; border-radius: 16px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {SUPERFICIE}; }}
.stButton > button, .stDownloadButton > button {{ min-height: 2.75rem; border-radius: 10px; font-weight: 600; }}
.stButton > button[kind="primary"] {{ background: {PRIMARIA}; border-color: {PRIMARIA}; color: #fff;
  min-height:3.2rem; box-shadow:0 5px 12px #0065ff18; transition:background 150ms, box-shadow 150ms; }}
.stButton > button[kind="primary"]:hover {{ background:#0057db; border-color:#0057db; box-shadow:0 7px 18px #0065ff25; }}
.stButton > button[kind="primary"]:disabled {{ background: {BORDA}; border-color: {BORDA}; color: {TEXTO}; cursor: not-allowed; }}
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="radio"]:focus-visible,
[data-baseweb="select"] *:focus-visible, a:focus-visible {{ outline: 3px solid {PRIMARIA} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ .stButton > button {{ transition:none !important; }} }}
@media (max-width: 640px) {{ .block-container {{ padding:0.7rem 1rem 2rem; }}
  .gx-nav {{ gap:0.5rem; padding-bottom:1rem; }} .gx-brand {{ font-size:1rem; gap:0.35rem; }}
  .gx-brand-sub {{ display:none; }} .gx-academic {{ font-size:0.55rem; padding:0.35rem 0.45rem; }}
  .gx-hero {{ padding:1.6rem 0 1.1rem; }} .gx-hero p {{ font-size:0.9rem; }}
  .gx-eyebrow {{ font-size:0.59rem; letter-spacing:0.09em; }} .gx-desktop-break {{ display:none; }}
  .gx-steps {{ gap:0.5rem 1rem; font-size:0.7rem; }}
  .st-key-gx_workspace {{ padding:0.3rem !important; }}
  .gx-numero {{ font-size:2.7rem; }} }}
</style>
"""
