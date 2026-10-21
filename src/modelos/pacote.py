"""Módulo consolidado. Reúne:
- Pacote de modelos servido pelo agente: q10, q50 e q90 verificados, cobertura observada e metadados.
- Manifesto versionado do pacote ATIVO e reversão (rollback) para a versão servida anterior.
- Previsão central (q50), faixa q10–q90 e contribuições do q50 para uma linha de inferência."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

from src.agente.contratos import Codigo, ErroAgente
from src.avaliacao import limiar as dl
from src.extracao.rubrica import gravar_json_atomico
from src.modelos import refit as rt, treino as fp, treino as md
from src.proveniencia import iso_utc as _iso, sha256_arquivo as _sha256


# ──── Pacote de modelos servido pelo agente: q10, q50 e q90 verificados, cobertura observada e metadados.
MODALIDADE_COM_MODELO = "video"
ROTA_VIDEO_LOCAL = "video_local"  # chave do pacote local no manifesto e valor de `rota` no `pacote.json`
NIVEL_VIDEO = "completo (desenvolvimento, provisório)"
LIMITACAO_MODALIDADE = (
    "Não há pacote de modelo configurado para {modalidade} neste serviço. O vídeo usa o modelo completo servido; "
    "imagem e texto usam pacotes próprios, carregados da pasta de modelos quando existem. Confira se o pacote de "
    "{modalidade} foi treinado e está na pasta esperada."
)


@dataclass(frozen=True)
class Pacote:
    q10: md.ModeloTreinado
    q50: md.ModeloTreinado
    q90: md.ModeloTreinado
    pasta_q50: Path
    pasta_faixa: Path
    declaracao_faixa: dict
    metadados_q50: dict
    resultado_faixa: dict
    registro: Mapping | None = None  # `pacote.json` do pacote final; None nos pacotes de desenvolvimento (rollback)
    limiar: dl.LimiarDeLargura | None = None  # só no vídeo final, vinculado à identidade dos três modelos
    identidade_sha256: str | None = None
    detalhe_do_limiar: Mapping | None = None  # origem e leitura do diagnóstico de desenvolvimento (do manifesto)

    @property
    def eh_final(self) -> bool:
        return self.registro is not None

    @property
    def eh_local(self) -> bool:
        """Preditor local de vídeo: atributos audiovisuais extraídos no próprio equipamento, sem Gemini."""
        return self.registro is not None and self.registro.get("rota") == ROTA_VIDEO_LOCAL

    @property
    def versao(self) -> str:
        if self.registro is not None:
            return self.registro["versao_servida"]
        return f"{self.pasta_q50.parent.name}/{self.pasta_q50.name}"

    @property
    def modalidade(self) -> str:
        """`video` para o pacote servido (sem modalidade gravada); texto ou imagem nos pacotes reduzidos."""
        if self.eh_local:
            return MODALIDADE_COM_MODELO
        return self.metadados_q50.get("modalidade", MODALIDADE_COM_MODELO)

    @property
    def nivel(self) -> str:
        if self.registro is not None:
            return self.registro["nivel_do_modelo"]
        return NIVEL_VIDEO if self.modalidade == MODALIDADE_COM_MODELO else self.metadados_q50["nivel_do_modelo"]

    @property
    def identificacao(self) -> dict:
        extra = ({"identidade_sha256": self.identidade_sha256, "versao_servida": self.registro["versao_servida"],
                  "refit_final": True, "limiar_vinculado": self.limiar is not None}
                 if self.registro is not None else {})
        if self.eh_local:
            extra |= {"rota": ROTA_VIDEO_LOCAL, "usa_gemini": False,
                      "extrator_identidade_sha256": identidade_do_extrator(self.registro),
                      "avaliacao": self.registro.get("avaliacao"),
                      **{c: self.registro[c] for c in ("superou_m0_em_erro", "superou_media_historica_em_erro",
                                                       "ordenacao_demonstrada") if c in self.registro}}
        return extra | {
            "q50_artefato": self.pasta_q50.name, "faixa_artefato": self.pasta_faixa.name,
            "sha256_q50": self.metadados_q50["sha256_modelo"], "parametros": self.metadados_q50["parametros"],
            "n_treino": self.metadados_q50["populacao"]["n_treino"],
            "n_validacao": self.metadados_q50["populacao"]["n_validacao"],
            "dataset_sha256": self.metadados_q50["dataset"]["sha256"],
        }

    def modelo_para(self, modalidade: str) -> "Pacote":
        if modalidade != MODALIDADE_COM_MODELO:
            raise ErroAgente(Codigo.MODELO_INDISPONIVEL, LIMITACAO_MODALIDADE.format(modalidade=modalidade),
                             {"modalidade": modalidade, "modalidade_com_modelo": MODALIDADE_COM_MODELO})
        return self


def _json(arquivo: Path) -> dict:
    return json.loads(arquivo.read_text(encoding="utf-8"))


def carregar_pacote(pasta_q50: Path, pasta_faixa: Path) -> Pacote:
    """Carrega os três modelos (hash e versões conferidos por `md.carregar`) e confere o vínculo entre eles."""
    pasta_q50, pasta_faixa = Path(pasta_q50), Path(pasta_faixa)
    meta_q50 = _json(pasta_q50 / md.ARQUIVO_METADADOS)
    resultado = _json(pasta_faixa / "resultado.json")
    if not resultado.get("recarga_identica"):
        raise md.ErroModelo("a faixa não tem recarga idêntica verificada")
    modelos = {"q50": md.carregar(pasta_q50)}
    for nome in ("q10", "q90"):
        meta = _json(pasta_faixa / nome / md.ARQUIVO_METADADOS)
        if meta.get("q50_sha256_modelo") != meta_q50["sha256_modelo"]:
            raise md.ErroModelo(f"{nome} não foi treinado a partir do q50 informado")
        if meta.get("dataset_sha256") != meta_q50["dataset"]["sha256"]:
            raise md.ErroModelo(f"{nome} usa outro dataset que o q50")
        modelos[nome] = md.carregar(pasta_faixa / nome)
    espec = modelos["q50"].especificacao
    for nome in ("q10", "q90"):
        outra = modelos[nome].especificacao
        if (list(outra.escalares) != list(espec.escalares) or list(outra.categoricas) != list(espec.categoricas)
                or dict(outra.embeddings) != dict(espec.embeddings)):
            raise md.ErroModelo(f"{nome} usa lista de atributos diferente da do q50")
    faixa = resultado["faixa"]
    n = int(faixa["n"])
    n_dentro = round(float(faixa["cobertura"]) * n)
    return Pacote(modelos["q10"], modelos["q50"], modelos["q90"], pasta_q50, pasta_faixa,
                  fp.declaracao_da_faixa(n_dentro=n_dentro, n=n), meta_q50, resultado)


def _modelos_conferidos(pasta: Path, registro: Mapping) -> tuple[dict, str]:
    """Os três modelos do pacote final (hash por `md.carregar`), a identidade conferida e a mesma lista de atributos."""
    modelos = {n: md.carregar(pasta / n) for n in ("q10", "q50", "q90")}
    identidade = dl.identidade_do_pacote({n: registro["modelos"][n]["sha256_modelo"] for n in modelos})
    if identidade != registro["identidade_sha256"]:
        raise md.ErroModelo("a identidade do pacote não confere com os hashes dos três modelos")
    espec = modelos["q50"].especificacao
    for nome in ("q10", "q90"):
        outra = modelos[nome].especificacao
        if (list(outra.escalares) != list(espec.escalares) or list(outra.categoricas) != list(espec.categoricas)
                or dict(outra.embeddings) != dict(espec.embeddings)):
            raise md.ErroModelo(f"{nome} usa lista de atributos diferente da do q50")
    return modelos, identidade


def carregar_pacote_final(pasta: Path, *, entrada_do_manifesto: Mapping | None = None) -> Pacote:
    """Pacote do refit final (`final=v1/<id>`): modelos conferidos por hash, identidade dos três e limiar vinculado.

    O limiar só é aceito se o arquivo for válido (hash do conteúdo), de papel `servido` e vinculado à identidade DESTE
    pacote (`LimiarIncompativel` caso contrário). A declaração de cobertura é a do procedimento de avaliação gravada no
    `pacote.json`, nunca uma medida do artefato."""
    from src.modelos import refit as rc

    pasta = Path(pasta)
    registro = _json(pasta / "pacote.json")
    modelos, identidade = _modelos_conferidos(pasta, registro)
    declaracao = registro["cobertura_do_procedimento"]
    rc.conferir(declaracao)
    limiar = _limiar_vinculado(pasta, identidade, (entrada_do_manifesto or {}).get("limiar"))
    detalhe = ((entrada_do_manifesto or {}).get("limiar") or {}).get("diagnostico_de_desenvolvimento")
    meta_q50 = _json(pasta / "q50" / md.ARQUIVO_METADADOS)
    return Pacote(modelos["q10"], modelos["q50"], modelos["q90"], pasta / "q50", pasta, declaracao, meta_q50,
                  {"recarga_identica": True, "cobertura_do_procedimento": declaracao}, registro=registro,
                  limiar=limiar, identidade_sha256=identidade, detalhe_do_limiar=detalhe)


# --- preditor local de vídeo ---------------------------------------------------------------------------------------

COBERTURA_NAO_MEDIDA = (
    "A cobertura da faixa q10–q90 não foi medida para este artefato: nenhum número de cobertura de outro modelo é "
    "usado aqui. A faixa vai de q10 a q90 com nível pretendido de {nivel:.0%}, sem garantia de cobertura.")


def declaracao_local(registro: Mapping) -> dict:
    """Declaração da faixa do preditor local.

    Usa `cobertura_do_procedimento` do `pacote.json` (a avaliação exploratória gravada para ESTE artefato; deve afirmar que
    não foi medida no artefato servido nem garante cobertura). Aceita também `avaliacao.cobertura = {n, n_dentro}`. Sem
    nenhuma das duas, a cobertura é declarada não medida; a de outro pacote nunca é usada."""
    declarada = registro.get("cobertura_do_procedimento")
    if declarada:
        from src.modelos import refit as rc

        rc.conferir(declarada)
        return dict(declarada)
    avaliacao = registro.get("avaliacao") or {}
    medida = avaliacao.get("cobertura")
    rotulo = avaliacao.get("rotulo") or "avaliação do artefato"
    base = {"nome": "faixa de previsão q10–q90", "nivel_nominal": fp.COBERTURA_NOMINAL, "origem": "avaliacao_do_artefato",
            "medida_no_artefato_servido": medida is not None, "garantia_de_cobertura": False}
    if not medida:
        return base | {"cobertura_observada": None, "n_dentro": None, "n": None, "abaixo_do_pretendido": None,
                       "origem": "nao_medida", "aviso": COBERTURA_NAO_MEDIDA.format(nivel=fp.COBERTURA_NOMINAL)}
    n, n_dentro = int(medida["n"]), int(medida["n_dentro"])
    if not 0 <= n_dentro <= n or n <= 0:
        raise md.ErroModelo("avaliacao.cobertura do pacote local é inconsistente")
    declarada = fp.declaracao_da_faixa(n_dentro=n_dentro, n=n)
    aviso = (f"{declarada['aviso']} Medida sobre {n} vídeos ({rotulo}); não é desempenho em vídeos novos nem "
             "garantia de cobertura.")
    return declarada | base | {"aviso": aviso, "rotulo_da_avaliacao": rotulo}


def _limiar_local(registro: Mapping, identidade: str) -> dl.LimiarDeLargura | None:
    """Limiar de largura gravado no `pacote.json` (`limiar_largura = {valor, n_lim, id_artefato}`).

    Ausente ou sem valor: `None` (confiança indisponível). Vinculado a outra identidade: `LimiarIncompativel`."""
    bruto = registro.get("limiar_largura")
    if not bruto or bruto.get("valor") is None or bruto.get("estado") == "indisponivel":
        return None
    valor, n_lim = float(bruto["valor"]), int(bruto["n_lim"])
    if not math.isfinite(valor) or valor < 0 or n_lim <= 0:
        raise md.ErroModelo("limiar_largura do pacote local é inválido")
    limiar = dl.LimiarDeLargura("servido", str(bruto["id_artefato"]), str(bruto.get("ids_sha256") or ""), n_lim,
                                float(bruto.get("percentil", 80.0)), valor)
    dl._exigir_compativel(limiar, identidade)
    return limiar


HEX64 = frozenset("0123456789abcdef")


def identidade_do_extrator(registro: Mapping) -> str | None:
    """SHA-256 da configuração do extrator que gerou os atributos: `extrator.identidade_sha256`, ou `extrator.config_id`
    quando este é o próprio SHA-256 (64 hexadecimais). Sem nenhum dos dois, `None`."""
    extrator = registro.get("extrator") or {}
    for chave in ("identidade_sha256", "config_id"):
        valor = extrator.get(chave)
        if isinstance(valor, str) and len(valor) == 64 and set(valor) <= HEX64:
            return valor
    return None


def _conferir_atributos_locais(registro: Mapping, espec: md.EspecificacaoAtributos, nomes_saida: list[str]) -> None:
    colunas = [*espec.escalares, *espec.categoricas, *espec.embeddings]
    if any(c.startswith("gem_") for c in colunas):
        raise md.ErroModelo("o preditor local não pode usar colunas gem_*: ele roda sem Gemini")
    ordem = registro.get("ordem_atributos")
    if isinstance(ordem, Mapping):  # {entrada_candidata, entrada_efetiva, ..., saida_do_preprocessador}
        if ordem.get("saida_do_preprocessador") is not None and list(ordem["saida_do_preprocessador"]) != list(nomes_saida):
            raise md.ErroModelo("ordem_atributos do pacote local difere da saída do pré-processador dos modelos")
        if ordem.get("entrada_candidata") is not None and list(ordem["entrada_candidata"]) != colunas:
            raise md.ErroModelo("ordem_atributos do pacote local difere da especificação dos modelos")
    elif ordem is not None and list(ordem) != colunas and list(ordem) != list(nomes_saida):
        raise md.ErroModelo("ordem_atributos do pacote local difere da especificação dos modelos")
    if not registro.get("familias"):
        raise md.ErroModelo("o pacote local não declara as famílias de atributos")


def carregar_pacote_video_local(pasta: Path, *, entrada_do_manifesto: Mapping | None = None) -> Pacote:
    """Pacote do preditor local de vídeo (`final=v2/video_local-...`): mesma estrutura do pacote final de vídeo.

    Exige `rota = "video_local"`, `usa_gemini = false`, identidade do extrator e nenhuma coluna `gem_*`. A cobertura e o
    limiar são sempre os DESTE artefato (ou ausentes), nunca os de outro pacote."""
    pasta = Path(pasta)
    registro = _json(pasta / "pacote.json")
    if registro.get("rota") != ROTA_VIDEO_LOCAL or registro.get("usa_gemini") is not False:
        raise md.ErroModelo("o pacote não é um preditor local de vídeo sem Gemini")
    if identidade_do_extrator(registro) is None:
        raise md.ErroModelo("o pacote local não registra a identidade (SHA-256) do extrator")
    modelos, identidade = _modelos_conferidos(pasta, registro)
    q50 = modelos["q50"]
    _conferir_atributos_locais(registro, q50.especificacao, q50.nomes_atributos)
    limiar = _limiar_local(registro, identidade)
    if limiar is None and (entrada_do_manifesto or {}).get("limiar"):
        limiar = _limiar_vinculado(pasta, identidade, entrada_do_manifesto["limiar"])
    declaracao = declaracao_local(registro)
    meta_q50 = _json(pasta / "q50" / md.ARQUIVO_METADADOS)
    return Pacote(modelos["q10"], q50, modelos["q90"], pasta / "q50", pasta, declaracao, meta_q50,
                  {"recarga_identica": True, "cobertura": declaracao}, registro=registro, limiar=limiar,
                  identidade_sha256=identidade)


def _limiar_vinculado(pasta: Path, identidade: str, registrado: Mapping | None) -> dl.LimiarDeLargura | None:
    if registrado is None:
        return None
    limiar = dl.ler_limiar(pasta / registrado["arquivo"])
    if limiar.papel != "servido":
        raise dl.LimiarIncompativel(f"o limiar do pacote servido deve ter papel 'servido', não {limiar.papel!r}")
    dl._exigir_compativel(limiar, identidade)
    return limiar


# ──── Manifesto versionado do pacote ATIVO e reversão (rollback) para a versão servida anterior.
ARQUIVO_MANIFESTO = Path("data/models/pacote_servido=v1.json")
ARQUIVO_MANIFESTO_V2 = Path("data/models/pacote_servido=v2.json")  # se existir, vale; senão, o v1 (rollback)
PASTA_HISTORICO = Path("data/models/pacote_servido=v1.historico")
VERSOES_DO_MANIFESTO = (1, 2)
VERSAO_DO_MANIFESTO = 1
MODALIDADES = ("video", "imagem", "texto")
MODALIDADES_OPCIONAIS = ("video_local",)  # preditor local de vídeo, ao lado dos três pacotes obrigatórios
LAYOUTS = ("final", "desenvolvimento")
FORMATO_UTC = "%Y-%m-%dT%H:%M:%SZ"


class ErroManifesto(RuntimeError):
    """Manifesto de pacote ativo ausente, inválido ou com artefato que não confere."""


def caminho_do_manifesto(raiz: Path) -> Path:
    """Manifesto vigente: `pacote_servido=v2.json` se existir; senão `pacote_servido=v1.json`.

    Remover ou renomear o v2 devolve o serviço ao v1, que nunca é alterado por esta escolha."""
    v2 = Path(raiz) / ARQUIVO_MANIFESTO_V2
    v1 = Path(raiz) / ARQUIVO_MANIFESTO
    if v2.is_file() or v1.is_file():
        return v2 if v2.is_file() else v1
    return Path(raiz) / "data/execucao/manifesto.json"


def ler(raiz: Path) -> dict:
    """Manifesto da raiz, com a estrutura conferida (não os arquivos; ver `validar_versao`)."""
    arquivo = caminho_do_manifesto(raiz)
    if not arquivo.is_file():
        raise ErroManifesto(f"manifesto de pacote ativo ausente ({ARQUIVO_MANIFESTO.as_posix()}); o agente não serve "
                            "modelos sem ele e não volta aos modelos de desenvolvimento em silêncio")
    try:
        manifesto = json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError) as falha:
        raise ErroManifesto(f"manifesto de pacote ativo ilegível: {type(falha).__name__}") from falha
    if manifesto.get("tipo") != "pacote_servido" or manifesto.get("versao_do_manifesto") not in VERSOES_DO_MANIFESTO:
        raise ErroManifesto("manifesto de pacote ativo de tipo ou versão desconhecidos")
    for versao in (manifesto.get("ativo"), *manifesto.get("anteriores", [])):
        _conferir_estrutura(versao)
    return manifesto


def _conferir_estrutura(versao: object) -> None:
    if not isinstance(versao, dict) or versao.get("layout") not in LAYOUTS or not versao.get("rotulo"):
        raise ErroManifesto("versão servida sem rótulo ou com layout desconhecido")
    pacotes = set(versao.get("pacotes", {}))
    if not set(MODALIDADES) <= pacotes or not pacotes <= {*MODALIDADES, *MODALIDADES_OPCIONAIS}:
        raise ErroManifesto(f"versão {versao['rotulo']!r} deve ter pacotes de {list(MODALIDADES)} "
                            f"(e, opcionalmente, {list(MODALIDADES_OPCIONAIS)})")
    for chave in ("dataset_de_referencia", "base_historica"):
        if not versao.get(chave, {}).get("pasta"):
            raise ErroManifesto(f"versão {versao['rotulo']!r} sem {chave}")


def versao_ativa(raiz: Path) -> dict:
    return ler(raiz)["ativo"]


# --- conferência dos artefatos -------------------------------------------------------------------------------------


def validar_versao(raiz: Path, versao: Mapping) -> None:
    """Confere pastas e hashes da versão; qualquer divergência é `ErroManifesto` (nada é servido pela metade)."""
    raiz = Path(raiz)
    for chave in ("dataset_de_referencia", "base_historica"):
        if not (raiz / versao[chave]["pasta"]).is_dir():
            raise ErroManifesto(f"{chave} ausente: {versao[chave]['pasta']}")
    dataset = versao["dataset_de_referencia"]
    if dataset.get("sha256_manifesto") and _sha256(raiz / dataset["pasta"] / "manifesto.json") != dataset["sha256_manifesto"]:
        raise ErroManifesto("o dataset de referência não confere com o manifesto registrado")
    conferir = _conferir_final if versao["layout"] == "final" else _conferir_desenvolvimento
    for modalidade, entrada in versao["pacotes"].items():
        try:
            conferir(raiz, modalidade, entrada)
        except (rt.ErroPacote, md.ErroModelo, dl.LimiarIncompativel, OSError, ValueError, KeyError) as falha:
            raise ErroManifesto(f"pacote de {modalidade} da versão {versao['rotulo']!r} não confere: {falha}") from falha


def _conferir_final(raiz: Path, modalidade: str, entrada: Mapping) -> None:
    pasta = raiz / entrada["pasta"]
    if not pasta.is_dir():
        raise rt.ErroPacote(f"pasta inexistente: {entrada['pasta']}")
    if _sha256(pasta / rt.ARQUIVO_PACOTE) != entrada["sha256_pacote_json"]:
        raise rt.ErroPacote("pacote.json não confere com o hash do manifesto")
    pacote = json.loads((pasta / rt.ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    local = modalidade == "video_local"
    modalidade_do_pacote = "video" if local else modalidade
    if pacote["modalidade"] not in (modalidade_do_pacote, modalidade) or pacote["identidade_sha256"] != entrada["identidade_sha256"]:
        raise rt.ErroPacote("modalidade ou identidade do pacote diferem do manifesto")
    if local and (pacote.get("rota") != "video_local" or pacote.get("usa_gemini") is not False):
        raise rt.ErroPacote("o pacote declarado como video_local não é um preditor local sem Gemini")
    rt.verificar_integridade(pasta)
    limiar = entrada.get("limiar")
    if limiar is None:
        return
    if modalidade not in ("video", "video_local"):
        raise rt.ErroPacote("só os pacotes de vídeo têm limiar de baixa confiança")
    arquivo = pasta / limiar["arquivo"]
    if _sha256(arquivo) != limiar["sha256_arquivo"]:
        raise rt.ErroPacote("o arquivo do limiar não confere com o manifesto")
    lido = dl.ler_limiar(arquivo)
    if lido.papel != "servido" or lido.n != dl.N_LIM:
        raise rt.ErroPacote("o limiar do pacote servido deve ter papel 'servido' e 100 vídeos da LIM")
    dl._exigir_compativel(lido, pacote["identidade_sha256"])  # recusa limiar de outro pacote


def _conferir_desenvolvimento(raiz: Path, modalidade: str, entrada: Mapping) -> None:
    for chave in ("pasta_q50", "pasta_faixa"):
        if not (raiz / entrada[chave]).is_dir():
            raise rt.ErroPacote(f"pasta inexistente: {entrada[chave]}")
    meta = json.loads((raiz / entrada["pasta_q50"] / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
    if meta["sha256_modelo"] != entrada["sha256_modelo_q50"]:
        raise rt.ErroPacote("o q50 não confere com o hash do manifesto")
    if _sha256(raiz / entrada["pasta_q50"] / md.ARQUIVO_MODELO) != meta["sha256_modelo"]:
        raise rt.ErroPacote("o arquivo do modelo q50 não confere com o hash registrado nos metadados")


# --- construção, ativação e reversão -----------------------------------------------------------------------------


def entrada_final(raiz: Path, pasta: Path, *, diagnostico_limiar: Mapping | None = None) -> dict:
    """Entrada do manifesto para um pacote final (hashes lidos do disco); o limiar, se existir, vai vinculado."""
    raiz, pasta = Path(raiz), Path(pasta)
    pacote = json.loads((pasta / rt.ARQUIVO_PACOTE).read_text(encoding="utf-8"))
    entrada = {"pasta": pasta.relative_to(raiz).as_posix(), "sha256_pacote_json": _sha256(pasta / rt.ARQUIVO_PACOTE),
               "identidade_sha256": pacote["identidade_sha256"], "versao_servida": pacote["versao_servida"],
               "limiar": None}
    arquivo = pasta / "limiar" / dl.ARQUIVO_LIMIAR.format(papel="servido")
    if arquivo.is_file():
        lido = dl.ler_limiar(arquivo)
        entrada["limiar"] = {
            "arquivo": arquivo.relative_to(pasta).as_posix(), "sha256_arquivo": _sha256(arquivo),
            "valor_log": lido.valor, "percentil": lido.percentil, "n": lido.n, "papel": lido.papel,
            "sha256_pacote": lido.sha256_pacote, "ids_sha256": lido.ids_sha256,
            "diagnostico_de_desenvolvimento": dict(diagnostico_limiar or {})}
    return entrada


def resumo_do_diagnostico(arquivo: Path) -> dict:
    """Leitura honesta do diagnóstico de desenvolvimento do limiar, copiada do arquivo gravado (nada digitado)."""
    d = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    return {"arquivo": Path(arquivo).name, "sha256_arquivo": _sha256(arquivo), "leitura": d["leitura"],
            "limiar_de_desenvolvimento_log": d["limiar"], "pacote_de_desenvolvimento": d["pacote"]["sha256_pacote"],
            "n_sinalizados": d["n_sinalizados"], "n_nao_sinalizados": d["n_nao_sinalizados"],
            "mae_sinalizados": d["mae_sinalizados"], "mae_nao_sinalizados": d["mae_nao_sinalizados"],
            "ic95_da_diferenca": d["bootstrap"]["ic95"], "n_validacao": d["populacao"]["n_validacao"]}


def montar_versao_final(raiz: Path, *, rotulo: str, pastas: Mapping[str, Path], pasta_dataset: Path, pasta_base: Path,
                        arquivo_diagnostico_limiar: Path | None, motivo: str) -> dict:
    """`pastas` leva `video`, `imagem` e `texto` e, opcionalmente, `video_local` (preditor local de vídeo)."""
    raiz = Path(raiz)
    diagnostico = resumo_do_diagnostico(arquivo_diagnostico_limiar) if arquivo_diagnostico_limiar else None
    dataset = json.loads((Path(pasta_dataset) / "manifesto.json").read_text(encoding="utf-8"))
    return {
        "rotulo": rotulo, "layout": "final", "motivo": motivo,
        "pacotes": {m: entrada_final(raiz, Path(pastas[m]), diagnostico_limiar=diagnostico if m == "video" else None)
                    for m in (*MODALIDADES, *(o for o in MODALIDADES_OPCIONAIS if o in pastas))},
        "dataset_de_referencia": {"pasta": Path(pasta_dataset).relative_to(raiz).as_posix(),
                                  "sha256_dataset": dataset["sha256_dataset"],
                                  "sha256_manifesto": _sha256(Path(pasta_dataset) / "manifesto.json"),
                                  "papel": "coleção de referência dos comparáveis (refit final; sem LIM)"},
        "base_historica": {"pasta": Path(pasta_base).relative_to(raiz).as_posix(),
                           "sha256_manifesto": _sha256(Path(pasta_base) / "MANIFESTO_SHA256.txt")},
    }


def _gravar(raiz: Path, manifesto: Mapping, *, agora: datetime, destino: Path | None = None) -> Path:
    arquivo = Path(raiz) / destino if destino is not None else caminho_do_manifesto(raiz)
    if arquivo.is_file():
        copia = Path(raiz) / PASTA_HISTORICO / f"{agora.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        copia.parent.mkdir(parents=True, exist_ok=True)
        with open(copia, "x", encoding="utf-8") as saida:  # cópia do manifesto anterior; nunca sobrescreve
            saida.write(arquivo.read_text(encoding="utf-8"))
    return gravar_json_atomico(arquivo, dict(manifesto))


def ativar(raiz: Path, versao: Mapping, *, agora: datetime | None = None, motivo: str = "",
           anterior_inicial: Mapping | None = None, destino: Path | None = None) -> dict:
    """Ativa `versao`; a ativa atual (se houver) passa à frente da lista de anteriores. Valida antes de gravar.

    `destino` (relativo à raiz) escolhe o arquivo gravado; `ARQUIVO_MANIFESTO_V2` cria o manifesto v2 a partir do
    vigente sem tocar no v1, que continua sendo o retorno do rollback.

    Sem manifesto prévio, `anterior_inicial` (a versão servida até então) entra como a primeira para rollback."""
    agora = agora or datetime.now(UTC)
    raiz = Path(raiz)
    _conferir_estrutura(dict(versao))
    validar_versao(raiz, versao)
    atual = ler(raiz) if caminho_do_manifesto(raiz).is_file() else None
    if atual is None and anterior_inicial is not None:
        _conferir_estrutura(dict(anterior_inicial))
        validar_versao(raiz, anterior_inicial)
        atual = {"ativo": dict(anterior_inicial), "anteriores": [], "historico": [
            {"em_utc": _iso(agora), "acao": "registrar_anterior", "de": None, "para": anterior_inicial["rotulo"],
             "motivo": "versão servida antes do manifesto, mantida para rollback"}]}
    nova = dict(versao) | {"ativado_em_utc": _iso(agora)}
    anteriores = [] if atual is None else [atual["ativo"], *[a for a in atual["anteriores"]
                                                           if a["rotulo"] != versao["rotulo"]]]
    anteriores = [a for a in anteriores if a["rotulo"] != nova["rotulo"]]
    historico = ([] if atual is None else list(atual.get("historico", []))) + [
        {"em_utc": _iso(agora), "acao": "ativar", "de": atual["ativo"]["rotulo"] if atual else None,
         "para": nova["rotulo"], "motivo": motivo}]
    manifesto = {"tipo": "pacote_servido",
                 "versao_do_manifesto": 2 if destino == ARQUIVO_MANIFESTO_V2 else VERSAO_DO_MANIFESTO, "ativo": nova,
                 "anteriores": anteriores, "historico": historico}
    _gravar(raiz, manifesto, agora=agora, destino=destino)
    return manifesto


def reverter(raiz: Path, *, rotulo: str | None = None, agora: datetime | None = None, motivo: str = "") -> dict:
    """Rollback: torna ativa a versão anterior (a mais recente, ou a de `rotulo`), depois de validá-la inteira.

    A versão que estava ativa passa a ser a primeira das anteriores, de modo que o rollback também se desfaz. Os pacotes
    em disco não são tocados."""
    agora = agora or datetime.now(UTC)
    manifesto = ler(raiz)
    anteriores = manifesto["anteriores"]
    if not anteriores:
        raise ErroManifesto("não há versão anterior mantida para rollback")
    escolhida = next((a for a in anteriores if rotulo in (None, a["rotulo"])), None)
    if escolhida is None:
        raise ErroManifesto(f"versão anterior inexistente: {rotulo!r}; disponíveis: {[a['rotulo'] for a in anteriores]}")
    validar_versao(raiz, escolhida)
    restantes = [a for a in anteriores if a["rotulo"] != escolhida["rotulo"]]
    novo = {"tipo": "pacote_servido", "versao_do_manifesto": manifesto["versao_do_manifesto"],
            "ativo": dict(escolhida) | {"ativado_em_utc": _iso(agora)}, "anteriores": [manifesto["ativo"], *restantes],
            "historico": [*manifesto.get("historico", []),
                          {"em_utc": _iso(agora), "acao": "reverter", "de": manifesto["ativo"]["rotulo"],
                           "para": escolhida["rotulo"], "motivo": motivo}]}
    _gravar(raiz, novo, agora=agora)
    return novo


def versao_de_desenvolvimento_servida_ate_18_10(raiz: Path) -> dict:
    """A versão servida antes do refit (pacotes de desenvolvimento, dataset de 453 linhas), registrada para rollback."""
    raiz = Path(raiz)
    nomes = {"video": "6f75e1255f73", "imagem": "2a4739584168", "texto": "18be8a9f7e86"}
    pacotes = {}
    for modalidade, nome in nomes.items():
        q50 = Path("data/models/provisorio=v1") / (f"{nome}-c62b81db")
        meta = json.loads((raiz / q50 / md.ARQUIVO_METADADOS).read_text(encoding="utf-8"))
        pacotes[modalidade] = {"pasta_q50": q50.as_posix(), "pasta_faixa": f"data/models/faixa=v1/{nome}-{nome}-c62b81db",
                               "sha256_modelo_q50": meta["sha256_modelo"]}
    dataset = Path("data/features/snapshot=20261014T021437Z/desenvolvimento=v1/6f75e1255f73")
    return {
        "rotulo": "desenvolvimento=v1-servido-ate-18-10", "layout": "desenvolvimento",
        "motivo": "pacotes de desenvolvimento servidos antes do refit final (vídeo 6f75e1255f73 com 346 linhas de treino; "
                  "imagem e texto com 684); mantidos só para rollback",
        "pacotes": pacotes,
        "dataset_de_referencia": {"pasta": dataset.as_posix(),
                                  "sha256_manifesto": _sha256(raiz / dataset / "manifesto.json")},
        "base_historica": {"pasta": "data/processed/snapshot=20261014T021437Z/curadoria=v1"},
    }



def main(argv: list[str] | None = None) -> int:
    """`mostrar`, `ativar-final` e `reverter`. 0 = feito; 2 = recusado (mensagem no stderr)."""
    import argparse
    import sys

    from src.config import RAIZ_PROJETO

    parser = argparse.ArgumentParser(prog="python -m src.modelos.pacote", description=__doc__)
    parser.add_argument("--raiz", type=Path, default=RAIZ_PROJETO)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("mostrar", help="resume a versão ativa e as anteriores")
    ativar_p = sub.add_parser("ativar-final", help="ativa o refit final e registra a versão de desenvolvimento servida até então")
    for modalidade in MODALIDADES:
        ativar_p.add_argument(f"--{modalidade}", type=Path, required=True, help=f"pasta do pacote final de {modalidade}")
    ativar_p.add_argument("--video-local", type=Path, help="pasta do preditor local de vídeo (opcional)")
    ativar_p.add_argument("--manifesto-v2", action="store_true",
                          help="grava pacote_servido=v2.json em vez de alterar o v1")
    ativar_p.add_argument("--dataset", type=Path, required=True)
    ativar_p.add_argument("--base", type=Path, required=True)
    ativar_p.add_argument("--diagnostico-limiar", type=Path, required=True)
    ativar_p.add_argument("--rotulo", default="final=v1")
    ativar_p.add_argument("--motivo", required=True)
    reverter_p = sub.add_parser("reverter", help="rollback para a versão anterior mantida")
    reverter_p.add_argument("--para", help="rótulo da versão anterior (padrão: a mais recente)")
    reverter_p.add_argument("--motivo", required=True)
    args = parser.parse_args(argv)
    raiz = Path(args.raiz).resolve()
    try:
        if args.comando == "mostrar":
            manifesto = ler(raiz)
        elif args.comando == "reverter":
            manifesto = reverter(raiz, rotulo=args.para, motivo=args.motivo)
        else:
            versao = montar_versao_final(
                raiz, rotulo=args.rotulo,
                pastas={m: getattr(args, m).resolve() for m in MODALIDADES}
                | ({"video_local": args.video_local.resolve()} if args.video_local else {}),
                pasta_dataset=args.dataset.resolve(), pasta_base=args.base.resolve(),
                arquivo_diagnostico_limiar=args.diagnostico_limiar.resolve(), motivo=args.motivo)
            inicial = None if caminho_do_manifesto(raiz).is_file() else versao_de_desenvolvimento_servida_ate_18_10(raiz)
            manifesto = ativar(raiz, versao, motivo=args.motivo, anterior_inicial=inicial,
                               destino=ARQUIVO_MANIFESTO_V2 if args.manifesto_v2 else None)
    except (ErroManifesto, rt.ErroPacote, OSError, ValueError, KeyError) as erro:
        print(f"Manifesto recusado ({type(erro).__name__}): {erro}", file=sys.stderr)
        return 2
    print(json.dumps({"ativo": manifesto["ativo"]["rotulo"], "ativado_em_utc": manifesto["ativo"].get("ativado_em_utc"),
                      "anteriores": [a["rotulo"] for a in manifesto["anteriores"]],
                      "pacotes": {m: p.get("versao_servida") or p.get("pasta_q50")
                                  for m, p in manifesto["ativo"]["pacotes"].items()}}, ensure_ascii=False, indent=1))
    return 0


# ──── Previsão central (q50), faixa q10–q90 e contribuições do q50 para uma linha de inferência.
def faixa_da_linha(pacote: Pacote, linha: Mapping) -> dict:
    """Valores exibidos (central, faixa, razão ao canal) a partir dos três modelos; sem SHAP."""
    linhas = [dict(linha)]
    faixa = fp.prever_faixa(linhas, pacote.q10.prever_relativo(linhas), pacote.q50.prever_relativo(linhas),
                            pacote.q90.prever_relativo(linhas))
    ref_log = float(linha["ref_log"])
    referencia = round(math.expm1(ref_log))
    central = int(faixa["curtidas_central"][0])
    return {
        "curtidas_central": central, "curtidas_inferior": int(faixa["curtidas_inferior"][0]),
        "curtidas_superior": int(faixa["curtidas_superior"][0]),
        "central_log": float(faixa["central_log"][0]), "inferior_log": float(faixa["inferior_log"][0]),
        "superior_log": float(faixa["superior_log"][0]), "q10_log": float(faixa["q10_log"][0]),
        "q50_log": float(faixa["q50_log"][0]), "q90_log": float(faixa["q90_log"][0]),
        "cruzou": bool(faixa["cruzou"][0]), "cortada_em_zero": bool(faixa["cortada_em_zero"][0]),
        "ref_log": ref_log, "curtidas_referencia": referencia,
        "variacao_log_vs_canal": float(faixa["variacao_log_vs_canal"][0]),
        "relative_to_channel": (1 + central) / (1 + referencia),
    }


def prever_linha(pacote: Pacote, linha: Mapping) -> dict:
    """`faixa_da_linha` mais as contribuições SHAP do q50 na escala do modelo."""
    shap = fp.contribuicoes_q50(pacote.q50, [dict(linha)])
    return faixa_da_linha(pacote, linha) | {"shap": {
        "nomes": list(shap["nomes"]), "valores": [float(v) for v in shap["contribuicoes"][0]],
        "base": float(shap["base"]), "saida_bruta_relativa": float(shap["saida_bruta_relativa"][0]),
        "cortada_em_zero": bool(shap["cortada_em_zero"][0]), "transformacoes": list(shap["transformacoes"]),
    }}


if __name__ == "__main__":
    raise SystemExit(main())
