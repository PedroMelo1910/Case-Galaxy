"""Módulo consolidado. Reúne:
- Avaliação descritiva única do preditor local no teste temporal já exposto (exploratória / teste reutilizado).
- Protocolo do experimento audiovisual local: definições, rascunho, congelamento e verificação por SHA-256."""

import json
import os
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

from src.avaliacao import (
    audiovisual as ae,
    experimentos as af,
    final as cf,
    metricas as bs,
    metricas as od,
    relatorio as rl,
    validacao as ad,
    validacao as ah,
    validacao as apo,
)
from src.dados import historico as hs
from src.modelos import modalidades as mm
from src.proveniencia import sha256_arquivo, sha256_arquivo as _sha, sha256_bytes


# ──── Avaliação descritiva única do preditor local no teste temporal já exposto (exploratória / teste reut
ROTULO = "exploratória / teste reutilizado"
BLOCOS_TESTE = ("teste_1", "teste_2", "teste_3")
BLOCOS_ANTERIORES = ("treino", "validacao")
MARCADOR = "teste_reutilizado.iniciado.json"
RESULTADO = "avaliacao_teste_reutilizado.json"
BRACOS_FIXOS = ("M0",)
MAX_AJUSTES_TESTE = 2 * len(BLOCOS_TESTE) * 3  # dois braços aprendidos × três dobras × três quantis


class TesteJaConsultado(RuntimeError):
    """O marcador da avaliação única já existe: a consulta ao teste reutilizado não se repete."""


class PortaoDoTesteFechado(RuntimeError):
    """Protocolo não congelado, resultado do desenvolvimento ausente ou hash divergente."""


NOME_RESULTADO_DESENVOLVIMENTO = "resultado_experimento.json"
ESTADOS_REABRIVEIS = ("falha_tecnica",)


@dataclass(frozen=True)
class TesteAberto:
    """Prova de que o marcador foi criado; só `abrir_teste` a produz."""

    sha256_protocolo: str
    configuracao_servida: str
    marcador: Path
    pasta: Path
    sha256_resultado_desenvolvimento: str = ""
    _chave: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._chave is not _CHAVE:
            raise TypeError("TesteAberto só é criado por abrir_teste")

    def marcar(self, estado: str | None = None, **extra) -> dict:
        """Atualiza o marcador de forma atômica (arquivo temporário + substituição)."""
        corpo = json.loads(self.marcador.read_text(encoding="utf-8"))
        corpo |= ({"estado": estado} if estado else {}) | extra
        _gravar_atomico(self.marcador, corpo)
        return corpo

    def metricas_produzidas(self) -> bool:
        return bool(json.loads(self.marcador.read_text(encoding="utf-8")).get("metricas_produzidas"))


_CHAVE = object()


def _gravar_atomico(arquivo: Path, corpo: Mapping) -> None:
    temporario = arquivo.with_name(arquivo.name + ".tmp")
    temporario.write_text(json.dumps(corpo, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporario, arquivo)


def conferir_livro_de_ajustes(livro: Path, feitos: int) -> int:
    """Relê o livro append-only e compara com os ajustes declarados no resultado (sequência 1..n sem lacuna)."""
    linhas = [json.loads(l) for l in Path(livro).read_text(encoding="utf-8").splitlines() if l.strip()]
    if [l["ajuste"] for l in linhas] != list(range(1, len(linhas) + 1)):
        raise PortaoDoTesteFechado("o livro de ajustes tem lacuna, repetição ou ordem alterada")
    if len(linhas) != feitos:
        raise PortaoDoTesteFechado(f"o livro registra {len(linhas)} ajustes e o resultado declara {feitos}")
    return len(linhas)


def ler_resultado_verificado(pasta_protocolo: Path, arquivo_resultado: Path, *, protocolo: str = "protocolo.json",
                             sha256_esperado: str | None = None, livro: Path | None = None) -> tuple[dict, str, str]:
    """(corpo, sha256 do resultado, sha256 do protocolo) depois de conferir tudo que liga o resultado ao protocolo.

    Exige protocolo congelado e íntegro, nome canônico do resultado, `sha256_protocolo` igual ao congelado, os 48
    ajustes feitos e uma configuração escolhida elegível."""
    pasta = Path(pasta_protocolo)
    arquivo = pasta / protocolo
    if not arquivo.is_file() or not (pasta / f"{protocolo}.sha256").is_file():
        raise PortaoDoTesteFechado("protocolo congelado ou seu SHA-256 ausente")
    sha = _sha(arquivo)
    if (pasta / f"{protocolo}.sha256").read_text(encoding="utf-8").split()[0] != sha:
        raise PortaoDoTesteFechado("o SHA-256 gravado não confere com o protocolo")
    if sha256_esperado is not None and sha != sha256_esperado:
        raise PortaoDoTesteFechado("o protocolo difere do hash esperado")
    if json.loads(arquivo.read_text(encoding="utf-8")).get("estado") != "congelado":
        raise PortaoDoTesteFechado("o protocolo não está congelado")
    if Path(arquivo_resultado).name != NOME_RESULTADO_DESENVOLVIMENTO:
        raise PortaoDoTesteFechado(f"o resultado do desenvolvimento deve se chamar {NOME_RESULTADO_DESENVOLVIMENTO}")
    corpo = json.loads(Path(arquivo_resultado).read_text(encoding="utf-8"))
    if corpo.get("sha256_protocolo") != sha:
        raise PortaoDoTesteFechado("o resultado do desenvolvimento pertence a outro protocolo")
    if corpo.get("resultado", {}).get("ajustes", {}).get("feitos") != ae.MAX_AJUSTES:
        raise PortaoDoTesteFechado(f"o resultado não registra os {ae.MAX_AJUSTES} ajustes do experimento")
    if corpo.get("selecao", {}).get("escolhida") not in af.ELEGIVEIS_PARA_SERVIR:
        raise PortaoDoTesteFechado("configuração escolhida inválida no resultado do desenvolvimento")
    if livro is not None:
        conferir_livro_de_ajustes(livro, corpo["resultado"]["ajustes"]["feitos"])
    return corpo, _sha(arquivo_resultado), sha


def abrir_teste(pasta_protocolo: Path, arquivo_resultado_desenvolvimento: Path, *, protocolo: str = "protocolo.json",
                sha256_esperado: str | None = None, entradas_externas: Mapping[str, str] | None = None,
                livro: Path | None = None) -> TesteAberto:
    """Confere protocolo e resultado do desenvolvimento e cria (ou, após falha técnica sem métrica, reabre) o marcador.

    A reexecução técnica só é permitida se o marcador estiver em `falha_tecnica`, o resultado final não existir e nenhuma
    métrica tiver sido produzida ou exibida. Qualquer outro estado recusa. O chamador deve ter tomado o lock e
    conferido a memória ANTES de chamar esta função."""
    pasta = Path(pasta_protocolo)
    corpo, sha_resultado, sha = ler_resultado_verificado(pasta, arquivo_resultado_desenvolvimento, protocolo=protocolo,
                                                         sha256_esperado=sha256_esperado, livro=livro)
    escolhida = corpo["selecao"]["escolhida"]
    marcador = pasta / MARCADOR
    inicio = {"iniciado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256_protocolo": sha,
              "configuracao_servida": escolhida, "sha256_resultado_desenvolvimento": sha_resultado,
              "entradas_externas_sha256": dict(entradas_externas or {}), "rotulo": ROTULO, "estado": "aberto",
              "tentativas": 1, "metricas_produzidas": False,
              "nota": "criado antes de construir qualquer contexto com alvo de teste; a consulta não se repete"}
    try:
        with open(marcador, "x", encoding="utf-8") as saida:
            json.dump(inicio, saida, ensure_ascii=False, indent=1)
    except FileExistsError:
        anterior = json.loads(marcador.read_text(encoding="utf-8"))
        reabrivel = (anterior.get("estado") in ESTADOS_REABRIVEIS and not anterior.get("metricas_produzidas")
                     and not (pasta / RESULTADO).exists())
        if not reabrivel:
            raise TesteJaConsultado("a avaliação no teste reutilizado já foi iniciada") from None
        if (anterior["sha256_protocolo"], anterior["sha256_resultado_desenvolvimento"]) != (sha, sha_resultado):
            raise TesteJaConsultado("a reexecução técnica exige o mesmo protocolo e o mesmo resultado") from None
        _gravar_atomico(marcador, anterior | {"estado": "aberto", "tentativas": anterior.get("tentativas", 1) + 1})
    return TesteAberto(sha, escolhida, marcador, pasta, sha_resultado, _CHAVE)


@contextmanager
def sessao(aberto: TesteAberto) -> Iterator[TesteAberto]:
    """Registra a falha no marcador: `falha_tecnica` (reabrível) se nenhuma métrica foi produzida; senão
    `falha_apos_metricas` (definitivo)."""
    try:
        yield aberto
    except BaseException as erro:
        estado = "falha_apos_metricas" if aberto.metricas_produzidas() else "falha_tecnica"
        aberto.marcar(estado, ultimo_erro=type(erro).__name__)
        raise


# --- contextos ----------------------------------------------------------------------------------------------------


def permitidos_ate(bloco: str) -> tuple[str, ...]:
    k = BLOCOS_TESTE.index(bloco) + 1
    return (*BLOCOS_ANTERIORES, *BLOCOS_TESTE[:k])


def contexto_de_teste(aberto: TesteAberto, bloco: str, posts: Sequence[hs.Post], canais: Mapping[str, hs.Canal],
                      atribuicao: Mapping[str, str], particao: Mapping, tabela_local: Mapping[str, Mapping],
                      ids_ativos: Collection[str], *, defasagem_dias: int = hs.DEFASAGEM_PADRAO,
                      intervalo_dias: int = 30) -> ae.DobraPronta:
    """Dobra do bloco de teste: N1 no corte do bloco, colunas locais coladas, auditoria falha fechada.

    `ids_ativos`: IDs da amostra ativa com extração local completa (os demais saem de todos os braços). O corte vem
    da partição original congelada, mesmo que uma substituição mude a primeira publicação do bloco."""
    if not isinstance(aberto, TesteAberto):
        raise PortaoDoTesteFechado("o contexto com alvo de teste exige a prova de abertura")
    corte = cf.corte_de_treino(particao, bloco, intervalo_dias)
    ctx = hs.ContextoTreino(f"teste_reutilizado_{bloco}", corte, frozenset())
    linhas = cf.n1_do_contexto(posts, canais, atribuicao, ctx, avaliados=(bloco,), permitidos=permitidos_ate(bloco),
                               defasagem_dias=defasagem_dias, ids_amostra=set(ids_ativos))
    faltam = sorted(l["video_id"] for l in linhas if l["video_id"] not in tabela_local)
    if faltam:
        raise ae.ErroExperimento(f"linhas sem atributos locais no contexto {bloco}: {faltam[:3]} ({len(faltam)})")
    colunas_n1 = set(linhas[0]) if linhas else set()
    juntas = [{**{c: v for c, v in tabela_local[l["video_id"]].items() if c not in colunas_n1}, **l} for l in linhas]
    auditoria = ah.auditar_dobra(posts, canais, juntas, ctx, defasagem_dias=defasagem_dias)
    if not auditoria["aprovado"]:
        raise ae.ErroExperimento(f"auditoria do histórico reprovada em {bloco}")
    treino, avaliacao, sem_treino, sem_aval, excluidas = [], [], [], [], 0
    for l in juntas:
        if l["uso_no_contexto"] == "intervalo_excluido":
            excluidas += 1
        elif not ae._tem_alvo(l):
            (sem_aval if l["uso_no_contexto"] == "avaliacao" else sem_treino).append(l["video_id"])
        elif l["uso_no_contexto"] == "avaliacao":
            avaliacao.append({**l, "dobra": bloco})
        else:
            treino.append({**l, "dobra": bloco})
    return ae.DobraPronta(BLOCOS_TESTE.index(bloco) + 1, treino, avaliacao, corte,
                          {"treino_sem_referencia_ou_alvo": sorted(sem_treino),
                           "avaliacao_sem_referencia_ou_alvo": sorted(sem_aval), "intervalo_excluido": excluidas},
                          auditoria)


# --- população ----------------------------------------------------------------------------------------------------


def identificar_populacao(ids_avaliados: Sequence[str], ids_originais_do_teste: Collection[str],
                          substituicoes: Mapping[str, str], ids_ativos_do_teste: Collection[str]) -> dict:
    """Quem foi avaliado: originais mantidos, substitutos e originais fora (substituídos ou sem extração local)."""
    avaliados, originais, ativos = set(ids_avaliados), set(ids_originais_do_teste), set(ids_ativos_do_teste)
    substitutos = set(substituicoes.values())
    ausentes = originais - avaliados
    return {
        "rotulo": ROTULO,
        "n_avaliados": len(avaliados), "n_originais_do_teste": len(originais),
        "n_originais_mantidos": len(avaliados & originais), "n_substitutos_avaliados": len(avaliados & substitutos),
        "n_avaliados_fora_de_original_e_substituto": len(avaliados - originais - substitutos),
        "n_originais_substituidos": len(originais & set(substituicoes)),
        "n_originais_nao_avaliados": len(ausentes),
        "n_originais_nao_avaliados_e_nao_substituidos": len(ausentes - set(substituicoes)),
        "n_ativos_do_teste_sem_avaliacao": len(ativos - avaliados),
        "sha256_ids_avaliados": ad.sha256_ids(sorted(avaliados)),
        "sha256_ids_originais_do_teste": ad.sha256_ids(sorted(originais)),
        "populacao_difere_da_original": avaliados != originais,
        "aviso": ("A população avaliada não é a dos 452 vídeos do teste original quando há substituição ou ausência "
                  "de extração local; diferenças de resultado em relação ao teste original podem vir da troca de "
                  "IDs e não do modelo."),
    }


def reconciliar_ids_de_teste(ids_ativos_do_teste: Collection[str], motivos_de_extracao: Mapping[str, str],
                             dobras: Sequence[ae.DobraPronta]) -> dict:
    """Todo ID ativo de bloco de teste foi avaliado ou tem exclusão listada com motivo; descarte silencioso é erro."""
    avaliados = {l["video_id"] for d in dobras for l in d.avaliacao}
    sem_referencia = {v for d in dobras for v in d.exclusoes["avaliacao_sem_referencia_ou_alvo"]}
    excluidos, silenciosos = {}, []
    for v in sorted(set(ids_ativos_do_teste) - avaliados):
        if v in motivos_de_extracao:
            excluidos[v] = f"extracao_local: {motivos_de_extracao[v]}"
        elif v in sem_referencia:
            excluidos[v] = "sem_referencia_de_historico_ou_alvo"
        else:
            silenciosos.append(v)
    if silenciosos:
        raise ae.ErroExperimento(f"{len(silenciosos)} IDs ativos de teste descartados sem motivo (ex.: {silenciosos[:3]})")
    fora = sorted(avaliados - set(ids_ativos_do_teste))
    if fora:
        raise ae.ErroExperimento(f"{len(fora)} IDs avaliados não são ativos de bloco de teste (ex.: {fora[:3]})")
    return {"n_ativos": len(set(ids_ativos_do_teste)), "n_avaliados": len(avaliados), "excluidos_com_motivo": excluidos,
            "por_motivo": {m.split(":")[0]: sum(x.split(":")[0] == m.split(":")[0] for x in excluidos.values())
                           for m in excluidos.values()}}


# --- execução -----------------------------------------------------------------------------------------------------


def avaliar_dobras_de_teste(dobras: Sequence[ae.DobraPronta], total: Mapping, configuracao_servida: str, *,
                            n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE) -> dict:
    """Braço servido, M0 e referências nas mesmas linhas, com contrastes pareados por canal."""
    if len(dobras) != len(BLOCOS_TESTE):
        raise ae.ErroExperimento(f"o teste tem {len(BLOCOS_TESTE)} dobras; recebi {len(dobras)}")
    S = configuracao_servida
    configuracoes = (S, *BRACOS_FIXOS)
    contrastes = [(S, "M0"), *((S, b) for b in ae.BASELINES), *(("M0", b) for b in ae.BASELINES)]
    nucleo = ae.avaliar_configuracoes(dobras, total, configuracoes=configuracoes, contrastes=contrastes,
                                      max_ajustes=MAX_AJUSTES_TESTE, n_repeticoes=n_repeticoes, semente=semente)
    return nucleo


def executar(aberto: TesteAberto, dobras: Sequence[ae.DobraPronta], total: Mapping, populacao: Mapping, *,
             saida: Path, n_repeticoes: int = bs.N_REPETICOES, semente: int = bs.SEMENTE) -> Path:
    """Roda os braços e grava `avaliacao_teste_reutilizado.json` uma única vez (modo exclusivo)."""
    if not isinstance(aberto, TesteAberto):
        raise PortaoDoTesteFechado("a avaliação exige a prova de abertura")
    resultado = avaliar_dobras_de_teste(dobras, total, aberto.configuracao_servida, n_repeticoes=n_repeticoes,
                                        semente=semente)
    aberto.marcar(metricas_produzidas=True)  # a partir daqui a execução não pode ser repetida
    corpo = {
        "rotulo": ROTULO, "natureza": "descritiva; teste já exposto; o modelo anterior viu todos os alvos originais",
        "sha256_protocolo": aberto.sha256_protocolo, "sha256_resultado_desenvolvimento":
        aberto.sha256_resultado_desenvolvimento, "configuracao_servida": aberto.configuracao_servida,
        "populacao_identificada": populacao, "resultado": resultado,
        "limitacoes": [
            "Teste já exposto: não é avaliação independente.",
            "A população pode diferir da original por substituição de IDs e por ausência de extração local.",
            "Avaliação retrospectiva: curtidas lidas na coleta; histórico com defasagem de 30 dias.",
            "Poucos canais por setor: intervalos aproximados; resultado nulo é inconclusivo."]}
    destino = Path(saida) / RESULTADO
    with open(destino, "x", encoding="utf-8") as arquivo:
        json.dump(rl.sanear(corpo), arquivo, ensure_ascii=False, indent=1, allow_nan=False)
    aberto.marcar("concluido", sha256_resultado_do_teste=_sha(destino))
    return destino


# ──── Protocolo do experimento audiovisual local: definições, rascunho, congelamento e verificação por SHA
VERSAO = "1"
ARQUIVO = "protocolo.json"
CAMPOS_PENDENTES_NO_RASCUNHO = ("dataset", "linhas", "dobras", "especificacao_total", "entradas_externas",
                                "ambiente")
BIBLIOTECAS_DO_AMBIENTE = ("lightgbm", "scikit-learn", "numpy")

MODULOS_DO_METODO = (
    "src/avaliacao/experimentos.py", "src/avaliacao/validacao.py",
    "src/avaliacao/validacao.py", "src/avaliacao/audiovisual.py",
    "src/avaliacao/protocolo_audiovisual.py", "src/avaliacao/protocolo_audiovisual.py",
    "src/avaliacao/protocolos.py", "src/modelos/preprocessamento.py",
    "src/modelos/modalidades.py", "src/modelos/refit.py", "src/modelos/refit.py",
    "src/dados/historico.py", "src/modelos/preprocessamento.py", "src/modelos/treino.py", "src/modelos/modalidades.py",
    "src/modelos/treino.py", "src/modelos/treino.py", "src/avaliacao/metricas.py", "src/avaliacao/experimentos.py",
    "src/avaliacao/metricas.py", "src/avaliacao/metricas.py", "src/avaliacao/metricas.py",
    "src/avaliacao/experimentos.py", "src/avaliacao/final.py", "src/avaliacao/limiar.py",
    "src/avaliacao/limiar.py", "src/extracao/pipeline.py", "src/avaliacao/execucao_audiovisual.py",
    "src/avaliacao/validacao.py", "src/dados/dataset.py", "src/dados/particoes.py", "src/dados/coleta.py",
    "src/modelos/refit.py", "src/modelos/refit.py", "src/avaliacao/relatorio.py",
    "src/dados/dataset_audiovisual.py")

EMENDAS = [
    {"id": "E1", "assunto": "população",
     "texto": "A população deixa de ser a das 474 linhas do dataset audiovisual v1. Passa a ser a visão de "
              "desenvolvimento do dataset audiovisual v2 sobre a amostra ativa (treino e validação originais; "
              "intervalo excluído, teste e LIM nunca entram), restrita às linhas com extração local concluída e sem "
              "estado de recusa. A lista de IDs é congelada antes de qualquer ajuste."},
    {"id": "E2", "assunto": "substituições",
     "texto": "IDs substitutos herdam o bloco pelas fronteiras de data da partição original; os cortes de treino das "
              "dobras de teste continuam os da partição original, mesmo que a substituição altere a primeira "
              "publicação de um bloco."},
    {"id": "E3", "assunto": "parâmetros",
     "texto": "O LightGBM usa `modelo_modalidades.CONFIGURACAO_A` (300 árvores, quantil, semente 20261015, "
              "min_child_samples = 10), a mesma dos modelos servidos, e não os parâmetros iniciais de "
              "desenvolvimento (min_child_samples = 20)."},
    {"id": "E4", "assunto": "histórico por dobra",
     "texto": "Histórico, referências (R0), alvo relativo e fallback de setor são recalculados no corte de treino de "
              "cada dobra com `historico.calcular`; as colunas do contexto de validação original não são usadas no "
              "ajuste nem na avaliação e servem apenas para o diagnóstico de segurança por dobra."},
    {"id": "E5", "assunto": "duração",
     "texto": "`duracao_s` (família histórico) faz parte de todas as configurações; `avl_duracao_s` é excluída por ser "
              "quase duplicata dela (`duracao_s` é o inteiro por teto da medida do arquivo, isto é, uma quantização). "
              "M1 a M3 medem o conteúdo visual e sonoro além da duração. No servimento a entrada `duracao_s` usa a "
              "mesma convenção (inteiro por teto da duração medida no arquivo), com teste de equivalência com a sonda "
              "do agente."},
    {"id": "E6", "assunto": "pré-processamento local por dobra",
     "texto": "Dentro de cada ajuste e só com as linhas de treino: PCA de 8 componentes por embedding, códigos de "
              "categoria, consolidação em 'outro' das categorias locais presentes em menos de 3 canais e descarte de "
              "colunas locais constantes. Nenhuma seleção supervisionada."},
    {"id": "E7", "assunto": "leitura de M4 a M7",
     "texto": "Todas as configurações são ajustadas e reportadas sempre; a leitura de M4 a M7 como contribuição do "
              "local acima de capa ou Gemini é secundária e descritiva, sem condicionar o cálculo."},
    {"id": "E8", "assunto": "preditor servível",
     "texto": "Podem ser servidas M1, M2 e M3: usam atributos locais, não exigem `gem_*` e só dependem de entradas "
              "obtidas de um arquivo de vídeo (duração, atributos locais, contexto e histórico). M4 e M5 exigem "
              "miniatura e legenda e M6 e M7 exigem Gemini: são avaliadas e não servidas."},
    {"id": "E9", "assunto": "teste reutilizado",
     "texto": "Uma única avaliação descritiva no teste temporal já exposto, depois de congelar e de escolher a "
              "configuração só com dados de desenvolvimento; rotulada 'exploratória / teste reutilizado'."},
    {"id": "E10", "assunto": "desfecho",
     "texto": "O item 1 do pré-registro v2 (acurácia par a par intra-canal como único desfecho confirmatório) é "
              "substituído: a métrica primária de seleção é o MAE em log; não há desfecho confirmatório; acurácia par a "
              "par e Spearman intra-canal são descritivos. Uma melhora de ordenação só pode ser afirmada se "
              "`ordenacao_demonstrada` for verdadeira (par a par e Spearman com IC inteiro favorável, mesmo sinal nas "
              "duas dobras e canais suficientes); caso contrário o texto é 'melhora de erro; ordenação não demonstrada'."},
    {"id": "E11", "assunto": "papel do estudo e política de servimento",
     "texto": "O estudo passa a decidir o preditor servido (antes era go/no-go descritivo). Princípio: servir a "
              "configuração escolhida mesmo sem superar M0 ou a média histórica, com o aviso correspondente e com a "
              "referência do canal exibida ao lado. M0 (histórico e contexto) poderia ser servida; fica de fora "
              "porque o requisito é um preditor que use atributos locais. Excluir M4 e M5 do servimento é decisão de "
              "produto (a rota de vídeo por arquivo não traz miniatura nem legenda). O desempate entre M1, M2 e M3 é "
              "decisão de engenharia sob incerteza, não estatisticamente sustentada."},
    {"id": "E12", "assunto": "portões do pré-registro v2",
     "texto": "Cada portão do pré-registro v2 está em `portoes_do_pre_registro_v2` como aplicado ou 'não aplicado, "
              "motivo'. Nenhum foi omitido."},
]

DEFINICOES = {
    "versao_do_protocolo": VERSAO,
    "natureza": "experimento exploratório de desenvolvimento, limitado e congelado; sem busca de hiperparâmetros",
    "modelo": {"algoritmo": "LightGBM", "parametros": dict(mm.CONFIGURACAO_A), "ignorados_na_comparacao": ["n_jobs"],
               "quantis": {"q10": 0.1, "q50": 0.5, "q90": 0.9}, "semente": 20261015,
               "busca": False, "catboost": False, "seis_modelos": False},
    "configuracoes": {c: {"descricao": cfg.descricao, "familias": list(cfg.familias), "aprendida": cfg.aprendida}
                      for c, cfg in af.CONFIGURACOES.items()},
    "referencias_r0": {"media_aritmetica": "pred_media_aritmetica", "mediana": "pred_mediana",
                       "media_movel_exponencial": "pred_mme_log"},
    "familias_em_ordem": list(af.FAMILIAS),
    "colunas_locais_fora_dos_preditores": af.colunas_locais_excluidas(),
    "dobras": {"fracao_inicial": ad.FRACAO_INICIAL, "n_janelas": ad.N_JANELAS, "intervalo_dias": ad.INTERVALO_DIAS,
               "ordem": "(publicação, video_id)", "janelas": "mesma contagem; a última recebe o resto",
               "treino_da_dobra": "linhas anteriores à janela publicadas até 30 dias antes da primeira publicação da "
                                  "janela", "embaralhamento": False},
    "historico": {"defasagem_dias": 30, "recalculo_por_dobra": True, "fallback_de_setor":
                  "posts até min(corte de treino da dobra, publicação − defasagem); alvo nunca entra",
                  "auditoria": "reconstrução independente de cada linha; qualquer violação interrompe"},
    "mesmas_linhas": "todas as configurações e referências usam exatamente as mesmas linhas em cada dobra",
    "metrica_primaria": "mae_log (agregado nas janelas de avaliação)",
    "metricas": ["mae_log", "rmse_log", "mape_adaptado (razão e %, interpretação: sensível a poucas curtidas)",
                 "spearman_intra_canal (células canal × dobra com ≥ 5 vídeos; mediana)",
                 "acuracia_par_a_par_distinguiveis (max ≥ 1,2 × min; empate 0,5)", "f1_macro_3_faixas (tercis do treino)",
                 "cobertura e largura da faixa q10–q90 corrigida (q50 central)", "r2 do alvo relativo"],
    "ordenacao": {"min_videos_por_celula": od.MIN_VIDEOS_CANAL, "fator_distinguivel": od.FATOR_DISTINGUIVEL},
    "bootstrap": {"n_repeticoes": bs.N_REPETICOES, "semente": bs.SEMENTE, "unidade": "canal inteiro nas duas dobras",
                  "ic": "percentil 2,5–97,5", "multiplicidade": "sem correção; resultado nulo é inconclusivo"},
    "contrastes_primarios": [list(c) for c in ae.CONTRASTES_PRIMARIOS],
    "contrastes_secundarios": [list(c) for c in ae.CONTRASTES_SECUNDARIOS] + [["configuração", "cada referência R0"]],
    "erro_e_ordenacao": "melhora de erro e melhora de ordenação são lidas em blocos separados; ordenação não se "
                        "deduz de MAE",
    "maximo_de_ajustes": {"experimento": ae.MAX_AJUSTES, "teste_reutilizado": MAX_AJUSTES_TESTE,
                          "refit_do_servido": 3, "total": ae.MAX_AJUSTES + MAX_AJUSTES_TESTE + 3,
                          "reexecucao": "só técnica, sem mudar método, antes de qualquer resultado gravado"},
    "selecao_do_preditor_servido": {
        "elegiveis": list(af.ELEGIVEIS_PARA_SERVIR),
        "regra": "menor MAE em log pooled entre as elegíveis; empate (≤ 0,005) vai à de menos famílias de extração "
                 "local e depois à ordem M1, M2, M3",
        "tolerancia_de_empate_mae_log": ae.TOLERANCIA_DE_EMPATE_MAE,
        "parada": "uma passagem; sem repetir, sem nova configuração; resultados do teste não alimentam a escolha",
        "sem_melhora": "se não superar M0 ou a média histórica, é servida mesmo assim, com o aviso correspondente"},
    "teste_reutilizado": {"rotulo": ROTULO, "blocos": list(BLOCOS_TESTE), "janela": "expansiva",
                          "intervalo_dias": 30, "defasagem_dias": 30, "bracos": ["configuração escolhida", "M0", "R0"],
                          "consulta_unica": True, "corte_de_treino": "partição original congelada"},
    "pacote": {"layout": "final=v2/video_local-<id12>-<config>", "refit": "todas as linhas completas da amostra ativa, "
               "LIM fora; contexto final", "limiar": "p80 das larguras da faixa corrigida na LIM, do artefato exato; "
               "sem LIM completa, confiança insuficiente/indisponível"},
    "portoes": {
        "piso": dict(apo.PISOS), "teto_de_exclusoes": apo.LIMITE_DE_EXCLUSOES, "limiar_smd": apo.LIMIAR_SMD, "cosseno_de_duplicata": apo.COSSENO_DE_DUPLICATA,
        "folga_da_sonda": apo.FOLGA_DA_SONDA, "min_canais_para_ic": apo.MIN_CANAIS_PARA_IC,
        "falha_fechada_em_preparar": ["piso", "auditoria do histórico", "visões de desenvolvimento e completa"],
        "diagnosticos_registrados": ["teto de exclusões e SMD de incluídos × excluídos (só desenvolvimento)",
                                     "sonda avl_* → canal", "duplicatas por cosseno", "equivalência de duração"]},
    "portoes_do_pre_registro_v2": [
        {"item": "1", "estado": "substituído", "nota": "emenda E10"},
        {"item": "2(a)", "estado": "aplicado", "nota": "COLUNAS_PREDITORAS sem impressões digitais; sem avl_duracao_s"},
        {"item": "2(b)", "estado": "aplicado", "nota": "PCA de 8 componentes; menor que o número de canais"},
        {"item": "2(c)", "estado": "aplicado", "nota": "categorias locais em menos de 3 canais viram 'outro'"},
        {"item": "2(d)", "estado": "aplicado", "nota": "sonda avl_* → canal como diagnóstico, sem ajuste extra"},
        {"item": "2(e)", "estado": "não aplicado", "nota": "perfil único de aquisição foi definido na extração; não é "
                                                           "verificado aqui"},
        {"item": "3", "estado": "não aplicado", "nota": "o corte único dá lugar às dobras cronológicas deste protocolo"},
        {"item": "4", "estado": "aplicado", "nota": "lista congelada, teto de 10%, SMD, dedupe por cosseno ≥ 0,98, "
                                                    "política de recusa por estado de etapa"},
        {"item": "4 (dono do canal)", "estado": "não aplicado", "nota": "a base não tem campo de dono do canal; "
                                                                        "canais distintos contam como donos distintos"},
        {"item": "5", "estado": "aplicado", "nota": "piso na população pós-exclusão, por setor e por janela"},
        {"item": "5 (canais não vistos)", "estado": "não aplicado", "nota": "análise não existe: o conjunto tem menos "
                                                                            "de 27 canais"},
        {"item": "6", "estado": "declarado", "nota": "poder baixo; efeitos pequenos não são detectáveis"},
        {"item": "7", "estado": "declarado", "nota": "vale só para a configuração A em n pequeno"},
        {"item": "8", "estado": "aplicado", "nota": "bootstrap por canal e leave-one-channel-out; menos de 8 canais = "
                                                    "não avaliável"},
        {"item": "9", "estado": "aplicado", "nota": "faixa e limiar próprios do artefato; sem herdar"},
        {"item": "10", "estado": "aplicado", "nota": "versões das bibliotecas, SHA-256 do uv.lock e identidade do "
                                                      "extrator no protocolo congelado"},
        {"item": "10 (plataforma)", "estado": "não aplicado", "nota": "toda a extração é do Windows nativo; a "
                                                                      "equivalência Linux × Windows não é exigida aqui"},
        {"item": "11", "estado": "declarado", "nota": "não existe avaliação independente"},
        {"item": "12", "estado": "aplicado", "nota": "pacote próprio, sem fallback para imagem"}],
    "regras_de_parada": ["ajuste sem nenhum split", "métrica não finita", "dobra abaixo do piso",
                         "auditoria do histórico reprovada", "limite de ajustes atingido"],
    "livro_de_ajustes": "append-only; cada ajuste é registrado antes de começar",
    "reexecucao_tecnica_do_teste": "só se o marcador estiver em falha_tecnica, avaliacao_teste_reutilizado.json não "
                                   "existir e nenhuma métrica tiver sido produzida ou exibida; mesmo protocolo e mesmo "
                                   "resultado do desenvolvimento; cada tentativa é contada no marcador",
    "tolerancia_de_empate_e_desempate": "0,005 em MAE log; menos famílias de extração local; depois M1, M2, M3",
    "limitacoes_declaradas": [
        "A auditoria do histórico é uma reconstrução na mesma base de dados, não uma verificação totalmente "
        "independente da especificação das regras.",
        "Os hiperparâmetros (configuração A) foram escolhidos em dados que se sobrepõem aos usados aqui; o candidato "
        "com min_child_samples = 40 foi recusado antes. O resultado vale para a configuração A em n pequeno.",
        "Exigir extração local concluída cria viés de sobrevivência: a população inclui só vídeos que puderam ser "
        "baixados e processados.",
        "Os intervalos são aproximados (poucos canais, percentil de reamostragem de canais) e não há correção de "
        "multiplicidade; resultado nulo é inconclusivo."],
    "emendas": EMENDAS,
}


class ErroProtocolo(ValueError):
    """Protocolo incompleto, divergente do disco ou já gravado."""


def hashes_dos_modulos(raiz: Path, modulos: Sequence[str] = MODULOS_DO_METODO) -> dict[str, str | None]:
    """SHA-256 dos módulos do método; módulo ainda inexistente aparece como `None`."""
    return {m: (sha256_arquivo(Path(raiz) / m) if (Path(raiz) / m).is_file() else None) for m in modulos}


def ambiente_atual(raiz: Path, identidade_do_extrator: str | None = None) -> dict:
    """Versões das bibliotecas do ajuste, SHA-256 do `uv.lock` e identidade do extrator."""
    from importlib import metadata
    lock = Path(raiz) / "uv.lock"
    return {"bibliotecas": {n: metadata.version(n) for n in BIBLIOTECAS_DO_AMBIENTE},
            "uv_lock_sha256": sha256_arquivo(lock) if lock.is_file() else None,
            "identidade_do_extrator": identidade_do_extrator}


def montar(raiz: Path, *, dataset: Mapping | None = None, linhas: Mapping | None = None, dobras: Mapping | None = None,
           especificacao_total: Mapping | None = None, entradas_externas: Mapping | None = None,
           ambiente: Mapping | None = None) -> dict:
    """Protocolo em rascunho (campos pendentes `None`) ou completo."""
    protocolo = {
        "estado": "rascunho", "gerado_em_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "definicoes": DEFINICOES, "dataset": dataset, "linhas": linhas, "dobras": dobras,
        "especificacao_total": especificacao_total, "entradas_externas": entradas_externas, "ambiente": ambiente,
        "modulos_sha256": hashes_dos_modulos(raiz),
    }
    protocolo["pendentes"] = [c for c in CAMPOS_PENDENTES_NO_RASCUNHO if protocolo[c] is None]
    return protocolo


def _canonico(protocolo: Mapping) -> bytes:
    return json.dumps(protocolo, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8")


def gravar(protocolo: Mapping, pasta: Path, nome: str = ARQUIVO) -> tuple[Path, str]:
    """Grava o JSON e o `.sha256` ao lado, sem sobrescrever. Devolve (arquivo, sha256)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    dados = _canonico(protocolo)
    sha = sha256_bytes(dados)
    arquivo = pasta / nome
    with open(arquivo, "xb") as saida:
        saida.write(dados)
    (pasta / f"{nome}.sha256").write_text(f"{sha}  {nome}\n", encoding="utf-8")
    return arquivo, sha


def congelar(raiz: Path, pasta: Path, *, dataset: Mapping, linhas: Mapping, dobras: Mapping,
             especificacao_total: Mapping, entradas_externas: Mapping, ambiente: Mapping,
             nome: str = ARQUIVO) -> tuple[Path, str]:
    """Exige tudo que faltava no rascunho, recalcula os hashes de módulos e grava `estado = congelado`."""
    vazios = [n for n, v in (("dataset", dataset), ("linhas", linhas), ("dobras", dobras),
                             ("especificacao_total", especificacao_total), ("entradas_externas", entradas_externas),
                             ("ambiente", ambiente)) if not v]
    if vazios:
        raise ErroProtocolo(f"não é possível congelar sem: {vazios}")
    if not ambiente.get("uv_lock_sha256") or not ambiente.get("identidade_do_extrator"):
        raise ErroProtocolo("o ambiente congelado exige o SHA-256 do uv.lock e a identidade do extrator")
    ausentes = sorted(m for m, h in hashes_dos_modulos(raiz).items() if h is None)
    if ausentes:
        raise ErroProtocolo(f"módulos do método ausentes: {ausentes}")
    protocolo = montar(raiz, dataset=dataset, linhas=linhas, dobras=dobras, especificacao_total=especificacao_total,
                       entradas_externas=entradas_externas, ambiente=ambiente)
    protocolo["estado"] = "congelado"
    return gravar(protocolo, pasta, nome)


def verificar(raiz: Path, pasta: Path, nome: str = ARQUIVO) -> dict:
    """Confere SHA-256 do arquivo, dos módulos, das bibliotecas e do `uv.lock`; devolve as divergências."""
    pasta = Path(pasta)
    arquivo = pasta / nome
    conteudo = json.loads(arquivo.read_text(encoding="utf-8"))
    gravado = (pasta / f"{nome}.sha256").read_text(encoding="utf-8").split()[0]
    atual = hashes_dos_modulos(raiz, list(conteudo["modulos_sha256"]))
    divergentes = sorted(m for m, h in conteudo["modulos_sha256"].items() if atual.get(m) != h)
    ambiente_divergente = []
    if conteudo.get("ambiente"):
        agora = ambiente_atual(raiz)
        ambiente_divergente = sorted(
            [f"biblioteca:{n}" for n, v in conteudo["ambiente"]["bibliotecas"].items() if agora["bibliotecas"].get(n) != v]
            + (["uv_lock"] if agora["uv_lock_sha256"] != conteudo["ambiente"]["uv_lock_sha256"] else []))
    sha_ok = sha256_arquivo(arquivo) == gravado
    return {"estado": conteudo["estado"], "sha256_confere": sha_ok, "sha256": sha256_arquivo(arquivo),
            "modulos_divergentes": divergentes, "ambiente_divergente": ambiente_divergente,
            "integro": sha_ok and not divergentes and not ambiente_divergente}


def descrever_linhas(ids_elegiveis: Sequence[str], ids_por_dobra: Mapping[str, Mapping[str, str]]) -> dict:
    """Bloco `linhas` do protocolo: contagem e hashes das listas congeladas."""
    return {"n_elegiveis": len(ids_elegiveis), "sha256_ids_elegiveis": ad.sha256_ids(sorted(ids_elegiveis)),
            "por_dobra": dict(ids_por_dobra)}
