"""Fornecedor de contextos da avaliação final sobre os dados reais (protocolo final §3 a §6).

É o adaptador entre o orquestrador (`final.executar`, que só conhece `fornecedor(espec)`) e a materialização do contexto
(`contextos_finais.materializar_contexto`). Fecha o portão ANTES de delegar: um contexto com bloco de teste só chega ao
materializador com a consulta única aberta, mesmo que alguém chame o adaptador direto. No ensaio de desenvolvimento
(validação no lugar do teste) nenhum contexto usa teste e `consulta` fica vazia.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.avaliacao import contextos_finais as cf
from src.avaliacao.protocolo_final import ConsultaAberta


@dataclass(frozen=True)
class FornecedorReal:
    pasta_base: Path
    arquivo_particao: Path
    destino: Path
    pasta_legenda_capa: Path
    arquivo_amostra: Path
    carregar_anotacoes: Callable[[Sequence[str]], Mapping[str, Mapping]]
    codificador: object
    cache_embeddings: Path
    consulta: ConsultaAberta | None = None

    def __call__(self, espec: cf.EspecContexto) -> Mapping:
        cf.exigir_portao(espec, consulta=self.consulta)
        return cf.materializar_contexto(
            espec, pasta_base=self.pasta_base, arquivo_particao=self.arquivo_particao, destino=self.destino,
            pasta_legenda_capa=self.pasta_legenda_capa, arquivo_amostra=self.arquivo_amostra,
            carregar_anotacoes=self.carregar_anotacoes, codificador=self.codificador,
            cache_embeddings=self.cache_embeddings, consulta=self.consulta)
