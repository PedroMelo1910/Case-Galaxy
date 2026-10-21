"""Estado do envio: impede análise duplicada.

Opera sobre qualquer mapeamento mutável (`st.session_state` na interface, um `dict` nos testes). O botão chama
`solicitar` como callback, antes de a página ser redesenhada: a partir daí `executando` é verdadeiro, o botão e os
campos aparecem desabilitados e novos pedidos são ignorados até `concluir`.
"""

from collections.abc import MutableMapping

_EXECUTANDO = "envio_executando"
_SOLICITADO = "envio_solicitado"


class ControleEnvio:
    def __init__(self, estado: MutableMapping) -> None:
        self._estado = estado
        self._estado.setdefault(_EXECUTANDO, False)
        self._estado.setdefault(_SOLICITADO, False)

    @property
    def executando(self) -> bool:
        return bool(self._estado[_EXECUTANDO])

    def solicitar(self) -> bool:
        """Registra um pedido de análise. Falso (e nada muda) se já há uma análise em andamento."""
        if self.executando:
            return False
        self._estado[_EXECUTANDO] = True
        self._estado[_SOLICITADO] = True
        return True

    def consumir_solicitacao(self) -> bool:
        """Verdadeiro uma única vez por solicitação aceita: quem executa a análise chama isto antes de começar."""
        pendente = bool(self._estado[_SOLICITADO])
        self._estado[_SOLICITADO] = False
        return pendente

    def concluir(self) -> None:
        self._estado[_EXECUTANDO] = False
        self._estado[_SOLICITADO] = False
