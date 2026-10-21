"""Compatibilidade de serialização: caminho antigo de `Preprocessador`.

Os modelos já treinados e gravados (`final=v1`, `final=v2`, `faixa=v1`, provisórios) guardam a classe como
`src.preprocessador.Preprocessador`. A implementação vive em `src.modelos.preprocessamento`; este módulo apenas a reexporta
para que `joblib.load` continue resolvendo o caminho gravado, sem regravar nenhum artefato. Não adicione lógica aqui.
"""

from src.modelos.preprocessamento import Preprocessador, _numero

__all__ = ["Preprocessador", "_numero"]
