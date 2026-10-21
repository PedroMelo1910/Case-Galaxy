"""Compatibilidade de serialização: caminho antigo de `PreprocessadorAudiovisual`.

O preditor local de vídeo (`final=v2/video_local-...`) guarda a classe como
`src.refit_audiovisual_preprocessador.PreprocessadorAudiovisual`. A implementação vive em `src.modelos.preprocessamento`; este
módulo apenas a reexporta para que `joblib.load` continue resolvendo o caminho gravado, sem regravar o artefato. Não adicione
lógica aqui.
"""

from src.modelos.preprocessamento import PreprocessadorAudiovisual, _presente

__all__ = ["PreprocessadorAudiovisual", "_presente"]
