"""Configurações compartilhadas do pipeline da Sprint 2.

Manter caminhos e parâmetros neste módulo evita que os experimentos tenham
valores divergentes espalhados pelos scripts e notebooks.
"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "the-verdict.txt"

SPECIAL_TOKENS = ("<|endoftext|>", "<|unk|>")


@dataclass(frozen=True)
class Sprint2Config:
    """Parâmetros padrão para os primeiros experimentos."""

    batch_size: int = 8
    context_length: int = 4
    stride: int = 4
    embedding_dim: int = 256
    shuffle: bool = False
    drop_last: bool = True


DEFAULT_CONFIG = Sprint2Config()
