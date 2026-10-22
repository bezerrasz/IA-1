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


@dataclass(frozen=True)
class Sprint4Config:
    """Configuração compacta usada como referência nos experimentos do GPT."""

    context_length: int = 16
    embedding_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    feed_forward_dim: int = 64
    dropout: float = 0.0
    max_new_tokens: int = 4


DEFAULT_SPRINT4_CONFIG = Sprint4Config()


@dataclass(frozen=True)
class TrainingConfig:
    """Parâmetros de um primeiro treinamento pequeno, executável na CPU."""

    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    train_fraction: float = 0.8
    stride: int = 8
    seed: int = 123
    patience: int = 5
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        import math

        for name in ("epochs", "batch_size", "stride", "patience"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} deve ser maior que zero")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction deve estar no intervalo (0, 1)")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser positivo e finito")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay deve ser não negativo e finito")
