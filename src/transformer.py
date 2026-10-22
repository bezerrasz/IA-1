"""Componentes de um Transformer Block em configuração pre-norm."""

from __future__ import annotations

import torch
from torch import nn

try:
    from .attention import MultiHeadAttention
except ImportError:  # Permite importar o módulo diretamente a partir de src/.
    from attention import MultiHeadAttention


class FeedForwardNetwork(nn.Module):
    """MLP aplicado independentemente a cada posição da sequência."""

    def __init__(self, embedding_dim: int, feed_forward_dim: int, dropout: float):
        super().__init__()
        if embedding_dim <= 0 or feed_forward_dim <= 0:
            raise ValueError("as dimensões da FeedForward devem ser maiores que zero")
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Bloco pre-norm: LayerNorm, atenção, FFN e conexões residuais.

    A entrada e a saída possuem forma ``[B, T, D]``. A atenção mistura as
    posições, enquanto a FFN aplica a transformação não linear posição a
    posição. As duas subcamadas preservam a dimensão ``D`` para permitir a
    soma residual.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser maior que zero")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar no intervalo [0, 1)")

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            d_in=embedding_dim,
            d_out=embedding_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForwardNetwork(
            embedding_dim, feed_forward_dim, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processa uma sequência mantendo sua forma `[B, T, D]`."""

        if x.ndim != 3:
            raise ValueError("x deve possuir forma [batch, sequence, features]")
        if x.shape[-1] != self.norm1.normalized_shape[0]:
            raise ValueError("a última dimensão de x não coincide com embedding_dim")

        attention_input = self.norm1(x)
        x = x + self.residual_dropout(self.attention(attention_input))

        feed_forward_input = self.norm2(x)
        x = x + self.residual_dropout(self.feed_forward(feed_forward_input))
        return x


__all__ = ["FeedForwardNetwork", "TransformerBlock"]
