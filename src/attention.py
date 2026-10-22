"""Atenção multi-head causal usada pela arquitetura GPT da Sprint 4.

O módulo recebe representações com forma ``[B, T, D]`` e impede que uma
posição consulte tokens futuros. Ele também pode ser usado de forma isolada
nos experimentos da atenção da Sprint 3.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Calcula atenção escalada com várias cabeças e máscara causal."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in e d_out devem ser maiores que zero")
        if context_length <= 0:
            raise ValueError("context_length deve ser maior que zero")
        if num_heads <= 0:
            raise ValueError("num_heads deve ser maior que zero")
        if d_out % num_heads != 0:
            raise ValueError("d_out deve ser divisível por num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar no intervalo [0, 1)")

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.query = nn.Linear(d_in, d_out, bias=bias)
        self.key = nn.Linear(d_in, d_out, bias=bias)
        self.value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna a representação contextualizada com forma ``[B, T, D]``."""

        if x.ndim != 3:
            raise ValueError("x deve possuir forma [batch, sequence, features]")
        batch_size, sequence_length, feature_dim = x.shape
        if feature_dim != self.d_in:
            raise ValueError(
                f"A última dimensão de x deve ser {self.d_in}, recebeu {feature_dim}"
            )
        if sequence_length > self.context_length:
            raise ValueError(
                "A sequência excede o context_length configurado para a atenção"
            )

        queries = self.query(x).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        keys = self.key(x).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        values = self.value(x).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~mask, torch.finfo(x.dtype).min)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = attention_weights @ values
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_out
        )
        return self.out_proj(context)


__all__ = ["MultiHeadAttention"]
