"""Embeddings de tokens e de posição para a entrada do modelo."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Converte Token IDs em vetores aprendíveis."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size deve ser maior que zero")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser maior que zero")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Fornece um vetor aprendível para cada posição do contexto."""

    def __init__(self, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length deve ser maior que zero")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve ser maior que zero")
        self.context_length = context_length
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                "A sequência possui mais posições que o context_length configurado"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        return self.embedding(positions)


class InputEmbedding(nn.Module):
    """Soma embeddings de token e posição para formar a entrada do modelo."""

    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEmbedding(context_length, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(token_ids)
        return token_vectors + position_vectors


__all__ = ["InputEmbedding", "PositionalEmbedding", "TokenEmbedding"]
