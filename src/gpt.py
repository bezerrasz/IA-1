"""Arquitetura GPT simplificada construída com os componentes do projeto."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from .embeddings import InputEmbedding
    from .transformer import TransformerBlock
except ImportError:  # Permite importar o módulo diretamente a partir de src/.
    from embeddings import InputEmbedding
    from transformer import TransformerBlock


@dataclass(frozen=True)
class GPTConfig:
    """Hiperparâmetros necessários para construir um GPT pequeno."""

    vocab_size: int
    context_length: int
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feed_forward_dim: int | None = None
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.feed_forward_dim is None:
            object.__setattr__(self, "feed_forward_dim", 4 * self.embedding_dim)


class GPTModel(nn.Module):
    """Modelo GPT sem treinamento, com logits e geração autoregressiva.

    A entrada possui forma ``[B, T]`` e contém Token IDs. A saída do `forward`
    possui forma ``[B, T, V]`, com um logit para cada token do vocabulário em
    cada posição.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feed_forward_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.config = GPTConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            bias=bias,
        )
        if self.config.vocab_size <= 0:
            raise ValueError("vocab_size deve ser maior que zero")
        if self.config.context_length <= 0:
            raise ValueError("context_length deve ser maior que zero")
        if self.config.num_layers <= 0:
            raise ValueError("num_layers deve ser maior que zero")

        self.token_position_embedding = InputEmbedding(
            vocab_size=self.config.vocab_size,
            context_length=self.config.context_length,
            embedding_dim=self.config.embedding_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=self.config.embedding_dim,
                    num_heads=self.config.num_heads,
                    feed_forward_dim=self.config.feed_forward_dim,
                    context_length=self.config.context_length,
                    dropout=self.config.dropout,
                    bias=self.config.bias,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.embedding_dim)
        self.lm_head = nn.Linear(
            self.config.embedding_dim, self.config.vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: GPTConfig) -> "GPTModel":
        """Constrói o modelo a partir de um `GPTConfig` explícito."""

        return cls(**config.__dict__)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Processa Token IDs e retorna logits com forma `[B, T, V]`."""

        if token_ids.ndim != 2:
            raise ValueError("token_ids deve possuir forma [batch, sequence]")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError(
                "A sequência excede o context_length configurado para o GPT"
            )

        hidden_states = self.token_position_embedding(token_ids)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    def parameter_count(self, trainable_only: bool = True) -> int:
        """Retorna o número de parâmetros do modelo."""

        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """Gera tokens anexando uma previsão ao contexto a cada iteração.

        Quando `do_sample=False`, escolhe o maior logit (greedy decoding).
        Quando `do_sample=True`, aplica temperatura e, opcionalmente, mantém
        apenas os `top_k` logits antes de amostrar a distribuição resultante.
        O contexto usado pelo modelo é sempre limitado à janela mais recente.
        """

        if token_ids.ndim != 2:
            raise ValueError("token_ids deve possuir forma [batch, sequence]")
        if token_ids.shape[1] == 0:
            raise ValueError("a sequência inicial não pode ser vazia")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens não pode ser negativo")
        if temperature <= 0:
            raise ValueError("temperature deve ser maior que zero")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k deve estar entre 1 e vocab_size")

        was_training = self.training
        self.eval()
        generated = token_ids.clone()
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.context_length :]
            logits = self(context)[:, -1, :]

            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k, dim=-1)
                cutoff = top_values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            if do_sample:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)

        if was_training:
            self.train()
        return generated


__all__ = ["GPTConfig", "GPTModel"]
