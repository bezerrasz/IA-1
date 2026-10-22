"""Dataset para pares de entrada e alvo do modelo autorregressivo."""

from __future__ import annotations

from typing import Protocol

import torch
from torch.utils.data import Dataset


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Cria janelas de contexto e alvos deslocados em um corpus."""

    def __init__(self, txt: str, tokenizer: Encoder, max_length: int, stride: int):
        self._build_windows(tokenizer.encode(txt), max_length, stride)

    @classmethod
    def from_token_ids(
        cls, token_ids: list[int], max_length: int, stride: int
    ) -> "GPTDatasetV1":
        """Cria janelas de IDs já codificados, sem tokenizar novamente.

        Permite separar treino e validação antes de construir janelas, evitando
        que uma amostra atravesse a fronteira entre os dois conjuntos.
        """

        dataset = cls.__new__(cls)
        dataset._build_windows(token_ids, max_length, stride)
        return dataset

    def _build_windows(
        self, token_ids: list[int], max_length: int, stride: int
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length deve ser maior que zero")
        if stride <= 0:
            raise ValueError("stride deve ser maior que zero")

        if len(token_ids) < max_length + 1:
            raise ValueError(
                "O texto precisa possuir pelo menos max_length + 1 tokens "
                "para formar entrada e alvo."
            )

        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        for start in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[start : start + max_length]
            target_chunk = token_ids[start + 1 : start + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


__all__ = ["GPTDatasetV1"]
