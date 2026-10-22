"""Dataset para pares de entrada e alvo do modelo autorregressivo."""

from __future__ import annotations

from typing import Protocol

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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


class GPTDialogueDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Um diálogo por amostra; somente a resposta contribui para a perda."""

    def __init__(
        self, token_ids: list[int], context_length: int,
        assistant_id: int, end_id: int,
    ) -> None:
        self.samples = []
        start = 0
        for index, token_id in enumerate(token_ids):
            if token_id != end_id:
                continue
            document = token_ids[start:index + 1]
            start = index + 1
            if assistant_id not in document:
                raise ValueError("Todo diálogo precisa do marcador de assistente")
            assistant_position = document.index(assistant_id)
            if assistant_position >= len(document) - 2:
                raise ValueError("Todo diálogo precisa de uma resposta antes do fim")
            if len(document) - 1 > context_length:
                raise ValueError("Um diálogo excede o contexto; aumente --context-length")
            inputs = torch.tensor(document[:-1], dtype=torch.long)
            targets = torch.tensor(document[1:], dtype=torch.long)
            targets[:assistant_position] = -100
            self.samples.append((inputs, targets))
        if start != len(token_ids):
            raise ValueError("O corpus de diálogo precisa terminar com o marcador de fim")
        if not self.samples:
            raise ValueError("Não há diálogos completos no conjunto")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_dialogues(batch, pad_token_id: int):
    """Preenchimento à direita; os alvos de padding são ignorados pela perda."""

    inputs, targets = zip(*batch)
    return (
        pad_sequence(inputs, batch_first=True, padding_value=pad_token_id),
        pad_sequence(targets, batch_first=True, padding_value=-100),
    )


__all__ = ["GPTDatasetV1", "GPTDialogueDataset", "collate_dialogues"]
