"""Persistência dos pesos, da configuração e do vocabulário do GPT."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .gpt import GPTConfig, GPTModel
from .tokenizer import SimpleTokenizerV2


@dataclass
class LoadedCheckpoint:
    model: GPTModel
    tokenizer: SimpleTokenizerV2
    epoch: int
    validation_loss: float
    metadata: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: GPTModel,
    tokenizer: SimpleTokenizerV2,
    *,
    epoch: int,
    validation_loss: float,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Salva atomicamente um checkpoint carregável com `weights_only=True`."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_config": asdict(model.config),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "vocabulary": dict(tokenizer.str_to_int),
        "special_tokens": list(tokenizer.special_tokens),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "metadata": metadata or {},
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(checkpoint, temporary_file)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> LoadedCheckpoint:
    """Reconstrói exatamente a arquitetura e o vocabulário salvos no treino."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Formato de checkpoint não reconhecido")
    config = GPTConfig(**checkpoint["model_config"])
    vocabulary = checkpoint["vocabulary"]
    if len(vocabulary) != config.vocab_size or set(vocabulary.values()) != set(
        range(config.vocab_size)
    ):
        raise ValueError("O vocabulário não corresponde à configuração salva")
    tokenizer = SimpleTokenizerV2(vocabulary, checkpoint["special_tokens"])
    model = GPTModel.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        epoch=checkpoint["epoch"],
        validation_loss=checkpoint["validation_loss"],
        metadata=checkpoint.get("metadata", {}),
    )
