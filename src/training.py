"""Treinamento supervisionado de próximo token com validação separada."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import SPECIAL_TOKENS, TrainingConfig
from .dataset import GPTDatasetV1, GPTDialogueDataset, collate_dialogues
from .gpt import GPTModel
from .tokenizer import SimpleTokenizerV2, build_vocab, normalize_text, tokenize_text


@dataclass
class TrainingData:
    tokenizer: SimpleTokenizerV2
    train_loader: DataLoader
    train_eval_loader: DataLoader
    validation_loader: DataLoader
    summary: dict


def prepare_training_data(
    text: str, context_length: int, config: TrainingConfig,
    *, normalization: str | None = None,
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    document_separator: str | None = None,
    task: str = "continuation",
) -> TrainingData:
    """Divide tokens contíguos antes das janelas e aprende o vocabulário no treino."""

    if context_length <= 0:
        raise ValueError("context_length deve ser maior que zero")
    tokens = tokenize_text(normalize_text(text, normalization), special_tokens)
    split = int(len(tokens) * config.train_fraction)
    if document_separator is not None:
        boundaries = [i + 1 for i, token in enumerate(tokens[:split])
                      if token == document_separator]
        if not boundaries:
            raise ValueError("Não há separador de documento antes da divisão de treino")
        split = boundaries[-1]
    if min(split, len(tokens) - split) < context_length + 1:
        raise ValueError(
            "Treino e validação precisam ter pelo menos context_length + 1 tokens cada"
        )
    train_tokens, validation_tokens = tokens[:split], tokens[split:]
    vocabulary = build_vocab(train_tokens, special_tokens)
    tokenizer = SimpleTokenizerV2(vocabulary, special_tokens, normalization)
    unknown_id = vocabulary[tokenizer.unknown_token]
    train_ids = [vocabulary[token] for token in train_tokens]
    validation_ids = [vocabulary.get(token, unknown_id) for token in validation_tokens]

    collate_fn = None
    if task == "dialogue":
        assistant_id = vocabulary.get("<|assistant|>")
        if assistant_id is None or document_separator != "<|endoftext|>":
            raise ValueError("Diálogos exigem marcadores de assistente e fim de texto")
        end_id = vocabulary["<|endoftext|>"]
        train_dataset = GPTDialogueDataset(train_ids, context_length, assistant_id, end_id)
        validation_dataset = GPTDialogueDataset(validation_ids, context_length, assistant_id, end_id)
        collate_fn = partial(collate_dialogues, pad_token_id=end_id)
    elif task == "continuation":
        train_dataset = GPTDatasetV1.from_token_ids(train_ids, context_length, config.stride)
        validation_dataset = GPTDatasetV1.from_token_ids(validation_ids, context_length, context_length)
    else:
        raise ValueError("task deve ser 'continuation' ou 'dialogue'")
    generator = torch.Generator().manual_seed(config.seed)
    return TrainingData(
        tokenizer=tokenizer,
        train_loader=DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            drop_last=False, generator=generator,
            collate_fn=collate_fn,
        ),
        train_eval_loader=DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn),
        validation_loader=DataLoader(validation_dataset, batch_size=config.batch_size, collate_fn=collate_fn),
        summary={
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "total_tokens": len(tokens),
            "train_tokens": len(train_ids),
            "validation_tokens": len(validation_ids),
            "vocab_size": len(vocabulary),
            "train_windows": len(train_dataset),
            "validation_windows": len(validation_dataset),
            "validation_unknown_tokens": sum(token == unknown_id for token in validation_ids),
            "validation_unknown_rate": sum(
                token == unknown_id for token in validation_ids
            ) / len(validation_ids),
            "document_separator": document_separator,
            "task": task,
            "loss_scope": "assistant_tokens" if task == "dialogue" else "all_tokens",
        },
    )


@torch.no_grad()
def evaluate_loss(
    model: GPTModel, loader: DataLoader, device: str | torch.device = "cpu"
) -> float:
    """Cross-entropy média por token, preservando o modo anterior do modelo."""

    was_training = model.training
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    try:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="sum"
            ).item()
            total_tokens += (targets != -100).sum().item()
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Não há tokens para calcular a perda")
    loss = total_loss / total_tokens
    if not math.isfinite(loss):
        raise RuntimeError("A perda de avaliação deixou de ser finita")
    return loss


def fit(
    model: GPTModel,
    data: TrainingData,
    config: TrainingConfig,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    log: Callable[[str], None] | None = print,
    metadata: dict | None = None,
) -> dict:
    """Executa AdamW, clipping de gradientes e seleção pela perda de validação.

    O checkpoint inicial também é um candidato: se nenhuma época melhorar a
    validação, os pesos iniciais continuam sendo o melhor resultado observado.
    """

    model.to(device)
    if model.config.vocab_size != len(data.tokenizer.str_to_int):
        raise ValueError("O vocabulário do modelo difere do vocabulário dos dados")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial_train_loss = evaluate_loss(model, data.train_eval_loader, device)
    initial_validation_loss = evaluate_loss(model, data.validation_loader, device)
    best_validation_loss = initial_validation_loss
    best_epoch = 0
    stale_epochs = 0
    history = []
    metadata = {**(metadata or {}), "training_config": asdict(config), "data": data.summary}
    save_checkpoint(
        checkpoint_path, model, data.tokenizer, epoch=0,
        validation_loss=best_validation_loss, metadata=metadata, optimizer=optimizer,
    )
    if log:
        log(
            f"Inicial: treino={initial_train_loss:.4f}, "
            f"validacao={initial_validation_loss:.4f}"
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        for inputs, targets in data.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError("A perda de treinamento deixou de ser finita")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()

        train_loss = evaluate_loss(model, data.train_eval_loader, device)
        validation_loss = evaluate_loss(model, data.validation_loader, device)
        improved = validation_loss < best_validation_loss
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss, "saved_best": improved})
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path, model, data.tokenizer, epoch=epoch,
                validation_loss=validation_loss, metadata=metadata, optimizer=optimizer,
            )
        else:
            stale_epochs += 1
        if log:
            log(
                f"Epoca {epoch:02d}/{config.epochs}: treino={train_loss:.4f}, "
                f"validacao={validation_loss:.4f}"
                + (" [melhor checkpoint salvo]" if improved else "")
            )
        if stale_epochs >= config.patience:
            if log:
                log(f"Parada antecipada: {config.patience} epocas sem melhora na validacao.")
            break

    return {
        "initial_train_loss": initial_train_loss,
        "initial_validation_loss": initial_validation_loss,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "completed_epochs": len(history),
        "history": history,
        "data": data.summary,
        "training_config": asdict(config),
        "model_config": asdict(model.config),
    }
