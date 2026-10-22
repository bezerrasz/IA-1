"""Funções de alto nível para transformar prompts em texto gerado."""

from __future__ import annotations

from typing import Protocol

import torch

from .gpt import GPTModel


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


def generate_text(
    model: GPTModel,
    tokenizer: TextTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
) -> str:
    """Codifica um prompt, gera tokens e decodifica a sequência completa."""

    if not prompt.strip():
        raise ValueError("prompt não pode ser vazio")
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("o prompt não produziu tokens")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = model.generate(
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
    )
    return tokenizer.decode(generated_ids[0].tolist())


__all__ = ["generate_text"]
