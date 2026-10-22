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
    eos_token_id: int | None = None,
    include_prompt: bool = True,
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
        eos_token_id=eos_token_id,
    )
    output_ids = generated_ids[0].tolist()
    continuation = output_ids[len(prompt_ids):]
    if eos_token_id is not None and eos_token_id in continuation:
        continuation = continuation[:continuation.index(eos_token_id)]
    return tokenizer.decode((prompt_ids if include_prompt else []) + continuation)


def generate_reply(model: GPTModel, tokenizer, question: str, **options) -> str:
    """Formata um diálogo e decodifica apenas os tokens produzidos pelo GPT."""

    if not question.strip():
        raise ValueError("A pergunta não pode ser vazia")
    for marker in ("<|user|>", "<|assistant|>"):
        if marker not in tokenizer.str_to_int:
            raise ValueError("O checkpoint não possui os marcadores de diálogo")
    prompt = f"<|user|> {question.strip()} <|assistant|>"
    reply = generate_text(
        model, tokenizer, prompt, eos_token_id=tokenizer.str_to_int["<|endoftext|>"],
        include_prompt=False, **options,
    )
    for marker in ("<|user|>", "<|assistant|>"):
        reply = reply.split(marker, 1)[0]
    return reply.strip()


__all__ = ["generate_reply", "generate_text"]
