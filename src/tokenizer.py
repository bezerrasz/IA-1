"""Tokenização simples e vocabulário do projeto.

O tokenizador segue a abordagem didática do Capítulo 2: palavras e símbolos
de pontuação são separados por expressão regular. Tokens especiais são
reconhecidos antes dessa separação para permanecerem como unidades únicas.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

try:
    from .config import SPECIAL_TOKENS
except ImportError:  # Permite importar o módulo diretamente a partir de src/.
    from config import SPECIAL_TOKENS


_PUNCTUATION_PATTERN = r"[,.:;?_!\"()']|--|\s"


def _build_split_pattern(special_tokens: Iterable[str]) -> re.Pattern[str]:
    escaped_special_tokens = sorted(
        (re.escape(token) for token in special_tokens),
        key=len,
        reverse=True,
    )
    special_pattern = "|".join(escaped_special_tokens)
    return re.compile(f"({special_pattern}|{_PUNCTUATION_PATTERN})")


def tokenize_text(text: str, special_tokens: Iterable[str] = SPECIAL_TOKENS) -> list[str]:
    """Divide texto em tokens, preservando tokens especiais."""

    if not isinstance(text, str):
        raise TypeError("text deve ser uma string")

    split_pattern = _build_split_pattern(special_tokens)
    pieces = split_pattern.split(text)
    return [piece.strip() for piece in pieces if piece and piece.strip()]


def build_vocab(
    tokens: Iterable[str], special_tokens: Iterable[str] = SPECIAL_TOKENS
) -> dict[str, int]:
    """Cria um vocabulário determinístico e adiciona tokens especiais."""

    vocabulary_tokens = sorted(set(tokens))
    for special_token in special_tokens:
        if special_token not in vocabulary_tokens:
            vocabulary_tokens.append(special_token)
    return {token: token_id for token_id, token in enumerate(vocabulary_tokens)}


class SimpleTokenizerV2:
    """Converte texto em tokens/IDs e IDs de volta em texto."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        special_tokens: Iterable[str] = SPECIAL_TOKENS,
    ) -> None:
        self.str_to_int = dict(vocab)
        self.int_to_str = {token_id: token for token, token_id in vocab.items()}
        self.special_tokens = tuple(special_tokens)
        self.unknown_token = "<|unk|>"

        if self.unknown_token not in self.str_to_int:
            raise ValueError("O vocabulário deve conter <|unk|>")
        if len(self.str_to_int) != len(self.int_to_str):
            raise ValueError("Cada token deve possuir um ID único")

    def tokenize(self, text: str) -> list[str]:
        """Retorna os tokens textuais, sem convertê-los em IDs."""

        return tokenize_text(text, self.special_tokens)

    def encode(self, text: str) -> list[int]:
        """Converte texto em Token IDs, usando UNK para tokens desconhecidos."""

        tokens = self.tokenize(text)
        return [
            self.str_to_int.get(token, self.str_to_int[self.unknown_token])
            for token in tokens
        ]

    def decode(self, ids: Iterable[int]) -> str:
        """Reconstrói uma representação textual a partir de Token IDs."""

        tokens = []
        for token_id in ids:
            if token_id not in self.int_to_str:
                raise KeyError(f"Token ID desconhecido: {token_id}")
            tokens.append(self.int_to_str[token_id])

        text = " ".join(tokens)
        text = re.sub(r"\s+([,.:;?!\"')])", r"\1", text)
        text = re.sub(r"([('(])\s+", r"\1", text)
        return text


__all__ = ["SimpleTokenizerV2", "build_vocab", "tokenize_text"]
