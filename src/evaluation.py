"""Avaliação observável de respostas, separada da perda de próximo token."""

from __future__ import annotations

from .generation import generate_reply
from .tokenizer import normalize_text, tokenize_text


def answer_key(text: str) -> tuple[str, ...]:
    return tuple(tokenize_text(normalize_text(text, "nfc_lower"), split_math=True))


def evaluate_dialogues(model, tokenizer, cases: list[dict], max_new_tokens: int = 40) -> dict:
    """Compara geração greedy livre com os alvos do experimento sintético.

    Não fornece a resposta correta como contexto e não faz busca no corpus.
    A igualdade de tokens é um critério estrito, não uma medida geral de semântica.
    """

    rows = []
    for case in cases:
        question = case["question"]
        actual = generate_reply(model, tokenizer, question, max_new_tokens=max_new_tokens)
        unknown_id = tokenizer.str_to_int[tokenizer.unknown_token]
        rows.append({
            **case, "actual": actual,
            "correct": answer_key(actual) == answer_key(case["answer"]),
            "unknown_input_tokens": tokenizer.encode(question).count(unknown_id),
        })
    if not rows:
        raise ValueError("A avaliação precisa de pelo menos um exemplo")
    correct = sum(row["correct"] for row in rows)
    return {"cases": len(rows), "correct": correct, "exact_match": correct / len(rows),
            "rows": rows}
