"""Executa os experimentos exigidos pela Sprint 2.

Uso, a partir da raiz do repositório:

    .venv\\Scripts\\python.exe experiments\\sprint2_experimentos.py
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import CORPUS_PATH
from src.dataloader import create_dataloader_v1
from src.dataset import GPTDatasetV1
from src.embeddings import InputEmbedding
from src.tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text


RESULTS_PATH = ROOT / "docs" / "resultados_experimentos_sprint2.md"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def dataset_for(raw_text: str, tokenizer: SimpleTokenizerV2, context: int, stride: int):
    return GPTDatasetV1(raw_text, tokenizer, max_length=context, stride=stride)


def run_experiments() -> str:
    torch.manual_seed(123)
    raw_text = CORPUS_PATH.read_text(encoding="utf-8")
    corpus_tokens = tokenize_text(raw_text)
    vocabulary = build_vocab(corpus_tokens)
    tokenizer = SimpleTokenizerV2(vocabulary)
    token_ids = tokenizer.encode(raw_text)

    lines = [
        "# Resultados dos experimentos — Sprint 2",
        "",
        "Os experimentos foram executados com `torch.manual_seed(123)`, usando o "
        "corpus `the-verdict.txt`, o tokenizador simples do projeto e CPU.",
        "",
        "## 1. Estatísticas do corpus",
        "",
        markdown_table(
            ["Métrica", "Valor"],
            [
                ["Caracteres", len(raw_text)],
                ["Tokens produzidos", len(corpus_tokens)],
                ["Tamanho do vocabulário", len(vocabulary)],
                ["IDs codificados", len(token_ids)],
                ["Tokens especiais reservados", 2],
            ],
        ),
        "",
        "## 2. Tamanho do contexto e quantidade de amostras",
        "",
        "O stride foi mantido em 4 para permitir comparar diretamente o efeito do "
        "contexto. A quantidade de amostras segue `floor((N - (L + 1)) / S) + 1`.",
        "",
    ]

    context_rows = []
    for context_length in [4, 8, 16, 32]:
        stride = 4
        dataset = dataset_for(raw_text, tokenizer, context_length, stride)
        context_rows.append(
            [
                context_length,
                stride,
                len(dataset),
                len(create_dataloader_v1(
                    raw_text,
                    tokenizer,
                    batch_size=4,
                    max_length=context_length,
                    stride=stride,
                    shuffle=False,
                    drop_last=False,
                )),
            ]
        )
    lines.append(
        markdown_table(
            ["Context length", "Stride", "Amostras", "Lotes (batch 4)"],
            context_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 3. Tamanho do lote",
            "",
            "O contexto foi fixado em 8 e o stride em 4. `drop_last=False` foi "
            "usado para contabilizar também o último lote incompleto.",
            "",
        ]
    )

    batch_rows = []
    for batch_size in [2, 4, 8, 16]:
        loader = create_dataloader_v1(
            raw_text,
            tokenizer,
            batch_size=batch_size,
            max_length=8,
            stride=4,
            shuffle=False,
            drop_last=False,
        )
        inputs, targets = next(iter(loader))
        batch_rows.append(
            [batch_size, len(loader), tuple(inputs.shape), tuple(targets.shape)]
        )
    lines.append(
        markdown_table(
            ["Batch size", "Lotes", "Forma entrada", "Forma alvo"], batch_rows
        )
    )
    lines.extend(["", "## 4. Dimensão dos embeddings", ""])

    base_loader = create_dataloader_v1(
        raw_text,
        tokenizer,
        batch_size=4,
        max_length=8,
        stride=4,
        shuffle=False,
        drop_last=True,
    )
    inputs, _ = next(iter(base_loader))
    embedding_rows = []
    for embedding_dim in [16, 32, 64, 128]:
        layer = InputEmbedding(len(vocabulary), 8, embedding_dim)
        output = layer(inputs)
        embedding_rows.append(
            [
                embedding_dim,
                tuple(output.shape),
                output.numel(),
                f"{output.numel() * output.element_size() / 1024:.2f} KiB",
            ]
        )
    lines.extend(
        [
            "A entrada foi fixada em `batch_size=4` e `context_length=8`. O custo "
            "abaixo considera somente o tensor combinado em `float32`.",
            "",
            markdown_table(
                ["Embedding dim", "Forma combinada", "Elementos", "Memória"],
                embedding_rows,
            ),
        ]
    )

    lines.extend(["", "## 5. Textos e quantidade de tokens", ""])
    text_samples = [
        "Hello, do you like tea?",
        "In the sunlit terraces of the palace.",
        "Jack Gisburn painted a portrait.",
    ]
    text_rows = []
    for text in text_samples:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        unknown_id = vocabulary["<|unk|>"]
        unknown_count = sum(token_id == unknown_id for token_id in ids)
        text_rows.append([text, len(tokens), unknown_count, tokens])
    lines.append(
        markdown_table(
            ["Texto", "Tokens", "UNK", "Tokens produzidos"], text_rows
        )
    )

    lines.extend(
        [
            "",
            "## 6. Formas finais do pipeline",
            "",
            f"- Corpus codificado: `[1, {len(token_ids)}]` quando representado "
            "como uma sequência de IDs.",
            "- Lote de entrada: `[4, 8]`.",
            "- Lote de alvo: `[4, 8]`.",
            "- Embedding combinado com dimensão 64: `[4, 8, 64]`.",
            "",
            "## Reprodução",
            "",
            "```powershell",
            ".venv\\Scripts\\python.exe experiments\\sprint2_experimentos.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    report = run_experiments()
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"Resultados gravados em: {RESULTS_PATH}")
