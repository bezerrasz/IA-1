"""Executa os experimentos de arquitetura e geração da Sprint 4."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CORPUS_PATH, DEFAULT_SPRINT4_CONFIG  # noqa: E402
from src.generation import generate_text  # noqa: E402
from src.gpt import GPTModel  # noqa: E402
from src.tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text  # noqa: E402


ARCHITECTURE_EXPERIMENTS = [
    {
        "name": "base",
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "feed_forward_dim": 64,
        "context_length": 16,
    },
    {
        "name": "mais_blocos",
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 4,
        "feed_forward_dim": 64,
        "context_length": 16,
    },
    {
        "name": "mais_heads",
        "embedding_dim": 32,
        "num_heads": 8,
        "num_layers": 2,
        "feed_forward_dim": 64,
        "context_length": 16,
    },
    {
        "name": "embedding_maior",
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "feed_forward_dim": 128,
        "context_length": 16,
    },
    {
        "name": "ffn_maior",
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "feed_forward_dim": 128,
        "context_length": 16,
    },
    {
        "name": "contexto_maior",
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "feed_forward_dim": 64,
        "context_length": 32,
    },
]


def _load_tokenizer() -> tuple[str, SimpleTokenizerV2]:
    corpus = CORPUS_PATH.read_text(encoding="utf-8")
    vocabulary = build_vocab(tokenize_text(corpus))
    return corpus, SimpleTokenizerV2(vocabulary)


def _prompt_ids(tokenizer: SimpleTokenizerV2, context_length: int) -> torch.Tensor:
    ids = tokenizer.encode("The court")
    if not ids:
        raise RuntimeError("O prompt de experimento não produziu Token IDs")
    return torch.tensor([ids[-context_length:]], dtype=torch.long)


def _measurement_ids(
    tokenizer: SimpleTokenizerV2, corpus: str, context_length: int
) -> torch.Tensor:
    ids = tokenizer.encode(corpus)[:context_length]
    if len(ids) < context_length:
        raise RuntimeError("O corpus não possui tokens suficientes para a medição")
    return torch.tensor([ids], dtype=torch.long)


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _measure_forward(model: GPTModel, token_ids: torch.Tensor) -> tuple[float, torch.Size]:
    model.eval()
    with torch.no_grad():
        model(token_ids)
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            logits = model(token_ids)
            durations.append((time.perf_counter() - start) * 1000)
    return statistics.mean(durations), logits.shape


def run() -> str:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    corpus, tokenizer = _load_tokenizer()
    vocab_size = len(tokenizer.str_to_int)
    rows = []

    for settings in ARCHITECTURE_EXPERIMENTS:
        torch.manual_seed(123)
        model = GPTModel(
            vocab_size=vocab_size,
            context_length=settings["context_length"],
            embedding_dim=settings["embedding_dim"],
            num_heads=settings["num_heads"],
            num_layers=settings["num_layers"],
            feed_forward_dim=settings["feed_forward_dim"],
            dropout=DEFAULT_SPRINT4_CONFIG.dropout,
        )
        token_ids = _measurement_ids(tokenizer, corpus, settings["context_length"])
        elapsed_ms, logits_shape = _measure_forward(model, token_ids)
        score_elements = (
            token_ids.shape[0]
            * settings["num_heads"]
            * token_ids.shape[1]
            * token_ids.shape[1]
        )
        generated = generate_text(
            model,
            tokenizer,
            "The court",
            max_new_tokens=DEFAULT_SPRINT4_CONFIG.max_new_tokens,
        )
        rows.append(
            {
                **settings,
                "parameters": model.parameter_count(),
                "score_elements": score_elements,
                "forward_ms": elapsed_ms,
                "logits_shape": tuple(logits_shape),
                "generated": generated,
            }
        )

    torch.manual_seed(321)
    baseline = GPTModel(
        vocab_size=vocab_size,
        context_length=DEFAULT_SPRINT4_CONFIG.context_length,
        embedding_dim=DEFAULT_SPRINT4_CONFIG.embedding_dim,
        num_heads=DEFAULT_SPRINT4_CONFIG.num_heads,
        num_layers=DEFAULT_SPRINT4_CONFIG.num_layers,
        feed_forward_dim=DEFAULT_SPRINT4_CONFIG.feed_forward_dim,
        dropout=DEFAULT_SPRINT4_CONFIG.dropout,
    )
    generation_rows = []
    for name, temperature, top_k, do_sample in (
        ("greedy", 1.0, None, False),
        ("top_k_temperatura_baixa", 0.7, 10, True),
        ("top_k_temperatura_alta", 1.3, 10, True),
    ):
        torch.manual_seed(987)
        generated = generate_text(
            baseline,
            tokenizer,
            "The court",
            max_new_tokens=DEFAULT_SPRINT4_CONFIG.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )
        generation_rows.append(
            {
                "name": name,
                "temperature": temperature,
                "top_k": top_k if top_k is not None else "—",
                "strategy": "amostragem" if do_sample else "greedy",
                "generated": generated,
            }
        )

    lines = [
        "# Resultados dos experimentos — Sprint 4",
        "",
        "## Ambiente e referência",
        "",
        f"- Corpus: `the-verdict.txt` ({_format_int(len(corpus))} caracteres);",
        f"- Vocabulário: {_format_int(vocab_size)} tokens;",
        "- Pesos: inicialização aleatória controlada por `torch.manual_seed`;",
        "- Medição: CPU, três execuções após uma execução de aquecimento;",
        "- O tempo é indicativo e deve ser comparado dentro do mesmo ambiente.",
        "",
        "## Experimentos de arquitetura",
        "",
        "A coluna `score_elements` estima o tamanho do tensor de scores de atenção "
        "para o prompt usado (`B × H × T × T`). A coluna de parâmetros inclui "
        "embeddings, blocos, normalização e LM Head.",
        "",
        "| Configuração | D | H | Blocos | F | Contexto | Parâmetros | Scores | Forward (ms) | Logits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['embedding_dim']} | {row['num_heads']} | "
            f"{row['num_layers']} | {row['feed_forward_dim']} | {row['context_length']} | "
            f"{_format_int(row['parameters'])} | {row['score_elements']} | "
            f"{row['forward_ms']:.3f} | `{row['logits_shape']}` |"
        )
    lines.extend(
        [
            "",
            "### Saídas observadas",
            "",
            "As continuações abaixo são diagnósticas: o modelo ainda não foi treinado.",
            "",
            "| Configuração | Continuação para `The court` |",
            "|---|---|",
        ]
    )
    for row in rows:
        lines.append(f"| {row['name']} | `{row['generated']}` |")

    lines.extend(
        [
            "",
            "## Experimentos de geração",
            "",
            "Todos usam a mesma arquitetura base, o mesmo prompt e `max_new_tokens=4`.",
            "",
            "| Estratégia | Temperatura | Top-k | Resultado |",
            "|---|---:|---:|---|",
        ]
    )
    for row in generation_rows:
        lines.append(
            f"| {row['strategy']} ({row['name']}) | {row['temperature']} | "
            f"{row['top_k']} | `{row['generated']}` |"
        )

    output = "\n".join(lines) + "\n"
    report_path = PROJECT_ROOT / "docs" / "resultados_experimentos_sprint4.md"
    report_path.write_text(output, encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    run()
