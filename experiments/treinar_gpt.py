"""Treina um GPT pequeno e registra o melhor checkpoint e as métricas."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint
from src.config import CORPUS_PATH, DEFAULT_SPRINT4_CONFIG, TrainingConfig
from src.generation import generate_text
from src.gpt import GPTModel
from src.training import fit, prepare_training_data


def write_report(path: Path, result: dict) -> None:
    """Gera um registro legível da execução, incluindo as limitações dos dados."""

    data = result["data"]
    lines = [
        "# Resultado do treinamento inicial do GPT", "",
        "Registro gerado por `experiments/treinar_gpt.py`. Este é um experimento",
        "adicional ao escopo da Sprint 4.", "",
        "## Configuração e dados", "",
        f"- Corpus: `{result['corpus']}`;",
        f"- Dispositivo: `{result['device']}`; Python {result['python_version']}; "
        f"PyTorch {result['torch_version']}; {result['threads']} thread(s);",
        f"- Tempo observado: {result['elapsed_seconds']:.2f} segundos;",
        f"- Parâmetros: {result['parameter_count']};",
        f"- Tokens: {data['train_tokens']} no treino e {data['validation_tokens']} "
        "na validação, separados antes da criação das janelas;",
        f"- Vocabulário: {data['vocab_size']} tokens, construído apenas no treino;",
        f"- Tokens desconhecidos na validação: {data['validation_unknown_tokens']} "
        f"({data['validation_unknown_rate']:.2%});",
        f"- SHA-256 do corpus: `{data['corpus_sha256']}`.", "",
        "Configuração do modelo:", "", "```json",
        json.dumps(result["model_config"], indent=2, ensure_ascii=False), "```", "",
        "Configuração do treinamento:", "", "```json",
        json.dumps(result["training_config"], indent=2, ensure_ascii=False), "```", "",
        "## Perdas observadas", "",
        "Cross-entropy média por token. As perdas são medidas com dropout desativado.",
        "O checkpoint escolhido é o de menor perda de validação, incluindo a época 0.",
        "", "| Época | Treino | Validação | Novo melhor checkpoint |",
        "|---:|---:|---:|---|",
        f"| 0 | {result['initial_train_loss']:.4f} | "
        f"{result['initial_validation_loss']:.4f} | Inicial |",
    ]
    for row in result["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | "
            f"{row['validation_loss']:.4f} | {'Sim' if row['saved_best'] else 'Não'} |"
        )
    lines.extend([
        "", f"Melhor época: **{result['best_epoch']}**. Perda de validação: "
        f"**{result['best_validation_loss']:.4f}**.", "",
        "## Geração antes e depois", "",
        "Mesma sequência inicial e seleção greedy nas duas execuções.", "",
        f"Prompt: `{result['prompt']}`", "", "Antes:", "", "```text",
        result["generation_before"], "```", "", "Depois (melhor checkpoint):", "",
        "```text", result["generation_after"], "```", "",
        "## Interpretação e limitações", "",
        "A queda da perda de treino demonstra ajuste aos exemplos apresentados.",
        "A validação usa um trecho separado e orienta a escolha do checkpoint e",
        "a parada antecipada. Se a perda de treino continuar caindo enquanto a",
        "validação piora, há sinal de sobreajuste.", "",
        "O corpus é curto e em inglês. O modelo pode aprender repetições e",
        "memorizar trechos; este resultado não o torna um assistente de conversa.",
        "Palavras exclusivas da validação viram `<|unk|>`, portanto parte da",
        "perda avalia esse marcador, e não a previsão da palavra original.", "",
        "O checkpoint binário contém os pesos, a configuração e o vocabulário.",
        "Ele fica no computador e não é versionado no Git. As métricas completas",
        "ficam no JSON local produzido pelo mesmo comando.", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    defaults = TrainingConfig()
    architecture = DEFAULT_SPRINT4_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--train-fraction", type=float, default=defaults.train_fraction)
    parser.add_argument("--stride", type=int, default=defaults.stride)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--context-length", type=int, default=architecture.context_length)
    parser.add_argument("--embedding-dim", type=int, default=architecture.embedding_dim)
    parser.add_argument("--num-heads", type=int, default=architecture.num_heads)
    parser.add_argument("--num-layers", type=int, default=architecture.num_layers)
    parser.add_argument("--feed-forward-dim", type=int, default=architecture.feed_forward_dim)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prompt", default="I HAD always")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "experiments" / "gpt_treinado.pt")
    parser.add_argument("--metrics", type=Path,
                        default=PROJECT_ROOT / "experiments" / "treinamento_metricas.json")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_ROOT / "docs" / "resultados_treinamento.md")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads deve ser maior que zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA não está disponível neste ambiente; use --device cpu")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    try:
        config = TrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, train_fraction=args.train_fraction,
            stride=args.stride, patience=args.patience, seed=args.seed,
        )
        if not args.prompt.strip():
            raise ValueError("--prompt não pode ser vazio")
        torch.set_num_threads(args.threads)
        torch.manual_seed(config.seed)
        text = args.corpus.read_text(encoding="utf-8")
        data = prepare_training_data(text, args.context_length, config)
        model = GPTModel(
            vocab_size=len(data.tokenizer.str_to_int), context_length=args.context_length,
            embedding_dim=args.embedding_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, feed_forward_dim=args.feed_forward_dim,
            dropout=args.dropout,
        ).to(device)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Dispositivo: {device}; parametros: {model.parameter_count()}", flush=True)
    print(
        f"Treino: {data.summary['train_tokens']} tokens; "
        f"validacao: {data.summary['validation_tokens']} tokens; "
        f"vocabulario: {data.summary['vocab_size']}", flush=True,
    )
    generation_before = generate_text(model, data.tokenizer, args.prompt, max_new_tokens=24)
    started = time.perf_counter()
    result = fit(
        model, data, config, args.checkpoint, device=device,
        log=lambda message: print(message, flush=True),
    )
    elapsed = time.perf_counter() - started
    loaded = load_checkpoint(args.checkpoint, device)
    generation_after = generate_text(
        loaded.model, loaded.tokenizer, args.prompt, max_new_tokens=24
    )
    result.update({
        "corpus": args.corpus.name, "device": device,
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "threads": args.threads, "parameter_count": model.parameter_count(),
        "elapsed_seconds": elapsed, "prompt": args.prompt,
        "generation_before": generation_before, "generation_after": generation_after,
    })
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    write_report(args.report, result)
    print(f"Melhor epoca: {loaded.epoch}; validacao: {loaded.validation_loss:.4f}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Relatorio: {args.report.resolve()}")
    print(f"Antes: {generation_before}")
    print(f"Depois: {generation_after}")


if __name__ == "__main__":
    main()
