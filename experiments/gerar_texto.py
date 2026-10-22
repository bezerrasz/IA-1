"""Gera continuações usando os pesos e o vocabulário de um checkpoint treinado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint
from src.config import LANGUAGE_PROFILES
from src.generation import generate_reply, generate_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(LANGUAGE_PROFILES), default="en")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--interactive", action="store_true",
                        help="Permite digitar várias frases; /sair encerra")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--sample", action="store_true",
                        help="Amostra tokens; sem esta opção usa greedy")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    profile = LANGUAGE_PROFILES[args.language]
    if args.checkpoint is None:
        args.checkpoint = profile["checkpoint"]
    if args.threads <= 0:
        parser.error("--threads deve ser maior que zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA não está disponível neste ambiente; use --device cpu")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch.set_num_threads(args.threads)
    if not args.checkpoint.is_file():
        parser.error(
            "Checkpoint não encontrado. Execute primeiro "
            f"experiments/treinar_gpt.py --language {args.language} ou informe --checkpoint."
        )
    try:
        loaded = load_checkpoint(args.checkpoint, device)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    torch.manual_seed(args.seed)
    language = loaded.metadata.get("language", args.language)
    if args.prompt is None:
        args.prompt = LANGUAGE_PROFILES.get(language, profile)["prompt"]
    dialogue = loaded.metadata.get("task") == "dialogue"
    if args.max_new_tokens < 0 or args.temperature <= 0:
        parser.error("max-new-tokens deve ser não negativo e temperature deve ser positiva")
    if args.top_k is not None and not 1 <= args.top_k <= loaded.model.config.vocab_size:
        parser.error("top-k deve estar entre 1 e o tamanho do vocabulário")
    print(
        f"Modelo carregado: {args.checkpoint.name}; epoca {loaded.epoch}; "
        f"validacao {loaded.validation_loss:.4f}; dispositivo {device}."
    )
    if dialogue:
        print("Modo: diálogo didático em português; as respostas são geradas pelo modelo.")

    def complete(prompt: str) -> None:
        if not prompt.strip():
            raise ValueError("O prompt não pode ser vazio")
        unknown_id = loaded.tokenizer.str_to_int[loaded.tokenizer.unknown_token]
        unknown_count = loaded.tokenizer.encode(prompt).count(unknown_id)
        if unknown_count:
            print(f"Aviso: {unknown_count} token(s) do prompt serao representados por <|unk|>.")
        generate = generate_reply if dialogue else generate_text
        response = generate(
            loaded.model, loaded.tokenizer, prompt,
            max_new_tokens=args.max_new_tokens, do_sample=args.sample,
            temperature=args.temperature, top_k=args.top_k,
        )
        print(response or "(O modelo encerrou a geração sem produzir uma resposta.)")

    if args.interactive:
        print("Digite uma pergunta." if dialogue else "Digite uma frase para continuar.")
        print("Use /sair para encerrar.")
        try:
            while True:
                prompt = input("Você> " if dialogue else "Frase> ")
                if prompt.strip().lower() == "/sair":
                    break
                if not prompt.strip():
                    continue
                complete(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrado.")
    else:
        try:
            complete(args.prompt)
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
