"""Compara respostas livres com referências e registra limitações do domínio."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint
from src.config import LANGUAGE_PROFILES
from src.evaluation import evaluate_dialogues


# Casos observados na interação do usuário. Estão no novo treino, não são teste inédito.
REGRESSION_CASES = [
    {"question": "Oi", "answer": "olá! tudo bem. como posso ajudar?"},
    {"question": "O que você pode fazer?", "answer": "posso ajudar com explicações simples sobre estudos e o projeto."},
    {"question": "ok", "answer": "certo. pode perguntar."},
    {"question": "Quanto é 1+1?", "answer": "2."},
    {"question": "cara, fala direto", "answer": "certo. vou responder de forma curta."},
    {"question": "Como está?", "answer": "estou pronto para ajudar. como posso ajudar você?"},
    {"question": "olá, tudo bem?", "answer": "olá! tudo bem. como posso ajudar?"},
    {"question": "o que é um token?", "answer": "um token é uma unidade de texto, como uma palavra ou um sinal."},
]


def write_quality_report(path: Path, result: dict) -> None:
    test = result["test"]
    regression = result["regression"]
    sums = [row for row in test["rows"] if row.get("intent", "").startswith("soma_")]
    language = [row for row in test["rows"] if not row.get("intent", "").startswith("soma_")]
    lines = [
        "# Avaliação de respostas do modelo em português", "",
        f"Checkpoint: `{result['checkpoint']}`; época {result['epoch']}.", "",
        f"SHA-256 do conjunto de casos: `{result['cases_sha256']}`.", "",
        "Geração greedy livre, sem fornecer a resposta esperada ao modelo.",
        "A comparação normaliza caixa e separa pontuação e operadores.", "",
        f"- Regressão (perguntas vistas no treino): {regression['correct']}/{regression['cases']};",
        f"- Teste controlado: {test['correct']}/{test['cases']} "
        f"({test['exact_match']:.2%} de correspondência exata);", "",
        f"- Somas do teste: {sum(row['correct'] for row in sums)}/{len(sums)};",
        f"- Demais perguntas do teste: {sum(row['correct'] for row in language)}/{len(language)};", "",
        "A pontuação agregada é influenciada pelos exemplos de soma. A taxa nas",
        "demais perguntas é registrada separadamente para não ocultar dificuldades",
        "com variações de linguagem.", "",
        "## Perguntas observadas no terminal", "",
        "Estas entradas estão no treino e verificam reprodução dos exemplos.", "",
        "| Pergunta | Resposta gerada | Igual à referência |",
        "|---|---|---|",
    ]
    for row in regression["rows"]:
        lines.append(f"| {row['question']} | {row['actual']} | {'Sim' if row['correct'] else 'Não'} |")
    lines.extend([
        "", "## Limites da avaliação", "",
        "O teste contém perguntas diferentes das de treino e validação após",
        "tokenização, mas compartilha intenções, perguntas-base e respostas.",
        "Várias entradas são versões com prefixos diferentes de uma pergunta-base.",
        "Portanto, o resultado mede variações controladas desse corpus sintético,",
        "não capacidade de conversa geral nem generalização para assuntos novos.", "",
        "As somas de números de 0 a 9 foram incluídas como exemplos supervisionados.",
        "A inferência é feita apenas pelo GPT; acertar essas somas não demonstra",
        "um algoritmo de cálculo, e contas fora desse domínio podem falhar.", "",
        "Correspondência exata pode rejeitar uma paráfrase correta. Por outro lado,",
        "memorizar respostas pode produzir boa pontuação sem compreensão ampla.", "",
        "## Exemplos do teste controlado", "",
        "| Pergunta | Referência | Resposta gerada | Igual |",
        "|---|---|---|---|",
    ])
    for row in test["rows"][:30]:
        lines.append(f"| {row['question']} | {row['answer']} | {row['actual']} | "
                     f"{'Sim' if row['correct'] else 'Não'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=LANGUAGE_PROFILES["pt"]["checkpoint"])
    parser.add_argument("--cases", type=Path,
                        default=PROJECT_ROOT / "experiments" / "avaliacao_portugues_casos.json")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_ROOT / "docs" / "avaliacao_portugues.md")
    parser.add_argument("--metrics", type=Path,
                        default=PROJECT_ROOT / "experiments" / "avaliacao_portugues_resultado.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    loaded = load_checkpoint(args.checkpoint)
    cases_text = args.cases.read_text(encoding="utf-8")
    cases = json.loads(cases_text)
    regression = evaluate_dialogues(loaded.model, loaded.tokenizer, REGRESSION_CASES)
    print(f"Regressao: {regression['correct']}/{regression['cases']}", flush=True)
    test = evaluate_dialogues(loaded.model, loaded.tokenizer, cases)
    result = {"checkpoint": args.checkpoint.name, "epoch": loaded.epoch,
              "regression": regression, "test": test,
              "cases_sha256": hashlib.sha256(cases_text.encode("utf-8")).hexdigest()}
    args.metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_quality_report(args.report, result)
    print(f"Teste controlado: {test['correct']}/{test['cases']} ({test['exact_match']:.2%})")
    for row in regression["rows"]:
        print(f"{row['question']} -> {row['actual']}")


if __name__ == "__main__":
    main()
