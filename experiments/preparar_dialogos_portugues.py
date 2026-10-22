"""Prepara um corpus sintético controlado, com perguntas distintas entre divisões."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DIALOGUE_TOKENS
from src.tokenizer import normalize_text, tokenize_text


# Respostas abaixo são ALVOS DE TREINO. Não são importadas pela geração.
# Cada grupo possui resposta, perguntas de treino, de validação e de teste.
INTENTS = [
    ("cumprimento", "olá! tudo bem. como posso ajudar?",
     ("oi", "olá", "ola, tudo bem?", "olá, tudo bem?", "bom dia"),
     ("oi, como vai?", "olá, como você está?"),
     ("oi, tudo bem?", "bom dia, tudo bem?")),
    ("agradecimento", "de nada! espero ter ajudado.",
     ("obrigado", "obrigada", "muito obrigado", "valeu"),
     ("obrigado pela ajuda",), ("muito obrigada pela ajuda",)),
    ("confirmacao", "certo. pode perguntar.",
     ("ok", "certo", "entendi", "beleza"),
     ("ok, entendi",), ("certo, obrigado",)),
    ("identidade", "sou um modelo de linguagem didático do projeto da faculdade.",
     ("quem é você?", "qual é seu nome?", "quem é voce?", "como você se chama?"),
     ("pode dizer quem você é?",), ("me conte quem é você",)),
    ("capacidade", "posso ajudar com explicações simples sobre estudos e o projeto.",
     ("o que você pode fazer?", "o que você faz?", "como pode ajudar?", "o que voce faz?"),
     ("qual ajuda você pode oferecer?",), ("como você pode me ajudar com o projeto?",)),
    ("estado", "estou pronto para ajudar. como posso ajudar você?",
     ("como está?", "como você está?", "como voce esta?", "tudo bem com você?"),
     ("como você está hoje?",), ("como está você hoje?",)),
    ("direto", "certo. vou responder de forma curta.",
     ("cara, fala direto", "fale direto", "seja direto", "responda de forma curta"),
     ("pode responder de forma curta?",), ("quero uma resposta curta",)),
    ("despedida", "até logo! bons estudos.",
     ("tchau", "até logo", "até mais", "vou sair"),
     ("tchau, obrigado",), ("até logo, obrigado",)),
    ("idioma", "sim. posso gerar pequenas respostas em português.",
     ("você fala português?", "voce fala portugues?", "fale em português", "responda em português"),
     ("pode falar em português?",), ("você pode me responder em português?",)),
    ("ajuda", "escreva uma pergunta sobre estudos ou sobre o projeto.",
     ("pode me ajudar?", "preciso de ajuda", "me ajude", "você pode me ajudar?"),
     ("eu preciso de ajuda",), ("quero ajuda com meus estudos",)),
    ("token", "um token é uma unidade de texto, como uma palavra ou um sinal.",
     ("o que é um token?", "explique token", "o que são tokens?", "defina token"),
     ("pode explicar o que é um token?",), ("qual é a definição de token?",)),
    ("treinamento", "treinar é ajustar os pesos usando exemplos para reduzir o erro.",
     ("o que é treinamento?", "como funciona o treinamento?", "o que significa treinar?", "explique treinamento"),
     ("pode explicar o treinamento?",), ("como o modelo é treinado?",)),
    ("atencao", "a atenção combina informações dos tokens do contexto.",
     ("o que é atenção?", "explique atenção", "para que serve a atenção?", "como funciona a atenção?"),
     ("pode explicar a atenção?",), ("qual é a função da atenção?",)),
    ("checkpoint", "um checkpoint salva os pesos, a configuração e o vocabulário.",
     ("o que é um checkpoint?", "explique checkpoint", "para que serve o checkpoint?", "o que um checkpoint salva?"),
     ("pode explicar o checkpoint?",), ("o que fica salvo no checkpoint?",)),
    ("epoca", "uma época é uma passagem pelas amostras de treinamento.",
     ("o que é uma época?", "explique época", "o que significa época?", "o que são épocas?"),
     ("pode explicar uma época?",), ("o que é uma época de treinamento?",)),
    ("limites", "sou um modelo pequeno e posso errar. confira as respostas.",
     ("você sabe tudo?", "você pode errar?", "quais são suas limitações?", "suas respostas são sempre corretas?"),
     ("você sabe todas as respostas?",), ("posso confiar em todas as suas respostas?",)),
]

TRAIN_PREFIXES = ("", "por favor, ", "me diga: ", "minha dúvida: ")
VALIDATION_PREFIX = "minha pergunta: "
TEST_PREFIX = "por favor, responda: "


def question_key(question: str) -> tuple[str, ...]:
    return tuple(tokenize_text(normalize_text(question, "nfc_lower"),
                               DIALOGUE_TOKENS, split_math=True))


def prepare_records(base_text: str) -> dict[str, list[dict]]:
    """Expande perguntas de treino e mantém contextos de avaliação distintos.

    Os três conjuntos compartilham intenções e respostas. A avaliação mede
    consistência em variações controladas, não conhecimento fora desse domínio.
    """

    groups = list(INTENTS)
    manual_keys = {question_key(question) for _, _, *parts in groups
                   for part in parts for question in part}
    for number, document in enumerate(base_text.split("<|endoftext|>")):
        if not document.strip():
            continue
        question, answer = document.split("<|assistant|>", 1)
        question = question.split("<|user|>", 1)[1].strip().lower()
        answer = answer.strip().lower()
        if question_key(question) not in manual_keys:
            groups.append((f"base_{number}", answer, (question,), (), ()))

    # A soma é usada apenas para criar rótulos supervisionados, nunca na inferência.
    for left in range(10):
        for right in range(10):
            groups.append((
                f"soma_{left}_{right}", f"{left + right}.",
                (f"quanto é {left}+{right}?", f"some {left} e {right}"), (), (),
            ))

    records = {"treino": [], "validacao": [], "teste": []}
    seen = {}

    def add(split, intent, question, answer):
        key = question_key(question)
        if key in seen:
            if seen[key] != (split, answer):
                raise ValueError(f"Pergunta duplicada ou ambígua: {question}")
            return
        seen[key] = (split, answer)
        records[split].append({"intent": intent, "question": question, "answer": answer})

    for intent, answer, training, validation, testing in groups:
        for question in training:
            for prefix in TRAIN_PREFIXES:
                add("treino", intent, prefix + question, answer)
        for question in validation:
            add("validacao", intent, question, answer)
        add("validacao", intent, VALIDATION_PREFIX + training[0], answer)
        for question in testing:
            add("teste", intent, question, answer)
        add("teste", intent, TEST_PREFIX + training[0], answer)

    for split, rows in records.items():
        random.Random(123).shuffle(rows)
    return records


def render_corpus(records: list[dict]) -> str:
    return "\n\n".join(
        f"<|user|> {row['question']}\n<|assistant|> {row['answer']} <|endoftext|>"
        for row in records
    ) + "\n"


def main() -> None:
    base = (PROJECT_ROOT / "corpus_portugues.txt").read_text(encoding="utf-8")
    records = prepare_records(base)
    for split, rows in records.items():
        path = PROJECT_ROOT / f"corpus_dialogos_pt_{split}.txt"
        path.write_text(render_corpus(rows), encoding="utf-8")
        print(f"{split}: {len(rows)} dialogos -> {path.name}")
    cases_path = PROJECT_ROOT / "experiments" / "avaliacao_portugues_casos.json"
    cases_path.write_text(json.dumps(records["teste"], ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


if __name__ == "__main__":
    main()
