import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.preparar_dialogos_portugues import prepare_records, question_key
from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import DIALOGUE_TOKENS, LANGUAGE_PROFILES, TrainingConfig
from src.evaluation import answer_key, evaluate_dialogues
from src.gpt import GPTModel
from src.tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text
from src.training import prepare_training_data


class DialogueQualityTests(unittest.TestCase):
    def test_arithmetic_tokenization_is_optional_and_ignores_spacing(self):
        text = "quanto é 1+1? 2. <|user|> <|assistant|>"
        tokens = tokenize_text(text, DIALOGUE_TOKENS, split_math=True)
        tokenizer = SimpleTokenizerV2(build_vocab(tokens, DIALOGUE_TOKENS), DIALOGUE_TOKENS,
                                      normalization="nfc_lower", split_math=True)
        self.assertEqual(tokenizer.encode("Quanto é 1+1?"), tokenizer.encode("quanto é 1 + 1?"))
        self.assertNotIn(tokenizer.str_to_int["<|unk|>"], tokenizer.encode("Quanto é 1+1?"))
        self.assertIn("1+1", tokenize_text("1+1"))

    def test_split_math_survives_checkpoint_loading(self):
        tokenizer = SimpleTokenizerV2(build_vocab(["1", "+", "2"]), split_math=True)
        model = GPTModel(vocab_size=len(tokenizer.str_to_int), context_length=4,
                         embedding_dim=8, num_heads=2, num_layers=1, feed_forward_dim=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, tokenizer, epoch=1, validation_loss=1.0)
            loaded = load_checkpoint(path)
        self.assertEqual(loaded.tokenizer.encode("1+1"), tokenizer.encode("1 + 1"))
        self.assertTrue(loaded.tokenizer.split_math)

    def test_external_validation_does_not_enter_the_training_vocabulary(self):
        train = "<|user|> aaa <|assistant|> bbb bbb <|endoftext|> " * 8
        validation = "<|user|> ccc <|assistant|> ddd ddd <|endoftext|> " * 4
        data = prepare_training_data(
            train, context_length=8, config=TrainingConfig(), validation_text=validation,
            special_tokens=DIALOGUE_TOKENS, task="dialogue", document_separator="<|endoftext|>",
        )
        self.assertEqual(data.summary["train_windows"], 8)
        self.assertEqual(data.summary["validation_windows"], 4)
        self.assertNotIn("ccc", data.tokenizer.str_to_int)
        self.assertNotIn("ddd", data.tokenizer.str_to_int)
        self.assertEqual(data.summary["split_strategy"], "explicit_validation")

    def test_corpus_generator_is_reproducible_and_splits_have_distinct_questions(self):
        base = Path("corpus_portugues.txt").read_text(encoding="utf-8")
        records = prepare_records(base)
        self.assertEqual(records, prepare_records(base))
        keys = {split: {question_key(row["question"]) for row in rows}
                for split, rows in records.items()}
        self.assertFalse(keys["treino"] & keys["validacao"])
        self.assertFalse(keys["treino"] & keys["teste"])
        self.assertFalse(keys["validacao"] & keys["teste"])
        self.assertIn(question_key("oi"), keys["treino"])
        self.assertIn(question_key("quanto é 1+1?"), keys["treino"])

    def test_duplicate_questions_in_external_validation_are_rejected(self):
        text = "<|user|> aaa <|assistant|> bbb bbb <|endoftext|> " * 8
        with self.assertRaisesRegex(ValueError, "Perguntas iguais"):
            prepare_training_data(
                text, context_length=8, config=TrainingConfig(), validation_text=text,
                special_tokens=DIALOGUE_TOKENS, task="dialogue", document_separator="<|endoftext|>",
            )

    def test_quality_metric_uses_generated_answers_without_revealing_the_reference(self):
        tokenizer = SimpleTokenizerV2(build_vocab(["oi", "token"]))
        cases = [{"question": "oi", "answer": "olá!"},
                 {"question": "token", "answer": "unidade de texto."}]
        with patch("src.evaluation.generate_reply", side_effect=["olá!", "de nada!"]) as generate:
            result = evaluate_dialogues(object(), tokenizer, cases)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["exact_match"], 0.5)
        self.assertEqual(generate.call_args_list[0].args[2], "oi")
        self.assertEqual(generate.call_args_list[1].args[2], "token")
        self.assertEqual(answer_key("1+1=2."), answer_key("1 + 1 = 2."))
