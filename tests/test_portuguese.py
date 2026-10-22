import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import DIALOGUE_TOKENS, LANGUAGE_PROFILES, TrainingConfig
from src.generation import generate_reply
from src.gpt import GPTModel
from src.dataset import GPTDialogueDataset, collate_dialogues
from src.tokenizer import SimpleTokenizerV2, build_vocab, normalize_text, tokenize_text
from src.training import evaluate_loss, prepare_training_data


class PortugueseTests(unittest.TestCase):
    def make_tokenizer(self):
        text = "<|user|> olá, tudo bem? <|assistant|> olá! <|endoftext|>"
        return SimpleTokenizerV2(
            build_vocab(tokenize_text(text, DIALOGUE_TOKENS), DIALOGUE_TOKENS),
            DIALOGUE_TOKENS, normalization="nfc_lower",
        )

    def make_model(self, vocabulary_size):
        return GPTModel(vocab_size=vocabulary_size, context_length=16,
                        embedding_dim=8, num_heads=2, num_layers=1, feed_forward_dim=16)

    def test_portuguese_normalization_handles_case_and_decomposed_accents(self):
        tokenizer = self.make_tokenizer()
        self.assertEqual(tokenizer.encode("OLÁ, TUDO BEM?"), tokenizer.encode("ola\u0301, tudo bem?"))
        self.assertEqual(tokenizer.decode(tokenizer.encode("OLÁ, TUDO BEM?")), "olá, tudo bem?")
        self.assertNotIn(tokenizer.str_to_int["<|unk|>"], tokenizer.encode("OLÁ, TUDO BEM?"))

    def test_legacy_tokenizer_remains_case_sensitive(self):
        tokenizer = SimpleTokenizerV2(build_vocab(["Hello"]))
        self.assertEqual(normalize_text("Hello"), "Hello")
        self.assertNotEqual(tokenizer.encode("Hello"), tokenizer.encode("hello"))

    def test_role_markers_are_single_tokens(self):
        tokenizer = self.make_tokenizer()
        self.assertEqual(tokenizer.tokenize("<|user|> OLÁ <|assistant|>"),
                         ["<|user|>", "olá", "<|assistant|>"])

    def test_checkpoint_restores_normalization_roles_and_language(self):
        tokenizer = self.make_tokenizer()
        model = self.make_model(len(tokenizer.str_to_int))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portuguese.pt"
            save_checkpoint(path, model, tokenizer, epoch=1, validation_loss=1.0,
                            metadata={"language": "pt", "task": "dialogue"})
            loaded = load_checkpoint(path)
        self.assertEqual(loaded.tokenizer.encode("OLÁ!"), tokenizer.encode("olá!"))
        self.assertEqual(loaded.tokenizer.special_tokens, DIALOGUE_TOKENS)
        self.assertEqual(loaded.metadata["language"], "pt")

    def test_old_checkpoint_without_tokenizer_options_is_supported(self):
        tokenizer = SimpleTokenizerV2(build_vocab(["Hello"]))
        model = self.make_model(len(tokenizer.str_to_int))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, model, tokenizer, epoch=0, validation_loss=2.0)
            payload = torch.load(path, weights_only=True)
            del payload["tokenizer_options"]
            torch.save(payload, path)
            loaded = load_checkpoint(path)
        self.assertIsNone(loaded.tokenizer.normalization)
        self.assertEqual(loaded.tokenizer.encode("Hello"), tokenizer.encode("Hello"))

    def test_reply_uses_model_tokens_and_stops_at_end_marker(self):
        tokenizer = self.make_tokenizer()
        model = self.make_model(len(tokenizer.str_to_int))
        next_tokens = [tokenizer.str_to_int[token] for token in ("olá", "!", "<|endoftext|>")]

        def predict(ids):
            logits = torch.full((1, ids.shape[1], len(tokenizer.str_to_int)), -100.0)
            logits[:, -1, next_tokens.pop(0)] = 100.0
            return logits

        with patch.object(model, "forward", side_effect=predict) as forward:
            reply = generate_reply(model, tokenizer, "olá, tudo bem?", max_new_tokens=10)
        self.assertEqual(reply, "olá!")
        self.assertEqual(forward.call_count, 3)
        self.assertTrue(model.training)

    def test_batch_generation_finishes_each_sequence_independently(self):
        model = self.make_model(6)
        predictions = [torch.tensor([5, 2]), torch.tensor([1, 5])]

        def predict(ids):
            logits = torch.full((2, ids.shape[1], 6), -100.0)
            logits[torch.arange(2), -1, predictions.pop(0)] = 100.0
            return logits

        with patch.object(model, "forward", side_effect=predict):
            result = model.generate(torch.tensor([[0], [0]]), 10, eos_token_id=5)
        self.assertTrue(torch.equal(result, torch.tensor([[0, 5, 5], [0, 2, 5]])))

    def test_corpus_has_known_greetings_and_separates_complete_dialogues(self):
        profile = LANGUAGE_PROFILES["pt"]
        text = profile["corpus"].read_text(encoding="utf-8")
        data = prepare_training_data(
            text, context_length=64, config=TrainingConfig(),
            normalization=profile["normalization"], special_tokens=DIALOGUE_TOKENS,
            document_separator="<|endoftext|>",
            task="dialogue",
        )
        tokens = tokenize_text(normalize_text(text, "nfc_lower"), DIALOGUE_TOKENS)
        split = data.summary["train_tokens"]
        self.assertEqual(tokens[split - 1], "<|endoftext|>")
        self.assertEqual(tokens[split], "<|user|>")
        unknown = data.tokenizer.str_to_int["<|unk|>"]
        for prompt in ("olá, tudo bem?", "ola, tudo bem?", "O que você faz?"):
            self.assertNotIn(unknown, data.tokenizer.encode(prompt))
        # Não há perguntas iguais nos dois conjuntos após a normalização.
        train_questions = text.lower().split("<|endoftext|>")
        questions = [item.split("<|assistant|>")[0].strip() for item in train_questions if item.strip()]
        self.assertEqual(len(questions), len(set(questions)))

    def test_dialogue_masks_questions_and_right_padding(self):
        dataset = GPTDialogueDataset([0, 1, 2, 3, 4, 5, 0, 2, 4, 5],
                                     context_length=8, assistant_id=2, end_id=5)
        inputs, targets = collate_dialogues([dataset[0], dataset[1]], pad_token_id=5)
        self.assertTrue(torch.equal(targets[0], torch.tensor([-100, -100, 3, 4, 5])))
        self.assertTrue(torch.equal(targets[1], torch.tensor([-100, 4, 5, -100, -100])))
        self.assertEqual(tuple(inputs.shape), (2, 5))

    def test_evaluation_averages_only_unmasked_response_tokens(self):
        dataset = GPTDialogueDataset([0, 1, 2, 3, 4, 5, 0, 2, 4, 5],
                                     context_length=8, assistant_id=2, end_id=5)
        collate = partial(collate_dialogues, pad_token_id=5)
        inputs, targets = collate([dataset[0], dataset[1]])
        model = self.make_model(6).eval()
        with torch.no_grad():
            expected = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten()).item()
        measured = evaluate_loss(model, DataLoader(dataset, batch_size=1, collate_fn=collate))
        self.assertAlmostEqual(measured, expected, places=6)


if __name__ == "__main__":
    unittest.main()
