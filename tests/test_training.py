import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import TrainingConfig
from src.dataset import GPTDatasetV1
from src.gpt import GPTModel
from src.tokenizer import SimpleTokenizerV2
from src.training import evaluate_loss, fit, prepare_training_data


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(123)

    def make_model(self, vocab_size=6, context_length=4):
        return GPTModel(
            vocab_size=vocab_size, context_length=context_length,
            embedding_dim=16, num_heads=2, num_layers=1,
            feed_forward_dim=32, dropout=0.0,
        )

    def test_split_and_vocabulary_do_not_read_validation_tokens(self):
        config = TrainingConfig(train_fraction=0.5, stride=2)
        data = prepare_training_data(
            " ".join(f"word{i}" for i in range(40)), context_length=3, config=config
        )
        self.assertEqual(data.summary["train_tokens"], 20)
        self.assertEqual(data.summary["validation_tokens"], 20)
        self.assertIn("word19", data.tokenizer.str_to_int)
        self.assertNotIn("word20", data.tokenizer.str_to_int)
        self.assertNotIn("word39", data.tokenizer.str_to_int)
        self.assertEqual(data.summary["validation_unknown_rate"], 1.0)
        unknown_id = data.tokenizer.str_to_int["<|unk|>"]
        for inputs, targets in data.validation_loader:
            self.assertTrue(torch.all(inputs == unknown_id))
            self.assertTrue(torch.all(targets == unknown_id))

    def test_pretokenized_dataset_preserves_shift_and_stride(self):
        dataset = GPTDatasetV1.from_token_ids(list(range(10)), max_length=3, stride=3)
        self.assertEqual(len(dataset), 3)
        inputs, targets = dataset[2]
        self.assertTrue(torch.equal(inputs, torch.tensor([6, 7, 8])))
        self.assertTrue(torch.equal(targets, torch.tensor([7, 8, 9])))

    def test_evaluation_weights_tokens_in_an_incomplete_batch(self):
        dataset = GPTDatasetV1.from_token_ids(
            list(range(6)) * 3, max_length=3, stride=3
        )
        loader = DataLoader(dataset, batch_size=3)
        model = self.make_model(context_length=3)
        inputs = torch.stack(dataset.input_ids)
        targets = torch.stack(dataset.target_ids)
        with torch.no_grad():
            expected = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten()).item()
        measured = evaluate_loss(model, loader)
        self.assertAlmostEqual(measured, expected, places=6)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_training_learns_a_predictable_sequence(self):
        config = TrainingConfig(epochs=4, learning_rate=0.02, stride=4, patience=4)
        data = prepare_training_data("a b c d " * 64, context_length=4, config=config)
        model = self.make_model(len(data.tokenizer.str_to_int))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            result = fit(model, data, config, path, log=None)
            checkpoint = load_checkpoint(path)
            measured = evaluate_loss(checkpoint.model, data.validation_loader)
        self.assertGreater(result["best_epoch"], 0)
        self.assertLess(result["best_validation_loss"], result["initial_validation_loss"] * 0.5)
        self.assertAlmostEqual(measured, result["best_validation_loss"], places=6)

    def test_checkpoint_restores_exact_logits_and_vocabulary(self):
        vocabulary = {"a": 0, "b": 1, "c": 2, "d": 3,
                      "<|unk|>": 4, "<|endoftext|>": 5}
        tokenizer = SimpleTokenizerV2(vocabulary)
        model = self.make_model().eval()
        inputs = torch.tensor([[0, 1, 2]])
        expected = model(inputs).detach()
        expected_generation = model.generate(inputs, max_new_tokens=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, tokenizer, epoch=2, validation_loss=1.25,
                            metadata={"seed": 123})
            restored = load_checkpoint(path)
        self.assertTrue(torch.equal(restored.model(inputs).detach(), expected))
        self.assertTrue(torch.equal(restored.model.generate(inputs, 4), expected_generation))
        self.assertEqual(restored.tokenizer.str_to_int, vocabulary)
        self.assertEqual(restored.model.config, model.config)
        self.assertEqual(restored.epoch, 2)
        self.assertEqual(restored.metadata, {"seed": 123})
        self.assertFalse(restored.model.training)

    def test_early_stopping_keeps_the_best_epoch_instead_of_the_last(self):
        config = TrainingConfig(epochs=4, stride=4, patience=1)
        data = prepare_training_data("a b c d " * 16, context_length=4, config=config)
        model = self.make_model(len(data.tokenizer.str_to_int))
        # Perdas inicial, época 1 e época 2: treino melhora, validação piora.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            with patch("src.training.evaluate_loss", side_effect=[7.0, 6.0, 5.0, 4.0, 3.0, 4.5]):
                result = fit(model, data, config, path, log=None)
            checkpoint = load_checkpoint(path)
        self.assertEqual(result["completed_epochs"], 2)
        self.assertEqual(checkpoint.epoch, 1)
        self.assertEqual(checkpoint.validation_loss, 4.0)
        self.assertTrue(any(
            not torch.equal(value, checkpoint.model.state_dict()[name])
            for name, value in model.state_dict().items()
        ))

    def test_rejects_corpus_too_short_for_separate_windows(self):
        with self.assertRaisesRegex(ValueError, "Treino e validação"):
            prepare_training_data("a b c d", context_length=4, config=TrainingConfig())

    def test_rejects_invalid_learning_rate(self):
        for learning_rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(learning_rate=learning_rate), self.assertRaises(ValueError):
                TrainingConfig(learning_rate=learning_rate)


if __name__ == "__main__":
    unittest.main()
