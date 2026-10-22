import unittest

import torch

from src.gpt import GPTConfig, GPTModel


class GPTTests(unittest.TestCase):
    def make_model(self):
        return GPTModel(
            vocab_size=30,
            context_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feed_forward_dim=32,
            dropout=0.0,
        )

    def test_logits_have_vocabulary_dimension(self):
        model = self.make_model()
        token_ids = torch.randint(0, 30, (2, 5))
        self.assertEqual(tuple(model(token_ids).shape), (2, 5, 30))

    def test_blocks_are_stacked(self):
        model = self.make_model()
        self.assertEqual(len(model.transformer_blocks), 2)
        self.assertGreater(model.parameter_count(), 0)

    def test_config_applies_default_feed_forward_dimension(self):
        config = GPTConfig(vocab_size=20, context_length=4, embedding_dim=12)
        self.assertEqual(config.feed_forward_dim, 48)
        model = GPTModel.from_config(config)
        self.assertEqual(tuple(model(torch.tensor([[1, 2]])).shape), (1, 2, 20))

    def test_forward_rejects_context_longer_than_configured(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 9), dtype=torch.long))

    def test_generate_appends_tokens_and_keeps_ids_valid(self):
        model = self.make_model()
        prompt = torch.tensor([[1, 2, 3]])
        generated = model.generate(prompt, max_new_tokens=4)
        self.assertEqual(tuple(generated.shape), (1, 7))
        self.assertTrue(torch.all((generated >= 0) & (generated < 30)))


if __name__ == "__main__":
    unittest.main()
