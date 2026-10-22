import unittest

import torch

from src.gpt import GPTModel
from src.generation import generate_text
from src.tokenizer import SimpleTokenizerV2


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.vocab = {
            "hello": 0,
            "world": 1,
            ",": 2,
            "<|unk|>": 3,
            "<|endoftext|>": 4,
        }
        self.tokenizer = SimpleTokenizerV2(self.vocab)
        self.model = GPTModel(
            vocab_size=len(self.vocab),
            context_length=4,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            feed_forward_dim=16,
            dropout=0.0,
        )

    def test_generate_text_preserves_prompt_and_adds_tokens(self):
        generated = generate_text(
            self.model,
            self.tokenizer,
            "hello",
            max_new_tokens=2,
        )
        self.assertTrue(generated.startswith("hello"))
        self.assertNotEqual(generated, "hello")

    def test_top_k_sampling_returns_valid_sequence(self):
        torch.manual_seed(12)
        generated = self.model.generate(
            torch.tensor([[0, 1]]),
            max_new_tokens=3,
            temperature=0.8,
            top_k=2,
            do_sample=True,
        )
        self.assertEqual(tuple(generated.shape), (1, 5))
        self.assertTrue(torch.all((generated >= 0) & (generated < 5)))

    def test_invalid_generation_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.generate(torch.tensor([[0]]), 1, temperature=0)
        with self.assertRaises(ValueError):
            self.model.generate(torch.tensor([[0]]), 1, top_k=6)


if __name__ == "__main__":
    unittest.main()
