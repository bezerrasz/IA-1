import unittest

import torch

from src.embeddings import InputEmbedding, PositionalEmbedding, TokenEmbedding


class EmbeddingTests(unittest.TestCase):
    def test_token_embedding_shape(self):
        layer = TokenEmbedding(vocab_size=20, embedding_dim=6)
        token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        self.assertEqual(tuple(layer(token_ids).shape), (2, 4, 6))

    def test_positional_embedding_shape_and_sequence_limit(self):
        layer = PositionalEmbedding(context_length=4, embedding_dim=6)
        token_ids = torch.zeros((2, 4), dtype=torch.long)
        self.assertEqual(tuple(layer(token_ids).shape), (4, 6))
        with self.assertRaises(ValueError):
            layer(torch.zeros((1, 5), dtype=torch.long))

    def test_positions_change_representation_for_same_token(self):
        torch.manual_seed(123)
        layer = InputEmbedding(vocab_size=20, context_length=4, embedding_dim=6)
        token_ids = torch.tensor([[3, 3, 3, 3]])
        vectors = layer(token_ids)
        self.assertFalse(torch.allclose(vectors[:, 0, :], vectors[:, 1, :]))

    def test_input_embedding_shape(self):
        layer = InputEmbedding(vocab_size=20, context_length=4, embedding_dim=6)
        token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        self.assertEqual(tuple(layer(token_ids).shape), (2, 4, 6))


if __name__ == "__main__":
    unittest.main()
