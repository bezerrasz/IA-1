import unittest

import torch

from src.attention import MultiHeadAttention
from src.transformer import FeedForwardNetwork, TransformerBlock


class TransformerTests(unittest.TestCase):
    def test_attention_output_shape(self):
        attention = MultiHeadAttention(
            d_in=16, d_out=16, context_length=8, num_heads=4
        )
        x = torch.randn(2, 5, 16)
        self.assertEqual(tuple(attention(x).shape), (2, 5, 16))

    def test_causal_attention_does_not_read_future_tokens(self):
        torch.manual_seed(7)
        attention = MultiHeadAttention(
            d_in=8, d_out=8, context_length=4, num_heads=2
        ).eval()
        original = torch.randn(1, 4, 8)
        changed_future = original.clone()
        changed_future[:, 2:, :] += 100.0

        original_output = attention(original)
        changed_output = attention(changed_future)
        self.assertTrue(torch.allclose(original_output[:, :2], changed_output[:, :2]))

    def test_feed_forward_returns_embedding_dimension(self):
        network = FeedForwardNetwork(embedding_dim=12, feed_forward_dim=32, dropout=0.0)
        self.assertEqual(tuple(network(torch.randn(2, 4, 12)).shape), (2, 4, 12))

    def test_transformer_block_preserves_shape(self):
        block = TransformerBlock(
            embedding_dim=16,
            num_heads=4,
            feed_forward_dim=64,
            context_length=8,
            dropout=0.0,
        )
        x = torch.randn(2, 5, 16)
        self.assertEqual(tuple(block(x).shape), (2, 5, 16))

    def test_invalid_head_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TransformerBlock(
                embedding_dim=10,
                num_heads=3,
                feed_forward_dim=20,
                context_length=8,
            )


if __name__ == "__main__":
    unittest.main()
