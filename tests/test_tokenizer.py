import unittest

from src.tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        tokens = tokenize_text("Hello, world! <|endoftext|> Hello")
        self.vocab = build_vocab(tokens)
        self.tokenizer = SimpleTokenizerV2(self.vocab)

    def test_tokenize_preserves_special_token(self):
        tokens = self.tokenizer.tokenize("Hello <|endoftext|> world")
        self.assertEqual(tokens, ["Hello", "<|endoftext|>", "world"])

    def test_encode_and_decode_round_trip(self):
        ids = self.tokenizer.encode("Hello, world!")
        self.assertEqual(self.tokenizer.decode(ids), "Hello, world!")

    def test_unknown_token_uses_unknown_id(self):
        unknown_id = self.vocab["<|unk|>"]
        self.assertEqual(self.tokenizer.encode("not-in-vocabulary"), [unknown_id])

    def test_vocabulary_ids_are_unique_and_deterministic(self):
        self.assertEqual(len(self.vocab), len(set(self.vocab.values())))
        self.assertEqual(self.vocab, build_vocab(tokenize_text("Hello, world! <|endoftext|> Hello")))


if __name__ == "__main__":
    unittest.main()
