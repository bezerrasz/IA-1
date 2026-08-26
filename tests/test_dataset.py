import unittest

try:
    import torch
except ModuleNotFoundError:  # Permite executar o teste de tokenizer sem PyTorch.
    torch = None

if torch is not None:
    from src.dataset import GPTDatasetV1
else:
    GPTDatasetV1 = None


@unittest.skipIf(torch is None, "PyTorch ainda não está instalado")
class DatasetTests(unittest.TestCase):
    class FixedTokenizer:
        def encode(self, text):
            return list(range(len(text.split())))

    def test_input_and_target_are_shifted(self):
        dataset = GPTDatasetV1(
            "a b c d e f",
            self.FixedTokenizer(),
            max_length=3,
            stride=1,
        )

        inputs, targets = dataset[0]
        self.assertTrue(torch.equal(inputs, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(targets, torch.tensor([1, 2, 3])))

    def test_stride_controls_number_of_windows(self):
        dataset = GPTDatasetV1(
            "a b c d e f g h",
            self.FixedTokenizer(),
            max_length=3,
            stride=2,
        )
        self.assertEqual(len(dataset), 3)

    def test_rejects_short_text(self):
        with self.assertRaises(ValueError):
            GPTDatasetV1("a b c", self.FixedTokenizer(), max_length=3, stride=1)


if __name__ == "__main__":
    unittest.main()
