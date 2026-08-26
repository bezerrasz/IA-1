import unittest

import torch

from src.dataloader import create_dataloader_v1


class DataLoaderTests(unittest.TestCase):
    class FixedTokenizer:
        def encode(self, text):
            return list(range(len(text.split())))

    def test_batch_shapes_and_target_shift(self):
        loader = create_dataloader_v1(
            "a b c d e f g h i j",
            self.FixedTokenizer(),
            batch_size=2,
            max_length=4,
            stride=4,
            shuffle=False,
            drop_last=True,
        )

        inputs, targets = next(iter(loader))
        self.assertEqual(tuple(inputs.shape), (2, 4))
        self.assertEqual(tuple(targets.shape), (2, 4))
        self.assertTrue(torch.equal(targets[:, :-1], inputs[:, 1:]))

    def test_default_parameters_are_accepted(self):
        loader = create_dataloader_v1(
            "a b c d e f g h i j k l",
            self.FixedTokenizer(),
            batch_size=2,
            max_length=4,
            stride=4,
        )
        self.assertGreater(len(loader), 0)

    def test_rejects_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            create_dataloader_v1(
                "a b c d e",
                self.FixedTokenizer(),
                batch_size=0,
                max_length=3,
                stride=1,
            )


if __name__ == "__main__":
    unittest.main()
