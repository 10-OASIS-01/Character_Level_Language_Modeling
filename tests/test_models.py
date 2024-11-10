# tests/test_models.py

import unittest
import torch
from models.transformer import Transformer, ModelConfig
from models.rnn import RNN
from models.gru import GRU
from models.lstm import LSTM


class TestModels(unittest.TestCase):
    def test_transformer_forward(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_layer=2, n_head=2, n_embd=32)
        model = Transformer(config)
        idx = torch.randint(0, 100, (1, 10))
        logits, loss = model(idx, targets=idx)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")

        self.assertEqual(logits.shape, (1, 10, config.vocab_size), "Logits shape mismatch")

        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                self.assertIsNotNone(param.grad, "Gradient should be computed for all model parameters")

    def test_rnn_forward(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_embd=32, n_embd2=64)
        model = RNN(config)
        idx = torch.randint(0, 100, (1, 10))
        logits, loss = model(idx, targets=idx)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")

        self.assertEqual(logits.shape, (1, 10, config.vocab_size), "Logits shape mismatch")

        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                self.assertIsNotNone(param.grad, "Gradient should be computed for all model parameters")

    def test_gru_forward(self):
        config = ModelConfig(vocab_size=100, block_size=10, n_embd=32, n_embd2=64)
        model = GRU(config)
        idx = torch.randint(0, 100, (1, 10))
        logits, loss = model(idx, targets=idx)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")

        self.assertEqual(logits.shape, (1, 10, config.vocab_size), "Logits shape mismatch")

        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                self.assertIsNotNone(param.grad, "Gradient should be computed for all model parameters")


if __name__ == '__main__':
    unittest.main()
