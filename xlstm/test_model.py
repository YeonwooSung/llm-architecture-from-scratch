import unittest

import torch

from xlstm.config import XLSTMConfig
from xlstm.model import MLSTMLayer, XLSTM, XLSTMBlock


class XLSTMBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = XLSTMConfig(
            embedding_dim=16, num_heads=2, num_blocks=2, vocab_size=32,
            ffn_round_up_to_multiple_of=8,
        )

    def test_model_forward_and_backward(self):
        model = XLSTM(self.config)
        logits = model(torch.randint(0, self.config.vocab_size, (2, 5)))
        self.assertEqual(logits.shape, (2, 5, self.config.vocab_size))
        logits.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_causality_and_memory_reset(self):
        block = XLSTMBlock(self.config).eval()
        x = torch.randn(2, 6, self.config.embedding_dim)
        output = block(x)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
        torch.testing.assert_close(block(changed)[:, :3], output[:, :3])
        torch.testing.assert_close(block(x[:, :3]), output[:, :3])
        torch.testing.assert_close(block(x), output)

    def test_memory_matches_explicit_causal_weighted_sum(self):
        layer = MLSTMLayer(self.config).double()
        q = torch.randn(2, 5, 2, 4, dtype=torch.float64)
        k = torch.randn_like(q)
        v = torch.randn(2, 5, 2, 8, dtype=torch.float64)
        i = torch.randn(2, 5, 2, dtype=torch.float64)
        f = torch.randn_like(i)
        actual = layer._recurrent_memory(q, k, v, i, f)
        for t in range(5):
            # Independently expand all historical writes and their forget
            # weights, without using the recurrent matrix state.
            weights = torch.stack([
                i[:, j].exp() * f[:, j + 1:t + 1].sigmoid().prod(dim=1)
                for j in range(t + 1)
            ], dim=1)
            scores = (q[:, t:t + 1] * k[:, :t + 1]).sum(dim=-1) / 2
            scores = scores * weights
            numerator = (scores.unsqueeze(-1) * v[:, :t + 1]).sum(dim=1)
            denominator = scores.sum(dim=1).abs().clamp_min(1).unsqueeze(-1)
            # The stabilized kernel adds epsilon in its rescaled domain.
            torch.testing.assert_close(actual[:, t], numerator / denominator, atol=2e-5, rtol=2e-5)

    def test_extreme_gates_have_finite_outputs_and_gradients(self):
        layer = MLSTMLayer(self.config)
        with torch.no_grad():
            layer.input_gate.bias.fill_(1000)
            layer.forget_gate.bias.fill_(-1000)
        x = torch.randn(2, 8, 16, requires_grad=True)
        output = layer(x)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_bias_and_value_dimensions(self):
        self.config.use_bias = True
        self.config.v_dim_factor = 1.5
        block = XLSTMBlock(self.config)
        self.assertEqual(block.ffn.up_proj.out_features, 48)
        self.assertEqual(block.mlstm.v_head_dim, 12)
        self.assertEqual(block(torch.randn(2, 1, 16)).shape, (2, 1, 16))

    def test_bfloat16(self):
        block = XLSTMBlock(self.config).to(torch.bfloat16)
        x = torch.randn(2, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        output = block(x)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_empty_sequence_and_invalid_shapes(self):
        block = XLSTMBlock(self.config)
        self.assertEqual(block(torch.empty(2, 0, 16)).shape, (2, 0, 16))
        with self.assertRaises(ValueError):
            block(torch.randn(2, 16))
        self.config.num_heads = 3
        with self.assertRaises(ValueError):
            XLSTMBlock(self.config)


if __name__ == "__main__":
    unittest.main()
