import unittest

import torch
from torch.nn import functional as F

from mistral_3_1.config import Mistral3Config, Mistral3TextConfig, Mistral3VisionConfig
from mistral_3_1.model import Mistral3Attention, Mistral3Model, Mistral3PatchMerger


def small_config():
    return Mistral3Config(
        text_config=Mistral3TextConfig(
            vocab_size=32, hidden_size=24, intermediate_size=40,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=32,
        ),
        vision_config=Mistral3VisionConfig(
            hidden_size=16, intermediate_size=24, num_hidden_layers=2,
            num_attention_heads=2, head_dim=8, image_size=12, patch_size=2,
        ),
    )


class Mistral3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = small_config()
        self.model = Mistral3Model(self.config).eval()

    def test_text_causality_and_output_head(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 8, 9]])
        logits = self.model(ids)
        self.assertEqual(logits.shape, (1, 4, 32))
        torch.testing.assert_close(logits[:, :2], self.model(changed)[:, :2])
        torch.testing.assert_close(logits[:, :2], self.model(ids[:, :2]))
        self.assertIsNot(self.model.token_embedding.weight, self.model.output_layer.linear.weight)

    def test_gqa_matches_sdpa_reference(self):
        attention = Mistral3Attention(self.config).eval()
        x = torch.randn(2, 5, 24)
        positions = torch.tensor([[0, 1, 4, 5, 8], [2, 3, 4, 5, 6]])
        q = attention.q_proj(x).view(2, 5, 4, 4).transpose(1, 2)
        k = attention.k_proj(x).view(2, 5, 2, 4).transpose(1, 2)
        v = attention.v_proj(x).view(2, 5, 2, 4).transpose(1, 2)
        # Independent complex-number rotation of paired split-half coordinates.
        frequencies = self.config.text_config.rope_theta ** (-torch.arange(0, 4, 2) / 4)
        rotations = torch.polar(torch.ones(2, 5, 2), positions[..., None] * frequencies)[:, None]

        def rotate(tensor):
            paired = torch.complex(tensor[..., :2], tensor[..., 2:]) * rotations
            return torch.cat((paired.real, paired.imag), dim=-1)

        expected = F.scaled_dot_product_attention(
            rotate(q), rotate(k).repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True,
        ).transpose(1, 2).reshape(2, 5, 16)
        torch.testing.assert_close(attention(x, position_ids=positions), attention.o_proj(expected))

    def test_left_padding_and_fully_masked_backward(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.tensor([[0, 0, 1, 1]])
        positions = torch.tensor([[0, 0, 0, 1]])
        logits = self.model(ids, attention_mask=mask, position_ids=positions)
        torch.testing.assert_close(logits[:, 2:], self.model(ids[:, 2:]))
        logits.square().mean().backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(torch.isfinite(self.model(ids, attention_mask=torch.zeros_like(ids))).all())

    def test_sliding_window(self):
        self.config.text_config.sliding_window = 2
        attention = Mistral3Attention(self.config).eval()
        x = torch.randn(1, 5, 24)
        changed = x.clone()
        changed[:, 0] += 100
        torch.testing.assert_close(attention(x)[:, 2:], attention(changed)[:, 2:])

    def test_vision_isolates_images_and_ignores_pixel_padding(self):
        pixels = torch.randn(2, 3, 8, 12)
        sizes = torch.tensor([[8, 12], [4, 8]])
        encoded = self.model.vision_encoder(pixels, sizes)
        self.assertEqual(encoded.shape, (32, 16))
        first = self.model.vision_encoder(pixels[:1])
        second = self.model.vision_encoder(pixels[1:, :, :4, :8])
        torch.testing.assert_close(encoded, torch.cat((first, second)))
        changed = pixels.clone()
        changed[1, :, 4:] = 100
        changed[1, :, :, 8:] = 100
        torch.testing.assert_close(encoded, self.model.vision_encoder(changed, sizes))

    def test_patch_merger_channel_and_spatial_order(self):
        merger = Mistral3PatchMerger(self.config)
        features = torch.arange(3 * 5 * 16).float().reshape(3, 5, 16)
        # Enumerate complete 2x2 blocks, channel first; incomplete edges are omitted.
        blocks = torch.stack([features[:2, col:col + 2].permute(2, 0, 1).reshape(-1)
                              for col in (0, 2)])
        expected = F.linear(blocks, merger.merging_layer.weight)
        torch.testing.assert_close(merger(features.reshape(-1, 16), torch.tensor([[6, 10]])), expected)

    def test_multimodal_forward_and_backward(self):
        pixels = torch.randn(2, 3, 8, 12, requires_grad=True)
        sizes = torch.tensor([[8, 12], [4, 8]])
        ids = torch.tensor([[1, 10, 10, 10, 10, 10, 10, 2, 3],
                            [1, 10, 10, 2, 3, 4, 5, 6, 7]])
        logits = self.model(ids, pixels, image_sizes=sizes)
        self.assertEqual(logits.shape, (2, 9, 32))
        F.cross_entropy(logits[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1)).backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(pixels.grad.abs().sum().item(), 0)
        changed = pixels.detach().clone()
        changed[0] += 1
        # Earlier text cannot read future image placeholders in the decoder.
        torch.testing.assert_close(logits.detach()[:, :1], self.model(ids, changed, image_sizes=sizes)[:, :1])

    def test_vision_feature_layer_selection(self):
        self.config.vision_feature_layer = 0
        model = Mistral3Model(self.config)
        pixels = torch.randn(1, 3, 8, 8)
        expected = model.vision_encoder.ln_pre(model.vision_encoder.patch_conv(pixels).flatten(2).transpose(1, 2))
        torch.testing.assert_close(model.vision_encoder(pixels), expected.squeeze(0))

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "require pixel_values"):
            self.model(torch.tensor([[1, 10]]))
        with self.assertRaisesRegex(ValueError, "placeholders"):
            self.model(torch.tensor([[1, 10]]), torch.randn(1, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, "Image sizes"):
            self.model.vision_encoder(torch.randn(1, 3, 8, 8), torch.tensor([[7, 8]]))
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            self.model(torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 3))
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            self.config.text_config.num_key_value_heads = 3
            Mistral3Model(self.config)

    def test_default_architecture_on_meta_device(self):
        with torch.device("meta"):
            model = Mistral3Model(Mistral3Config())
        self.assertEqual(len(model.layers), 40)
        self.assertEqual(len(model.vision_encoder.layers), 24)
        self.assertEqual(model.layers[0].self_attn.q_proj.weight.shape, (4096, 5120))
        self.assertEqual(model.layers[0].self_attn.k_proj.weight.shape, (1024, 5120))
        self.assertEqual(model.multi_modal_projector.patch_merger.merging_layer.weight.shape, (1024, 4096))


if __name__ == "__main__":
    unittest.main()
