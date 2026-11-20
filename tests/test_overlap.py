"""Check overlap coverage, shared initialization, and differentiable sampling."""

import unittest

import torch
from torch.nn import functional as F

from atfg.diffusion import GaussianDiffusion
from atfg.model import FusionDenoiser
from atfg.overlap import OverlapFusionDenoiser


torch.set_num_threads(1)


class OverlapTests(unittest.TestCase):
    def test_merge_recovers_endpoints_and_a_sharp_step(self):
        for length, patch_size in ((32, 8), (8, 8), (12, 2)):
            model = OverlapFusionDenoiser(length, 2, patch_size, 16, 2, 1)
            head = model.short.output
            signal = torch.arange(length, dtype=torch.float32)[None, :, None].repeat(2, 1, 2)
            signal[:, length // 2:] += 30
            patches = F.pad(signal.transpose(1, 2), (head.padding, head.padding))
            patches = patches.unfold(2, head.span, head.stride).permute(0, 2, 3, 1)
            torch.testing.assert_close(head.merge(patches), signal)
            self.assertTrue(torch.all(head.denominator > 0))

    def test_shared_weights_and_training_random_stream_are_preserved(self):
        torch.manual_seed(42)
        original = FusionDenoiser(32, 2, 8, 16, 2, 1)
        original_rng = torch.get_rng_state()
        torch.manual_seed(42)
        overlap = OverlapFusionDenoiser(32, 2, 8, 16, 2, 1)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        current = dict(overlap.named_parameters())
        for name, parameter in original.named_parameters():
            if not name.startswith(("short.output.", "long.output.")):
                torch.testing.assert_close(parameter, current[name], rtol=0, atol=0)
        for before, after in ((original.short, overlap.short), (original.long, overlap.long)):
            start = after.output.padding * 2
            torch.testing.assert_close(after.output.projection.weight[start:start + 16],
                                       before.output[1].weight, rtol=0, atol=0)

    def test_overlap_model_trains_and_samples_with_original_router_shapes(self):
        model = OverlapFusionDenoiser(32, 2, 8, 16, 2, 1)
        noisy = torch.randn(3, 32, 2)
        values, weights = model(noisy, torch.tensor([0, 3, 7]))
        self.assertEqual(values.shape, noisy.shape)
        self.assertEqual(weights.shape, (3, 4, 3))
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 4))
        values.square().mean().backward()
        for expert in (model.short, model.long):
            gradient = expert.output.projection.weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        generated, _, _ = GaussianDiffusion(8).sample(model.eval(), noisy, 4)
        self.assertTrue(torch.isfinite(generated).all())


if __name__ == "__main__":
    unittest.main()
