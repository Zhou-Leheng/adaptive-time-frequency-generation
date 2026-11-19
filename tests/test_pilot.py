"""Focused checks for spectral alignment, data leakage, and checkpoint recovery."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from atfg.data import prepare_data
from atfg.diffusion import GaussianDiffusion
from atfg.metrics import distribution_metrics
from atfg.model import FusionDenoiser
from run import parser, run_experiment


torch.set_num_threads(1)


class PilotTests(unittest.TestCase):
    def test_all_fusion_modes_start_identically_and_can_backpropagate(self):
        x = torch.randn(3, 32, 2)
        times = torch.tensor([0, 3, 7])
        predictions = []
        for mode in ("fixed", "global", "patch"):
            torch.manual_seed(42)
            model = FusionDenoiser(32, 2, 8, 16, 2, 1, mode)
            predicted, weights = model(x, times)
            self.assertEqual(predicted.shape, x.shape)
            self.assertEqual(weights.shape, (3, 4, 3))
            torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, 4))
            predictions.append(predicted.detach())
            predicted.square().mean().backward()
            self.assertTrue(torch.isfinite(model.frequency.output[-1].weight.grad).all())
            if mode != "fixed":
                self.assertGreater(model.router[-1].weight.grad.abs().sum().item(), 0)
            sampled, routes, steps = GaussianDiffusion(8).sample(model.eval(), x, 4)
            self.assertEqual(sampled.shape, x.shape)
            self.assertEqual(routes.shape, (3, 4, 4, 3))
            self.assertEqual(len(steps), 4)
            self.assertTrue(torch.isfinite(sampled).all())
        torch.testing.assert_close(predictions[0], predictions[1])
        torch.testing.assert_close(predictions[0], predictions[2])

    def test_spectral_round_trip_handles_odd_length(self):
        model = FusionDenoiser(33, 2, 3, 16, 2, 1, "patch")
        output, _ = model(torch.randn(2, 33, 2), torch.tensor([0, 5]))
        self.assertEqual(output.shape, (2, 33, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_train_statistics_exclude_held_out_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "data.csv"
            values = np.arange(2000, dtype=float).reshape(1000, 2)
            np.savetxt(source, values, delimiter=",", header="a,b", comments="")
            original, metadata = prepare_data(source, 32, 16, 8, 42)
            values[700:] += 1e6
            np.savetxt(source, values, delimiter=",", header="a,b", comments="")
            changed, changed_metadata = prepare_data(source, 32, 16, 8, 42)
            np.testing.assert_array_equal(original[0], changed[0])
            self.assertEqual(metadata["normalization_mean"], changed_metadata["normalization_mean"])

    def test_resume_matches_uninterrupted_training_and_replays_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "data.csv"
            time = np.arange(1000)
            values = np.stack((np.sin(time / 8), np.cos(time / 13)), axis=1)
            np.savetxt(source, values, delimiter=",", header="a,b", comments="")
            common = ["--data", str(source), "--modes", "patch", "--device", "cpu",
                      "--length", "32", "--patch-size", "8", "--width", "16", "--heads", "2",
                      "--layers", "1", "--steps", "4", "--batch-size", "4", "--train-windows", "16",
                      "--eval-windows", "8", "--samples", "8", "--diffusion-steps", "8",
                      "--sampling-steps", "4", "--check-every", "2", "--log-every", "1"]
            uninterrupted = parser().parse_args(common + ["--output", str(root / "full")])
            resumed = parser().parse_args(common + ["--output", str(root / "resumed")])
            run_experiment(uninterrupted)
            original_step = torch.optim.AdamW.step
            calls = 0

            def interrupt_on_third_step(optimizer, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt
                return original_step(optimizer, *args, **kwargs)

            with patch.object(torch.optim.AdamW, "step", interrupt_on_third_step):
                with self.assertRaises(KeyboardInterrupt):
                    run_experiment(resumed)
            run_experiment(resumed)
            for name in ("model", "ema"):
                full = torch.load(root / "full/patch/last.pt", weights_only=False)[name]
                partial = torch.load(root / "resumed/patch/last.pt", weights_only=False)[name]
                for key in full:
                    torch.testing.assert_close(full[key], partial[key], rtol=0, atol=0)
            expected = np.load(root / "resumed/patch/samples_normalized.npy")
            (root / "resumed/patch/metrics.json").unlink()
            run_experiment(resumed)
            np.testing.assert_array_equal(expected, np.load(root / "resumed/patch/samples_normalized.npy"))
            summary = json.loads((root / "resumed/summary.json").read_text())
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["trained_steps"], 4)
            run_experiment(resumed)

    def test_distribution_metrics_have_no_paired_sample_requirement(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(8, 32, 2))
        metrics = distribution_metrics(data, data[::-1])
        for key in ("swd", "psd_relative_l1", "acf_mae", "delta_w1", "cross_corr_mae"):
            self.assertAlmostEqual(metrics[key], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
