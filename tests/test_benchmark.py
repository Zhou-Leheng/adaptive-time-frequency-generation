"""Integration checks for adding a baseline to already completed fusion experiments."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from atfg.diffusion import GaussianDiffusion
from atfg.upstream import OfficialDiffusionTS
from compare import compare, parser as comparison_parser
from run import parser as training_parser, run_experiment


torch.set_num_threads(1)


class BenchmarkTests(unittest.TestCase):
    def test_official_objective_and_shared_noise_schedule(self):
        model = OfficialDiffusionTS(32, 2, 8, 16, 2, 1, "diffusion_ts",
                                    diffusion_steps=8, encoder_layers=1, decoder_layers=1)
        diffusion = GaussianDiffusion(8)
        torch.testing.assert_close(diffusion.alpha_bar, model.core.alphas_cumprod)
        clean, noise = torch.randn(4, 32, 2), torch.randn(4, 32, 2)
        times = torch.tensor([0, 2, 5, 7])
        expected = model.core._train_loss(clean, times, noise=noise)
        actual = model.training_loss(clean, times, noise)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_full_comparison_preserves_existing_samples_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            times = np.arange(1000)
            values = np.stack((np.sin(times / 8), np.cos(times / 13)), axis=1)
            np.savetxt(source, values, delimiter=",", header="a,b", comments="")
            output = root / "pilot"
            original_args = training_parser().parse_args([
                "--data", str(source), "--output", str(output), "--device", "cpu",
                "--steps", "2", "--length", "32", "--patch-size", "8", "--width", "16",
                "--heads", "2", "--layers", "1", "--batch-size", "4", "--train-windows", "16",
                "--eval-windows", "8", "--samples", "8", "--diffusion-steps", "8",
                "--sampling-steps", "4", "--check-every", "1",
            ])
            run_experiment(original_args)
            hashes = {mode: hashlib.sha256((output / mode / "samples_normalized.npy").read_bytes()).hexdigest()
                      for mode in ("fixed", "global", "patch")}
            args = comparison_parser().parse_args([
                "--results", str(output), "--data", str(source), "--device", "cpu",
                "--baseline-encoder-layers", "1", "--baseline-decoder-layers", "1",
                "--encoder-iterations", "2", "--discriminator-iterations", "2",
                "--predictor-iterations", "2", "--repeats", "1",
            ])
            compare(args)
            work = output / "benchmark_diffusion_ts"
            result = json.loads((work / "comparison.json").read_text())
            self.assertEqual([row["method"] for row in result], ["fixed", "global", "patch", "diffusion_ts"])
            for row in result:
                for key in ("context_fid", "correlational_score", "discriminative_mean", "predictive_mean"):
                    self.assertTrue(np.isfinite(row[key]), (row["method"], key))
            for mode, before in hashes.items():
                self.assertEqual(before, hashlib.sha256((output / mode / "samples_normalized.npy").read_bytes()).hexdigest())
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Unexpected retraining")):
                compare(args)
            for mode in hashes:
                self.assertTrue((work / "evaluation" / f"{mode}.json").is_file())


if __name__ == "__main__":
    unittest.main()
