"""Load checksum-verified modules from a pinned official Diffusion-TS revision."""

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import torch
from torch import nn


REVISION = "566307e6cf2d8095e58de4c6e3a6ae965b69b5b5"
REPOSITORY = "Y-debug-sys/Diffusion-TS"


def ensure_upstream(cache_directory=None):
    root = Path(cache_directory or Path(__file__).resolve().parents[1] / "data/upstream")
    root = root / f"Diffusion-TS-{REVISION}"
    manifest = json.loads(Path(__file__).with_name("upstream_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        destination = root / relative
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            continue
        print(f"Downloading official Diffusion-TS module {relative}", flush=True)
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{relative}"
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Upstream checksum mismatch for {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
    path = str(root.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return root


class OfficialDiffusionTS(nn.Module):
    """Official architecture and L1/Fourier objective within the shared pilot protocol."""

    def __init__(self, length, channels, patch_size, width, heads, layers, mode,
                 diffusion_steps=200, encoder_layers=3, decoder_layers=2):
        super().__init__()
        ensure_upstream()
        from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

        self.length, self.channels, self.patch_size = length, channels, patch_size
        self.core = Diffusion_TS(
            seq_length=length, feature_size=channels, n_layer_enc=encoder_layers,
            n_layer_dec=decoder_layers, d_model=width, n_heads=heads,
            timesteps=diffusion_steps, sampling_timesteps=diffusion_steps,
            loss_type="l1", beta_schedule="cosine", mlp_hidden_times=4,
            attn_pd=0.0, resid_pd=0.0, kernel_size=1, padding_size=0, use_ff=True,
        )

    def forward(self, noisy, times):
        prediction = self.core.output(noisy, times)
        placeholder = noisy.new_ones((len(noisy), self.length // self.patch_size, 1))
        return prediction, placeholder

    def training_loss(self, clean, times, noise):
        return self.core._train_loss(x_start=clean, t=times, noise=noise)
