"""Three denoising experts with fixed, sequence-wide, or patch-wise fusion."""

import math

import torch
from torch import nn


def sinusoidal_embedding(positions, width):
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=positions.device) / max(half - 1, 1)
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def transformer_layers(width, heads, layers):
    # Construct each layer separately so layers have independent initial weights.
    return nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        for _ in range(layers)
    ])


class TimeExpert(nn.Module):
    def __init__(self, length, channels, patch_size, width, heads, layers, local):
        super().__init__()
        self.length, self.channels, self.patch_size = length, channels, patch_size
        count = length // patch_size
        self.input = nn.Linear(patch_size * channels, width)
        self.layers = transformer_layers(width, heads, layers)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, patch_size * channels))
        self.register_buffer("position", sinusoidal_embedding(torch.arange(count), width))
        self.register_buffer("mask", (torch.arange(count)[:, None] - torch.arange(count)).abs() > 1)
        self.local = local

    def forward(self, noisy, step_embedding):
        hidden = self.input(noisy.reshape(noisy.shape[0], -1, self.patch_size * self.channels))
        hidden = hidden + self.position + step_embedding[:, None, :]
        for layer in self.layers:
            hidden = layer(hidden, src_mask=self.mask if self.local else None)
        return self.output(hidden).reshape_as(noisy)


class FrequencyExpert(nn.Module):
    def __init__(self, length, channels, width, heads, layers):
        super().__init__()
        self.length, self.channels = length, channels
        frequencies = length // 2 + 1
        self.input = nn.Linear(2 * channels, width)
        self.layers = transformer_layers(width, heads, layers)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * channels))
        self.register_buffer("position", sinusoidal_embedding(torch.arange(frequencies), width))
        imaginary_mask = torch.ones(1, frequencies, 1)
        imaginary_mask[:, 0] = 0
        if length % 2 == 0:
            imaginary_mask[:, -1] = 0
        self.register_buffer("imaginary_mask", imaginary_mask)

    def forward(self, noisy, step_embedding):
        spectrum = torch.fft.rfft(noisy.float(), dim=1, norm="ortho")
        hidden = self.input(torch.cat((spectrum.real, spectrum.imag), dim=-1))
        hidden = hidden + self.position + step_embedding[:, None, :]
        for layer in self.layers:
            hidden = layer(hidden)
        real, imaginary = self.output(hidden).chunk(2, dim=-1)
        spectrum = torch.complex(real, imaginary * self.imaginary_mask)
        return torch.fft.irfft(spectrum, n=self.length, dim=1, norm="ortho")


class FusionDenoiser(nn.Module):
    def __init__(self, length=128, channels=7, patch_size=8, width=64, heads=4,
                 layers=2, mode="patch"):
        super().__init__()
        if length < 2 or patch_size < 1 or length % patch_size:
            raise ValueError("length must be at least 2 and divisible by patch_size")
        if width < 4 or width % 2 or heads < 1 or width % heads or layers < 1 or channels < 1:
            raise ValueError("Use an even width divisible by heads and positive channels/layers")
        if mode not in ("fixed", "global", "patch"):
            raise ValueError(f"Unknown fusion mode {mode}")
        self.length, self.channels = length, channels
        self.patch_size, self.width, self.mode = patch_size, width, mode
        self.step_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.short = TimeExpert(length, channels, patch_size, width, heads, layers, local=True)
        self.long = TimeExpert(length, channels, patch_size, width, heads, layers, local=False)
        self.frequency = FrequencyExpert(length, channels, width, heads, layers)
        self.router_input = nn.Linear(patch_size * channels, width)
        self.router = nn.Sequential(nn.Linear(3 * width, 32), nn.SiLU(), nn.Linear(32, 3))
        # All three modes start at the same equal mixture and share expert initialization.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, noisy, steps):
        if noisy.shape[1:] != (self.length, self.channels):
            raise ValueError("Expected noisy input shaped [batch, length, channels]")
        step_embedding = self.step_mlp(sinusoidal_embedding(steps, self.width))
        experts = torch.stack((
            self.short(noisy, step_embedding),
            self.long(noisy, step_embedding),
            self.frequency(noisy, step_embedding),
        ), dim=-1)
        patches = noisy.reshape(noisy.shape[0], -1, self.patch_size * self.channels)
        local_features = self.router_input(patches)
        global_features = local_features.mean(dim=1, keepdim=True).expand_as(local_features)
        if self.mode == "fixed":
            weights = noisy.new_full((*patches.shape[:2], 3), 1 / 3)
        else:
            features = local_features if self.mode == "patch" else global_features
            features = torch.cat((features, global_features,
                                  step_embedding[:, None, :].expand_as(features)), dim=-1)
            weights = self.router(features).softmax(dim=-1)
        point_weights = weights.repeat_interleave(self.patch_size, dim=1)
        estimate = (experts * point_weights[:, :, None, :]).sum(dim=-1)
        return estimate, weights
