"""Overlapping time-expert output heads with the original fusion router."""

import torch
from torch import nn
from torch.nn import functional as F

from atfg.model import FusionDenoiser


class OverlapAddHead(nn.Module):
    def __init__(self, original, length, channels, patch_size):
        super().__init__()
        if patch_size < 2 or patch_size % 2:
            raise ValueError("Overlap output requires an even patch size of at least two")
        self.length, self.channels, self.stride = length, channels, patch_size
        self.span, self.padding = 2 * patch_size, patch_size // 2
        self.norm = original[0]
        previous = original[1]
        self.projection = nn.Linear(previous.in_features, self.span * channels,
                                    device=previous.weight.device, dtype=previous.weight.dtype)
        # Keep the original predictions in the central patch; initialize the new halo normally.
        start = self.padding * channels
        with torch.no_grad():
            self.projection.weight[start:start + patch_size * channels].copy_(previous.weight)
            self.projection.bias[start:start + patch_size * channels].copy_(previous.bias)
        taper = torch.hann_window(self.span + 2, periodic=False)[1:-1]
        self.register_buffer("taper", taper.reshape(1, 1, self.span, 1))
        denominator = self._fold(taper[:, None].expand(self.span, length // patch_size)[None])
        if not torch.all(denominator > 0):
            raise ValueError("Overlap output leaves uncovered time positions")
        self.register_buffer("denominator", denominator)

    def _fold(self, columns):
        return F.fold(columns, output_size=(1, self.length), kernel_size=(1, self.span),
                      padding=(0, self.padding), stride=(1, self.stride))

    def merge(self, patches):
        # [batch, token, output position, variable] -> normalized overlap-add.
        columns = patches.permute(0, 3, 2, 1) * self.taper
        columns = columns.reshape(patches.shape[0], self.channels * self.span, patches.shape[1])
        values = self._fold(columns) / self.denominator
        return values.squeeze(2).transpose(1, 2)

    def forward(self, hidden):
        patches = self.projection(self.norm(hidden))
        return self.merge(patches.reshape(hidden.shape[0], hidden.shape[1], self.span, self.channels))


class OverlapFusionDenoiser(FusionDenoiser):
    def __init__(self, length=128, channels=7, patch_size=8, width=64, heads=4,
                 layers=2, mode="patch"):
        super().__init__(length, channels, patch_size, width, heads, layers, mode)
        # Preserve all shared initial weights and the original post-construction RNG state.
        with torch.random.fork_rng(devices=[]):
            for expert in (self.short, self.long):
                expert.output = OverlapAddHead(expert.output, length, channels, patch_size)
