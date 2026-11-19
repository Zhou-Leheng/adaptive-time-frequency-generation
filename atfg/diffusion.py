"""Cosine forward diffusion and deterministic DDIM with x0 prediction."""

import math

import torch
from torch import nn


class GaussianDiffusion(nn.Module):
    def __init__(self, steps=200):
        super().__init__()
        if steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        self.steps = steps
        times = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        cumulative = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999)
        self.register_buffer("alpha_bar", (1 - betas).cumprod(dim=0).float())

    def add_noise(self, clean, times, noise):
        alpha = self.alpha_bar[times, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, initial_noise, sampling_steps=50, trace_samples=16):
        if not 1 <= sampling_steps <= self.steps:
            raise ValueError("sampling_steps must be between 1 and diffusion_steps")
        times = torch.linspace(self.steps - 1, -1, sampling_steps + 1).round().long().tolist()
        sample = initial_noise.clone()
        traces = []
        for current, following in zip(times[:-1], times[1:]):
            batch_steps = torch.full((len(sample),), current, device=sample.device, dtype=torch.long)
            clean, weights = model(sample, batch_steps)
            traces.append(weights[:trace_samples].cpu())
            if following < 0:
                sample = clean
            else:
                alpha = self.alpha_bar[current]
                next_alpha = self.alpha_bar[following]
                noise = (sample - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                sample = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
        return sample, torch.stack(traces, dim=1), times[:-1]
