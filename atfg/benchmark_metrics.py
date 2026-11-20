"""Common generation metrics using TS2Vec and post-hoc GRU evaluators."""

import numpy as np
from scipy.linalg import sqrtm
import torch
from torch import nn
from torch.nn import functional as F

from atfg.upstream import ensure_upstream


def fit_metric_scaler(training):
    minimum = training.min(axis=(0, 1), keepdims=True)
    span = np.maximum(training.max(axis=(0, 1), keepdims=True) - minimum, 1e-6)
    return minimum, span


def frechet_distance(real_features, fake_features):
    mean_real, mean_fake = real_features.mean(0), fake_features.mean(0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_fake = np.cov(fake_features, rowvar=False)
    covariance_mean = sqrtm(cov_real @ cov_fake)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    distance = np.square(mean_real - mean_fake).sum() + np.trace(cov_real + cov_fake - 2 * covariance_mean)
    if not np.isfinite(distance):
        raise FloatingPointError("Context-FID produced a non-finite result")
    return float(max(0, distance))


def official_correlational_score(real, generated):
    ensure_upstream()
    from Utils.cross_correlation import CrossCorrelLoss

    real_tensor = torch.as_tensor(real, dtype=torch.float32)
    fake_tensor = torch.as_tensor(generated, dtype=torch.float32)
    result = float(CrossCorrelLoss(real_tensor, name="CrossCorrelLoss").compute(fake_tensor))
    if not np.isfinite(result):
        raise FloatingPointError("Correlational score is non-finite. Check for collapsed, constant variables.")
    return result


class PostHocGRU(nn.Module):
    def __init__(self, channels, hidden, sequence_output=False):
        super().__init__()
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
        self.sequence_output = sequence_output

    def forward(self, values):
        output, hidden = self.gru(values)
        return self.output(output if self.sequence_output else hidden[-1])


def check_stop(stop):
    if stop["requested"]:
        raise InterruptedError("Evaluation stopped. Completed evaluator repeats are saved.")


def discriminative_score(real, generated, seed, iterations, device, stop):
    """TimeGAN-style |accuracy - 0.5| with a one-layer GRU and an 80/20 evaluator split."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    real_order, fake_order = rng.permutation(len(real)), rng.permutation(len(generated))
    split_real, split_fake = int(0.8 * len(real)), int(0.8 * len(generated))
    real_train = torch.as_tensor(real[real_order[:split_real]], dtype=torch.float32, device=device)
    fake_train = torch.as_tensor(generated[fake_order[:split_fake]], dtype=torch.float32, device=device)
    real_test = torch.as_tensor(real[real_order[split_real:]], dtype=torch.float32, device=device)
    fake_test = torch.as_tensor(generated[fake_order[split_fake:]], dtype=torch.float32, device=device)
    if min(len(real_train), len(fake_train), len(real_test), len(fake_test)) < 1:
        raise ValueError("The discriminator needs enough windows for its 80/20 split")
    model = PostHocGRU(real.shape[-1], max(real.shape[-1] // 2, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    batch = min(128, len(real_train), len(fake_train))
    for iteration in range(iterations):
        check_stop(stop)
        real_ids = torch.as_tensor(rng.choice(len(real_train), batch, replace=False), device=device)
        fake_ids = torch.as_tensor(rng.choice(len(fake_train), batch, replace=False), device=device)
        real_logits, fake_logits = model(real_train[real_ids]), model(fake_train[fake_ids])
        loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (iteration + 1) % 500 == 0:
            print(f"  Discriminator {iteration + 1}/{iterations}", flush=True)
    model.eval()
    with torch.no_grad():
        correct_real = (model(real_test) > 0).sum().item()
        correct_fake = (model(fake_test) <= 0).sum().item()
    accuracy = (correct_real + correct_fake) / (len(real_test) + len(fake_test))
    return float(abs(accuracy - 0.5)), float(accuracy)


def predictive_score(real, generated, seed, iterations, device, stop):
    """Train on synthetic windows and predict the last real feature one step ahead."""
    if real.shape[-1] < 2:
        raise ValueError("The TimeGAN-style predictive metric requires at least two variables")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    training = torch.as_tensor(generated, dtype=torch.float32, device=device)
    model = PostHocGRU(real.shape[-1] - 1, max(real.shape[-1] // 2, 1), sequence_output=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    batch = min(128, len(training))
    for iteration in range(iterations):
        check_stop(stop)
        indices = torch.as_tensor(rng.choice(len(training), batch, replace=False), device=device)
        windows = training[indices]
        prediction = model(windows[:, :-1, :-1]).sigmoid()
        loss = F.l1_loss(prediction, windows[:, 1:, -1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (iteration + 1) % 1000 == 0:
            print(f"  Predictor {iteration + 1}/{iterations}", flush=True)
    model.eval()
    total, elements = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(real), 128):
            windows = torch.as_tensor(real[start:start + 128], dtype=torch.float32, device=device)
            prediction = model(windows[:, :-1, :-1]).sigmoid()
            total += F.l1_loss(prediction, windows[:, 1:, -1:], reduction="sum").item()
            elements += windows[:, 1:, -1:].numel()
    return total / elements
