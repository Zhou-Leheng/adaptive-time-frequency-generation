"""Distribution comparisons for unconditional generation."""

import numpy as np


def autocorrelation(windows, lags=32):
    centered = windows - windows.mean(axis=1, keepdims=True)
    variance = (centered ** 2).mean(axis=1).clip(min=1e-8)
    return np.stack([
        (centered[:, lag:] * centered[:, :-lag]).mean(axis=1) / variance
        for lag in range(1, min(lags, windows.shape[1] - 1) + 1)
    ], axis=1).mean(axis=0)


def distribution_metrics(real, generated, seed=1234, projections=128):
    if real.ndim != 3 or generated.ndim != 3 or real.shape[1:] != generated.shape[1:]:
        raise ValueError("Expected real and generated arrays shaped [samples, length, channels]")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("Evaluation received NaN or infinite samples")
    count = min(len(real), len(generated))
    if count < 2:
        raise ValueError("Evaluation needs at least two real and generated windows")
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), count, replace=False)].astype(np.float64)
    generated = generated[rng.choice(len(generated), count, replace=False)].astype(np.float64)
    flat_real, flat_generated = real.reshape(count, -1), generated.reshape(count, -1)
    directions = rng.standard_normal((flat_real.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    swd = np.abs(np.sort(flat_real @ directions, axis=0)
                 - np.sort(flat_generated @ directions, axis=0)).mean()
    power_real = (np.abs(np.fft.rfft(real, axis=1, norm="ortho")) ** 2).mean(axis=0)
    power_generated = (np.abs(np.fft.rfft(generated, axis=1, norm="ortho")) ** 2).mean(axis=0)
    psd = np.abs(power_real - power_generated).sum() / (power_real.sum() + 1e-8)
    acf = np.abs(autocorrelation(real) - autocorrelation(generated)).mean()
    delta = np.abs(np.sort(np.abs(np.diff(real, axis=1)).ravel())
                   - np.sort(np.abs(np.diff(generated, axis=1)).ravel())).mean()
    corr_real = np.nan_to_num(np.corrcoef(real.reshape(-1, real.shape[-1]), rowvar=False))
    corr_generated = np.nan_to_num(np.corrcoef(generated.reshape(-1, generated.shape[-1]), rowvar=False))
    correlation = np.abs(corr_real - corr_generated).mean()
    return dict(swd=float(swd), psd_relative_l1=float(psd), acf_mae=float(acf),
                delta_w1=float(delta), cross_corr_mae=float(correlation), evaluated_samples=count)


def save_preview(real, generated, routes, diffusion_times, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = min(real.shape[-1], 3)
    figure, axes = plt.subplots(2, channels, figsize=(4 * channels, 5), squeeze=False)
    for channel in range(channels):
        for sample in range(min(4, len(real), len(generated))):
            axes[0, channel].plot(real[sample, :, channel], alpha=0.65, linewidth=1)
            axes[1, channel].plot(generated[sample, :, channel], alpha=0.65, linewidth=1)
        axes[0, channel].set_title(f"Held-out windows, variable {channel + 1}")
        axes[1, channel].set_title(f"Generated windows, variable {channel + 1}")
    figure.tight_layout()
    figure.savefig(destination / "samples.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3), squeeze=False, layout="constrained")
    for axis, index in zip(axes[0], (0, len(diffusion_times) // 2, len(diffusion_times) - 1)):
        image = axis.imshow(routes[0, index].T, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axis.set_title(f"Diffusion step {diffusion_times[index]}")
        axis.set_yticks([0, 1, 2], ["Short", "Long", "Frequency"])
        axis.set_xlabel("Patch")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Fusion weight", fraction=0.025)
    figure.savefig(destination / "router.png", dpi=140, bbox_inches="tight")
    plt.close(figure)
