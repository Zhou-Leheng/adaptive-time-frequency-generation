"""Train the three fusion controls, generate samples, and write a shared summary."""

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import signal
import time

from filelock import FileLock
import numpy as np
import torch
from torch.nn import functional as F

from atfg.data import download_etth1, prepare_data
from atfg.diffusion import GaussianDiffusion
from atfg.metrics import distribution_metrics, save_preview
from atfg.model import FusionDenoiser


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def validation_loss(model, diffusion, data, batch_size, seed, device):
    generator = torch.Generator().manual_seed(seed)
    steps = torch.randint(diffusion.steps, (len(data),), generator=generator)
    noise = torch.randn(data.shape, generator=generator)
    total = 0.0
    model.eval()
    for start in range(0, len(data), batch_size):
        clean = data[start:start + batch_size].to(device)
        indices = steps[start:start + batch_size].to(device)
        noisy = diffusion.add_noise(clean, indices, noise[start:start + batch_size].to(device))
        predicted, _ = model(noisy, indices)
        total += F.mse_loss(predicted, clean, reduction="sum").item()
    return total / data.numel()


def train_model(args, mode, train, validation, configuration, directory, device, stop,
                model_factory=FusionDenoiser):
    set_seed(args.seed)
    model = model_factory(
        length=args.length, channels=train.shape[-1], patch_size=args.patch_size,
        width=args.width, heads=args.heads, layers=args.layers, mode=mode,
    ).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(args.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    batch_generator = torch.Generator().manual_seed(args.seed + 100)
    step, best_loss, best_step, trained_seconds = 0, float("inf"), 0, 0.0
    checkpoint_path = directory / "last.pt"
    best_path = directory / "best.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["configuration"] != configuration:
            raise ValueError(f"Existing checkpoint configuration differs in {directory}. Choose a new --output.")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move optimizer moments to the active device after loading a CPU checkpoint.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        batch_generator.set_state(checkpoint["batch_rng"])
        restore_rng(checkpoint["rng"])
        step, best_loss, best_step = checkpoint["step"], checkpoint["best_loss"], checkpoint["best_step"]
        trained_seconds = checkpoint["trained_seconds"]
        print(f"Resume {mode} at step {step}/{args.steps}", flush=True)

    start_time = time.monotonic()

    def save_last():
        atomic_checkpoint(checkpoint_path, dict(
            configuration=configuration, model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), step=step, best_loss=best_loss, best_step=best_step,
            batch_rng=batch_generator.get_state(), rng=rng_state(),
            trained_seconds=trained_seconds + time.monotonic() - start_time,
        ))

    while step < args.steps:
        if stop["requested"]:
            if step:
                save_last()
            raise InterruptedError("Stop requested. Rerun the same command to resume.")
        model.train()
        indices = torch.randint(len(train), (args.batch_size,), generator=batch_generator)
        clean = train[indices].to(device)
        times = torch.randint(diffusion.steps, (len(clean),), device=device)
        noise = torch.randn_like(clean)
        if hasattr(model, "training_loss"):
            loss = model.training_loss(clean, times, noise)
        else:
            noisy = diffusion.add_noise(clean, times, noise)
            prediction, _ = model(noisy, times)
            loss = F.mse_loss(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for averaged, current in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(current, 1 - args.ema_decay)
        step += 1
        if step % args.log_every == 0 or step == 1:
            label = "train_loss" if hasattr(model, "training_loss") else "train_mse"
            print(f"{mode} step={step}/{args.steps} {label}={loss.item():.7f}", flush=True)
        if step % args.check_every == 0 or step == args.steps:
            val = validation_loss(ema, diffusion, validation, args.batch_size, args.seed + 200, device)
            if not np.isfinite(val):
                raise FloatingPointError(f"Non-finite validation loss at step {step}")
            if val < best_loss:
                best_loss, best_step = val, step
                atomic_checkpoint(best_path, dict(
                    configuration=configuration, model=ema.state_dict(), step=step, validation_mse=val,
                ))
            save_last()
            print(f"{mode} step={step} val_mse={val:.7f} best_step={best_step} best_val={best_loss:.7f}",
                  flush=True)
    if stop["requested"]:
        raise InterruptedError("Stop requested. Training is saved; rerun for sampling.")
    if not best_path.is_file():
        raise RuntimeError(f"Missing validation-selected checkpoint {best_path}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    ema.load_state_dict(best["model"])
    ema.eval()
    return ema, diffusion, dict(
        trained_steps=step, best_step=best_step, validation_mse=best_loss,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        training_seconds=trained_seconds + time.monotonic() - start_time,
    )


def generate_and_evaluate(args, model, diffusion, test, directory, device, stop):
    generator = torch.Generator().manual_seed(args.seed + 300)
    # The same CPU-generated initial noise bank is used for all fusion modes.
    initial_noise = torch.randn((args.samples, args.length, test.shape[-1]), generator=generator)
    generated, traces = [], []
    start_time = time.monotonic()
    for start in range(0, args.samples, args.batch_size):
        if stop["requested"]:
            raise InterruptedError("Sampling stopped. Rerun the same command to regenerate.")
        samples, weights, times = diffusion.sample(
            model, initial_noise[start:start + args.batch_size].to(device), args.sampling_steps,
        )
        generated.append(samples.cpu().numpy())
        if start == 0:
            traces = weights.numpy()
    sampling_seconds = time.monotonic() - start_time
    generated = np.concatenate(generated)
    result = distribution_metrics(test, generated, seed=args.seed + 400)
    result["sampling_seconds"] = sampling_seconds
    np.save(directory / "samples_normalized.npy", generated)
    np.savez_compressed(directory / "router_weights.npz", weights=traces, diffusion_times=times,
                        experts=np.array(["short", "long", "frequency"]))
    save_preview(test, generated, traces, times, directory)
    result["mean_router_weights"] = traces.mean(axis=(0, 1, 2)).tolist()
    result["patch_router_std"] = float(traces.std(axis=2).mean())
    return result


def write_summary(output):
    rows = []
    for mode in ("fixed", "global", "patch"):
        path = output / mode / "metrics.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    atomic_json(output / "summary.json", rows)
    columns = ["mode", "seed", "trained_steps", "best_step", "validation_mse", "swd",
               "psd_relative_l1", "acf_mae", "delta_w1", "cross_corr_mae", "parameters",
               "training_seconds", "sampling_seconds", "peak_gpu_mib"]
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")


def run_experiment(args):
    for name in ("steps", "batch_size", "train_windows", "check_every", "log_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.samples < 2 or args.eval_windows < 2 or args.lr <= 0 or not 0 <= args.ema_decay < 1:
        raise ValueError("Use at least two samples/eval_windows, positive lr, and EMA decay in [0,1)")
    if not 1 <= args.sampling_steps <= args.diffusion_steps:
        raise ValueError("sampling_steps must be between 1 and diffusion_steps")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("Fusion modes must be unique")
    data_path, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if args.download and not data_path.is_file():
        print(f"Downloading ETTh1 to {data_path}", flush=True)
        download_etth1(data_path)
    windows, metadata = prepare_data(data_path, args.length, args.train_windows, args.eval_windows, args.seed)
    train, validation = (torch.from_numpy(window) for window in windows[:2])
    settings = {key: value for key, value in vars(args).items()
                if key not in ("data", "output", "device", "download", "modes", "log_every")}
    settings["data"] = metadata
    with FileLock(str(output / ".run.lock"), timeout=0):
        settings_path = output / "settings.json"
        if settings_path.is_file() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise ValueError(f"Settings differ from the saved run. Choose a new --output directory.")
        atomic_json(settings_path, settings)
        stop = {"requested": False}
        previous_handler = signal.getsignal(signal.SIGTERM)

        def request_stop(signum, frame):
            stop["requested"] = True

        signal.signal(signal.SIGTERM, request_stop)
        try:
            print(f"Device={device}, window counts={metadata['window_counts']}, channels={len(metadata['columns'])}", flush=True)
            for mode in args.modes:
                directory = output / mode
                directory.mkdir(exist_ok=True)
                metrics_path = directory / "metrics.json"
                if metrics_path.is_file():
                    print(f"Skip completed mode={mode}", flush=True)
                    continue
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                configuration = dict(settings, mode=mode)
                model, diffusion, training = train_model(
                    args, mode, train, validation, configuration, directory, device, stop,
                )
                evaluation = generate_and_evaluate(args, model, diffusion, windows[2], directory, device, stop)
                result = dict(mode=mode, seed=args.seed, **training, **evaluation,
                              peak_gpu_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0)
                atomic_json(metrics_path, result)
                write_summary(output)
                print(f"Finished {mode}: SWD={result['swd']:.6f}, PSD={result['psd_relative_l1']:.6f}, "
                      f"ACF={result['acf_mae']:.6f}, delta_W1={result['delta_w1']:.6f}", flush=True)
                del model, diffusion
            write_summary(output)
            print(f"Results saved in {output.resolve()}", flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", default="data/ETTh1.csv")
    result.add_argument("--download", action="store_true")
    result.add_argument("--output", default="results/etth1_seed42")
    result.add_argument("--modes", nargs="+", choices=("fixed", "global", "patch"), default=["fixed", "global", "patch"])
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--length", type=int, default=128)
    result.add_argument("--patch-size", type=int, default=8)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--heads", type=int, default=4)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--steps", type=int, default=5000)
    result.add_argument("--diffusion-steps", type=int, default=200)
    result.add_argument("--sampling-steps", type=int, default=50)
    result.add_argument("--train-windows", type=int, default=4096)
    result.add_argument("--eval-windows", type=int, default=512)
    result.add_argument("--samples", type=int, default=512)
    result.add_argument("--ema-decay", type=float, default=0.999)
    result.add_argument("--check-every", type=int, default=500)
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument("--device", default="auto")
    return result


if __name__ == "__main__":
    try:
        run_experiment(parser().parse_args())
    except InterruptedError as error:
        print(error, flush=True)
        raise SystemExit(130)
