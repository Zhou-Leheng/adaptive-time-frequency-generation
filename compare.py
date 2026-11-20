"""Add official Diffusion-TS and evaluate saved fusion samples on one common protocol."""

import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import time

from filelock import FileLock
import numpy as np
import torch

from atfg.benchmark_metrics import (
    check_stop, discriminative_score, fit_metric_scaler, frechet_distance,
    official_correlational_score, predictive_score,
)
from atfg.data import download_etth1, prepare_data
from atfg.metrics import distribution_metrics
from atfg.upstream import ensure_upstream, OfficialDiffusionTS, REVISION
from run import atomic_json, set_seed, train_model


def atomic_array(path, values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_protocol(path, protocol):
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Saved protocol differs at {path}. Choose a new --name for a changed comparison.")
    atomic_json(path, protocol)


def load_shared_data(args, source_settings):
    data_path = Path(args.data)
    if args.download and not data_path.is_file():
        download_etth1(data_path)
    windows, metadata = prepare_data(
        data_path, source_settings["length"], source_settings["train_windows"],
        source_settings["eval_windows"], source_settings["seed"],
    )
    expected = source_settings["data"]
    for key in ("source_sha256", "columns", "split_rows", "window_counts"):
        if metadata[key] != expected[key]:
            raise ValueError(f"Dataset mismatch in {key}. Use the CSV from the existing fusion run.")
    for key in ("normalization_mean", "normalization_std"):
        np.testing.assert_allclose(metadata[key], expected[key], rtol=1e-10, atol=1e-10)
    return windows


def train_baseline(args, settings, windows, work, device, stop):
    directory = work / "diffusion_ts"
    directory.mkdir(exist_ok=True)
    sample_path, result_path = directory / "samples_normalized.npy", directory / "metrics.json"
    if sample_path.is_file() and result_path.is_file():
        print("Skip completed official Diffusion-TS baseline", flush=True)
        return
    training_args = SimpleNamespace(**settings)
    training_args.log_every = 100
    training_args.steps = args.baseline_steps or settings["steps"]
    training_args.lr = args.baseline_lr or settings["lr"]
    configuration = dict(
        source_settings=settings, upstream_revision=REVISION, baseline_steps=training_args.steps,
        baseline_lr=training_args.lr, encoder_layers=args.baseline_encoder_layers,
        decoder_layers=args.baseline_decoder_layers,
        objective="official L1 plus Fourier loss and official timestep weighting",
        protocol="shared z-score inputs, AdamW, validation-selected EMA, common unclipped DDIM",
    )
    factory = partial(OfficialDiffusionTS, diffusion_steps=settings["diffusion_steps"],
                      encoder_layers=args.baseline_encoder_layers, decoder_layers=args.baseline_decoder_layers)
    model, diffusion, training = train_model(
        training_args, "diffusion_ts", torch.from_numpy(windows[0]), torch.from_numpy(windows[1]),
        configuration, directory, device, stop, model_factory=factory,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(settings["seed"] + 300)
    noise = torch.randn((settings["samples"], settings["length"], windows[0].shape[-1]), generator=generator)
    generated = []
    start_time = time.monotonic()
    for start in range(0, len(noise), settings["batch_size"]):
        check_stop(stop)
        values, _, _ = diffusion.sample(
            model, noise[start:start + settings["batch_size"]].to(device), settings["sampling_steps"],
        )
        generated.append(values.cpu().numpy())
    sampling_seconds = time.monotonic() - start_time
    generated = np.concatenate(generated)
    result = dict(mode="diffusion_ts", seed=settings["seed"], **training,
                  **distribution_metrics(windows[2], generated, settings["seed"] + 400),
                  sampling_seconds=sampling_seconds, upstream_revision=REVISION)
    atomic_array(sample_path, generated)
    atomic_json(result_path, result)
    print(f"Official Diffusion-TS samples saved. SWD={result['swd']:.6f}", flush=True)


def shared_encoder(training, work, args, seed, device, stop):
    ensure_upstream()
    from Models.ts2vec.ts2vec import TS2Vec

    set_seed(seed)

    def progress(encoder, loss):
        check_stop(stop)
        if encoder.n_iters % 100 == 0:
            print(f"TS2Vec feature encoder {encoder.n_iters}/{args.encoder_iterations}", flush=True)

    encoder = TS2Vec(
        input_dims=training.shape[-1], output_dims=320, hidden_dims=64, depth=10,
        device=str(device), batch_size=8, lr=0.001, max_train_length=3000, after_iter_callback=progress,
    )
    path = work / "shared_ts2vec.pt"
    if path.is_file():
        encoder.load(path)
        print("Loaded the shared TS2Vec feature encoder", flush=True)
    else:
        encoder.fit(training, n_iters=args.encoder_iterations, verbose=False)
        temporary = path.with_suffix(".tmp")
        encoder.save(temporary)
        temporary.replace(path)
        print("Saved the shared TS2Vec feature encoder", flush=True)
    return encoder


def write_comparison(work, methods):
    rows = []
    for method in methods:
        path = work / "evaluation" / f"{method}.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if "summary" in value:
                rows.append(value["summary"])
    atomic_json(work / "comparison.json", rows)
    columns = ["method", "generator_seed", "evaluated_samples", "context_fid", "correlational_score",
               "discriminative_mean", "discriminative_std", "predictive_mean", "predictive_std",
               "evaluator_repeats", "swd", "psd_relative_l1", "acf_mae", "delta_w1", "cross_corr_mae",
               "trained_steps", "best_step", "parameters", "training_seconds", "sampling_seconds"]
    temporary = work / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(work / "comparison.csv")


def evaluate_method(method, paths, real, scaler, encoder, real_features, args, settings, work, device, stop):
    sample_path, generator_metrics = paths
    generated = np.load(sample_path, allow_pickle=False)
    if generated.shape != (settings["samples"], settings["length"], real.shape[-1]):
        raise ValueError(f"Unexpected saved sample shape for {method}: {generated.shape}")
    if not np.isfinite(generated).all():
        raise ValueError(f"Non-finite generated samples in {method}")
    minimum, span = scaler
    generated_scaled = ((generated - minimum) / span).astype(np.float32)
    input_sha = digest(sample_path)
    cache_path = work / "evaluation" / f"{method}.json"
    cache = dict(method=method, input_sha=input_sha, repeats=[])
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache["input_sha"] != input_sha:
            raise ValueError(f"The saved {method} samples changed. Choose a new --name.")
    if "context_fid" not in cache:
        check_stop(stop)
        features = encoder.encode(generated_scaled, encoding_window="full_series", batch_size=32)
        cache["context_fid"] = frechet_distance(real_features, features)
        cache["correlational_score"] = official_correlational_score(real, generated_scaled)
        atomic_json(cache_path, cache)
        print(f"{method}: Context-FID={cache['context_fid']:.6f}, correlation={cache['correlational_score']:.6f}", flush=True)
    for repetition in range(args.repeats):
        evaluator_seed = settings["seed"] + 1000 + repetition
        if len(cache["repeats"]) <= repetition:
            cache["repeats"].append(dict(seed=evaluator_seed))
        result = cache["repeats"][repetition]
        if "discriminative_score" not in result:
            print(f"{method}: discriminator repeat {repetition + 1}/{args.repeats}", flush=True)
            score, accuracy = discriminative_score(
                real, generated_scaled, evaluator_seed, args.discriminator_iterations, device, stop,
            )
            result.update(discriminative_score=score, discriminator_accuracy=accuracy)
            atomic_json(cache_path, cache)
        if "predictive_score" not in result:
            print(f"{method}: predictor repeat {repetition + 1}/{args.repeats}", flush=True)
            result["predictive_score"] = predictive_score(
                real, generated_scaled, evaluator_seed, args.predictor_iterations, device, stop,
            )
            atomic_json(cache_path, cache)
    discriminator = [row["discriminative_score"] for row in cache["repeats"]]
    predictor = [row["predictive_score"] for row in cache["repeats"]]
    real_zscore = real * span + minimum
    diagnostics = distribution_metrics(real_zscore, generated, settings["seed"] + 400)
    generator_result = json.loads(generator_metrics.read_text(encoding="utf-8"))
    training_keys = ("trained_steps", "best_step", "parameters", "training_seconds", "sampling_seconds")
    cache["summary"] = dict(
        method=method, generator_seed=settings["seed"],
        context_fid=cache["context_fid"], correlational_score=cache["correlational_score"],
        discriminative_mean=float(np.mean(discriminator)), discriminative_std=float(np.std(discriminator)),
        predictive_mean=float(np.mean(predictor)), predictive_std=float(np.std(predictor)),
        evaluator_repeats=args.repeats, **diagnostics,
        **{key: generator_result[key] for key in training_keys},
    )
    atomic_json(cache_path, cache)
    print(f"Completed evaluation of {method}", flush=True)


def compare(args):
    for key in ("encoder_iterations", "discriminator_iterations", "predictor_iterations", "repeats",
                "baseline_encoder_layers", "baseline_decoder_layers"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.baseline_steps is not None and args.baseline_steps < 1:
        raise ValueError("baseline_steps must be positive")
    if args.baseline_lr is not None and args.baseline_lr <= 0:
        raise ValueError("baseline_lr must be positive")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        raise ValueError("name must be a folder name")
    root = Path(args.results)
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    methods = ("fixed", "global", "patch", "diffusion_ts")
    paths = {method: (root / method / "samples_normalized.npy", root / method / "metrics.json")
             for method in methods[:-1]}
    for method, pair in paths.items():
        if not all(path.is_file() for path in pair):
            raise FileNotFoundError(f"Complete the existing {method} run first: {pair}")
    work = root / args.name
    work.mkdir(exist_ok=True)
    (work / "evaluation").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    protocol = dict(
        source_settings=settings, upstream_revision=REVISION,
        baseline_steps=args.baseline_steps or settings["steps"], baseline_lr=args.baseline_lr or settings["lr"],
        baseline_encoder_layers=args.baseline_encoder_layers, baseline_decoder_layers=args.baseline_decoder_layers,
        encoder_iterations=args.encoder_iterations, discriminator_iterations=args.discriminator_iterations,
        predictor_iterations=args.predictor_iterations, evaluator_repeats=args.repeats,
        metric_scaling="shared min-max fitted on real training windows, no clipping",
        context_encoder="official TS2Vec, trained once on real training windows, shared by all methods",
        correlation="official Diffusion-TS CrossCorrelLoss",
        posthoc_evaluators="PyTorch single-layer GRU, hidden=floor(channels/2), TimeGAN-style definitions",
        sampling=f"common {settings['diffusion_steps']}-level cosine diffusion, DDIM eta=0 without clipping",
    )
    with FileLock(str(root / ".run.lock"), timeout=0):
        same_protocol(work / "protocol.json", protocol)
        windows = load_shared_data(args, settings)
        if min(len(windows[2]), settings["samples"]) < 8:
            raise ValueError("This comparison needs at least eight real and generated windows")
        stop = {"requested": False}
        previous = signal.getsignal(signal.SIGTERM)

        def request_stop(signum, frame):
            stop["requested"] = True

        signal.signal(signal.SIGTERM, request_stop)
        try:
            ensure_upstream()
            print(f"Comparison device={device}. Existing three fusion sample sets will be reused.", flush=True)
            train_baseline(args, settings, windows, work, device, stop)
            paths["diffusion_ts"] = (work / "diffusion_ts/samples_normalized.npy", work / "diffusion_ts/metrics.json")
            scaler = fit_metric_scaler(windows[0])
            minimum, span = scaler
            training_scaled = ((windows[0] - minimum) / span).astype(np.float32)
            real_scaled = ((windows[2] - minimum) / span).astype(np.float32)
            encoder = shared_encoder(training_scaled, work, args, settings["seed"] + 500, device, stop)
            real_features = encoder.encode(real_scaled, encoding_window="full_series", batch_size=32)
            for method in methods:
                check_stop(stop)
                evaluate_method(method, paths[method], real_scaled, scaler, encoder, real_features,
                                args, settings, work, device, stop)
                write_comparison(work, methods)
            print(f"Finished comparison: {work / 'comparison.csv'}", flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results", required=True, help="Existing folder containing settings.json and fixed/global/patch")
    result.add_argument("--data", default="data/ETTh1.csv")
    result.add_argument("--download", action="store_true")
    result.add_argument("--name", default="benchmark_diffusion_ts")
    result.add_argument("--baseline-steps", type=int)
    result.add_argument("--baseline-lr", type=float)
    result.add_argument("--baseline-encoder-layers", type=int, default=3)
    result.add_argument("--baseline-decoder-layers", type=int, default=2)
    result.add_argument("--encoder-iterations", type=int, default=600)
    result.add_argument("--discriminator-iterations", type=int, default=2000)
    result.add_argument("--predictor-iterations", type=int, default=5000)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--device", default="auto")
    return result


if __name__ == "__main__":
    try:
        compare(parser().parse_args())
    except InterruptedError as error:
        print(error, flush=True)
        raise SystemExit(130)
