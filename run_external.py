"""Train three official-method baselines and score them with the existing ETTh1 protocol."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from filelock import FileLock
import numpy as np
import torch

from atfg.benchmark_metrics import check_stop, fit_metric_scaler
from atfg.external import INPUT_TRANSFORMS, REPOSITORIES, SAMPLING_STEPS
from compare import digest, evaluate_method, load_shared_data, same_protocol, shared_encoder
from run import atomic_json
from run_overlap import boundary_diagnostics, load_reference


BASE = Path(__file__).resolve().parent
BENCHMARK = "benchmark_diffusion_ts"
DEFAULT_NAME = "benchmark_three_baselines"


def upstream_checkout(method, cache):
    repo, revision = REPOSITORIES[method]
    destination = cache / f"{method}-{revision[:12]}"
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.is_dir():
            return destination
        print(f"[{method}] Download {repo} at {revision}", flush=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            temporary = Path(folder)
            archive_path = temporary / "source.zip"
            url = f"https://codeload.github.com/{repo}/zip/{revision}"
            with urllib.request.urlopen(url, timeout=120) as remote, archive_path.open("wb") as local:
                shutil.copyfileobj(remote, local)
            extracted = temporary / "source"
            with zipfile.ZipFile(archive_path) as archive:
                for entry in archive.infolist():
                    parts = Path(entry.filename).parts[1:]
                    if entry.is_dir() or not parts:
                        continue
                    path = extracted.joinpath(*parts)
                    if not path.resolve().is_relative_to(extracted.resolve()):
                        raise ValueError("Unexpected path in the upstream code archive")
                    if path.suffix not in (".py", ".yaml") and path.name != "LICENSE":
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(entry))
            extracted.replace(destination)
    return destination


def load_existing(root, base_protocol, settings):
    locations = {
        "diffusion_ts": root / BENCHMARK,
        "patch_overlap": root / "overlap_output",
        "global_overlap": root / "overlap_routing",
    }
    rows = {}
    fingerprints = {}
    paths = {}
    for method, directory in locations.items():
        evaluation = directory / "evaluation" / f"{method}.json"
        if method == "global_overlap" and not evaluation.is_file():
            continue
        if method != "diffusion_ts":
            protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
            if (protocol["reference_protocol"] != base_protocol or protocol["method"] != method
                    or protocol["shared_encoder_sha"] != digest(root / BENCHMARK / "shared_ts2vec.pt")):
                raise ValueError(f"Saved {method} evaluation uses a different protocol")
        samples = directory / method / "samples_normalized.npy"
        metrics = directory / method / "metrics.json"
        value, fingerprints[method] = load_reference(samples, metrics, evaluation, method, settings["seed"])
        rows[method] = value["summary"]
        paths[method] = samples
    return rows, fingerprints, paths


def write_table(work, rows):
    columns = ["method", "generator_seed", "trained_steps", "best_step", "parameters", "evaluated_samples",
               "context_fid", "predictive_mean", "predictive_std", "correlational_score", "discriminative_mean",
               "discriminative_std", "swd", "psd_relative_l1", "acf_mae", "delta_w1", "cross_corr_mae",
               "patch_boundary_ratio", "sampling_steps", "training_seconds", "sampling_seconds", "training_device"]
    records = list(rows.values())
    atomic_json(work / "comparison.json", records)
    temporary = work / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(work / "comparison.csv")


def run_worker(source, arrays, directory, device, stop):
    command = [sys.executable, "-u", str(BASE / "external_worker.py"), "--source", str(source),
               "--input", str(arrays), "--output", str(directory), "--device", str(device)]
    environment = os.environ.copy()
    if device.type == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    with subprocess.Popen(command, env=environment) as process:
        while True:
            try:
                status = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if stop["requested"] and os.name != "nt":
                    process.terminate()
        check_stop(stop)
        if status != 0:
            raise subprocess.CalledProcessError(status, command)


def compare_external(args, stop):
    reserved = ("", ".", "..", "fixed", "global", "patch", BENCHMARK, "overlap_output", "overlap_routing")
    if Path(args.name).name != args.name or args.name in reserved:
        raise ValueError("Choose a new result folder name")
    if (args.train_steps is not None and args.train_steps < 1) or len(set(args.methods)) != len(args.methods):
        raise ValueError("Use a positive training budget and unique method names")
    root = args.results.resolve()
    work = root / args.name
    benchmark = root / BENCHMARK
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    base_protocol = json.loads((benchmark / "protocol.json").read_text(encoding="utf-8"))
    encoder = benchmark / "shared_ts2vec.pt"
    if base_protocol["source_settings"] != settings or not encoder.is_file():
        raise ValueError("Complete the original Diffusion-TS benchmark on this dataset first")
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    train_steps = args.train_steps if args.train_steps is not None else settings["steps"]
    native = {
        method: dict(method=method, repository=repo, revision=revision, seed=settings["seed"],
                     train_steps=train_steps, batch_size=settings["batch_size"], check_every=settings["check_every"],
                     samples=settings["samples"], length=settings["length"], channels=len(settings["data"]["columns"]),
                     sampling_steps=SAMPLING_STEPS[method], normalization=INPUT_TRANSFORMS[method])
        for method, (repo, revision) in REPOSITORIES.items()
    }
    args.repeats = base_protocol["evaluator_repeats"]
    for name in ("encoder_iterations", "discriminator_iterations", "predictor_iterations"):
        setattr(args, name, base_protocol[name])

    with FileLock(str(root / ".run.lock"), timeout=0), tempfile.TemporaryDirectory(prefix="atfg_ext_") as scratch:
        rows, fingerprints, paths = load_existing(root, base_protocol, settings)
        work.mkdir(exist_ok=True)
        (work / "evaluation").mkdir(exist_ok=True)
        same_protocol(work / "protocol.json", dict(
            source_settings=settings, evaluation_protocol=base_protocol, existing_results=fingerprints,
            shared_encoder_sha=digest(encoder), methods=native,
            trainer="common batches and gradient clip; upstream objectives, optimizer schedules and sampling",
            checkpoint_selection="lowest native objective on the fixed-noise held-out validation windows",
        ))
        windows = load_shared_data(args, settings)
        arrays = Path(scratch) / "training_windows.npz"
        # Test data stays in the evaluator process.
        np.savez(arrays, train=windows[0], validation=windows[1])
        for method, path in paths.items():
            diagnostic = boundary_diagnostics(np.load(path, allow_pickle=False), settings["patch_size"])
            rows[method]["patch_boundary_ratio"] = diagnostic["patch_boundary_ratio"]
            rows[method]["sampling_steps"] = settings["sampling_steps"]
        minimum, span = fit_metric_scaler(windows[0])
        training_scaled = ((windows[0] - minimum) / span).astype(np.float32)
        real = ((windows[2] - minimum) / span).astype(np.float32)
        feature_encoder = None
        failures = []
        if not (work / "comparison.csv").is_file():
            write_table(work, rows)
        for method, spec in native.items():
            check_stop(stop)
            directory = work / method
            metrics = directory / "metrics.json"
            samples = directory / "samples_normalized.npy"
            if method in args.methods and not (metrics.is_file() and samples.is_file()):
                source = upstream_checkout(method, BASE / "data" / "external_baselines")
                directory.mkdir(exist_ok=True)
                same_protocol(directory / "training_protocol.json", spec)
                try:
                    run_worker(source, arrays, directory, device, stop)
                except subprocess.CalledProcessError as error:
                    print(f"[{method}] Failed with exit code {error.returncode}. See the log above.", flush=True)
                    failures.append(method)
                    continue
            if not (metrics.is_file() and samples.is_file()):
                continue
            metadata = json.loads(metrics.read_text(encoding="utf-8"))
            if metadata["samples_sha"] != digest(samples):
                raise ValueError(f"Saved samples changed for {method}")
            cache = work / "evaluation" / f"{method}.json"
            value = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
            if value and value["input_sha"] != metadata["samples_sha"]:
                raise ValueError(f"Saved evaluation does not match {method} samples")
            if "summary" not in value and method in args.methods:
                if feature_encoder is None:
                    feature_encoder = shared_encoder(training_scaled, benchmark, args, settings["seed"] + 500,
                                                     device, stop)
                    real_features = feature_encoder.encode(real, encoding_window="full_series", batch_size=32)
                evaluate_method(method, (samples, metrics), real, (minimum, span), feature_encoder, real_features,
                                args, settings, work, device, stop)
                value = json.loads(cache.read_text(encoding="utf-8"))
            if "summary" in value:
                values = np.load(samples, allow_pickle=False)
                diagnostic = boundary_diagnostics(values, settings["patch_size"])
                rows[method] = dict(value["summary"], patch_boundary_ratio=diagnostic["patch_boundary_ratio"],
                                    sampling_steps=metadata["sampling_steps"], training_device=metadata["device"])
                write_table(work, rows)
        if failures:
            raise RuntimeError(f"Incomplete methods: {', '.join(failures)}. Completed rows: {work / 'comparison.csv'}")
        print(f"Finished comparison: {work / 'comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--data", default="data/ETTh1.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--methods", nargs="+", choices=tuple(REPOSITORIES), default=list(REPOSITORIES))
    args = parser.parse_args()
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    try:
        compare_external(args, stop)
    except InterruptedError as error:
        print(error, flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
