"""Train one overlap-output ablation and reuse the completed comparison evaluators."""

import argparse
import csv
import json
from pathlib import Path
import signal
from types import SimpleNamespace

from filelock import FileLock
import numpy as np
import torch

from atfg.benchmark_metrics import check_stop, fit_metric_scaler
from atfg.overlap import OverlapFusionDenoiser
from compare import digest, evaluate_method, load_shared_data, same_protocol, shared_encoder
from run import atomic_json, generate_and_evaluate, train_model


METHODS = ("fixed", "global", "patch", "diffusion_ts", "patch_overlap", "global_overlap")


def boundary_diagnostics(values, patch_size):
    difference = np.abs(np.diff(values.astype(np.float64), axis=1))
    boundary = np.arange(values.shape[1] - 1) % patch_size == patch_size - 1
    interior_mean = float(difference[:, ~boundary].mean())
    boundary_mean = float(difference[:, boundary].mean()) if boundary.any() else None
    return dict(
        patch_boundary_ratio=boundary_mean / max(interior_mean, 1e-12) if boundary.any() else None,
        boundary_mean_abs_change=boundary_mean, interior_mean_abs_change=interior_mean,
        mean_abs_change=float(difference.mean()),
        mean_within_window_std=float(values.std(axis=1).mean()),
        within_window_std_by_variable=values.std(axis=1).mean(axis=0).tolist(),
    )


def load_reference(samples, metrics, evaluation, method, seed):
    if not all(path.is_file() for path in (samples, metrics, evaluation)):
        raise FileNotFoundError(f"Complete the reference evaluation first: {method}")
    value = json.loads(evaluation.read_text(encoding="utf-8"))
    samples_sha = digest(samples)
    if value.get("input_sha") != samples_sha or "summary" not in value:
        raise ValueError(f"Reference evaluation is incomplete or stale for {method}")
    if value["summary"]["generator_seed"] != seed:
        raise ValueError(f"Reference seed differs for {method}")
    provenance = dict(samples_sha=samples_sha, metrics_sha=digest(metrics),
                      evaluation_sha=digest(evaluation))
    return value, provenance


def reference_files(root, benchmark, settings):
    paths = {method: (root / method / "samples_normalized.npy", root / method / "metrics.json")
             for method in METHODS[:3]}
    paths["diffusion_ts"] = (benchmark / "diffusion_ts/samples_normalized.npy",
                             benchmark / "diffusion_ts/metrics.json")
    references = {}
    cached = {}
    for method, (samples, metrics) in paths.items():
        evaluation = benchmark / "evaluation" / f"{method}.json"
        cached[method], references[method] = load_reference(
            samples, metrics, evaluation, method, settings["seed"],
        )
    return paths, cached, references


def write_summary(work, cached, diagnostics, methods):
    rows = []
    for method in methods:
        if method not in cached or "summary" not in cached[method]:
            continue
        row = dict(cached[method]["summary"])
        row.update({key: diagnostics[method][key] for key in
                    ("patch_boundary_ratio", "mean_abs_change", "mean_within_window_std")})
        rows.append(row)
    atomic_json(work / "comparison.json", rows)
    columns = ["method", "generator_seed", "context_fid", "discriminative_mean", "discriminative_std",
               "predictive_mean", "predictive_std", "correlational_score", "patch_boundary_ratio",
               "mean_abs_change", "mean_within_window_std", "swd", "psd_relative_l1", "acf_mae",
               "delta_w1", "cross_corr_mae", "evaluated_samples", "evaluator_repeats", "trained_steps",
               "best_step", "parameters", "training_seconds", "sampling_seconds"]
    temporary = work / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(work / "comparison.csv")


def run_overlap(args):
    method = f"{args.mode}_overlap"
    compare_routing = args.mode == "global"
    output_name = args.name or ("overlap_routing" if compare_routing else "overlap_output")
    methods = ("global_overlap", "patch_overlap") if compare_routing else METHODS[:-1]
    folder_names = (output_name, args.benchmark_name)
    if compare_routing:
        folder_names += (args.patch_reference,)
    for name in folder_names:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Output names must be folder names")
    if len(set(folder_names)) != len(folder_names) or output_name in METHODS:
        raise ValueError("Choose a separate folder for the overlap ablation")
    root = Path(args.results)
    benchmark = root / args.benchmark_name
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    original_protocol = json.loads((benchmark / "protocol.json").read_text(encoding="utf-8"))
    if original_protocol["source_settings"] != settings:
        raise ValueError("The original comparison and fusion settings differ")
    if settings["patch_size"] < 2 or settings["patch_size"] % 2:
        raise ValueError("Overlap output requires an even patch size")
    encoder_path = benchmark / "shared_ts2vec.pt"
    if not encoder_path.is_file():
        raise FileNotFoundError("Finish the original comparison to save its shared feature encoder")
    evaluation_args = SimpleNamespace(
        encoder_iterations=original_protocol["encoder_iterations"],
        discriminator_iterations=original_protocol["discriminator_iterations"],
        predictor_iterations=original_protocol["predictor_iterations"],
        repeats=original_protocol["evaluator_repeats"],
    )
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    with FileLock(str(root / ".run.lock"), timeout=0):
        paths, cached, references = reference_files(root, benchmark, settings)
        protocol = dict(
            source_settings=settings, reference_protocol=original_protocol, reference_files=references,
            shared_encoder_sha=digest(encoder_path), method="patch_overlap",
            input_patch_size=settings["patch_size"], output_patch_size=2 * settings["patch_size"],
            output_stride=settings["patch_size"], output_padding=settings["patch_size"] // 2,
            merge="positive Hann taper with per-position coverage normalization",
            router="original patch-constant softmax weights", frequency_expert="original unchanged",
            initialization="same shared initial parameters and training RNG; central output rows copied",
        )
        if compare_routing:
            patch_reference = root / args.patch_reference
            patch_protocol = patch_reference / "protocol.json"
            if not patch_protocol.is_file():
                raise FileNotFoundError(f"Complete patch_overlap first: {patch_protocol}")
            if json.loads(patch_protocol.read_text(encoding="utf-8")) != protocol:
                raise ValueError("The patch_overlap reference uses a different overlap or evaluation protocol")
            patch_dir = patch_reference / "patch_overlap"
            patch_paths = (patch_dir / "samples_normalized.npy", patch_dir / "metrics.json")
            patch_result, patch_provenance = load_reference(
                *patch_paths, patch_reference / "evaluation/patch_overlap.json", "patch_overlap", settings["seed"],
            )
            if patch_result["summary"]["trained_steps"] != settings["steps"]:
                raise ValueError("The patch_overlap reference uses a different training budget")
            paths = {"patch_overlap": patch_paths}
            cached = {"patch_overlap": patch_result}
            protocol.update(
                method=method, router="one learned softmax vector per sequence and diffusion step",
                patch_reference=dict(folder=args.patch_reference, **patch_provenance),
            )
        expected_seeds = [settings["seed"] + 1000 + i for i in range(evaluation_args.repeats)]
        for reference, value in cached.items():
            if (value["summary"]["evaluator_repeats"] != evaluation_args.repeats
                    or [row["seed"] for row in value["repeats"]] != expected_seeds
                    or any("predictive_score" not in row or "discriminative_score" not in row
                           for row in value["repeats"])):
                raise ValueError(f"Reference evaluator repeats differ or are incomplete for {reference}")
        work = root / output_name
        work.mkdir(exist_ok=True)
        (work / "evaluation").mkdir(exist_ok=True)
        same_protocol(work / "protocol.json", protocol)
        windows = load_shared_data(args, settings)
        stop = {"requested": False}
        previous = signal.getsignal(signal.SIGTERM)

        def request_stop(signum, frame):
            stop["requested"] = True

        signal.signal(signal.SIGTERM, request_stop)
        try:
            directory = work / method
            directory.mkdir(exist_ok=True)
            samples, metrics = directory / "samples_normalized.npy", directory / "metrics.json"
            if not (samples.is_file() and metrics.is_file()):
                training_args = SimpleNamespace(**settings, log_every=100)
                configuration = dict(settings, mode=args.mode, decoder="overlap_output_v1",
                                     output_patch_size=2 * settings["patch_size"])
                print(f"Train {method} from scratch: {settings['steps']} steps on {device}", flush=True)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                model, diffusion, training = train_model(
                    training_args, args.mode, torch.from_numpy(windows[0]), torch.from_numpy(windows[1]),
                    configuration, directory, device, stop, model_factory=OverlapFusionDenoiser,
                )
                evaluation = generate_and_evaluate(
                    training_args, model, diffusion, windows[2], directory, device, stop,
                )
                result = dict(mode=method, seed=settings["seed"], **training, **evaluation,
                              peak_gpu_mib=torch.cuda.max_memory_allocated(device) / 2**20
                              if device.type == "cuda" else 0.0)
                atomic_json(metrics, result)
                del model, diffusion
            else:
                print(f"Skip completed {method} training and sampling", flush=True)
            paths[method] = (samples, metrics)
            diagnostics = {
                "real_train": boundary_diagnostics(windows[0], settings["patch_size"]),
                "real_test": boundary_diagnostics(windows[2], settings["patch_size"]),
            }
            for label, (sample_path, _) in paths.items():
                values = np.load(sample_path, allow_pickle=False)
                expected = (settings["samples"], settings["length"], windows[0].shape[-1])
                if values.shape != expected or not np.isfinite(values).all():
                    raise ValueError(f"Invalid sample array for {label}")
                diagnostics[label] = boundary_diagnostics(values, settings["patch_size"])
            atomic_json(work / "boundary_diagnostics.json", diagnostics)
            cache_path = work / "evaluation" / f"{method}.json"
            current = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
            if current and current.get("input_sha") != digest(samples):
                raise ValueError("Saved overlap samples changed. Choose a new --name.")
            if "summary" not in current:
                write_summary(work, cached, diagnostics, methods)
                check_stop(stop)
                minimum, span = fit_metric_scaler(windows[0])
                train_scaled = ((windows[0] - minimum) / span).astype(np.float32)
                real_scaled = ((windows[2] - minimum) / span).astype(np.float32)
                encoder = shared_encoder(train_scaled, benchmark, evaluation_args,
                                         settings["seed"] + 500, device, stop)
                features = encoder.encode(real_scaled, encoding_window="full_series", batch_size=32)
                evaluate_method(method, paths[method], real_scaled, (minimum, span),
                                encoder, features, evaluation_args, settings, work, device, stop)
                current = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                print(f"Skip completed {method} evaluation", flush=True)
            cached[method] = current
            write_summary(work, cached, diagnostics, methods)
            reference = "patch_overlap" if compare_routing else "patch"
            before = diagnostics[reference]["patch_boundary_ratio"]
            after = diagnostics[method]["patch_boundary_ratio"]
            print(f"Patch boundary ratio: {reference}={before}, {method}={after}", flush=True)
            print(f"Finished overlap comparison: {work / 'comparison.csv'}", flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results", required=True, help="Completed original fusion experiment folder")
    result.add_argument("--mode", choices=("patch", "global"), default="patch")
    result.add_argument("--name", help="Output folder (default overlap_output or overlap_routing)")
    result.add_argument("--patch-reference", default="overlap_output", help="Completed patch_overlap for global mode")
    result.add_argument("--benchmark-name", default="benchmark_diffusion_ts")
    result.add_argument("--data", default="data/ETTh1.csv")
    result.add_argument("--download", action="store_true")
    result.add_argument("--device", default="auto")
    return result


if __name__ == "__main__":
    try:
        run_overlap(parser().parse_args())
    except InterruptedError as error:
        print(error, flush=True)
        raise SystemExit(130)
