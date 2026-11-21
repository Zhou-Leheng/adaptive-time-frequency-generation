"""Train and sample a single external baseline in an isolated Python process."""

import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from atfg.external import build_native
from compare import atomic_array, digest
from run import atomic_checkpoint, atomic_json, restore_rng, rng_state, set_seed


@torch.no_grad()
def validation_loss(recipe, model, validation, batch_size, seed, device):
    state = rng_state()
    model.eval()
    try:
        set_seed(seed)
        scores = [float(recipe.loss(model, batch.to(device), False)) * len(batch)
                  for batch in validation.split(batch_size)]
        return sum(scores) / len(validation)
    finally:
        restore_rng(state)


def work(args, stop):
    output = args.output
    settings = json.loads((output / "training_protocol.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    seed, steps = settings["seed"], settings["train_steps"]
    batch_size, sampling_steps = settings["batch_size"], settings["sampling_steps"]
    method = settings["method"]
    set_seed(seed)
    with np.load(args.input, allow_pickle=False) as arrays:
        training, validation = (torch.from_numpy(arrays[key]) for key in ("train", "validation"))
    recipe, training, validation = build_native(
        method, args.source.resolve(), training, validation, steps, sampling_steps, device, seed,
    )
    model, ema = recipe.model, recipe.ema
    selected = ema.ema_model if ema is not None else model
    batches = torch.Generator().manual_seed(seed + 100)
    step, best_step, best_loss, previous_seconds = 0, 0, float("inf"), 0.0
    best_model = None
    last, best = output / "last.pt", output / "best.pt"
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("Checkpoint settings differ from this run")
        model.load_state_dict(state["model"])
        recipe.optimizer.load_state_dict(state["optimizer"])
        recipe.scheduler.load_state_dict(state["scheduler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        batches.set_state(state["batches"])
        restore_rng(state["rng"])
        step, best_step = state["step"], state["best_step"]
        best_loss, previous_seconds = state["best_loss"], state["training_seconds"]
        best_model = state["best_model"]
        del state
        print(f"[{method}] Resume at {step}/{steps}", flush=True)
    else:
        print(f"[{method}] Train {steps} updates on {device}", flush=True)

    start = time.monotonic()

    def save_last():
        atomic_checkpoint(last, dict(
            settings=settings, model=model.state_dict(), optimizer=recipe.optimizer.state_dict(),
            scheduler=recipe.scheduler.state_dict(), ema=ema.state_dict() if ema is not None else None,
            batches=batches.get_state(), rng=rng_state(), step=step, best_step=best_step,
            best_loss=best_loss, best_model=best_model,
            training_seconds=previous_seconds + time.monotonic() - start,
        ))

    while step < steps:
        if stop["requested"]:
            if step:
                save_last()
            raise InterruptedError("Training stopped. Rerun to resume.")
        model.train()
        indices = torch.randint(len(training), (batch_size,), generator=batches)
        recipe.optimizer.zero_grad(set_to_none=True)
        loss = recipe.loss(model, training[indices].to(device), True)
        value = float(loss.detach())
        if not np.isfinite(value):
            raise FloatingPointError(f"[{method}] Non-finite training loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        recipe.optimizer.step()
        if recipe.schedule_on_loss:
            recipe.scheduler.step(value)
        else:
            recipe.scheduler.step()
        if ema is not None:
            ema.update()
        step += 1
        if step == 1 or step % 100 == 0:
            print(f"[{method}] step={step}/{steps} loss={value:.6g}", flush=True)
        if step % settings["check_every"] == 0 or step == steps:
            score = validation_loss(recipe, selected, validation, batch_size, seed + 200, device)
            if not np.isfinite(score):
                raise FloatingPointError(f"[{method}] Non-finite validation loss at step {step}")
            if score < best_loss:
                best_loss, best_step = score, step
                best_model = {key: value.detach().cpu().clone() for key, value in selected.state_dict().items()}
            save_last()
            print(f"[{method}] validation_step={step} loss={score:.6g} best_step={best_step}", flush=True)

    training_seconds = previous_seconds + time.monotonic() - start
    if best_model is None:
        raise RuntimeError("No validation-selected weights were saved")
    if not best.is_file():
        atomic_checkpoint(best, dict(model=best_model, step=best_step))
    selected.load_state_dict(best_model)
    selected.eval()
    set_seed(seed + 300)
    samples = []
    start = time.monotonic()
    with torch.no_grad():
        for start_index in range(0, settings["samples"], batch_size):
            if stop["requested"]:
                raise InterruptedError("Training is saved. Rerun to finish generating samples.")
            count = min(batch_size, settings["samples"] - start_index)
            batch = recipe.sample(selected, count).cpu().float().numpy()
            expected = (count, settings["length"], settings["channels"])
            if batch.shape != expected or not np.isfinite(batch).all():
                raise ValueError(f"[{method}] Invalid generated batch shape or values: {batch.shape}")
            samples.append(batch)
            print(f"[{method}] sampled {start_index + count}/{settings['samples']}", flush=True)
    sampling_seconds = time.monotonic() - start
    sample_path = output / "samples_normalized.npy"
    atomic_array(sample_path, np.concatenate(samples))
    atomic_json(output / "metrics.json", dict(
        mode=method, seed=seed, trained_steps=steps, best_step=best_step, validation_loss=best_loss,
        parameters=sum(param.numel() for param in model.parameters()), training_seconds=training_seconds,
        sampling_seconds=sampling_seconds, sampling_steps=sampling_steps,
        samples_sha=digest(sample_path), device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        torch_version=str(torch.__version__),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    try:
        work(args, stop)
    except InterruptedError as error:
        print(error, flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
