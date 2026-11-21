"""Adapters for the authors' training objectives and unconditional samplers."""

from dataclasses import dataclass
import sys
from typing import Callable

import torch
import yaml


REPOSITORIES = {
    "pacodi_sde": ("RongyaoCai/PaCoDi", "8c77d91f3056b9cfd9b62c3eea0ee40b65110b3a"),
    "freqdiff": ("JonathanCrabbe/FourierDiffusion", "e60d532c0d7e304fae7e23beab7de3b8bf023051"),
    "timebridge_gp": ("JinseongP/TimeBridge", "023a0aa9f6dbd24f66bd84eb8983bb78affd4e27"),
}

SAMPLING_STEPS = {"pacodi_sde": 500, "freqdiff": 1000, "timebridge_gp": 40}
INPUT_TRANSFORMS = {
    "pacodi_sde": "train-only min-max to [-1,1]; restore per-window moments drawn from training data",
    "freqdiff": "shared train-row z-score then upstream packed orthonormal DFT",
    "timebridge_gp": "train-only min-max to [-1,1]; upstream RBF GP prior fitted on training windows",
}


@dataclass
class NativeRun:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: object
    schedule_on_loss: bool
    ema: object | None
    loss: Callable
    sample: Callable


def _read_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _bounded_model_inputs(training, validation):
    minimum = training.amin(dim=(0, 1), keepdim=True)
    span = (training.amax(dim=(0, 1), keepdim=True) - minimum).clamp_min(1e-6)

    def transform(values):
        return (values - minimum) / span * 2 - 1

    def inverse(values):
        return (values.cpu() + 1) * 0.5 * span + minimum

    return transform(training), transform(validation), inverse


def _adam_and_plateau(model, config):
    from engine.lr_sch import ReduceLROnPlateauWithWarmup

    options = config["solver"]
    optimizer = torch.optim.Adam(model.parameters(), lr=options["base_lr"], betas=(0.9, 0.96))
    scheduler = ReduceLROnPlateauWithWarmup(optimizer, **options["scheduler"]["params"])
    return optimizer, scheduler


def _pacodi(source, training, validation, sampling_steps, device, seed):
    from models.pacodi.pacodi_sde import PaCoDi_sde
    from utils.ema import EMA

    config = _read_config(source / "config/unconditional/etth1.yaml")
    params = dict(config["model"]["params"], seq_length=training.shape[1], channels=training.shape[2],
                  sampling_steps=sampling_steps, device=device)
    model = PaCoDi_sde(**params)
    optimizer, scheduler = _adam_and_plateau(model, config)
    ema = EMA(model, beta=config["solver"]["ema"]["decay"],
              update_every=config["solver"]["ema"]["update_interval"]).to(device)
    training, validation, inverse = _bounded_model_inputs(training, validation)
    stats_rng = torch.Generator().manual_seed(seed + 351)

    def loss(net, values, is_training):
        return net(values)

    def sample(net, count):
        # The upstream sampler restores amplitude from window statistics. Use training data only.
        indices = torch.randint(len(training), (count,), generator=stats_rng)
        _, stats = net.instance_normalize(training[indices].to(device))
        return inverse(net.generate_unconditional(batch_size=count, instance_stats=stats))

    return NativeRun(model, optimizer, scheduler, True, ema, loss, sample), training, validation


def _freqdiff(source, training, validation, steps, sampling_steps, device):
    sys.path.insert(0, str(source / "src"))
    from fdiff.models.score_models import ScoreModule
    from fdiff.sampling.sampler import DiffusionSampler
    from fdiff.schedulers.sde import VPScheduler
    from fdiff.utils.dataclasses import DiffusableBatch
    from fdiff.utils.fourier import dft, idft

    options = _read_config(source / "cmd/conf/score_model/default.yaml")
    schedule = _read_config(source / "cmd/conf/score_model/noise_scheduler/vpsde.yaml")
    sde = VPScheduler(beta_min=float(schedule["beta_min"]), beta_max=float(schedule["beta_max"]),
                      eps=float(schedule["eps"]), fourier_noise_scaling=True)
    model = ScoreModule(
        n_channels=training.shape[2], max_len=training.shape[1], noise_scheduler=sde,
        d_model=options["d_model"], num_layers=options["num_layers"], n_head=options["n_head"],
        lr_max=float(options["lr_max"]), num_training_steps=steps, fourier_noise_scaling=True,
        likelihood_weighting=options["likelihood_weighting"],
    ).to(device)
    sde.set_noise_scaling(training.shape[1])
    optimizers = model.configure_optimizers()
    sampler = DiffusionSampler(model, sample_batch_size=1)
    sde.set_timesteps(sampling_steps)

    def loss(net, values, is_training):
        loss_function = net.training_loss_fn if is_training else net.validation_loss_fn
        return loss_function(net, DiffusableBatch(X=values))

    def sample(net, count):
        # Use the upstream reverse-SDE step, logging one line per batch in the worker.
        values = sampler.sample_prior(count)
        for timestep in sde.timesteps.tolist():
            times = torch.full((count,), timestep, dtype=torch.float32, device=device)
            values = sampler.reverse_diffusion_step(DiffusableBatch(X=values, timesteps=times))
        return idft(values.cpu())

    recipe = NativeRun(model, optimizers["optimizer"], optimizers["lr_scheduler"]["scheduler"],
                       False, None, loss, sample)
    return recipe, dft(training), dft(validation)


def _timebridge(source, training, validation, sampling_steps, device):
    from ema_pytorch import EMA
    from Models.bridge_diffusion.bridge_karras_diffusion import Diffusion_TS
    from Models.bridge_diffusion.model_utils import gp

    config = _read_config(source / "Config/etth.yaml")
    params = dict(config["model"]["params"], seq_length=training.shape[1], feature_size=training.shape[2],
                  sampling_timesteps=sampling_steps, pred_mode="vp", sampler="sde", prior="gp",
                  sigma_min=0.0001, sigma_max=1, sigma_data=0.1, beta_min=0.2, beta_d=10,
                  model_matching="D_matching", use_img_transform=False)
    model = Diffusion_TS(**params).to(device)
    optimizer, scheduler = _adam_and_plateau(model, config)
    ema = EMA(model, beta=config["solver"]["ema"]["decay"],
              update_every=config["solver"]["ema"]["update_interval"]).to(device)
    training, validation, inverse = _bounded_model_inputs(training, validation)
    prior = gp(training, kernel_type="rbf", bw="GAUSSIAN", var=0.5, device=device)

    def loss(net, values, is_training):
        return net(values, prior_info=prior)

    def sample(net, count):
        values, _, _ = net.generate_mts(batch_size=count, prior_info=prior)
        return inverse(values)

    return NativeRun(model, optimizer, scheduler, True, ema, loss, sample), training, validation


def build_native(method, source, training, validation, steps, sampling_steps, device, seed):
    # Each model runs in a separate process because upstream packages reuse top-level names.
    sys.path.insert(0, str(source))
    if method == "pacodi_sde":
        return _pacodi(source, training, validation, sampling_steps, device, seed)
    if method == "freqdiff":
        return _freqdiff(source, training, validation, steps, sampling_steps, device)
    if method == "timebridge_gp":
        return _timebridge(source, training, validation, sampling_steps, device)
    raise ValueError(f"Unknown external model: {method}")
