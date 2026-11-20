# Adaptive Time-Frequency Generation

A small unconditional time-series generation experiment for a Colab GPU with approximately 16 GB of memory. Three denoising experts estimate the same clean sequence at every diffusion step. A router combines their estimates at the original time positions.

| Expert | Input representation | Context |
| --- | --- | --- |
| Short time | Patches of 8 consecutive points | Adjacent patches in each attention layer |
| Long time | Patches of 8 consecutive points | All patches in the window |
| Frequency | Joint real and imaginary FFT coefficients | All frequency positions, followed by inverse FFT |

The three controls use the same expert architecture and initial expert weights. `fixed` assigns one third to each expert. `global` learns one mixture per sequence and diffusion step. `patch` learns a mixture for each patch and diffusion step. All three experts run in each control.

The implementation uses cosine Gaussian diffusion, predicts the clean sequence, and samples with deterministic DDIM. Training starts from scratch. The diffusion and transformer modules are implemented directly in this repository. The method uses the original time positions throughout, with a patch-wise fusion layer after the spectral estimate is transformed back to time.

## Colab

Open [colab.ipynb](https://colab.research.google.com/github/Zhou-Leheng/adaptive-time-frequency-generation/blob/main/colab.ipynb) and select a GPU runtime. The first code cell downloads public repository code directly. For a private repository, authorize GitHub when opening the notebook. The download cell reads a GitHub token from the Colab secret `GITHUB_TOKEN`, or from a hidden input prompt. Grant that token repository contents read access.

Run the notebook cells in order. Training starts in the background with `nohup`, so the next cells can read the log and results while it runs. Checkpoints and result files are written to `MyDrive/ATFG/etth1_seed42`. The code and downloaded ETTh1 CSV stay in the Colab runtime.

The notebook contains the full launch command and a stop command. Rerunning the launch cell resumes completed checkpoint intervals and skips controls whose metrics are already saved. Keep the output name for a resume. Give a changed experiment a new `RUN_NAME`.

## Command line

```bash
pip install -r requirements.txt
python -u run.py --download --output results/etth1_seed42
```

Run all three controls in one process, or choose one with `--modes patch`. For a local CSV, pass `--data /path/to/ETTh1.csv`. Every variable column must be numeric. A `date`, `time`, or `timestamp` column is excluded from the model input.

Default settings are length 128, patch size 8, width 64, 4 attention heads, 2 layers per expert, batch size 64, learning rate 0.0001, 5000 optimizer steps per control, 200 diffusion levels, 50 DDIM sampling steps, and seed 42. Training uses an unweighted clean-sequence MSE and gradient norm clipping at 1.0. The EMA decay is 0.999.

ETTh1 is split chronologically into 70% train, 10% validation, and 20% test before windows are extracted. Per-variable mean and standard deviation come from the training rows. The pilot samples up to 4096 training windows and 512 windows from each held-out split. Windows can overlap inside a split. The source CSV hash and selected data settings are saved with the run.

Each optimizer step draws a batch of training windows with replacement and one noise level per example. Every 500 optimizer steps, a fixed validation noise bank evaluates the EMA model. Sampling uses the EMA checkpoint with the lowest validation MSE. The three controls receive matching training batches, noise streams, and initial sampling noise banks. A test set is used to evaluate each completed control.

## Results

`summary.csv` and `summary.json` update after each control finishes. The generative metrics compare two sets of windows. All five distance metrics below favor lower values.

| Metric | Comparison |
| --- | --- |
| `swd` | Sliced Wasserstein distance over standardized whole-window vectors, with 128 fixed random projection directions |
| `psd_relative_l1` | L1 difference of mean power spectra, divided by the total real-data power |
| `acf_mae` | Mean absolute difference of average autocorrelations at lags 1 through 32 |
| `delta_w1` | Wasserstein distance between distributions of absolute adjacent-point differences |
| `cross_corr_mae` | Mean absolute difference between variable correlation matrices |

Every control directory contains these files.

```text
last.pt                  model, EMA, optimizer, and RNG state for resuming
best.pt                  validation-selected EMA weights
metrics.json             quality metrics, timings, and peak GPU memory
samples_normalized.npy   generated standardized windows [sample, time, variable]
router_weights.npz       weights from the first sampling batch, at every DDIM step
samples.png              real and generated example windows
router.png               patch weights at early, middle, and late diffusion steps
```

`router_weights.npz` stores up to 16 traces, shaped `[sample, sampling_step, patch, expert]`, and the corresponding diffusion times. Expert order is short time, long time, frequency. Inverse-transform samples with `sample * normalization_std + normalization_mean`, using the values in `settings.json` under `data`.

Checkpoints are saved at validation intervals and after a graceful `SIGTERM` at an optimizer-step boundary. A sudden runtime loss resumes from the last saved interval. A lock prevents two processes from writing into the same experiment directory simultaneously.

## Compare against official Diffusion-TS

Open [compare_colab.ipynb](https://colab.research.google.com/github/Zhou-Leheng/adaptive-time-frequency-generation/blob/main/compare_colab.ipynb), select a GPU runtime, and run its first three code cells. The default input is the completed `MyDrive/ATFG/etth1_seed42` experiment. It trains one additional Diffusion-TS baseline and reuses the existing three fusion sample files. The log and result cells show progress and the comparison table.

For a local run, install the comparison dependencies and launch the command below.

```bash
pip install -r requirements-benchmark.txt
python -u compare.py --results results/etth1_seed42 --download
```

The comparison downloads checksum-verified model and TS2Vec modules from official Diffusion-TS commit `566307e6cf2d8095e58de4c6e3a6ae965b69b5b5`. The cached source includes its upstream MIT license. The baseline uses the released 3-encoder/2-decoder architecture and its original L1, Fourier, and timestep-weighted training objective. It follows this project's shared z-score inputs, AdamW optimizer, validation-selected EMA, diffusion levels, unclipped DDIM sampling, and training budget. Its initial sampling noise bank matches the fusion runs. This produces a common-protocol pilot comparison; published paper tables use their own protocols.

The evaluation reports Context-FID, the official CrossCorrelLoss, and TimeGAN-style discriminative and predictive scores. A 320-dimensional official TS2Vec encoder is trained once on the real training windows for 600 updates, then reused for all four methods. All metric inputs use a shared per-variable min-max transform fitted on real training windows. Values outside the training range are retained. DS and PS use a PyTorch port of the released single-layer GRU definitions, with hidden size `floor(channels / 2)`, 2000 discriminator updates, and 5000 predictor updates. The predictor learns from generated data to predict the next value of the last variable from the other variables. Evaluators repeat with three seeds. The DS/PS standard deviations describe evaluator randomness for one generator seed.

Outputs are saved in `benchmark_diffusion_ts` under the existing experiment folder. `comparison.csv` and `comparison.json` update after each method is evaluated. `protocol.json` records the shared protocol. `diffusion_ts/last.pt` supports baseline resume, and `shared_ts2vec.pt` stores the feature encoder. Completed DS and PS stages are cached individually in `evaluation/`. A restart reuses these stages. A feature encoder interrupted during its 600 training updates restarts that stage. The original fusion samples are read directly from their existing files.

`training_seconds` and `sampling_seconds` carry each run's recorded timings. Compare timings on matching GPU hardware. The code checks the source CSV hash, saved split boundaries, window counts, and normalization statistics before training the baseline. It locks the source experiment while running the comparison.

## Checks

```bash
python -m unittest discover -s tests -v
```

The focused checks cover aligned expert outputs and gradients, odd-length inverse FFT, train-only normalization, RNG-exact checkpoint recovery on CPU, reproducible sampling, and sample-order-invariant distribution metrics.

The conceptual references are [DDPM](https://arxiv.org/abs/2006.11239), [DDIM](https://arxiv.org/abs/2010.02502), and [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS). The ETTh1 CSV downloads from the authors' [ETDataset repository](https://github.com/zhouhaoyi/ETDataset).
