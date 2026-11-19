"""Chronological splits, train-only normalization, and bounded window sets."""

import csv
import hashlib
from pathlib import Path
import urllib.request

import numpy as np


ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def prepare_data(csv_path, length, train_windows, eval_windows, seed):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset {path}. Use --download to fetch ETTh1.")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = [name for name in reader.fieldnames if name.lower() not in ("date", "time", "timestamp")]
        values = np.asarray([[float(row[name]) for name in columns] for row in reader], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("CSV must contain finite numeric variables and an optional date column")
    train_end, val_end = int(len(values) * 0.7), int(len(values) * 0.8)
    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0).clip(min=1e-6)
    normalized = ((values - mean) / std).astype(np.float32)
    splits = (normalized[:train_end], normalized[train_end:val_end], normalized[val_end:])
    windows = []
    counts = (train_windows, eval_windows, eval_windows)
    for index, (split, limit) in enumerate(zip(splits, counts)):
        available = len(split) - length + 1
        if available < 1:
            raise ValueError(f"Split {index} has {len(split)} rows, fewer than length={length}")
        rng = np.random.default_rng(seed + index)
        starts = np.sort(rng.choice(available, size=min(limit, available), replace=False))
        windows.append(np.stack([split[start:start + length] for start in starts]))
    metadata = {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "columns": columns, "split_rows": [0, train_end, val_end, len(values)],
        "window_counts": [len(window) for window in windows],
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
    }
    return windows, metadata


def download_etth1(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(ETTH1_URL, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
