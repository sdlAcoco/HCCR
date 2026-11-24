"""Read multivariate time series and expose stride-one, time-aligned windows."""
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


SOURCE_CHANNELS = {"MSL": 55, "SMAP": 25, "SWaT": 25, "PSM": 25, "SMD": 33, "WADI": 123}


class SeriesWindows(Dataset):
    """Return (observations, calendar, seasonal residual, point labels)."""

    def __init__(self, arrays, window_length):
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        if len({len(array) for array in arrays}) != 1:
            raise ValueError("observations, calendar, residuals and labels must be aligned")
        self.arrays = arrays
        self.window_length = window_length
        self.channel_count = arrays[0].shape[1]
        self.time_width = arrays[1].shape[1]

    def __len__(self):
        return max(0, len(self.arrays[0]) - self.window_length + 1)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = slice(index, index + self.window_length)
        return tuple(array[window] for array in self.arrays)


def calendar_coordinates(timestamps):
    """Apply the calendar encoding used to train the released weights."""
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps).reshape(-1)))
    return np.column_stack((
        dates.minute / 59 - 0.5,
        dates.hour / 23 - 0.5,
        dates.weekday / 6 - 0.5,
        dates.day / 30 - 0.5,
        dates.month / 365 - 0.5,
    ))


def select_channels(raw, name):
    count = SOURCE_CHANNELS[name]
    if raw.ndim != 2 or raw.shape[1] < count:
        raise ValueError(f"{name} needs at least {count} input columns; received {raw.shape}")
    keep = [i for i in range(count) if name != "SMD" or i not in (15, 16, 29)]
    values = raw[:, keep]
    if name in ("MSL", "SMAP"):
        values = (values[:, 0] + values[:, 1:].mean(axis=1))[:, None]
    return pd.DataFrame(values).fillna(0).to_numpy()


def make_splits(root, dataset_name, cycle=1440, train_fraction=0.8, window_length=64):
    """Build training, validation, calibration ('init') and labeled test windows.

    Accept either the parent data directory or the dataset's own directory.
    The calibration sequence is the full trimmed training sequence. The cycle
    controls the centered rolling median used to form the training target.
    """
    if dataset_name not in SOURCE_CHANNELS:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Choose from {list(SOURCE_CHANNELS)}")
    if cycle < 1 or not 0 < train_fraction < 1:
        raise ValueError("cycle must be positive and train_fraction must be between 0 and 1")
    folder = Path(root)
    if not (folder / f"{dataset_name}_train_data.npy").is_file():
        folder = folder / dataset_name

    def load(field):
        return np.load(folder / f"{dataset_name}_{field}.npy", allow_pickle=False)

    train_values = select_channels(load("train_data"), dataset_name)
    calendar = calendar_coordinates(load("train_date"))
    trend = pd.DataFrame(train_values).rolling(cycle, center=True).median().to_numpy()
    trim = slice(cycle // 2, -(cycle - cycle // 2))
    seasonal = (train_values - trend)[trim]
    calibration = (train_values[trim], calendar[trim], seasonal, np.zeros((len(seasonal), 1)))
    cut = int(len(seasonal) * train_fraction)

    test_values = select_channels(load("test_data"), dataset_name)
    test = (test_values, calendar_coordinates(load("test_date")), np.zeros_like(test_values),
            load("test_label").reshape(-1, 1))

    sequences = {
        "train": tuple(a[:cut] for a in calibration),
        "valid": tuple(a[cut:] for a in calibration),
        "init": calibration,
        "test": test,
    }
    return {name: SeriesWindows(arrays, window_length) for name, arrays in sequences.items()}
