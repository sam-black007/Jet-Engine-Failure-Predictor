# data_loader.py
"""Utility functions to load the NASA C-MAPSS dataset.

The raw files are located under ``dataset/raw/CMAPSSData/`` after extraction.
There are four subsets (FD001‑FD004). Each subset has:
- ``train_FDx.txt``   : training trajectories (run‑to‑failure)
- ``test_FDx.txt``    : testing trajectories (truncated before failure)
- ``RUL_FDx.txt``     : ground‑truth remaining useful life for each test engine

The files contain space‑separated columns:
1. engine_number (int)
2. cycle (int)
3‑5 operational settings (float)
6‑26 sensor measurements (float) – 21 sensors

This module provides a ``load_data`` function that returns a ``pandas.DataFrame``
with appropriate column names.
"""

import os
import pandas as pd

# Column names based on the C‑MAPSS dataset specification
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
COLUMN_NAMES = ["engine_id", "cycle"] + OP_COLS + SENSOR_COLS


def _load_txt(file_path: str) -> pd.DataFrame:
    """Read a space‑separated txt file and assign column names.

    Parameters
    ----------
    file_path: str
        Path to the .txt file.

    Returns
    -------
    pd.DataFrame
        Data with the standard column names.
    """
    df = pd.read_csv(file_path, sep="\s+", header=None, engine="python")
    df.columns = COLUMN_NAMES
    return df


def load_dataset(subset: str, kind: str = "train") -> pd.DataFrame:
    """Load a specific C‑MAPSS subset.

    Parameters
    ----------
    subset: str
        One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``.
    kind: str, default "train"
        ``"train"`` loads ``train_FDx.txt``;
        ``"test"`` loads ``test_FDx.txt``.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    subset = subset.upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Invalid subset '{subset}'. Expected FD001‑FD004.")
    if kind not in {"train", "test"}:
        raise ValueError("kind must be 'train' or 'test'.")

    base_dir = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "raw", "CMAPSSData"
    )
    filename = f"{kind}_{subset}.txt"
    file_path = os.path.abspath(os.path.join(base_dir, filename))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Ensure the dataset is extracted.")
    return _load_txt(file_path)


def load_rul(subset: str) -> pd.Series:
    """Load the ground‑truth RUL values for a test subset.

    Returns a ``pd.Series`` indexed by engine number (starting from 1).
    """
    subset = subset.upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("Invalid subset.")
    base_dir = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "raw", "CMAPSSData"
    )
    file_path = os.path.abspath(os.path.join(base_dir, f"RUL_{subset}.txt"))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    rul = pd.read_csv(file_path, header=None, names=["RUL"])
    # The file contains one RUL per test engine in order of appearance in test set.
    return rul["RUL"]

# Example usage (commented out for production use)
# if __name__ == "__main__":
#     df_train = load_dataset("FD001", kind="train")
#     print(df_train.head())
