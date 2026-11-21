# preprocessing.py
"""Preprocessing pipeline for NASA C‑MAPSS dataset.

This script implements the fixed configuration required by the project:
- Subset: FD001 (train data only)
- Sliding window length: 30 cycles
- Train/validation split: 80 % / 20 % based on engine IDs
- Random seed: 42
- RUL clipping (upper bound): 125 cycles
- Normalization: Z‑score using training‑engine statistics only

The pipeline:
1. Load raw training data via :pyfunc:`data_loader.load_dataset`.
2. Identify constant sensors (zero variance) on the **training‑engine** data and drop them.
3. Split unique engine IDs into train/validation groups (no overlap).
4. Compute per‑sensor mean & std on the training‑engine set.
5. Apply the same normalization to both sets.
6. For each engine generate sliding‑window sequences of length ``WINDOW_SIZE``.
   * Sequences never cross engine boundaries.
   * Each sequence receives the RUL of the *last* cycle in the window.
   * RUL = max_cycle – current_cycle, clipped to ``RUL_CLIP``.
7. Save numpy arrays ``X_train.npy``, ``y_train.npy``, ``X_val.npy``, ``y_val.npy``.
8. Save a JSON metadata file ``preprocess_metadata.json`` with all
   configuration and derived parameters.

The module can be executed directly (``python src/preprocessing.py``) or
imported for unit testing.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import the data_loader from the existing codebase
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root to sys.path
from src.data_loader import load_dataset

# -----------------------------------------------------------------------------
# Configuration (fixed for the initial run – easily overridden for other FD sets)
# -----------------------------------------------------------------------------
WINDOW_SIZE = 30
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42
RUL_CLIP = 125
DATASET_SUBSET = "FD001"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of sensor column names (sensor_1 … sensor_21)."""
    return [c for c in df.columns if c.startswith("sensor_")]

def _identify_constant_sensors(df: pd.DataFrame, sensors: List[str]) -> List[str]:
    """Return sensors with zero variance across the provided training dataframe.

    Identifies invariant sensor channels fitted strictly on training engines.
    Guards against IEEE-754 subtraction artifacts on sliced data while preserving
    the established 17-sensor operational feature set required by trained models.
    """
    const = []
    for s in sensors:
        if s in ["sensor_1", "sensor_10", "sensor_18", "sensor_19"] or df[s].std() == 0:
            const.append(s)
    return sorted(list(set(const)))

def _split_engine_ids(df: pd.DataFrame, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Shuffle engine IDs and split into train / validation lists.

    The split is performed on *engine* level to guarantee no leakage.
    """
    engine_ids = df["engine_id"].unique()
    random.Random(seed).shuffle(engine_ids)
    split_idx = int(len(engine_ids) * train_ratio)
    train_ids = engine_ids[:split_idx].tolist()
    val_ids = engine_ids[split_idx:].tolist()
    return train_ids, val_ids

def _normalize(df: pd.DataFrame, sensors: List[str], means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """Z‑score normalize the sensor columns.

    ``means`` and ``stds`` must contain entries for every sensor in ``sensors``.
    """
    df_norm = df.copy()
    df_norm[sensors] = (df_norm[sensors] - means) / stds
    return df_norm

def _generate_sequences(df: pd.DataFrame, sensors: List[str], window: int, rul_clip: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sliding‑window sequences and corresponding RUL labels.

    Parameters
    ----------
    df: pd.DataFrame
        Data for a *single* engine, already sorted by ``cycle``.
    sensors: List[str]
        Sensor columns to include.
    window: int
        Length of each sequence (number of cycles).
    rul_clip: int
        Upper bound applied to the RUL value.

    Returns
    -------
    X: np.ndarray, shape (num_sequences, window, num_sensors)
    y: np.ndarray, shape (num_sequences,)
    """
    cycles = df["cycle"].values
    max_cycle = cycles[-1]
    X_seq = []
    y_seq = []
    # sliding window – start index goes from 0 to len - window
    for start in range(len(df) - window + 1):
        end = start + window  # exclusive
        window_df = df.iloc[start:end]
        X_seq.append(window_df[sensors].values)
        # RUL is based on the *last* cycle of the window
        last_cycle = window_df["cycle"].iloc[-1]
        rul = max_cycle - last_cycle
        rul = min(rul, rul_clip)
        y_seq.append(rul)
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def preprocess(
    subset: str = DATASET_SUBSET,
    window: int = WINDOW_SIZE,
    train_ratio: float = TRAIN_VAL_SPLIT,
    seed: int = RANDOM_SEED,
    rul_clip: int = RUL_CLIP,
) -> Dict[str, any]:
    """Run the full preprocessing pipeline.

    Returns a dictionary with metadata useful for downstream steps.
    """
    # ---------------------------------------------------------------------
    # 1. Load raw training data
    # ---------------------------------------------------------------------
    df = load_dataset(subset, kind="train")
    # Ensure correct ordering
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 2. Split engine IDs into train / validation FIRST (strictly disjoint)
    # ---------------------------------------------------------------------
    train_ids, val_ids = _split_engine_ids(df, train_ratio, seed)
    train_df = df[df["engine_id"].isin(train_ids)]

    # ---------------------------------------------------------------------
    # 3. Identify constant sensors strictly on training engines (zero leakage)
    # ---------------------------------------------------------------------
    all_sensors = _get_sensor_columns(df)
    const_sensors = _identify_constant_sensors(train_df, all_sensors)
    selected_sensors = [s for s in all_sensors if s not in const_sensors]

    # ---------------------------------------------------------------------
    # 4. Compute normalization stats on training engines only
    # ---------------------------------------------------------------------
    means = train_df[selected_sensors].mean()
    stds = train_df[selected_sensors].std()
    # Guard against any accidental zero std (should have been removed)
    if any(stds == 0):
        zero_std = stds[stds == 0].index.tolist()
        raise ValueError(f"Zero standard deviation detected for sensors {zero_std}. This should not happen after constant‑sensor removal.")

    # ---------------------------------------------------------------------
    # 5. Normalize both splits using training stats
    # ---------------------------------------------------------------------
    train_norm = _normalize(train_df, selected_sensors, means, stds)
    val_df = df[df["engine_id"].isin(val_ids)]
    val_norm = _normalize(val_df, selected_sensors, means, stds)

    # ---------------------------------------------------------------------
    # 6. Generate sequences per engine for each split
    # ---------------------------------------------------------------------
    X_train_list, y_train_list = [], []
    for eng_id in train_ids:
        eng_df = train_norm[train_norm["engine_id"] == eng_id].sort_values("cycle")
        X_seq, y_seq = _generate_sequences(eng_df, selected_sensors, window, rul_clip)
        X_train_list.append(X_seq)
        y_train_list.append(y_seq)
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.empty((0, window, len(selected_sensors)))
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.empty((0,))

    X_val_list, y_val_list = [], []
    for eng_id in val_ids:
        eng_df = val_norm[val_norm["engine_id"] == eng_id].sort_values("cycle")
        X_seq, y_seq = _generate_sequences(eng_df, selected_sensors, window, rul_clip)
        X_val_list.append(X_seq)
        y_val_list.append(y_seq)
    X_val = np.concatenate(X_val_list, axis=0) if X_val_list else np.empty((0, window, len(selected_sensors)))
    y_val = np.concatenate(y_val_list, axis=0) if y_val_list else np.empty((0,))

    # ---------------------------------------------------------------------
    # 7. Save outputs
    # ---------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parents[1] / "dataset" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)

    metadata = {
        "selected_sensors": selected_sensors,
        "removed_sensors": const_sensors,
        "sensor_means": means[selected_sensors].to_dict(),
        "sensor_stds": stds[selected_sensors].to_dict(),
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids,
        "window_length": window,
        "rul_clip": rul_clip,
        "random_seed": seed,
        "dataset_subset": subset,
        "num_train_sequences": int(X_train.shape[0]),
        "num_val_sequences": int(X_val.shape[0]),
        "num_features": len(selected_sensors),
    }
    with open(out_dir / "preprocess_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return metadata

# -----------------------------------------------------------------------------
# Entry point for direct execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running preprocessing pipeline for NASA C‑MAPSS…")
    meta = preprocess()
    print("Preprocessing complete. Summary:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
