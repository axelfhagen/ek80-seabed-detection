"""Data I/O helpers for the data cleaning pipeline."""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_patch_by_uid(uid, patch_dir):
    """Load a single patch + mask from disk given its uid.

    Parameters
    ----------
    uid : str, format '{npz_stem}::{patch_index}'
    patch_dir : path-like to the directory containing .npz files

    Returns
    -------
    patch : ndarray (3, H, W)
    mask  : ndarray (H, W)
    """
    stem, idx_str = uid.split("::")
    idx = int(idx_str)
    data = np.load(Path(patch_dir) / f"{stem}.npz")
    return data["patches"][idx], data["masks"][idx]


def load_quality_df(csv_path):
    """Load a quality metrics CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def save_exclusion_list(uids, path):
    """Save a list of excluded patch UIDs to a JSON file.

    Format: {"excluded_uids": ["uid1", "uid2", ...]}
    """
    with open(path, "w") as f:
        json.dump({"excluded_uids": sorted(uids)}, f, indent=2)


def load_exclusion_list(path):
    """Load an exclusion list from a JSON file.

    Returns a set of uid strings.
    """
    path = Path(path)
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get("excluded_uids", []))
