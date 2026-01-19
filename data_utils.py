"""
Utilities for loading CIC-IDS2017 parquet files.
"""

from pathlib import Path
from typing import List
import pandas as pd

from config import DATA_DIR


def get_parquet_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return a list of all parquet files in the data directory."""
    return sorted(data_dir.glob("*.parquet"))


def load_all_parquet(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load all parquet files under data_dir into a single DataFrame.
    """
    files = get_parquet_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    dfs = []
    for f in files:
        print(f"[data_utils] Loading {f.name}")
        df = pd.read_parquet(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"[data_utils] Loaded {len(data)} rows from {len(files)} files.")
    return data
