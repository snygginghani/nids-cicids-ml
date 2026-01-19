"""
Global configuration for the NIDS project.
Edit DATA_DIR to where your parquet files live.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Your actual data folder:
DATA_DIR = BASE_DIR / "data"   # remove /raw

# Your models folder:
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3

# Column names that should be dropped (identifiers / timestamps)
COLUMNS_TO_DROP = [
    "Flow ID",
    "Timestamp",
    "Src IP",
    "Dst IP",
    "Source IP",
    "Destination IP",
]
