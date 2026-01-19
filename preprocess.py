"""
Preprocessing utilities:
- Clean CIC-IDS2017 data
- Create binary & multi-class labels
- Split features/labels
- Build train/test splits
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    COLUMNS_TO_DROP,
    RANDOM_STATE,
    TEST_SIZE,
)


def find_label_column(df: pd.DataFrame) -> str:
    """
    Try to detect the label column name.
    Common names: 'Label', 'label', 'Attack', 'Tag'
    """
    candidates = ["Label", "label", "Attack", "Tag"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Could not find label column. Please set it manually.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop obvious non-feature columns (IDs, timestamps, IPs)
    - Replace inf with NaN
    - Drop rows with NaN
    """
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


def add_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Adds:
      - BinaryLabel: 0 = benign/normal, 1 = attack
      - MultiLabel: original label string
    """
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    def to_binary(label: str) -> int:
        low = label.lower()
        if "benign" in low or "normal" in low:
            return 0
        return 1

    df["BinaryLabel"] = df[label_col].apply(to_binary)
    df["MultiLabel"] = df[label_col]

    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns:
      X : numeric feature columns only (no labels)
      y_binary : 0/1
      y_multi  : original label strings
    """
    df = df.copy()

    # Find original label column (e.g. 'Label')
    label_col = find_label_column(df)

    # y targets
    y_binary = df["BinaryLabel"]
    y_multi = df["MultiLabel"]

    # Drop ALL label-related columns from features
    X = df.drop(columns=[label_col, "BinaryLabel", "MultiLabel"], errors="ignore")

    # Keep only numeric columns (RandomForest/XGBoost need numbers)
    X = X.select_dtypes(include=["number"])

    return X, y_binary, y_multi



def train_test_split_binary(
    X: pd.DataFrame,
    y_binary: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Binary classification split (normal vs attack) with stratification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_binary,
    )
    return X_train, X_test, y_train, y_test
