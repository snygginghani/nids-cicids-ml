"""
Simple "real-time" stream simulator:
- Loads a trained model
- Streams random flows and prints predictions
- Optionally writes them to CSV (for dashboards / ELK / Grafana)
"""

import time
import random
from pathlib import Path

import joblib
import pandas as pd

from config import MODELS_DIR
from data_utils import load_all_parquet
from preprocess import (
    find_label_column,
    clean_dataframe,
    add_labels,
    split_features_labels,
)


def simulate_stream(
    model_path: Path,
    delay_seconds: float = 0.2,
    n_samples: int = 50,
    save_csv: bool = True,
):
    df = load_all_parquet()
    label_col = find_label_column(df)
    df = clean_dataframe(df)
    df = add_labels(df, label_col)
    X, y_binary, y_multi = split_features_labels(df)

    model = joblib.load(model_path)
    print(f"[stream_simulator] Loaded model from {model_path}")

    records = []

    idxs = random.sample(range(len(X)), k=min(n_samples, len(X)))

    for i, idx in enumerate(idxs, 1):
        row = X.iloc[idx]
        true_bin = int(y_binary.iloc[idx])
        true_multi = str(y_multi.iloc[idx])

        pred = model.predict(row.values.reshape(1, -1))[0]

        record = {
            "index": idx,
            "true_binary": true_bin,
            "true_label": true_multi,
            "pred_binary": int(pred),
        }
        records.append(record)

        status = "ATTACK" if pred == 1 else "BENIGN"
        print(
            f"[{i}/{len(idxs)}] Flow #{idx} -> Pred: {status} "
            f"(true: {'ATTACK' if true_bin == 1 else 'BENIGN'})"
        )

        time.sleep(delay_seconds)

    if save_csv:
        out_path = MODELS_DIR / "stream_predictions.csv"
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"[stream_simulator] Saved stream predictions to {out_path}")


if __name__ == "__main__":
    # Use RandomForest binary model by default
    model_path = MODELS_DIR / "supervised_rf.joblib"
    simulate_stream(model_path, delay_seconds=0.1, n_samples=1000)
