"""
Train an IsolationForest anomaly detection model on BENIGN traffic only.
"""

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

from config import MODELS_DIR, RANDOM_STATE
from data_utils import load_all_parquet
from preprocess import (
    find_label_column,
    clean_dataframe,
    add_labels,
    split_features_labels,
    train_test_split_binary,
)


def train_iso_forest():
    # Load & clean
    df = load_all_parquet()
    label_col = find_label_column(df)
    df = clean_dataframe(df)
    df = add_labels(df, label_col)

    X, y_binary, _ = split_features_labels(df)

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split_binary(X, y_binary)

    # Train only on BENIGN rows from training set
    benign_mask = y_train == 0
    X_train_benign = X_train[benign_mask]

    print(f"[main_anomaly] Training IsolationForest on {len(X_train_benign)} benign flows...")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    iso.fit(X_train_benign)

    # Evaluate: map output (1=normal, -1=anomaly) to our labels
    preds_raw = iso.predict(X_test)
    # Convert to 0/1 representation: 0=benign, 1=attack
    preds_binary = np.where(preds_raw == 1, 0, 1)

    print("\n===== IsolationForest (Anomaly Detection) =====")
    print(classification_report(y_test, preds_binary, target_names=["Benign", "Attack"]))

    iso_path = MODELS_DIR / "iso_forest.joblib"
    joblib.dump(iso, iso_path)
    print(f"[main_anomaly] Saved IsolationForest model to {iso_path}")


if __name__ == "__main__":
    train_iso_forest()
