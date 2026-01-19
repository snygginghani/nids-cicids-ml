"""
Train supervised models (binary & optional multi-class) on CIC-IDS2017 parquet files.
Models:
  - RandomForest (binary)
  - XGBoost (binary, optional)
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import MODELS_DIR
from data_utils import load_all_parquet
from preprocess import (
    find_label_column,
    clean_dataframe,
    add_labels,
    split_features_labels,
    train_test_split_binary,
)
from metrics_utils import print_classification_report, plot_confusion_matrix


def train_binary_models():
    # 1. Load data
    df = load_all_parquet()

    # 2. Detect label column & clean
    label_col = find_label_column(df)
    df = clean_dataframe(df)
    df = add_labels(df, label_col)

    # 3. Split
    X, y_binary, y_multi = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_binary(X, y_binary)

    # 4. RandomForest (binary)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    print("[main_supervised] Training RandomForest (binary)...")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print_classification_report(y_test, y_pred_rf, title="RandomForest (Binary)")
    plot_confusion_matrix(
        y_test,
        y_pred_rf,
        target_names=["Benign", "Attack"],
        title="RF_Binary",
    )

    rf_path = MODELS_DIR / "supervised_rf.joblib"
    joblib.dump(rf, rf_path)
    print(f"[main_supervised] Saved RandomForest to {rf_path}")

    # 5. XGBoost (binary) – optional but recommended
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    print("[main_supervised] Training XGBoost (binary)...")
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    print_classification_report(y_test, y_pred_xgb, title="XGBoost (Binary)")
    plot_confusion_matrix(
        y_test,
        y_pred_xgb,
        target_names=["Benign", "Attack"],
        title="XGB_Binary",
    )

    xgb_path = MODELS_DIR / "supervised_xgb.joblib"
    joblib.dump(xgb, xgb_path)
    print(f"[main_supervised] Saved XGBoost to {xgb_path}")


def train_multiclass_xgb():
    """
    Optional: train multi-class XGBoost to predict specific attack types.
    """
    df = load_all_parquet()
    label_col = find_label_column(df)
    df = clean_dataframe(df)
    df = add_labels(df, label_col)

    X, _, y_multi = split_features_labels(df)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_multi)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded,
    )

    xgb = XGBClassifier(
        n_estimators=350,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    print("[main_supervised] Training XGBoost (multi-class)...")
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    target_names = le.classes_
    print_classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        title="XGBoost (Multi-class)",
    )
    plot_confusion_matrix(
        y_test,
        y_pred,
        target_names=target_names,
        title="XGB_Multiclass",
    )

    joblib.dump(xgb, MODELS_DIR / "supervised_xgb_multiclass.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    print("[main_supervised] Saved multi-class XGBoost and label encoder.")


if __name__ == "__main__":
    train_binary_models()
    # If you also want multi-class, uncomment:
    # train_multiclass_xgb()
