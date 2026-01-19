# web_app.py
"""
Simple web dashboard for the NIDS project using Streamlit.

Run with:
    streamlit run web_app.py
"""

import streamlit as st
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


@st.cache_resource
def load_models():
    rf_path = MODELS_DIR / "supervised_rf.joblib"
    xgb_path = MODELS_DIR / "supervised_xgb.joblib"

    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    return rf_model, xgb_model


@st.cache_data
def load_and_prepare_data(sample_size: int = 20000):
    """
    Load CIC-IDS2017 parquet files, clean them, add labels and
    return a sampled subset to keep things light.
    """
    df = load_all_parquet()
    label_col = find_label_column(df)
    df = clean_dataframe(df)
    df = add_labels(df, label_col)

    # sample to avoid loading 2M+ rows fully in the dashboard
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    X, y_binary, y_multi = split_features_labels(df)

    # keep the feature column order for the model
    feature_cols = X.columns.tolist()
    return df, X, y_binary, y_multi, feature_cols


def main():
    st.set_page_config(
        page_title="NIDS – CIC-IDS2017 Dashboard",
        layout="wide",
    )

    st.title("🛡️ Network Intrusion Detection System (CIC-IDS2017)")
    st.markdown(
        "This dashboard uses trained machine learning models to detect "
        "malicious network flows on the CIC-IDS2017 dataset."
    )

    # Sidebar controls
    st.sidebar.header("Settings")

    model_choice = st.sidebar.selectbox(
        "Choose model",
        ["RandomForest (binary)", "XGBoost (binary)"],
    )

    sample_size = st.sidebar.slider(
        "Number of flows to sample from dataset",
        min_value=2000,
        max_value=500000,
        step=2000,
        value=10000,
    )

    st.sidebar.info(
        "Tip: start with a smaller sample if your machine is not very strong."
    )

    # Load resources
    with st.spinner("Loading models and data..."):
        rf_model, xgb_model = load_models()
        df, X, y_binary, y_multi, feature_cols = load_and_prepare_data(sample_size)

    if model_choice.startswith("RandomForest"):
        model = rf_model
    else:
        model = xgb_model

    st.success(
        f"Loaded {len(df):,} flows. Using **{model_choice}** "
        f"on {len(feature_cols)} features."
    )

    # Button to run predictions
    if st.button("🚀 Run Detection"):
        with st.spinner("Running model on sampled flows..."):
            # Use only feature columns for prediction
            X_features = X[feature_cols]
            preds = model.predict(X_features)

            # 0 = benign, 1 = attack (from our BinaryLabel convention)
            df_results = df.copy()
            df_results["PredictedBinary"] = preds
            df_results["PredictedLabel"] = df_results["PredictedBinary"].map(
                {0: "Benign", 1: "Attack"}
            )

        st.subheader("Summary")

        total = len(df_results)
        n_attack_true = int((df_results["BinaryLabel"] == 1).sum())
        n_attack_pred = int((df_results["PredictedBinary"] == 1).sum())

        col1, col2, col3 = st.columns(3)

        col1.metric("Total flows", f"{total:,}")
        col2.metric("True attacks (from label)", f"{n_attack_true:,}")
        col3.metric("Predicted attacks", f"{n_attack_pred:,}")

        # Attack vs benign distribution
        st.markdown("### Distribution of predicted classes")
        dist = df_results["PredictedLabel"].value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        st.bar_chart(dist.set_index("Class"))

        # Show detailed table
        st.markdown("### Sample of flows with predictions")
        st.dataframe(
            df_results[
                [
                    "BinaryLabel",
                    "MultiLabel",
                    "PredictedLabel",
                ]
            ].head(200),
            use_container_width=True,
        )

        # Download predictions as CSV
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="⬇️ Download predictions as CSV",
            data=csv,
            file_name="nids_predictions_sample.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **🚀 Run Detection** to generate predictions.")


if __name__ == "__main__":
    main()

#cd "C:\Users\snygg\PycharmProjects\DL project\nids-cicids-ml"
#streamlit run web_app.py
