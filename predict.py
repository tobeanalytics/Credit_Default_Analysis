## predict.py — (Complete script) 

```python
#!/usr/bin/env python3
"""
predict.py

Simple CLI and importable script to load saved models and produce predictions
for new/cleaned data.

Usage (from project root):
    python predict.py --input data/new_customers.csv --output predictions.csv

Example (inside notebook):
    from predict import load_models, predict_df
    scaler, lr, rf = load_models()
    preds = predict_df(df_new, lr, rf, scaler=scaler)
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_models(model_dir=MODEL_DIR):
    """
    Load scaler (optional), logistic regression and random forest models.
    Returns (scaler_or_none, logistic_model, random_forest_model)
    """
    scaler = None
    # file names
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    lr_path = os.path.join(model_dir, "logistic_regression_model.pkl")
    rf_path = os.path.join(model_dir, "random_forest_model.pkl")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        print("No scaler found (expected if LR used raw features).")

    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"Logistic Regression model not found at {lr_path}")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Random Forest model not found at {rf_path}")

    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    print(f"Loaded LR from {lr_path}")
    print(f"Loaded RF from {rf_path}")

    return scaler, lr, rf

def prepare_features(df, scaler=None, drop_cols=None):
    """
    Accepts a cleaned DataFrame (same columns used for training).
    Applies scaler if provided (scaler is assumed to operate on numeric columns
    that were used for LR training). This function does not do heavy
    preprocessing — it expects the input to match training features.
    """
    # drop any unwanted columns if provided (e.g., ID)
    df_proc = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df_proc.columns:
                df_proc = df_proc.drop(columns=c)

    # if scaler provided, apply to numeric columns only
    if scaler is not None:
        # apply scaler to numeric columns; keep columns order
        num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found to scale.")
        df_proc.loc[:, num_cols] = scaler.transform(df_proc[num_cols])
    return df_proc

def predict_df(df, lr_model, rf_model, scaler=None, id_col=None):
    """
    Return DataFrame with predictions and probabilities for both models.
    - df: cleaned input DataFrame (columns must align with training features)
    - lr_model, rf_model: trained sklearn models
    - scaler: optional scaler for LR
    - id_col: optional column name that identifies rows (kept in output)
    """
    df_in = df.copy()
    ids = None
    if id_col and id_col in df_in.columns:
        ids = df_in[id_col]
    # Prepare feature matrix
    X = prepare_features(df_in, scaler=scaler, drop_cols=[id_col] if id_col else None)

    # Predictions & probs
    lr_preds = lr_model.predict(X)
    lr_probs = lr_model.predict_proba(X)[:, 1] if hasattr(lr_model, "predict_proba") else None

    rf_preds = rf_model.predict(X)
    rf_probs = rf_model.predict_proba(X)[:, 1] if hasattr(rf_model, "predict_proba") else None

    out = pd.DataFrame({
        "LR_PRED": lr_preds,
        "LR_PROB": lr_probs,
        "RF_PRED": rf_preds,
        "RF_PROB": rf_probs
    }, index=df.index)

    if ids is not None:
        out.insert(0, id_col, ids)

    return out

def main(args):
    # Load models
    scaler, lr, rf = load_models(args.model_dir)

    # Read input
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"Input shape: {df.shape}")

    # If user provided id column, preserve it
    id_col = args.id_col if args.id_col in df.columns else None
    preds = predict_df(df, lr, rf, scaler=scaler, id_col=id_col)

    # Save output
    out_path = args.output or "predictions.csv"
    preds.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using saved models.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file with cleaned features.")
    parser.add_argument("--output", type=str, required=False, default="predictions.csv",
                        help="Path to output CSV file.")
    parser.add_argument("--model_dir", type=str, required=False, default=MODEL_DIR,
                        help="Path to models directory.")
    parser.add_argument("--id_col", type=str, required=False, default=None,
                        help="Optional ID column in input to preserve in output.")
    args = parser.parse_args()
    main(args)