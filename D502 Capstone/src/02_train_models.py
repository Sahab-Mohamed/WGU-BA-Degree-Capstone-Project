"""
02_train_models.py
Train + evaluate churn prediction models using the Telco churn dataset in PostgreSQL.

KEY FIXES INCLUDED (bulletproof):
- Prevents target leakage by explicitly dropping churn-derived columns:
  churn, churn_label, churn_value, churn_score, churn_reason
- Uses churn_label as the target by default (works with your Excel-derived dataset)
- Safe drop logic (drops only columns that exist)
- Exports qualifying outputs:
  metrics table, ROC curve points, test predictions, feature importance (RF),
  and writes predictions back to PostgreSQL (capstone.model_predictions)

Run (PowerShell, from project root):
  python src\02_train_models.py
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------
# CONFIG
# -----------------------------
SCHEMA = "capstone"
SOURCE_TABLE = "telco_churn_clean"      # from your ETL
# will be created/overwritten in PostgreSQL
PRED_TABLE = "model_predictions"

# Target + leakage protection
TARGET_COL = "churn_label"              # dataset has churn_label
# Hard blocklist: if any of these exist, they MUST NOT be used as features
LEAKAGE_COLS = [
    "churn", "churn_label", "churn_value", "churn_score", "churn_reason"
]

# Common identifiers / non-predictive IDs
ID_COLS = ["customerid"]

# Benchmark
ROC_AUC_BENCHMARK = 0.80

# Output files
OUTPUT_DIR = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_env_db_url() -> str | None:
    '''Load DB_URL from .env in project root if present.'''
    existing = os.environ.get("DB_URL")
    if existing:
        return existing

    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return None

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("DB_URL="):
            value = line.split("=", 1)[1].strip().strip("'").strip('"')
            if value:
                os.environ["DB_URL"] = value
                return value
    return None


def _to_binary_churn(series: pd.Series) -> pd.Series:
    """
    Convert churn_label variants to 0/1 safely.
    Handles values like: Yes/No, True/False, 1/0, Churned/Stayed (best effort).
    """
    s = series.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1, "churned": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "stay": 0, "stayed": 0, "non-churn": 0,
        "nonchurn": 0, "not churn": 0, "not churned": 0
    }
    out = s.map(mapping)

    # If some entries didn't map (NaN), try numeric coercion
    if out.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        # numeric may still be NaN; fill only those missing
        out = out.fillna(numeric)

    # Final sanity: anything still missing becomes NaN, and we'll drop those rows later
    return out


def _safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop only columns that exist."""
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


def main():
    # -----------------------------
    # DB connection
    # -----------------------------
    db_url = os.environ.get("DB_URL")
    if not db_url:
        raise RuntimeError(
            "DB_URL environment variable not set.\n"
            "PowerShell example:\n"
            '$env:DB_URL="postgresql+psycopg2://postgres:Team.mas555@127.0.0.1:5432/telco_churn"'
        )

    engine = create_engine(db_url)

    # -----------------------------
    # Load dataset from PostgreSQL
    # -----------------------------
    sql = f"SELECT * FROM {SCHEMA}.{SOURCE_TABLE};"
    df = pd.read_sql(sql, engine)

    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # Resolve target column safely (no globals)
    if TARGET_COL in df.columns:
        target_col = TARGET_COL
    elif "churn" in df.columns:
        print("TARGET_COL 'churn_label' not found; falling back to 'churn'")
        target_col = "churn"
    else:
        raise KeyError(
            f"No valid churn target found. Available columns:\n{df.columns.tolist()}"
        )

    # -----------------------------
    # Build y (target) and protect against leakage
    # -----------------------------
    y_raw = df[target_col]
    y = _to_binary_churn(y_raw)

    # Drop rows where target could not be interpreted
    valid_mask = y.notna()
    df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    # -----------------------------
    # Build X (features): drop leakage + IDs safely
    # -----------------------------
    df_features = df.copy()
    # includes 'churn' if present
    df_features = _safe_drop(df_features, LEAKAGE_COLS)
    df_features = _safe_drop(df_features, ID_COLS)

    X = df_features

    # Debug prints (helpful if anything weird happens again)
    print(f"Rows: {len(df):,}")
    print(f"Target: {target_col} (positive class=1)")
    print("Leakage columns present & dropped:", [
          c for c in LEAKAGE_COLS if c in df.columns])
    print("ID columns present & dropped:", [
          c for c in ID_COLS if c in df.columns])
    print(f"Feature columns used (count={X.shape[1]}):")
    print(sorted(X.columns.tolist())[:30], "..." if X.shape[1] > 30 else "")

    # -----------------------------
    # Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Identify numeric vs categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocessing pipelines
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # OneHotEncoder can handle NaN as its own category; skip imputation to
    # avoid dtype conversion issues on mixed string/object data.
    categorical_pipe = Pipeline(steps=[
        ("onehot", onehot),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )

    # -----------------------------
    # Models
    # -----------------------------
    logreg = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])

    rf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    models = {
        "logistic_regression": logreg,
        "random_forest": rf,
    }

    # Storage for results
    metrics_rows = []
    preds_out = pd.DataFrame({"y_true": y_test.values}, index=X_test.index)

    # For ROC curve export
    roc_frames = []

    # -----------------------------
    # Train + Evaluate
    # -----------------------------
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)

        # predicted probabilities for ROC-AUC
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics_rows.append({
            "model": name,
            "roc_auc": float(auc),
            "meets_benchmark": bool(auc >= ROC_AUC_BENCHMARK),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc_benchmark": ROC_AUC_BENCHMARK
        })

        preds_out[f"proba_{name}"] = y_proba
        preds_out[f"pred_{name}"] = y_pred

        # ROC points
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_df = pd.DataFrame({
            "model": name,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds
        })
        roc_frames.append(roc_df)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        "roc_auc", ascending=False)

    # -----------------------------
    # Save outputs for Tableau
    # -----------------------------
    # 1) Metrics table
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # 2) ROC curve points
    roc_all = pd.concat(roc_frames, ignore_index=True)
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve_points_models.csv")
    roc_all.to_csv(roc_path, index=False)

    # 3) Test predictions (for segmentation visuals)
    preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    preds_out.reset_index(drop=True).to_csv(preds_path, index=False)

    # 4) Feature importance for Random Forest only (explainability)
    # Extract feature names after preprocessing
    rf_pipe = models["random_forest"]
    prep = rf_pipe.named_steps["prep"]
    rf_model = rf_pipe.named_steps["model"]

    # Numeric feature names remain the same
    feature_names = []
    feature_names.extend(numeric_cols)

    # One-hot feature names
    if categorical_cols and "cat" in prep.named_transformers_:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(ohe_names)

    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fi_path = os.path.join(OUTPUT_DIR, "feature_importance_rf.csv")
    fi_df.to_csv(fi_path, index=False)

    # -----------------------------
    # Write predictions back to PostgreSQL (rubric / traceability)
    # -----------------------------
    pred_db = preds_out.copy()
    pred_db["scored_at"] = datetime.utcnow()

    # Add simple risk bands from RF probabilities for business use
    if "proba_random_forest" in pred_db.columns:
        p = pred_db["proba_random_forest"]
        pred_db["risk_band_rf"] = np.where(p >= 0.8, "High",
                                           np.where(p >= 0.5, "Medium", "Low"))
    else:
        pred_db["risk_band_rf"] = None

    # Overwrite the prediction table
    pred_db.reset_index(drop=True).to_sql(
        PRED_TABLE,
        engine,
        schema=SCHEMA,
        if_exists="replace",
        index=False
    )

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\nTraining + evaluation complete OK\n")
    print("Top metrics:")
    print(metrics_df.to_string(index=False))

    print("\nSaved outputs:")
    print(f"- Metrics: {metrics_path}")
    print(f"- ROC points (ALL models): {roc_path}")
    print(f"- Test predictions: {preds_path}")
    print(f"- Feature importance (RF): {fi_path}")
    print(f"- PostgreSQL table: {SCHEMA}.{PRED_TABLE}")


if __name__ == "__main__":
    main()
