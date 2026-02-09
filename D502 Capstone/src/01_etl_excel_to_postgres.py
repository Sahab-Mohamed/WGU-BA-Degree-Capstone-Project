import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


# -----------------------------
# Config
# -----------------------------
DEFAULT_DB_URL = "postgresql+psycopg2://postgres:PASSWORD@127.0.0.1:5432/telco_churn"

DEFAULT_EXCEL_FILENAME = "Telco_customer_churn.xlsx"

SCHEMA_NAME = "capstone"
RAW_TABLE = "telco_churn_raw"
CLEAN_TABLE = "telco_churn_clean"

DEPENDENT_VIEWS = [
    "v_churn_rate_by_contract",
    "v_churn_rate_by_payment",
]


# -----------------------------
# Helpers
# -----------------------------
def yn_to_bool(x):
    """Convert common Yes/No style values to boolean."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    return None  # unknown / not applicable


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, trim, and replace spaces with underscores for stable references."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def get_project_root() -> Path:
    """Resolve project root as parent of src/ folder."""
    return Path(__file__).resolve().parents[1]


# -----------------------------
# Main ETL
# -----------------------------
def main():
    # Resolve paths
    base_dir = get_project_root()
    excel_path = os.environ.get("TELCO_XLSX_PATH")
    if excel_path:
        excel_path = Path(excel_path)
    else:
        excel_path = base_dir / "data" / DEFAULT_EXCEL_FILENAME

    db_url = os.environ.get("DB_URL", DEFAULT_DB_URL)

    print(f"Using Excel: {excel_path}")
    print(f"Using DB_URL host: {db_url.split('@')[-1].split('/')[0]}")

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at: {excel_path}")

    # Read Excel
    df = pd.read_excel(excel_path)
    df = normalize_columns(df)

    print("Columns found:", list(df.columns))

    required_cols = {
        "customerid",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "churn_label",
        "churn_value",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}. "
            f"Your file columns are: {list(df.columns)}"
        )

    # Connect DB
    engine = create_engine(db_url, future=True)

    with engine.begin() as conn:
        # Create schema
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};"))

        # Drop dependent views first (prevents "cannot drop table because other objects depend on it")
        for v in DEPENDENT_VIEWS:
            conn.execute(text(f"DROP VIEW IF EXISTS {SCHEMA_NAME}.{v};"))

    # -----------------------------
    # Load RAW (as-is after column normalization)
    # -----------------------------
    # Replace raw table each run
    df.to_sql(
        RAW_TABLE,
        engine,
        schema=SCHEMA_NAME,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=2000,
    )
    print(f"Loaded RAW table: {SCHEMA_NAME}.{RAW_TABLE} (rows={len(df):,})")

    # -----------------------------
    # Build CLEAN dataset
    # -----------------------------
    d = df.copy()

    # Target: churn_value (0/1) -> churn boolean
    d["churn_value"] = pd.to_numeric(
        d["churn_value"], errors="coerce").fillna(0).astype(int)
    d["churn"] = d["churn_value"].astype(bool)

    # Normalize churn_label to clean text
    d["churn_label"] = d["churn_label"].astype(str).str.strip()

    # Convert common Yes/No fields to boolean if present
    bool_cols = [
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
    ]
    for c in bool_cols:
        if c in d.columns:
            d[c] = d[c].apply(yn_to_bool)

    # Numeric conversions (safe)
    if "monthly_charges" in d.columns:
        d["monthly_charges"] = pd.to_numeric(
            d["monthly_charges"], errors="coerce")
    if "total_charges" in d.columns:
        d["total_charges"] = pd.to_numeric(d["total_charges"], errors="coerce")

    # Optional: rename tenure for convenience
    if "tenure_months" in d.columns:
        d["tenure"] = pd.to_numeric(d["tenure_months"], errors="coerce")

    # Drop leakage-prone column(s) if you want a “cleaner” predictive setup
    # churn_reason often reflects post-churn info and can cause leakage.
    # Comment this OUT if you intentionally want to keep it.
    if "churn_reason" in d.columns:
        d = d.drop(columns=["churn_reason"])

    # Select final columns for CLEAN table (keeps strong features + target)
    keep_cols = [
        # identifiers
        "customerid",
        # customer profile & services (if present)
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "tenure_months",
        "tenure",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "paperless_billing",
        "payment_method",
        # charges
        "monthly_charges",
        "total_charges",
        # churn fields
        "churn_label",
        "churn_value",
        "churn",
        # optional fields if present
        "churn_score",
        "cltv",
        "country",
        "state",
        "city",
        "zip_code",
        "latitude",
        "longitude",
    ]

    keep_cols = [c for c in keep_cols if c in d.columns]
    d_clean = d[keep_cols].copy()

    # Basic data quality summary
    dq = pd.DataFrame(
        {
            "column": d_clean.columns,
            "dtype": [str(x) for x in d_clean.dtypes],
            "null_count": [int(d_clean[c].isna().sum()) for c in d_clean.columns],
            "null_pct": [float(d_clean[c].isna().mean()) for c in d_clean.columns],
        }
    )

    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    dq_path = outputs_dir / "data_quality_report.csv"
    dq.to_csv(dq_path, index=False)
    print(f"Wrote data quality report: {dq_path}")

    # Load CLEAN table (replace each run)
    d_clean.to_sql(
        CLEAN_TABLE,
        engine,
        schema=SCHEMA_NAME,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=2000,
    )
    print(
        f"Loaded CLEAN table: {SCHEMA_NAME}.{CLEAN_TABLE} (rows={len(d_clean):,})")

    print("ETL complete")


if __name__ == "__main__":
    main()
