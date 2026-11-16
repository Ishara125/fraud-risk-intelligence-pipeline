"""Section A: Data audit — schema, dupes, cardinality, impossible values,
extreme observations, missingness-as-signal, and leakage notes."""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/transaction_fraud_risk_advanced_dataset.csv")
TRAIN = Path("data/train.csv")

def load():
    return pd.read_csv(RAW)

def check_schema(df):
    print("=" * 70)
    print("SCHEMA")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    ids = df["transaction_id"].str.extract(r"TXN(\d+)")[0].astype(int)
    expected = set(range(ids.min(), ids.max() + 1))
    missing = sorted(expected - set(ids))
    print(f"\ntransaction_id range: {ids.min()} .. {ids.max()}")
    print(f"Missing IDs in sequence: {missing}")

def check_duplicates(df):
    print("\n" + "=" * 70)
    print("DUPLICATES")
    print("=" * 70)
    print(f"Full-row duplicates: {df.duplicated().sum()}")
    print(f"Duplicate transaction_ids: {df['transaction_id'].duplicated().sum()}")

def check_cardinality(df):
    print("\n" + "=" * 70)
    print("CARDINALITY (categorical columns)")
    print("=" * 70)
    for col in df.select_dtypes(include=["object", "string"]).columns:
     if col == "transaction_id":
        print(f"{col}: {df[col].nunique()} unique (skipped, high-cardinality ID)")
        continue
    print(f"{col} ({df[col].nunique()} unique):")
    print(df[col].value_counts(dropna=False).to_string(), "\n")

def check_impossible(df):
    print("=" * 70)
    print("IMPOSSIBLE VALUES")
    print("=" * 70)
    checks = {
        "hour in [0,23]": df["hour"].between(0, 23).all(),
        "day_of_week in [0,6]": df["day_of_week"].between(0, 6).all(),
        "transaction_amount > 0": (df["transaction_amount"].dropna() > 0).all(),
        "distance >= 0": (df["distance_from_usual_location_km"].dropna() >= 0).all(),
        "merchant_risk_score in [0,1]": df["merchant_risk_score"].dropna().between(0, 1).all(),
        "ip_risk_score in [0,1]": df["ip_risk_score"].dropna().between(0, 1).all(),
        "failed_logins_24h >= 0": (df["failed_logins_24h"] >= 0).all(),
        "prior_chargebacks >= 0": (df["prior_chargebacks"] >= 0).all(),
        "account_age_days >= 0": (df["account_age_days"] >= 0).all(),
        "device_age_days >= 0": (df["device_age_days"] >= 0).all(),
        "email_age_days >= 0": (df["email_age_days"] >= 0).all(),
        "login_to_transaction_minutes >= 0": (df["login_to_transaction_minutes"].dropna() >= 0).all(),
        "fraud in [0,1]": df["fraud"].isin([0, 1]).all(),
        "new_device in [0,1]": df["new_device"].isin([0, 1]).all(),
        "billing_address_mismatch in [0,1]": df["billing_address_mismatch"].isin([0, 1]).all(),
    }
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")

def check_extremes(df):
    print("\n" + "=" * 70)
    print("EXTREME OBSERVATIONS (3x IQR rule)")
    print("=" * 70)
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n = ((s < lo) | (s > hi)).sum()
        print(f"{col:35s} outliers={n:4d}  min={s.min():12.2f}  max={s.max():12.2f}")

def check_missingness_signal(train):
    print("\n" + "=" * 70)
    print("MISSINGNESS AS SIGNAL (train only — avoids test leakage)")
    print("=" * 70)
    print(f"Overall fraud rate: {train['fraud'].mean():.4f}\n")
    for col in train.columns:
        if col == "fraud":
            continue
        miss = train[col].isna()
        n = miss.sum()
        if n == 0:
            continue
        f_miss = train.loc[miss, "fraud"].mean()
        f_pres = train.loc[~miss, "fraud"].mean()
        print(f"{col:35s} n={n:4d}  fraud|miss={f_miss:.4f}  "
              f"fraud|present={f_pres:.4f}  diff={f_miss - f_pres:+.4f}")

def leakage_notes():
    print("\n" + "=" * 70)
    print("LEAKAGE ANALYSIS (to be included in report)")
    print("=" * 70)
    print("""
1. prior_chargebacks — permissible ONLY if it counts chargebacks from
   strictly PRIOR transactions. If updated after current transaction is
   classified, it would leak the label. No data dictionary available;
   we assume prior-only and document this assumption.

2. No transaction timestamp present (only hour and day_of_week).
   A proper temporal holdout is impossible — this is a genuine
   limitation. We use random stratified split and document the risk.

3. All other features (amount, distance, velocity counts, ages, risk
   scores, auth level, channel, country, category) are available at
   decision time and encode no future information.

4. Row TXN5000485 in the raw file was truncated/malformed — removed
   during ingestion (documented in data quality section).

5. Raw file supplied in chat was itself truncated at row ~485; user
   substituted with locally-cached version of 2999 rows. This is
   documented as an ingestion note.
    """)

if __name__ == "__main__":
    df = load()
    check_schema(df)
    check_duplicates(df)
    check_cardinality(df)
    check_impossible(df)
    check_extremes(df)
    check_missingness_signal(pd.read_csv(TRAIN))
    leakage_notes()