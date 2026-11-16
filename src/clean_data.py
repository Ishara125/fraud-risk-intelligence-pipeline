import pandas as pd
from pathlib import Path

RAW = Path("data/transaction_fraud_risk_advanced_dataset.csv")
OUT = Path("data/clean_data.csv")

def main():
    df = pd.read_csv(RAW)

    # --- Numeric imputation (median) ---
    numeric_impute = [
        "transaction_amount",
        "distance_from_usual_location_km",
        "merchant_risk_score",
        "ip_risk_score",
        "login_to_transaction_minutes",
    ]
    for col in numeric_impute:
        median = df[col].median()
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(median)
        print(f"{col}: filled {n_missing} NaNs with median {median:.4f}")

    # --- Categorical imputation ---
    n_missing = df["authentication_level"].isna().sum()
    df["authentication_level"] = df["authentication_level"].fillna("Unknown")
    print(f"authentication_level: filled {n_missing} NaNs with 'Unknown'")

    # --- Drop the ID column (not a feature) ---
    df = df.drop(columns=["transaction_id"])

    # --- Verify zero missing ---
    remaining = df.isna().sum().sum()
    print(f"\nTotal remaining NaNs: {remaining}")
    assert remaining == 0, "Still have NaNs!"

    # --- Save ---
    df.to_csv(OUT, index=False)
    print(f"\nSaved cleaned data to {OUT}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    main()