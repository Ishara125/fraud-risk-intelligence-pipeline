"""Section C: Feature Engineering — 12 defensible derived features."""
import pandas as pd
import numpy as np
from pathlib import Path

CLEAN = Path("data/clean_data.csv")
OUT = Path("data/engineered_data.csv")

def main():
    df = pd.read_csv(CLEAN)
    print(f"Original shape: {df.shape}")

    # 1. Velocity ratio: Is the recent 1h activity unusually high compared to 24h?
    df["velocity_ratio_1h_24h"] = df["transactions_last_1h"] / (df["transactions_last_24h"] + 1)

    # 2. Log amount: Reduce skew of transaction amount
    df["amount_log"] = np.log1p(df["transaction_amount"])

    # 3. Amount per account day: Behavioral deviation. High amounts on new accounts.
    df["amount_per_account_day"] = df["transaction_amount"] / (df["account_age_days"] + 1)

    # 4. Combined risk score: Additive risk
    df["combined_risk_score"] = df["merchant_risk_score"] + df["ip_risk_score"]

    # 5. Risk interaction: Multiplicative risk (high merchant risk * high ip risk)
    df["risk_score_interaction"] = df["merchant_risk_score"] * df["ip_risk_score"]

    # 6. Amount x Merchant Risk: Explicit interaction from our EDA heatmap
    df["amount_x_merchant_risk"] = df["transaction_amount"] * df["merchant_risk_score"]

    # 7. Distance per account day: How far is the transaction relative to account maturity?
    df["distance_per_account_day"] = df["distance_from_usual_location_km"] / (df["account_age_days"] + 1)

    # 8. Night indicator: Based on EDA, early morning (0-6) has higher fraud
    df["is_night_txn"] = df["hour"].apply(lambda x: 1 if x in [0, 1, 2, 3, 4, 5, 6] else 0)

    # 9. Login speed: How fast did they go from login to transaction?
    # (fast logins might be automated)
    df["is_fast_login"] = (df["login_to_transaction_minutes"] < 2.0).astype(int)

    # 10. Device age ratio: Is the device very new relative to the account?
    df["device_to_account_age_ratio"] = df["device_age_days"] / (df["account_age_days"] + 1)

    # 11. New device x Distance: Interaction from PDF prompt
    df["new_device_x_distance"] = df["new_device"] * df["distance_from_usual_location_km"]

    # 12. Missing risk score indicator: From our missingness analysis!
    # Note: clean_data.csv has already imputed it, so we must use the raw file
    # to recover what was originally missing.
    raw = pd.read_csv("data/transaction_fraud_risk_advanced_dataset.csv")
    raw = raw[raw["transaction_id"] != "TXN5000485"].reset_index(drop=True)
    df["was_missing_merchant_risk"] = raw["merchant_risk_score"].isna().astype(int)

    print(f"Engineered shape: {df.shape}")
    df.to_csv(OUT, index=False)
    print(f"Saved to {OUT}")

if __name__ == "__main__":
    main()