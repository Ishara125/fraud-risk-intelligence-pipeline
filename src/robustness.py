"""Section J: Robustness & Stress Testing.
Simulates distribution shifts to see how the model degrades.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Operational thresholds from Step 25
T_BLOCK = 0.15
T_REVIEW = 0.15
COST_FN = 500.0
COST_REVIEW = 15.0
COST_BLOCK = 50.0

def simulate_policy(y_true, proba):
    """Calculates expected cost under the operational policy."""
    cost = 0
    for p, actual in zip(proba, y_true):
        if p >= T_BLOCK: cost += COST_BLOCK if actual == 0 else 0
        elif p >= T_REVIEW: cost += COST_REVIEW
        else: cost += 0 if actual == 0 else COST_FN
    return cost

def evaluate_shift(name, y_true, proba):
    auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    cost = simulate_policy(y_true, proba)
    print(f"{name:35s} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Op. Cost: ${cost:,.0f}")

def main():
    # 1. Load data and model
    df = pd.read_csv("data/engineered_data.csv")
    X = df.drop(columns=["fraud"])
    y = df["fraud"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = joblib.load("models/calibrated_model.joblib")
    
    print("="*80)
    print("ROBUSTNESS & STRESS TESTING")
    print("="*80)
    print(f"Baseline (No Shift)")

    # --- Baseline ---
    proba_base = model.predict_proba(X_test)[:, 1]
    evaluate_shift("Baseline", y_test, proba_base)

    # --- Shift 1: Economic Inflation (Amount +50%) ---
    X_amt = X_test.copy()
    X_amt["transaction_amount"] = X_amt["transaction_amount"] * 1.5
    # Must also update engineered features that depend on amount!
    X_amt["amount_log"] = np.log1p(X_amt["transaction_amount"])
    X_amt["amount_per_account_day"] = X_amt["transaction_amount"] / (X_amt["account_age_days"] + 1)
    X_amt["amount_x_merchant_risk"] = X_amt["transaction_amount"] * X_amt["merchant_risk_score"]
    
    proba_amt = model.predict_proba(X_amt)[:, 1]
    evaluate_shift("Shift 1: Amount +50% (Inflation)", y_test, proba_amt)

    # --- Shift 2: Velocity Attack (Transactions last 1h doubled) ---
    X_vel = X_test.copy()
    X_vel["transactions_last_1h"] = X_vel["transactions_last_1h"] * 2
    X_vel["velocity_ratio_1h_24h"] = X_vel["transactions_last_1h"] / (X_vel["transactions_last_24h"] + 1)
    
    proba_vel = model.predict_proba(X_vel)[:, 1]
    evaluate_shift("Shift 2: Velocity Attack (1h x2)", y_test, proba_vel)

    # --- Shift 3: Auth Service Outage (20% of auth_level missing) ---
    X_auth = X_test.copy()
    # Randomly remove 20% of auth levels
    mask = np.random.rand(len(X_auth)) < 0.2
    X_auth.loc[mask, "authentication_level"] = np.nan
    
    proba_auth = model.predict_proba(X_auth)[:, 1]
    evaluate_shift("Shift 3: Auth Outage (20% missing)", y_test, proba_auth)

    # --- Shift 4: Channel Mix Shift (20% of E-commerce moves to POS) ---
    X_channel = X_test.copy()
    ecom_mask = (X_channel["channel"] == "E-commerce") & (np.random.rand(len(X_channel)) < 0.2)
    X_channel.loc[ecom_mask, "channel"] = "POS"
    
    proba_channel = model.predict_proba(X_channel)[:, 1]
    evaluate_shift("Shift 4: Channel Mix (Ecom -> POS)", y_test, proba_channel)

if __name__ == "__main__":
    main()