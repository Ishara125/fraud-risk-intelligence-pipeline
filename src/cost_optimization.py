"""Section E: Cost-Sensitive Learning & Threshold Optimization."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# --- 1. Define Costs ---
COST_FN = 500.0  # Missed fraud
COST_REVIEW = 15.0  # Manual review cost
COST_BLOCK = 50.0  # Friction cost of blocking a legit customer

CAT_COLS = ["card_presence", "channel", "country", "device_type", "merchant_category", "authentication_level", "customer_tier"]

def build_model(X):
    num_cols = [c for c in X.columns if c not in CAT_COLS and c != "transaction_id"]
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, CAT_COLS),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])

def main():
    # Load Engineered data (since it gave the best LogReg PR-AUC)
    df = pd.read_csv("data/engineered_data.csv")
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train
    model = build_model(X_train)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    # --- 2. Threshold Sweep ---
    # We test thresholds from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.0, 0.01)
    results = []

    for t in thresholds:
        # Decision logic: 
        # Proba > t -> Block (Cost: 50 if legit, 0 if fraud)
        # Proba > t/2 -> Review (Cost: 15 if legit, 15 if fraud)
        # Proba <= t/2 -> Approve (Cost: 0 if legit, 500 if fraud)
        
        t_review = t / 2.0
        
        preds = []
        costs = []
        
        for p, actual in zip(proba, y_test):
            if p >= t:
                # Blocked
                cost = COST_BLOCK if actual == 0 else 0
                pred = "Block"
            elif p >= t_review:
                # Review
                cost = COST_REVIEW
                pred = "Review"
            else:
                # Approve
                cost = 0 if actual == 0 else COST_FN
                pred = "Approve"
            
            preds.append(pred)
            costs.append(cost)

        results.append({
            "threshold_block": t,
            "threshold_review": t_review,
            "total_cost": sum(costs),
            "avg_cost_per_txn": np.mean(costs),
            "n_block": preds.count("Block"),
            "n_review": preds.count("Review"),
            "n_approve": preds.count("Approve"),
        })

    res_df = pd.DataFrame(results)
    
    # --- 3. Find Optimal Threshold ---
    best = res_df.loc[res_df["total_cost"].idxmin()]
    
    print("=" * 60)
    print("COST-SENSITIVE THRESHOLD OPTIMIZATION")
    print("=" * 60)
    print(f"Assumed Costs: FN=${COST_FN}, Review=${COST_REVIEW}, Block=${COST_BLOCK}")
    print(f"\nOptimal Block Threshold: {best['threshold_block']:.2f}")
    print(f"Optimal Review Threshold: {best['threshold_review']:.2f}")
    print(f"\nMin Expected Cost on Test Set: ${best['total_cost']:.2f}")
    print(f"Avg Cost Per Transaction: ${best['avg_cost_per_txn']:.2f}")
    print(f"\nVolume Breakdown:")
    print(f"  Approved: {best['n_approve']} / {len(y_test)}")
    print(f"  Reviewed: {best['n_review']} / {len(y_test)}")
    print(f"  Blocked:  {best['n_block']} / {len(y_test)}")

    # --- 4. Compare to Default 0.5 ---
    default_row = res_df.loc[(res_df["threshold_block"] - 0.50).abs().idxmin()]
    print(f"\nComparison vs Default (0.50):")
    print(f"  Default Cost: ${default_row['total_cost']:.2f}")
    print(f"  Optimal Cost: ${best['total_cost']:.2f}")
    print(f"  Savings: ${default_row['total_cost'] - best['total_cost']:.2f}")

    # Save output for deliverable 05_threshold_cost_analysis.csv
    res_df.to_csv("reports/05_threshold_cost_analysis.csv", index=False)
    print("\nSaved reports/05_threshold_cost_analysis.csv")

    # Save the trained pipeline for later steps (Section F & H)
    joblib.dump(model, "models/logreg_pipeline.joblib")
    print("Saved models/logreg_pipeline.joblib")

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()