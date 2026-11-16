"""Section H: Explainable AI using SHAP.
Generates global beeswarm plot + 5 local waterfall plots for flagged transactions.
"""
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Load data and split
    df = pd.read_csv("data/engineered_data.csv")
    X = df.drop(columns=["fraud"])
    y = df["fraud"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Load calibrated model
    model = joblib.load("models/calibrated_model.joblib")
    
    # 3. Extract the preprocessor and base model from the calibrated wrapper
    # The calibrated model contains 5 fold models. We'll just use the first one for SHAP.
    base_pipe = model.calibrated_classifiers_[0].estimator
    prep = base_pipe.named_steps["prep"]
    logreg = base_pipe.named_steps["model"]

    # 4. Transform X_test using the preprocessor so SHAP can read feature names
    X_test_trans = prep.transform(X_test)
    feature_names = prep.get_feature_names_out()
    
    # 5. Create SHAP explainer
    # LinearExplainer is exact and fast for LogisticRegression
    explainer = shap.LinearExplainer(logreg, X_test_trans, feature_names=feature_names)
    shap_values = explainer(X_test_trans)

    # --- Global Summary Plot ---
    print("Generating Global Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(FIG / "17_shap_global_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Local Explanations (5 highest risk transactions) ---
    print("Generating Local Waterfall Plots...")
    # Find the indices of the 5 highest probabilities
    probas = model.predict_proba(X_test)[:, 1]
    top5_idx = np.argsort(probas)[-5:][::-1] # top 5 descending

    for rank, idx in enumerate(top5_idx, 1):
        # Get actual raw probability from calibrated model
        calib_proba = probas[idx]
        actual = y_test.iloc[idx]
        
        plt.figure()
        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
        plt.title(f"Rank {rank} Risk | True Fraud: {actual} | Calibrated Prob: {calib_proba:.3f}")
        plt.tight_layout()
        plt.savefig(FIG / f"18_shap_local_{rank}_txn.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    print(f"Saved SHAP plots to {FIG}/")

if __name__ == "__main__":
    main()