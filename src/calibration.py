"""Section F: Probability Calibration.
Compares Uncalibrated vs Platt (Sigmoid) vs Isotonic.
Produces calibration curves and Brier scores.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import joblib

# Re-use the pipeline builder from cost_optimization
import sys
sys.path.append("src")
from cost_optimization import build_model

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

def main():
    # Load data
    df = pd.read_csv("data/engineered_data.csv")
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Same split as before
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Base Model
    base_model = build_model(X_train)
    base_model.fit(X_train, y_train)
    prob_base = base_model.predict_proba(X_test)[:, 1]

    # 2. Platt Scaling (Sigmoid)
    platt = CalibratedClassifierCV(estimator=build_model(X_train), method="sigmoid", cv=5)
    platt.fit(X_train, y_train)
    prob_platt = platt.predict_proba(X_test)[:, 1]

    # 3. Isotonic Calibration
    iso = CalibratedClassifierCV(estimator=build_model(X_train), method="isotonic", cv=5)
    iso.fit(X_train, y_train)
    prob_iso = iso.predict_proba(X_test)[:, 1]

    # 4. Evaluate Brier Scores (lower is better)
    print("=" * 60)
    print("CALIBRATION METRICS (Test Set)")
    print("=" * 60)
    print(f"Brier Score (Uncalibrated): {brier_score_loss(y_test, prob_base):.4f}")
    print(f"Brier Score (Platt):        {brier_score_loss(y_test, prob_platt):.4f}")
    print(f"Brier Score (Isotonic):     {brier_score_loss(y_test, prob_iso):.4f}")

    # 5. Plot Calibration Curves
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    
    for prob, name, color in [
        (prob_base, "Uncalibrated", "gray"),
        (prob_platt, "Platt (Sigmoid)", "blue"),
        (prob_iso, "Isotonic", "green")
    ]:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob, n_bins=10, strategy="quantile"
        )
        ax1.plot(mean_predicted_value, fraction_of_positives, marker='o', label=name, color=color)

    ax1.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfectly calibrated")
    ax1.set_ylabel("Fraction of positives (Actual Fraud Rate)")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_title("Calibration Curves (Reliability Diagram)")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(FIG / "16_calibration_curves.png", dpi=150)
    plt.close()
    
    print("\nSaved reports/figures/16_calibration_curves.png")

    # 6. Save the best calibrated model
    # We'll use Platt if Brier is lowest, otherwise Isotonic
    if brier_score_loss(y_test, prob_iso) < brier_score_loss(y_test, prob_platt):
        print("Isotonic performed best. Saving Isotonic model.")
        joblib.dump(iso, "models/calibrated_model.joblib")
    else:
        print("Platt performed best. Saving Platt model.")
        joblib.dump(platt, "models/calibrated_model.joblib")
        
    print("Saved models/calibrated_model.joblib")

if __name__ == "__main__":
    main()