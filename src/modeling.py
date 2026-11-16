"""Section D: Cross-validated modeling pipeline.
Compares LR / RF / HGB on raw vs. engineered features using 5-fold stratified CV.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score

# ---- Scoring ----
def pr_auc(y_true, y_score):
    return average_precision_score(y_true, y_score)

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": make_scorer(pr_auc, response_method="predict_proba"),
    "f1": "f1",
}

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

CAT_COLS = [
    "card_presence", "channel", "country", "device_type",
    "merchant_category", "authentication_level", "customer_tier",
]

def build_preprocessor(X):
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, CAT_COLS),
    ])

def make_models():
    return {
        "LogReg": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=SEED,
        ),
    }

def evaluate_set(df, tag):
    # Drop ID and target from features
    y = df["fraud"].astype(int)
    X = df.drop(columns=["fraud"])
    for drop_col in ("transaction_id",):
        if drop_col in X.columns:
            X = X.drop(columns=[drop_col])

    print(f"\n{'=' * 70}\n{tag}  (n={len(df)}, n_features={X.shape[1]})\n{'=' * 70}")
    rows = []
    for name, model in make_models().items():
        pipe = Pipeline([
            ("prep", build_preprocessor(X)),
            ("model", model),
        ])
        cvres = cross_validate(
            pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1, error_score="raise",
        )
        row = {
            "model": name,
            "roc_auc_mean": cvres["test_roc_auc"].mean(),
            "roc_auc_std":  cvres["test_roc_auc"].std(),
            "pr_auc_mean":  cvres["test_pr_auc"].mean(),
            "pr_auc_std":   cvres["test_pr_auc"].std(),
            "f1_mean":      cvres["test_f1"].mean(),
            "f1_std":       cvres["test_f1"].std(),
        }
        rows.append(row)
        print(f"{name:15s} ROC-AUC: {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f} "
              f"| PR-AUC: {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} "
              f"| F1: {row['f1_mean']:.4f}")
    return pd.DataFrame(rows)

def main():
    raw = pd.read_csv("data/clean_data.csv")
    eng = pd.read_csv("data/engineered_data.csv")

    results_raw = evaluate_set(raw, "RAW features (cleaned data)")
    results_eng = evaluate_set(eng, "ENGINEERED features")

    results_raw.to_csv("reports/cv_results_raw.csv", index=False)
    results_eng.to_csv("reports/cv_results_engineered.csv", index=False)
    print("\nSaved reports/cv_results_raw.csv and reports/cv_results_engineered.csv")

if __name__ == "__main__":
    main()