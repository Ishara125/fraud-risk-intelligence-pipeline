"""Section G: Compare Class Weighting vs. Resampling."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

CAT_COLS = ["card_presence", "channel", "country", "device_type", "merchant_category", "authentication_level", "customer_tier"]

def get_preprocessor(X):
    num_cols = [c for c in X.columns if c not in CAT_COLS and c != "transaction_id"]
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

def evaluate(y_true, y_pred, y_proba, name):
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    print(f"{name:25s} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f}")

def main():
    df = pd.read_csv("data/engineered_data.csv")
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Class Weighting (Our current approach)
    pipe_weight = Pipeline([
        ("prep", get_preprocessor(X_train)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    pipe_weight.fit(X_train, y_train)
    evaluate(y_test, pipe_weight.predict(X_test), pipe_weight.predict_proba(X_test)[:, 1], "Class Weighting")

    # 2. Random Undersampling (Need ImbPipeline so sampling ONLY happens on train fold)
    pipe_resample = ImbPipeline([
        ("prep", get_preprocessor(X_train)),
        ("sampler", RandomUnderSampler(random_state=42)),
        ("model", LogisticRegression(max_iter=2000, random_state=42)) # No class_weight here
    ])
    pipe_resample.fit(X_train, y_train)
    evaluate(y_test, pipe_resample.predict(X_test), pipe_resample.predict_proba(X_test)[:, 1], "Random Undersampling")

if __name__ == "__main__":
    main()