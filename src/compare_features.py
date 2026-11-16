import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

def evaluate(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Simple shallow RF to see feature utility
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:, 1]
    
    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"{name:25s} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")

def main():
    # 1. Original cleaned data (one-hot encoded)
    df_orig = pd.read_csv("data/clean_data.csv")
    cat_cols = ["card_presence", "channel", "country", "device_type", 
                "merchant_category", "authentication_level", "customer_tier"]
    df_orig = pd.get_dummies(df_orig, columns=cat_cols)
    y = df_orig["fraud"]
    X_orig = df_orig.drop(columns=["fraud"])
    evaluate(X_orig, y, "Original Features")

    # 2. Engineered data (one-hot encoded)
    df_eng = pd.read_csv("data/engineered_data.csv")
    df_eng = pd.get_dummies(df_eng, columns=cat_cols)
    y = df_eng["fraud"]
    X_eng = df_eng.drop(columns=["fraud"])
    evaluate(X_eng, y, "Engineered Features")

if __name__ == "__main__":
    main()