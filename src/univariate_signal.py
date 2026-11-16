import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

TRAIN = Path("data/train.csv")
TARGET = "fraud"

df = pd.read_csv(TRAIN)
y = df[TARGET]
X = df.drop(columns=[TARGET])

rows = []
for col in X.columns:
    try:
        auc = roc_auc_score(y, X[col])
        # flip so AUC >= 0.5 (direction doesn't matter for "signal" check)
        auc = max(auc, 1 - auc)
        rows.append((col, auc))
    except ValueError:
        pass

signal = pd.DataFrame(rows, columns=["feature", "univariate_auc"]).sort_values(
    "univariate_auc", ascending=False
)
print(signal.to_string(index=False))