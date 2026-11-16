import pandas as pd

df = pd.read_csv("data/transaction_fraud_risk_advanced_dataset.csv")

print("Shape:", df.shape)
print("\nColumns:\n", list(df.columns))
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nFraud distribution:\n", df["fraud"].value_counts(dropna=False))
print("\nFirst 3 rows:\n", df.head(3))