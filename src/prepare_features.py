import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CLEAN = Path("data/clean_data.csv")
OUT_TRAIN = Path("data/train.csv")
OUT_TEST = Path("data/test.csv")

CATEGORICAL = [
    "card_presence",
    "channel",
    "country",
    "device_type",
    "merchant_category",
    "authentication_level",
    "customer_tier",
]

TARGET = "fraud"

def main():
    df = pd.read_csv(CLEAN)

    # --- One-hot encode categoricals ---
    print("Before encoding, shape:", df.shape)
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False)
    print("After encoding, shape:", df.shape)

    # --- Split features / target ---
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # --- Train/test split (stratified to preserve 12% fraud rate) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4f}")

    # --- Save as single files with target included ---
    train = X_train.copy()
    train[TARGET] = y_train.values
    test = X_test.copy()
    test[TARGET] = y_test.values

    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)
    print(f"\nSaved {OUT_TRAIN} and {OUT_TEST}")

if __name__ == "__main__":
    main()