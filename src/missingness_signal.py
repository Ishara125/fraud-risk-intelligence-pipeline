"""Investigate whether NaNs themselves carry fraud signal.
Run on the RAW dataset — BEFORE imputation."""
import pandas as pd
from pathlib import Path

RAW = Path("data/transaction_fraud_risk_advanced_dataset.csv")

def main():
    df = pd.read_csv(RAW)
    # Drop the known-malformed row if still present
    df = df[df["transaction_id"] != "TXN5000485"].copy()

    overall = df["fraud"].mean()
    print(f"Overall fraud rate (raw, n={len(df)}): {overall:.4f}\n")
    print(f"{'column':35s} {'n_miss':>7s} {'fraud|miss':>11s} "
          f"{'fraud|present':>14s} {'diff':>8s} {'flag':>6s}")
    print("-" * 90)

    any_flag = False
    for col in df.columns:
        if col in ("fraud", "transaction_id"):
            continue
        miss = df[col].isna()
        n = int(miss.sum())
        if n == 0:
            continue
        f_miss = df.loc[miss, "fraud"].mean()
        f_pres = df.loc[~miss, "fraud"].mean()
        diff = f_miss - f_pres
        # Flag if |diff| >= 0.05 AND at least 20 missing rows (noisy tiny groups)
        flag = "***" if (abs(diff) >= 0.05 and n >= 20) else ""
        if flag:
            any_flag = True
        print(f"{col:35s} {n:7d} {f_miss:11.4f} {f_pres:14.4f} "
              f"{diff:+8.4f} {flag:>6s}")

    print("\n" + ("!" * 90))
    if any_flag:
        print("VERDICT: At least one missingness indicator shows meaningful "
              "fraud-rate gap (|diff| >= 0.05). Consider adding explicit "
              "'was_missing_<col>' binary features.")
    else:
        print("VERDICT: No missingness indicator shows a meaningful "
              "fraud-rate gap. Imputation alone is defensible; no "
              "missingness-indicator features required.")
    print("!" * 90)

if __name__ == "__main__":
    main()