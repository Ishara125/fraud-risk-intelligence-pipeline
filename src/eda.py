"""Section B: Advanced EDA — at least 10 meaningful visualizations,
including at least two interaction-focused plots.

All plots save to reports/figures/ as high-res PNGs.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

def load():
    df = pd.read_csv("data/transaction_fraud_risk_advanced_dataset.csv")
    df = df[df["transaction_id"] != "TXN5000485"].copy()
    # For EDA we need to SEE the missingness, so we fill auth_level with "Unknown"
    # but do NOT drop/mutate missing numeric — those are handled per-plot.
    df["authentication_level"] = df["authentication_level"].fillna("Unknown")
    return df

def rate_by_category(df, col, fname, title, sort=True, min_n=20):
    g = df.groupby(col, dropna=False)["fraud"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= min_n]  # suppress tiny noisy groups
    if sort:
        g = g.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(g[col].astype(str), g["mean"], color="#3b6fb6")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel(col)
    ax.set_title(title)
    ax.axhline(df["fraud"].mean(), color="red", linestyle="--", label=f"Overall ({df['fraud'].mean():.3f})")
    ax.legend(loc="upper right")
    # annotate counts
    for bar, n in zip(bars, g["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={n}", ha="center", va="bottom", fontsize=7)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150, bbox_inches="tight")
    plt.close()

def rate_by_bin(df, col, fname, title, bins=10):
    s = df[col].dropna()
    try:
        binned = pd.qcut(s, q=bins, duplicates="drop")
    except ValueError:
        binned = pd.cut(s, bins=bins)
    g = df.loc[s.index].assign(_bin=binned).groupby("_bin", observed=True)["fraud"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = [f"{b.left:.1f}–{b.right:.1f}" for b in g.index]
    bars = ax.bar(x, g["mean"], color="#3b6fb6")
    ax.axhline(df["fraud"].mean(), color="red", linestyle="--", label=f"Overall ({df['fraud'].mean():.3f})")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel(col)
    ax.set_title(title)
    ax.legend()
    for bar, n in zip(bars, g["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{n}", ha="center", va="bottom", fontsize=7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150, bbox_inches="tight")
    plt.close()

def interaction_heatmap(df, x_col, y_col, fname, title, q=4):
    """2D binned fraud-rate heatmap — interaction-focused plot."""
    sub = df[[x_col, y_col, "fraud"]].dropna().copy()
    sub["_x"] = pd.qcut(sub[x_col], q=q, duplicates="drop", labels=False)
    sub["_y"] = pd.qcut(sub[y_col], q=q, duplicates="drop", labels=False)
    pivot = sub.groupby(["_y", "_x"])["fraud"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds", cbar_kws={"label": "Fraud rate"}, ax=ax)
    ax.set_xlabel(f"{x_col} (quartile bins, low→high)")
    ax.set_ylabel(f"{y_col} (quartile bins, low→high)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    df = load()
    print(f"Fraud rate: {df['fraud'].mean():.4f}  n={len(df)}")

    # --- Section B-1: fraud rate by category (5 plots) ---
    rate_by_category(df, "channel", "01_fraud_by_channel.png",
                     "Fraud rate by channel")
    rate_by_category(df, "merchant_category", "02_fraud_by_merchant_category.png",
                     "Fraud rate by merchant category")
    rate_by_category(df, "authentication_level", "03_fraud_by_auth_level.png",
                     "Fraud rate by authentication level")
    rate_by_category(df, "customer_tier", "04_fraud_by_customer_tier.png",
                     "Fraud rate by customer tier")
    rate_by_category(df, "device_type", "05_fraud_by_device_type.png",
                     "Fraud rate by device type")

    # --- Section B-2: time (2 plots) ---
    rate_by_category(df, "hour", "06_fraud_by_hour.png",
                     "Fraud rate by hour of day")
    rate_by_category(df, "day_of_week", "07_fraud_by_day_of_week.png",
                     "Fraud rate by day of week")

    # --- Section B-3: nonlinear effects (4 plots) ---
    rate_by_bin(df, "transaction_amount", "08_fraud_by_amount_bin.png",
                "Fraud rate vs transaction amount (deciles)")
    rate_by_bin(df, "transactions_last_1h", "09_fraud_by_velocity_1h.png",
                "Fraud rate vs transactions last 1h (quantile bins)")
    rate_by_bin(df, "distance_from_usual_location_km", "10_fraud_by_distance_bin.png",
                "Fraud rate vs distance from usual location (deciles)")
    rate_by_bin(df, "account_age_days", "11_fraud_by_account_age_bin.png",
                "Fraud rate vs account age days (deciles)")

    # --- Section B-4: interactions (3 plots, ≥2 required) ---
    interaction_heatmap(df, "transaction_amount", "merchant_risk_score",
                        "12_interaction_amount_x_merchant_risk.png",
                        "Interaction: transaction amount × merchant risk (quartiles)",
                        q=4)
    interaction_heatmap(df, "distance_from_usual_location_km", "account_age_days",
                        "13_interaction_distance_x_account_age.png",
                        "Interaction: distance × account age (quartiles)", q=4)

    # auth × channel — grouped bar
    sub = df.dropna(subset=["authentication_level"]).copy()
    pivot = sub.groupby(["channel", "authentication_level"])["fraud"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Fraud rate")
    ax.set_title("Interaction: authentication level × channel")
    ax.legend(title="auth level", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "14_interaction_auth_x_channel.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Section B-5: correlation heatmap (bonus, useful for modelling) ---
    num = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(num.corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlations (includes target)")
    plt.tight_layout()
    plt.savefig(FIG / "15_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved 15 figures to {FIG}/")
    for f in sorted(FIG.glob("*.png")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()