"""Exploratory data analysis: target distribution and feature/target correlations.

Uses TRAIN rows only so the analysis never peeks at validation/test data.
Writes figures to reports/ and prints the key numbers.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as cfg
from .transform import get_feature_sets

REPORTS = cfg.ROOT / "reports"


def run() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_parquet(cfg.PROCESSED_DIR / "dataset.parquet")
    fs = get_feature_sets(df)
    tr = df[df["split"] == "train"]

    # ---- target description ----
    r = tr["ret_fwd_24h"]
    print("ret_fwd_24h describe:\n", r.describe(percentiles=[.01, .05, .5, .95, .99]).round(4).to_string())
    print(f"\nP(up)={tr['y_up'].mean():.3f}   P(>1%)={tr['y_up_1pct'].mean():.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    r.clip(-0.3, 0.3).hist(bins=120, ax=ax[0])
    ax[0].set_title("24h forward return (train, clipped +-30%)")
    tr.groupby("symbol")["y_up"].mean().sort_values().plot.barh(ax=ax[1], figsize=(12, 9))
    ax[1].axvline(0.5, color="k", lw=.8)
    ax[1].set_title("P(up in 24h) by symbol")
    plt.tight_layout(); plt.savefig(REPORTS / "eda_target.png", dpi=110); plt.close()

    # ---- correlation with targets ----
    cols = fs["NUMERIC"]
    corr = tr[cols + ["ret_fwd_24h", "y_up"]].corr(method="spearman")[["ret_fwd_24h", "y_up"]]
    corr = corr.drop(["ret_fwd_24h", "y_up"]).sort_values("ret_fwd_24h")
    print("\nSpearman correlation with targets (train):\n", corr.round(4).to_string())
    corr.to_csv(REPORTS / "eda_correlations.csv")

    corr["ret_fwd_24h"].plot.barh(figsize=(8, 9))
    plt.title("Spearman corr: feature vs ret_fwd_24h (train)")
    plt.tight_layout(); plt.savefig(REPORTS / "eda_corr.png", dpi=110); plt.close()

    # ---- hour-of-day / fear-greed seasonality of the target ----
    print("\nMean fwd return by session:\n", tr.groupby("session")["ret_fwd_24h"].mean().round(5).to_string())
    print("\nMean fwd return by fear/greed regime:\n", tr.groupby("fg_regime")["ret_fwd_24h"].mean().round(5).to_string())

    # ---- feature correlation heatmap (redundancy) ----
    import seaborn as sns
    plt.figure(figsize=(12, 10))
    sns.heatmap(tr[cols].corr(), cmap="coolwarm", center=0, xticklabels=True, yticklabels=True)
    plt.tight_layout(); plt.savefig(REPORTS / "eda_feature_corr.png", dpi=100); plt.close()
    print("\nfigures saved to", REPORTS)


if __name__ == "__main__":
    run()
