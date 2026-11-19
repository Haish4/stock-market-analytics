"""Walk-forward validation of the daily market-neutral book.

The XGBoost regressor on the market-relative return is re-trained on an EXPANDING
window before every fold and then predicts the next, unseen window. Folds are
chained, so the concatenated predictions form one long out-of-sample history
(2024-07 .. latest) instead of a single test window. Hyperparameters are fixed
(not tuned on any fold). A 24h purge gap separates train and test.

Caveat: k=10 and the 7-day smoothing were originally chosen on the 2025-H2 window,
so folds from 2025-07 onward are not completely independent of that choice. The
sensitivity table (span 1/3/7) shows how much the conclusion depends on it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import config as cfg
from .simulate import COST, REPORTS, _daily_stats, benchmarks, daily_rebalance
from .transform import get_feature_sets

CUTOFFS = ["2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01", "2025-07-01",
           "2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]
PARAMS = dict(max_depth=4, learning_rate=0.05, n_estimators=300, subsample=0.8,
              colsample_bytree=0.7, min_child_weight=200)
COL = "pred_xgb_reg_rel"
COSTS = {"taker 0.15%/side": COST, "maker-ish 0.05%/side": 0.0005}


def make_oos_predictions() -> pd.DataFrame:
    df = pd.read_parquet(cfg.PROCESSED_DIR / "dataset.parquet")
    feats = get_feature_sets(df)["FEATURES"]
    cuts = [pd.Timestamp(c, tz="UTC") for c in CUTOFFS]
    ends = cuts[1:] + [df["open_time"].max() + pd.Timedelta(hours=1)]
    parts = []
    for i, (c, e) in enumerate(zip(cuts, ends)):
        tr = df[(df["open_time"] < c - pd.Timedelta(hours=24)) & df["ret_fwd_rel_24h"].notna()]
        te = df[(df["open_time"] >= c) & (df["open_time"] < e) & df["ret_fwd_24h"].notna()]
        m = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0, **PARAMS)
        m.fit(tr[feats].values, tr["ret_fwd_rel_24h"].values)
        te = te[["symbol", "open_time", "close", "ret_fwd_24h", "ret_fwd_rel_24h"]].copy()
        te[COL] = m.predict(df.loc[te.index, feats].values)
        te["fold"] = i
        ic = te[COL].corr(te["ret_fwd_rel_24h"], method="spearman")
        print(f"fold {i} {c.date()} -> {e.date()}: train rows {len(tr):,}, test rows {len(te):,}, rank-IC {ic:.4f}")
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def run() -> pd.DataFrame:
    REPORTS.mkdir(exist_ok=True)
    oos = make_oos_predictions()
    oos.to_parquet(cfg.PROCESSED_DIR / "predictions_walkforward.parquet", index=False)
    fold_of_date = oos.groupby(oos["open_time"].dt.normalize())["fold"].first()
    close_all = benchmarks(oos)

    rows, pooled = [], []
    curves = {}
    for span in (1, 3, 7):
        for cname, c in COSTS.items():
            per_hour = [daily_rebalance(oos, COL, 10, span, c, h)["net"] for h in range(24)]
            # ---- per fold: average the 24 rebalance-hour offsets ----
            for f in sorted(oos["fold"].unique()):
                st = []
                for net in per_hour:
                    n = net[fold_of_date.reindex(net.index.normalize()).to_numpy() == f]
                    if len(n) > 5:
                        st.append(_daily_stats(n))
                s = pd.DataFrame(st).mean()
                rows.append({"span_days": span, "cost": cname, "fold": f, **s.to_dict()})
            pooled.append({"span_days": span, "cost": cname,
                           **pd.DataFrame([_daily_stats(n) for n in per_hour]).mean().to_dict()})
            avg = pd.concat([n.set_axis(n.index.normalize()) for n in per_hour], axis=1).mean(axis=1)
            curves[(span, cname)] = (1 + avg).cumprod()

    folds = pd.DataFrame(rows)
    # benchmark return per fold
    for f in sorted(oos["fold"].unique()):
        part = oos[oos["fold"] == f]
        b = benchmarks(part)
        folds.loc[folds["fold"] == f, "btc_return"] = b["BTC buy&hold"].iloc[-1] / b["BTC buy&hold"].iloc[0] - 1
        folds.loc[folds["fold"] == f, "basket_return"] = (b["Equal-weight basket"].iloc[-1]
                                                          / b["Equal-weight basket"].iloc[0] - 1)
    folds["start"] = folds["fold"].map(lambda i: CUTOFFS[int(i)])
    folds.to_csv(REPORTS / "walkforward_folds.csv", index=False)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(REPORTS / "walkforward_pooled.csv", index=False)

    main = folds[(folds["span_days"] == 7) & (folds["cost"] == "taker 0.15%/side")]
    print("\n== per fold (k=10, 7-day smoothing, taker cost)")
    print(main[["start", "total_return", "sharpe", "max_drawdown", "btc_return", "basket_return"]]
          .round(3).to_string(index=False))
    print("\n== pooled out-of-sample (all folds chained), avg over 24 rebalance hours")
    print(pooled.round(3).to_string(index=False))
    print("\n== share of folds with positive return:")
    print(folds.groupby(["span_days", "cost"])["total_return"].apply(lambda s: round((s > 0).mean(), 2)).to_string())

    # ---- figure ----
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for (span, cname), eq in curves.items():
        if span == 7:
            eq.plot(ax=ax[0], label=f"book span={span}, {cname}", lw=2)
    for name, v in close_all.items():
        d = v.resample("D").last()
        (d / d.iloc[0]).plot(ax=ax[0], label=name, lw=1)
    for c in CUTOFFS:
        ax[0].axvline(pd.Timestamp(c, tz="UTC"), color="grey", lw=.5, ls=":")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
    ax[0].set_title("Walk-forward out-of-sample equity (log scale), dotted lines = re-training dates")
    x = np.arange(len(main))
    ax[1].bar(x - 0.27, main["total_return"], 0.27, label="book (taker cost)")
    ax[1].bar(x, main["btc_return"], 0.27, label="BTC buy&hold")
    ax[1].bar(x + 0.27, main["basket_return"], 0.27, label="equal-weight basket")
    ax[1].set_xticks(x); ax[1].set_xticklabels(main["start"], rotation=0, fontsize=8)
    ax[1].axhline(0, color="k", lw=.8); ax[1].legend(fontsize=8)
    ax[1].set_title("Return in each out-of-sample window")
    plt.tight_layout(); plt.savefig(REPORTS / "walkforward.png", dpi=110); plt.close()
    return folds


if __name__ == "__main__":
    run()
