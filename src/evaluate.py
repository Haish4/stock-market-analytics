"""Deep error analysis on the TEST period: where are predictions better or worse?

For a model's P(up) we report, per slice (symbol / month / hour of day / regime):
  auc        ranking quality inside the slice
  hit_top10  hit rate of the top-10 % most confident "up" calls (precision)
  base       share of rows that really went up (what a blind long would get)
  edge       hit_top10 - base
Outputs CSVs and a plot in reports/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg

REPORTS = cfg.ROOT / "reports"


def _slice_stats(d: pd.DataFrame, col: str) -> pd.Series:
    y = d["y_up"]
    thr = d[col].quantile(0.9)
    top = d[d[col] >= thr]
    auc = roc_auc_score(y, d[col]) if y.nunique() > 1 else np.nan
    return pd.Series({"rows": len(d), "auc": auc, "hit_top10": top["y_up"].mean(),
                      "base": y.mean(), "edge": top["y_up"].mean() - y.mean(),
                      "avg_ret_top10": top["ret_fwd_24h"].mean()})


def run(col: str = "pred_xgb") -> dict[str, pd.DataFrame]:
    REPORTS.mkdir(exist_ok=True)
    p = pd.read_parquet(cfg.PROCESSED_DIR / "predictions.parquet")
    t = p[p["split"] == "test"].copy()
    t["month"] = t["open_time"].dt.strftime("%Y-%m")
    t["hour"] = t["open_time"].dt.hour
    t["fg_regime"] = pd.cut(t["fear_greed"], [-1, 25, 45, 55, 75, 101],
                            labels=["extreme_fear", "fear", "neutral", "greed", "extreme_greed"])
    t["mkt_trend"] = np.where(t["btc_ret_24h"] > 0, "btc_up_24h", "btc_down_24h")

    out = {}
    for key in ["symbol", "month", "hour", "fg_regime", "mkt_trend"]:
        res = t.groupby(key, observed=True).apply(lambda d: _slice_stats(d, col), include_groups=False)
        res.to_csv(REPORTS / f"errors_by_{key}.csv")
        out[key] = res
        print(f"\n== {col} by {key}\n", res.round(3).sort_values("edge").to_string() if key == "symbol"
              else res.round(3).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    out["month"]["edge"].plot.bar(ax=ax[0], title="top-10% hit-rate edge by month")
    out["hour"]["edge"].plot.bar(ax=ax[1], title="edge by hour of day (UTC)")
    out["symbol"]["auc"].sort_values().plot.barh(ax=ax[2], title="AUC by symbol", figsize=(16, 9))
    ax[2].axvline(0.5, color="k", lw=.8)
    plt.tight_layout(); plt.savefig(REPORTS / "errors_overview.png", dpi=100); plt.close()

    # biggest wrong calls (debugging): highest-confidence ups that fell the most
    worst = t.nlargest(500, col).nsmallest(15, "ret_fwd_24h")[["open_time", "symbol", col, "ret_fwd_24h"]]
    worst.to_csv(REPORTS / "worst_calls.csv", index=False)
    print("\nWorst high-confidence calls:\n", worst.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    run()
