"""Live prediction: today's target book for the daily market-neutral strategy.

Uses the configuration selected on the validation period
(reports/best_strategy_daily.json: model, k, smoothing span) and the saved
relative-return model (models/xgb_reg_rel.joblib):

1. take the most recent hour in the dataset (the rebalance hour, UTC),
2. predict the same hour on each of the previous days (to build the smoothed signal),
3. smooth with an EWMA, rank the 50 coins, long the top k, short the bottom k
   (each side gets 50 % of capital, equal weights),
4. compare with yesterday's book to list what to open / close / keep.

Output: data/processed/latest_book.csv

Note: shorts are meant for perpetual futures. The book is only as fresh as the
last published archive day (data.binance.vision lags by up to a day). Re-train
the models periodically with `python -m src.train --target rel`.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .simulate import REPORTS, target_weights
from .train import MODELS_DIR
from .transform import get_feature_sets

HISTORY_DAYS = 40


def predict_live() -> pd.DataFrame:
    best = json.loads((REPORTS / "best_strategy_daily.json").read_text())
    k, span = int(best["k"]), int(best["span_days"])
    model = joblib.load(MODELS_DIR / f"{best['model'].replace('pred_', '')}.joblib")

    df = pd.read_parquet(cfg.PROCESSED_DIR / "dataset.parquet")
    feats = get_feature_sets(df)["FEATURES"]
    last = df["open_time"].max()
    sub = df[(df["open_time"].dt.hour == last.hour)
             & (df["open_time"] >= last - pd.Timedelta(days=HISTORY_DAYS))].copy()
    sub["signal"] = model.predict(sub[feats].values)

    S = sub.pivot(index="open_time", columns="symbol", values="signal").ewm(span=span, adjust=False).mean()
    W = target_weights(S, k)
    now, prev = W.iloc[-1], (W.iloc[-2] if len(W) > 1 else W.iloc[-1] * 0)
    price = sub[sub["open_time"] == last].set_index("symbol")["close"]

    book = pd.DataFrame({"signal": S.iloc[-1], "weight": now, "prev_weight": prev, "price": price})
    book = book[(book["weight"] != 0) | (book["prev_weight"] != 0)].copy()
    book["side"] = np.where(book["weight"] > 0, "long", np.where(book["weight"] < 0, "short", "-"))
    book["action"] = np.select(
        [(book["prev_weight"] == 0) & (book["weight"] != 0),
         (book["prev_weight"] != 0) & (book["weight"] == 0),
         np.sign(book["prev_weight"]) != np.sign(book["weight"])],
        ["OPEN", "CLOSE", "FLIP"], default="hold")
    book.insert(0, "as_of", last)
    book = (book.reset_index().rename(columns={"index": "symbol"})
            .sort_values(["side", "signal"], ascending=[True, False]))
    book.to_csv(cfg.PROCESSED_DIR / "latest_book.csv", index=False)
    return book


def format_message(book: pd.DataFrame) -> str:
    as_of = book["as_of"].iloc[0]
    trades = book[book["action"] != "hold"]
    keep = book[book["action"] == "hold"]
    lines = [f"Daily market-neutral book as of {as_of:%Y-%m-%d %H:%M} UTC",
             f"Positions: {int((book['weight'] > 0).sum())} long / {int((book['weight'] < 0).sum())} short "
             f"(equal weights, 50% capital per side)"]
    if len(trades):
        lines.append("Trades to make:")
        lines += [f"  {r.action:5s} {r.symbol:10s} {r.side:5s} @ {r.price:.6g}" for r in trades.itertuples()]
    else:
        lines.append("No changes vs yesterday.")
    lines.append(f"Unchanged: {', '.join(keep['symbol'])}" if len(keep) else "Unchanged: none")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_message(predict_live()))
