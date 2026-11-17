"""Merge raw sources into ONE table, engineer features and define the target.

Output: data/processed/dataset.parquet  (one row per symbol per hour)

Timing convention
-----------------
A row at ``open_time = t`` describes the candle [t, t+1h). All features use only
information available at the *close* of that candle, and the position is
assumed to be opened at that close price. Funding rate uses the last funding
event at or before ``t`` and Fear & Greed uses the *previous* day's value, so
nothing leaks from the future.

Target
------
ret_fwd_24h = close(t+24h) / close(t) - 1       (regression target)
y_up        = ret_fwd_24h > 0                    (classification target)
y_up_1pct   = ret_fwd_24h > 1 %                  (higher-probability events)
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import config as cfg

HORIZON = 24  # hours ahead

# Split boundaries (UTC). A 24h purge gap is removed before each boundary so no
# target window overlaps the next split.
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALID_END = pd.Timestamp("2026-01-01", tz="UTC")

TO_PREDICT = ["ret_fwd_24h", "y_up", "y_up_1pct", "ret_fwd_rel_24h", "y_up_rel"]

NUMERIC = [
    # price momentum
    "ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_72h", "ret_168h",
    # volatility / range
    "vol_24h", "vol_168h", "hl_range", "hl_range_24h",
    # volume / order flow
    "vol_z_168h", "trades_z_168h", "taker_buy_ratio", "taker_buy_ratio_24h",
    # technical indicators
    "rsi_14", "macd_hist", "bb_pos", "dist_high_168h", "dist_low_168h",
    "close_sma_24", "close_sma_168",
    # derivatives and sentiment
    "funding_rate", "funding_chg_24h", "fear_greed", "fg_chg_7d",
    # market context
    "btc_ret_1h", "btc_ret_24h", "corr_btc_168h", "mkt_ret_24h", "rel_ret_24h",
    "rank_ret_24h", "mkt_vol_168h",
]

# Dummy sets are created from these categorical/binned columns.
DUMMY_SOURCES = ["session", "dow", "vol_bucket", "fg_regime", "liq_bucket"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    klines = pd.concat(
        (pd.read_parquet(f) for f in sorted(glob.glob(str(cfg.KLINES_DIR / "*.parquet")))),
        ignore_index=True)
    funding = pd.concat(
        (pd.read_parquet(f) for f in sorted(glob.glob(str(cfg.FUNDING_DIR / "*.parquet")))),
        ignore_index=True)
    fg = pd.read_parquet(cfg.SENTIMENT_DIR / "fear_greed.parquet")
    return klines, funding, fg


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _symbol_features(g: pd.DataFrame, btc_close: pd.Series) -> pd.DataFrame:
    """Per-symbol time-series features. Input has one row per hour."""
    sym = g["symbol"].iloc[0]
    g = g.set_index("open_time").sort_index().asfreq("1h")  # make gaps explicit
    g["symbol"] = sym
    c, h, l = g["close"], g["high"], g["low"]

    for n in (1, 4, 12, 24, 72, 168):
        g[f"ret_{n}h"] = c / c.shift(n) - 1
    g["vol_24h"] = g["ret_1h"].rolling(24).std()
    g["vol_168h"] = g["ret_1h"].rolling(168).std()
    g["hl_range"] = (h - l) / c
    g["hl_range_24h"] = g["hl_range"].rolling(24).mean()

    qv = np.log1p(g["quote_volume"])
    g["vol_z_168h"] = (qv - qv.rolling(168).mean()) / qv.rolling(168).std()
    nt = np.log1p(g["n_trades"])
    g["trades_z_168h"] = (nt - nt.rolling(168).mean()) / nt.rolling(168).std()
    g["taker_buy_ratio"] = g["taker_buy_quote"] / g["quote_volume"].replace(0, np.nan)
    g["taker_buy_ratio_24h"] = g["taker_buy_ratio"].rolling(24).mean()
    g["quote_volume_24h"] = g["quote_volume"].rolling(24).sum()

    g["rsi_14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    g["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    g["bb_pos"] = (c - sma20) / (2 * std20)
    g["dist_high_168h"] = c / h.rolling(168).max() - 1
    g["dist_low_168h"] = c / l.rolling(168).min() - 1
    g["close_sma_24"] = c / c.rolling(24).mean() - 1
    g["close_sma_168"] = c / c.rolling(168).mean() - 1

    btc = btc_close.reindex(g.index)
    g["btc_ret_1h"] = btc / btc.shift(1) - 1
    g["btc_ret_24h"] = btc / btc.shift(24) - 1
    g["corr_btc_168h"] = g["ret_1h"].rolling(168).corr(g["btc_ret_1h"])

    g["ret_fwd_24h"] = c.shift(-HORIZON) / c - 1
    return g.reset_index()


def build_dataset() -> pd.DataFrame:
    klines, funding, fg = load_raw()
    klines = klines.drop(columns=["close_time", "ignore"], errors="ignore")

    btc_close = (klines[klines["symbol"] == "BTCUSDT"].set_index("open_time")["close"]
                 .sort_index())
    parts = [_symbol_features(g, btc_close) for _, g in klines.groupby("symbol")]
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["close"])  # drop the artificial gap rows

    # ---- funding rate (as-of join: last known funding event) ----
    f = funding.sort_values("calc_time")
    f["symbol"] = f["symbol"].astype(str)
    df = df.sort_values("open_time")
    df = pd.merge_asof(df, f, left_on="open_time", right_on="calc_time", by="symbol",
                       direction="backward", tolerance=pd.Timedelta("12h"))
    df = df.drop(columns="calc_time").sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    df["funding_chg_24h"] = df.groupby("symbol")["funding_rate"].diff(24).fillna(0.0)

    # ---- fear & greed (previous day's value => no look-ahead) ----
    fg = fg.sort_values("date").copy()
    fg["fg_chg_7d"] = fg["fear_greed"].diff(7)
    fg["date"] = fg["date"].dt.floor("D") + pd.Timedelta(days=1)  # available next day
    df["date"] = df["open_time"].dt.floor("D")
    df = df.merge(fg, on="date", how="left").drop(columns="date")
    df[["fear_greed", "fg_chg_7d"]] = df[["fear_greed", "fg_chg_7d"]].ffill()

    # ---- cross-sectional market features ----
    grp = df.groupby("open_time")
    df["mkt_ret_24h"] = grp["ret_24h"].transform("mean")
    df["rel_ret_24h"] = df["ret_24h"] - df["mkt_ret_24h"]
    df["rank_ret_24h"] = grp["ret_24h"].rank(pct=True)
    df["mkt_vol_168h"] = grp["vol_168h"].transform("mean")

    # ---- targets ----
    df["y_up"] = (df["ret_fwd_24h"] > 0).astype("float").where(df["ret_fwd_24h"].notna())
    df["y_up_1pct"] = (df["ret_fwd_24h"] > 0.01).astype("float").where(df["ret_fwd_24h"].notna())
    # market-relative target: beat the average coin over the next 24h (removes the market trend)
    df["ret_fwd_rel_24h"] = df["ret_fwd_24h"] - df.groupby("open_time")["ret_fwd_24h"].transform("mean")
    df["y_up_rel"] = (df["ret_fwd_rel_24h"] > 0).astype("float").where(df["ret_fwd_rel_24h"].notna())

    # ---- split (with purge gap); rows without a target are the 'live' rows ----
    gap = pd.Timedelta(hours=HORIZON)
    t = df["open_time"]
    df["split"] = np.select(
        [t < TRAIN_END - gap, (t >= TRAIN_END) & (t < VALID_END - gap), t >= VALID_END],
        ["train", "valid", "test"], default="purge")
    df.loc[df["ret_fwd_24h"].isna(), "split"] = "live"

    df = df.dropna(subset=NUMERIC).reset_index(drop=True)
    df = add_derived(df)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Binned variables + dummies. Bin edges come from TRAIN rows only."""
    tr = df["split"] == "train"
    hour = df["open_time"].dt.hour
    df["session"] = pd.cut(hour, [-1, 7, 15, 23], labels=["asia", "europe", "us"]).astype(str)
    df["dow"] = df["open_time"].dt.dayofweek.astype(str)

    q = df.loc[tr, "vol_168h"].quantile([1 / 3, 2 / 3]).values
    df["vol_bucket"] = pd.cut(df["vol_168h"], [-np.inf, q[0], q[1], np.inf],
                              labels=["low", "mid", "high"]).astype(str)
    df["fg_regime"] = pd.cut(df["fear_greed"], [-1, 25, 45, 55, 75, 101],
                             labels=["extreme_fear", "fear", "neutral", "greed",
                                     "extreme_greed"]).astype(str)
    liq = df.loc[tr].groupby("symbol")["quote_volume_24h"].mean()
    lq = liq.quantile([1 / 3, 2 / 3]).values
    liq_bucket = pd.cut(liq, [-np.inf, lq[0], lq[1], np.inf], labels=["low", "mid", "high"])
    df["liq_bucket"] = df["symbol"].map(liq_bucket.astype(str))
    df["liq_bucket"] = df["liq_bucket"].fillna("mid")

    d = pd.get_dummies(df[DUMMY_SOURCES], prefix=DUMMY_SOURCES, dtype="uint8")
    df = pd.concat([df, d], axis=1)
    df.attrs["dummies"] = list(d.columns)
    return df


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    dummies = [c for c in df.columns
               if any(c.startswith(s + "_") for s in DUMMY_SOURCES)]
    return {"TO_PREDICT": TO_PREDICT, "NUMERIC": NUMERIC, "DUMMIES": dummies,
            "FEATURES": NUMERIC + dummies}


def run() -> pd.DataFrame:
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df.to_parquet(cfg.PROCESSED_DIR / "dataset.parquet", index=False)
    fs = get_feature_sets(df)
    print(f"dataset: {df.shape}  symbols={df['symbol'].nunique()}")
    print(df["split"].value_counts().to_string())
    print(f"numeric={len(fs['NUMERIC'])}  dummies={len(fs['DUMMIES'])}")
    return df


if __name__ == "__main__":
    run()
