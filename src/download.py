"""Download raw data from the public Binance bulk archive (data.binance.vision).

Sources
-------
* spot klines (1h OHLCV)               -> data/raw/klines/<SYMBOL>.parquet
* USD-M futures funding rate           -> data/raw/funding/<SYMBOL>.parquet
* Fear & Greed index (alternative.me)  -> data/raw/sentiment/fear_greed.parquet

The download is incremental: existing parquet files are read, only the missing
months/days are fetched, and the result is merged and de-duplicated. Re-running
the script therefore only pulls new data (this is what the cron job uses).

Usage
-----
    python -m src.download                       # all symbols, from START_MONTH
    python -m src.download --symbols BTCUSDT ETHUSDT --start 2024-06
"""
from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from . import config as cfg

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

_session = requests.Session()


def _fetch_zip_csv(url: str, **read_kwargs) -> pd.DataFrame | None:
    """Download a zipped CSV. Returns None if the file does not exist (404)."""
    for attempt in range(3):
        try:
            r = _session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    return pd.read_csv(f, **read_kwargs)
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == 2:
                raise
    return None


def _month_range(start: str, end: date) -> list[str]:
    """List of 'YYYY-MM' strings from start up to and including end's month."""
    y, m = map(int, start.split("-"))
    out = []
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _to_datetime(s: pd.Series) -> pd.Series:
    """Binance switched spot timestamps from ms to us in 2025; handle both."""
    s = pd.to_numeric(s)
    unit = "us" if s.iloc[0] > 1e14 else "ms"
    return pd.to_datetime(s, unit=unit, utc=True)


def _parse_klines(df: pd.DataFrame) -> pd.DataFrame:
    # Newer files carry a header row; older ones do not.
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df = df.iloc[:, :11].copy()
    df.columns = KLINE_COLS[:11]
    df["open_time"] = _to_datetime(df["open_time"])
    df["close_time"] = _to_datetime(df["close_time"])
    num = [c for c in df.columns if c not in ("open_time", "close_time")]
    df[num] = df[num].astype("float64")
    df["n_trades"] = df["n_trades"].astype("int64")
    return df


# --------------------------------------------------------------------- klines
def download_klines(symbol: str, start: str = cfg.START_MONTH) -> int:
    """Incrementally download hourly spot klines for one symbol. Returns #new rows."""
    cfg.KLINES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.KLINES_DIR / f"{symbol}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    last_ts = old["open_time"].max() if old is not None else None

    today = date.today()
    frames = []
    for month in _month_range(start, today):
        y, m = map(int, month.split("-"))
        is_current = (y, m) == (today.year, today.month)
        if not is_current:
            month_end = pd.Timestamp(year=y, month=m, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
            if last_ts is not None and last_ts >= month_end.normalize() + pd.Timedelta(hours=23):
                continue  # month already complete locally
            url = f"{cfg.VISION_BASE}/spot/monthly/klines/{symbol}/{cfg.INTERVAL}/{symbol}-{cfg.INTERVAL}-{month}.zip"
            df = _fetch_zip_csv(url, header=None)
            if df is not None:
                frames.append(_parse_klines(df))
        else:
            # current month: daily files, up to yesterday (today is not published yet)
            first = date(y, m, 1)
            if last_ts is not None:
                first = max(first, last_ts.date())
            d = first
            while d < today:
                url = (f"{cfg.VISION_BASE}/spot/daily/klines/{symbol}/{cfg.INTERVAL}/"
                       f"{symbol}-{cfg.INTERVAL}-{d.isoformat()}.zip")
                df = _fetch_zip_csv(url, header=None)
                if df is not None:
                    frames.append(_parse_klines(df))
                d += timedelta(days=1)

    if not frames:
        return 0
    new = pd.concat(frames, ignore_index=True)
    merged = pd.concat([old.drop(columns="symbol"), new], ignore_index=True) if old is not None else new
    merged = (merged.drop_duplicates("open_time").sort_values("open_time")
              .reset_index(drop=True))
    merged.insert(0, "symbol", symbol)
    merged.to_parquet(path, index=False)
    return len(merged) - (len(old) if old is not None else 0)


# -------------------------------------------------------------------- funding
def download_funding(symbol: str, start: str = cfg.START_MONTH) -> int:
    """Incrementally download USD-M futures funding rates (published monthly)."""
    cfg.FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FUNDING_DIR / f"{symbol}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    fsym = cfg.FUTURES_SYMBOL_OVERRIDES.get(symbol, symbol)

    today = date.today()
    frames = []
    for month in _month_range(start, today):
        y, m = map(int, month.split("-"))
        if (y, m) == (today.year, today.month):
            continue  # monthly file only exists once the month is over
        if old is not None and old["calc_time"].max() >= pd.Timestamp(
                year=y, month=m, day=1, tz="UTC") + pd.offsets.MonthEnd(1) - pd.Timedelta(days=1):
            continue
        url = f"{cfg.VISION_BASE}/futures/um/monthly/fundingRate/{fsym}/{fsym}-fundingRate-{month}.zip"
        df = _fetch_zip_csv(url)
        if df is None or df.empty:
            continue
        df = df.rename(columns={"last_funding_rate": "funding_rate"})
        df["calc_time"] = _to_datetime(df["calc_time"])
        frames.append(df[["calc_time", "funding_rate"]])

    if not frames:
        return 0
    new = pd.concat(frames, ignore_index=True)
    merged = pd.concat([old.drop(columns="symbol"), new], ignore_index=True) if old is not None else new
    merged = (merged.drop_duplicates("calc_time").sort_values("calc_time")
              .reset_index(drop=True))
    merged.insert(0, "symbol", symbol)
    merged.to_parquet(path, index=False)
    return len(merged) - (len(old) if old is not None else 0)


# ------------------------------------------------------------------ sentiment
def download_fear_greed() -> int:
    """Download the full daily Fear & Greed history (small, overwritten each run)."""
    cfg.SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    r = _session.get(cfg.FEAR_GREED_URL, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
    df["fear_greed"] = df["value"].astype(int)
    df = df[["date", "fear_greed"]].sort_values("date").reset_index(drop=True)
    df.to_parquet(cfg.SENTIMENT_DIR / "fear_greed.parquet", index=False)
    return len(df)


# ----------------------------------------------------------------------- main
def _one(symbol: str, start: str) -> tuple[str, int, int]:
    try:
        return symbol, download_klines(symbol, start), download_funding(symbol, start)
    except Exception as e:  # keep the batch going if one symbol fails
        print(f"[WARN] {symbol} failed: {e}")
        return symbol, -1, -1


def run(symbols: list[str] | None = None, start: str = cfg.START_MONTH, workers: int = 6) -> None:
    symbols = symbols or cfg.SYMBOLS
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, nk, nf in ex.map(lambda s: _one(s, start), symbols):
            print(f"{sym:10s} +{nk} klines  +{nf} funding")
    print("fear_greed rows:", download_fear_greed())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", help="override symbol list, e.g. BTCUSDT ETHUSDT")
    ap.add_argument("--start", default=cfg.START_MONTH, help="first month YYYY-MM")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.symbols, a.start, a.workers)
