"""Central configuration for the crypto hourly pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KLINES_DIR = RAW_DIR / "klines"
FUNDING_DIR = RAW_DIR / "funding"
SENTIMENT_DIR = RAW_DIR / "sentiment"
PROCESSED_DIR = DATA_DIR / "processed"

# Public bulk archive: the only host that is reachable without a geo-block.
VISION_BASE = "https://data.binance.vision/data"

# Earliest month to load (about 3 years of hourly history).
START_MONTH = "2023-11"

INTERVAL = "1h"

# 50 liquid USDT pairs, all listed before START_MONTH. Pairs whose history is
# shorter or missing are skipped gracefully by the downloader.
BASES = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE", "TRX", "LINK", "AVAX",
    "DOT", "LTC", "BCH", "ATOM", "UNI", "ETC", "XLM", "NEAR", "FIL", "APT",
    "ARB", "OP", "INJ", "HBAR", "VET", "ALGO", "AAVE", "SAND", "MANA", "AXS",
    "GRT", "THETA", "EGLD", "SHIB", "ICP", "SUI", "RUNE", "MKR", "SNX", "CRV",
    "LDO", "IMX", "FET", "GALA", "CHZ", "ENJ", "KAVA", "XTZ", "EOS", "FLOW",
]
SYMBOLS = [f"{b}USDT" for b in BASES]

# Futures perpetuals use a different ticker for some coins.
FUTURES_SYMBOL_OVERRIDES = {"SHIBUSDT": "1000SHIBUSDT"}

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"
