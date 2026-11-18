"""Trading simulations on top of model predictions.

Two levels
----------
1. Vector simulations (fast, approximate). For every prediction column we add
   ``sim1_<model>`` ... ``sim4_<model>`` fields holding the net return of the
   trade opened at that row (NaN = no trade):
       sim1  long every signal above the threshold
       sim2  long only the top-k signals per hour
       sim3  long-short: top-k longs + bottom-k shorts per hour
       sim4  sim2 only in "fear" market regimes (Fear&Greed < 50)
   Capital is approximated with 24 equal tranches (one per hour of the holding
   window), each reinvested when it is released.

2. Exact simulation (``exact_sim``): an event loop over hours (iterating rows)
   with a single capital pool, reinvestment of gains, max N concurrent
   positions, position sizing = equity / N (optionally larger for stronger
   signals), stop-loss / take-profit checked on hourly high/low, fees and
   slippage on both sides.

Thresholds are quantiles of the VALID-period predictions ("only trade the top
q % most confident signals"); the best configuration is selected on VALID and
reported on TEST.

Costs: 0.10 % fee + 0.05 % slippage per side.
"""
from __future__ import annotations

import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

FEE, SLIP = 0.001, 0.0005
COST = FEE + SLIP          # per side
HOURS_PER_YEAR = 8760
HOLD = 24
REPORTS = cfg.ROOT / "reports"


# --------------------------------------------------------------------- metrics
def metrics(equity: pd.Series) -> dict:
    """Standard performance statistics from an hourly equity curve."""
    equity = equity.dropna()
    r = equity.pct_change().dropna()
    n = len(r)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = n / HOURS_PER_YEAR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else np.nan
    sharpe = r.mean() / r.std() * np.sqrt(HOURS_PER_YEAR) if r.std() > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "max_drawdown": dd,
            "calmar": cagr / abs(dd) if dd < 0 else np.nan}


def benchmarks(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df.pivot(index="open_time", columns="symbol", values="close").ffill()
    btc = close["BTCUSDT"] / close["BTCUSDT"].iloc[0]
    basket = (1 + close.pct_change().mean(axis=1).fillna(0)).cumprod()
    return {"BTC buy&hold": btc, "Equal-weight basket": basket}


# ------------------------------------------------------------ vector simulation
def add_vector_sims(df: pd.DataFrame, col: str, thr_hi: float, thr_lo: float, k: int = 3) -> pd.DataFrame:
    """Add sim1_..sim4_ net-return columns for prediction column `col`."""
    m = col.replace("pred_", "")
    long_net = df["ret_fwd_24h"] - 2 * COST
    short_net = -df["ret_fwd_24h"] - 2 * COST
    rank_hi = df.groupby("open_time")[col].rank(ascending=False, method="first")
    rank_lo = df.groupby("open_time")[col].rank(ascending=True, method="first")

    is_long = df[col] >= thr_hi
    top_long = is_long & (rank_hi <= k)
    top_short = (df[col] <= thr_lo) & (rank_lo <= k)
    fear = df["fear_greed"] < 50

    df[f"sim1_{m}"] = long_net.where(is_long)
    df[f"sim2_{m}"] = long_net.where(top_long)
    df[f"sim3_{m}"] = long_net.where(top_long).fillna(short_net.where(top_short))
    df[f"sim4_{m}"] = long_net.where(top_long & fear)
    return df


def tranche_equity(df: pd.DataFrame, simcol: str) -> pd.Series:
    """Approximate capital curve: 24 tranches, each invested for HOLD hours."""
    idx = pd.date_range(df["open_time"].min(), df["open_time"].max() + pd.Timedelta(hours=HOLD), freq="1h")
    tr = df.groupby("open_time")[simcol].mean().reindex(idx).fillna(0.0)   # 0 => tranche in cash
    port = tr.shift(HOLD).fillna(0.0) / HOLD
    return (1 + port).cumprod()


def vector_summary(df: pd.DataFrame, col: str, q: float, k: int = 3) -> pd.DataFrame:
    thr_hi, thr_lo = df[col].quantile(q), df[col].quantile(1 - q)
    d = add_vector_sims(df.copy(), col, thr_hi, thr_lo, k)
    rows = []
    for s in [c for c in d.columns if c.startswith("sim") and c.endswith(col.replace("pred_", ""))]:
        t = d[s].dropna()
        row = {"sim": s, "trades": len(t), "win_rate": (t > 0).mean() if len(t) else np.nan,
               "avg_net_trade": t.mean() if len(t) else np.nan}
        row.update(metrics(tranche_equity(d, s)))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- exact simulation
def exact_sim(df: pd.DataFrame, col: str, n_slots: int = 3, thr_hi: float = 0.5,
              thr_lo: float | None = None, long_short: bool = False,
              sl: float | None = None, tp: float | None = None, hold: int = HOLD,
              prob_sizing: bool = False, init: float = 1.0, neutral: bool = False):
    """Event-driven simulation iterating hour by hour with one capital pool.

    At each hour: (1) exit positions (stop-loss / take-profit on hourly low/high,
    otherwise after `hold` hours at the close), (2) mark equity, (3) fill free
    slots with the strongest signals, each sized equity / n_slots (up to +50 %
    for stronger signals if `prob_sizing`). Gains are reinvested automatically
    because sizing always uses current equity.
    Returns (equity Series, trades DataFrame).
    """
    wide = lambda c: df.pivot(index="open_time", columns="symbol", values=c)
    P = wide(col)
    times, syms = P.index, P.columns.to_numpy()
    avail = ~np.isnan(P.to_numpy())
    P = P.to_numpy(dtype="float64")
    C = wide("close").ffill().to_numpy(dtype="float64")
    H = wide("high").ffill().to_numpy(dtype="float64")
    L = wide("low").ffill().to_numpy(dtype="float64")
    T = len(times)
    sd = np.nanstd(P) or 1.0

    cash, pos, trades = init, [], []
    eq = np.empty(T)

    def mtm(t):
        return cash + sum(p["net"] * max(0.0, 1 + p["d"] * (C[t, p["i"]] / p["px"] - 1)) for p in pos)

    for t in range(T):
        # ---- 1. exits ----
        still = []
        for p in pos:
            i, d, age = p["i"], p["d"], t - p["t0"]
            exit_px = reason = None
            if age >= 1:
                if d == 1:
                    if sl is not None and L[t, i] <= p["px"] * (1 - sl):
                        exit_px, reason = p["px"] * (1 - sl), "stop"
                    elif tp is not None and H[t, i] >= p["px"] * (1 + tp):
                        exit_px, reason = p["px"] * (1 + tp), "take_profit"
                else:
                    if sl is not None and H[t, i] >= p["px"] * (1 + sl):
                        exit_px, reason = p["px"] * (1 + sl), "stop"
                    elif tp is not None and L[t, i] <= p["px"] * (1 - tp):
                        exit_px, reason = p["px"] * (1 - tp), "take_profit"
                if exit_px is None and age >= hold:
                    exit_px, reason = C[t, i], "time"
            if exit_px is None:
                still.append(p)
                continue
            proceeds = p["net"] * max(0.0, 1 + d * (exit_px / p["px"] - 1)) * (1 - COST)
            cash += proceeds
            trades.append({"symbol": syms[i], "entry_time": times[p["t0"]], "exit_time": times[t],
                           "side": "long" if d == 1 else "short", "size": p["alloc"],
                           "ret": proceeds / p["alloc"] - 1, "reason": reason})
        pos = still

        # ---- 2/3. entries into free slots ----
        free = n_slots - len(pos)
        if free > 0 and t < T - 1:
            equity_now = mtm(t)
            held = {p["i"] for p in pos}
            row = P[t]
            cands = [(row[i] - thr_hi, i, 1) for i in np.where(avail[t] & (row >= thr_hi))[0]]
            if long_short and thr_lo is not None:
                cands += [(thr_lo - row[i], i, -1) for i in np.where(avail[t] & (row <= thr_lo))[0]]
            cands.sort(reverse=True)
            # market-neutral book: at most n_slots/2 longs and n_slots/2 shorts
            cap = ({1: n_slots // 2 - sum(p["d"] == 1 for p in pos),
                    -1: n_slots // 2 - sum(p["d"] == -1 for p in pos)} if neutral
                   else {1: free, -1: free})
            for excess, i, d in cands:
                if free == 0:
                    break
                if i in held or np.isnan(C[t, i]) or cap[d] <= 0:
                    continue
                cap[d] -= 1
                mult = 1 + min(0.5, excess / (2 * sd)) if prob_sizing else 1.0
                alloc = min(equity_now / n_slots * mult, cash)
                if alloc <= 1e-9:
                    break
                cash -= alloc
                pos.append({"i": i, "d": d, "t0": t, "px": C[t, i], "alloc": alloc,
                            "net": alloc * (1 - COST)})
                held.add(i)
                free -= 1
        eq[t] = mtm(t)

    return pd.Series(eq, index=times), pd.DataFrame(trades)


def exact_summary(eq: pd.Series, trades: pd.DataFrame) -> dict:
    s = metrics(eq)
    s["trades"] = len(trades)
    s["win_rate"] = (trades["ret"] > 0).mean() if len(trades) else np.nan
    s["avg_trade"] = trades["ret"].mean() if len(trades) else np.nan
    return s


# ---------------------------------------------------------------------- driver
def run(models: list[str] | None = None) -> dict:
    REPORTS.mkdir(exist_ok=True)
    preds = pd.read_parquet(cfg.PROCESSED_DIR / "predictions.parquet")
    models = models or [c for c in preds.columns if c.startswith("pred_")]
    valid = preds[preds["split"] == "valid"]
    test = preds[preds["split"] == "test"]

    # ---- vector sims on TEST for every model (sim1..sim4) ----
    vec = []
    for col in models:
        v = vector_summary(test, col, q=0.9)
        v.insert(0, "model", col)
        vec.append(v)
    vec = pd.concat(vec, ignore_index=True)
    vec.to_csv(REPORTS / "sim_vector_test.csv", index=False)
    print("== vector sims (test, top-10% signals)\n",
          vec[["sim", "trades", "win_rate", "avg_net_trade", "cagr", "sharpe", "max_drawdown"]]
          .round(4).to_string(index=False))

    # ---- exact sim grid: choose on VALID, report on TEST ----
    grid = list(itertools.product(models, [1, 3, 5], [0.8, 0.9, 0.95], [False, True],
                                  [None, 0.05], [False, True]))
    rows = []
    for col, n, q, ls, sl, ps in grid:
        cfgd = {"model": col, "n_slots": n, "q": q, "long_short": ls, "sl": sl, "prob_sizing": ps}
        res = {}
        for name, part in (("valid", valid), ("test", test)):
            hi, lo = valid[col].quantile(q), valid[col].quantile(1 - q)   # thresholds fixed on VALID
            eq, tr = exact_sim(part, col, n, hi, lo, ls, sl, None, HOLD, ps)
            res[name] = exact_summary(eq, tr)
        rows.append({**cfgd, **{f"valid_{k}": v for k, v in res["valid"].items()},
                     **{f"test_{k}": v for k, v in res["test"].items()}})
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(REPORTS / "sim_grid.csv", index=False)

    ok = grid_df[grid_df["valid_trades"] >= 50]
    best = ok.sort_values("valid_sharpe", ascending=False).iloc[0].to_dict()
    print("\n== best config by VALID Sharpe\n", {k: best[k] for k in
          ["model", "n_slots", "q", "long_short", "sl", "prob_sizing"]})
    print({k: round(float(best[k]), 4) for k in best if k.startswith(("valid_", "test_"))
           and k.split("_", 1)[1] in ("cagr", "sharpe", "max_drawdown", "total_return", "trades")})

    # ---- final run on TEST with best config + plots ----
    col = best["model"]
    hi, lo = valid[col].quantile(best["q"]), valid[col].quantile(1 - best["q"])
    sl = None if pd.isna(best["sl"]) else float(best["sl"])
    eq, tr = exact_sim(test, col, int(best["n_slots"]), hi, lo, bool(best["long_short"]), sl,
                       None, HOLD, bool(best["prob_sizing"]))
    bm = benchmarks(test)
    bm_stats = {k: metrics(v) for k, v in bm.items()}
    final = {"strategy": exact_summary(eq, tr), "benchmarks": bm_stats,
             "config": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                        for k, v in best.items() if k in
                        ["model", "n_slots", "q", "long_short", "sl", "prob_sizing"]},
             "thr_hi": float(hi), "thr_lo": float(lo)}
    (REPORTS / "best_strategy.json").write_text(json.dumps(final, indent=2, default=float))
    tr.to_csv(REPORTS / "best_strategy_trades.csv", index=False)
    print("\n== TEST: strategy vs benchmarks")
    print(pd.DataFrame({"strategy": final["strategy"], **bm_stats}).T
          [["total_return", "cagr", "sharpe", "max_drawdown"]].round(4).to_string())

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    eq.plot(ax=ax[0], label=f"strategy ({col})", lw=2)
    for k, v in bm.items():
        (v / v.iloc[0]).plot(ax=ax[0], label=k, lw=1)
    ax[0].set_title("Equity curve, TEST period"); ax[0].legend()
    (eq.pct_change(24 * 30)).plot(ax=ax[1], label="strategy")
    for k, v in bm.items():
        v.pct_change(24 * 30).plot(ax=ax[1], label=k, lw=1)
    ax[1].set_title("Rolling 30-day return"); ax[1].legend()
    plt.tight_layout(); plt.savefig(REPORTS / "equity_test.png", dpi=110); plt.close()
    return final


def target_weights(S: pd.DataFrame, k: int) -> pd.DataFrame:
    """Market-neutral weights from a signal table (rows = days, columns = symbols).

    +1/(2k) for the k highest signals, -1/(2k) for the k lowest, 0 otherwise.
    """
    rk = S.rank(axis=1, ascending=False)
    n = S.notna().sum(axis=1)
    return ((rk <= k).astype(float) - rk.gt(n - k, axis=0).astype(float)) / (2 * k)


def daily_rebalance(preds: pd.DataFrame, col: str, k: int = 5, span: int = 1,
                    cost: float = COST, hour: int = 0) -> pd.DataFrame:
    """Low-turnover market-neutral book rebalanced once a day at `hour` UTC.

    Long the k best, short the k worst (50 % of capital each side). Only the CHANGE
    in weights is charged (`cost` per side), so a coin that stays in the book is free.
    `span` > 1 smooths the signal with an EWMA over days to cut turnover further.
    Shorting is assumed via perpetual futures (funding paid/received is ignored).
    """
    d = preds[preds["open_time"].dt.hour == hour]
    S = d.pivot(index="open_time", columns="symbol", values=col)
    R = d.pivot(index="open_time", columns="symbol", values="ret_fwd_24h").fillna(0.0)
    if span > 1:
        S = S.ewm(span=span, adjust=False).mean()
    W = target_weights(S, k)
    gross = (W * R).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    return pd.DataFrame({"gross": gross, "turnover": turnover, "net": gross - cost * turnover})


def _daily_stats(net: pd.Series) -> dict:
    eq = (1 + net).cumprod()
    years = len(net) / 365
    dd = (eq / eq.cummax() - 1).min()
    return {"total_return": eq.iloc[-1] - 1,
            "cagr": eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0,
            "sharpe": net.mean() / net.std() * np.sqrt(365) if net.std() > 0 else 0.0,
            "max_drawdown": dd}


def run_daily() -> pd.DataFrame:
    """Grid over model / k / smoothing / cost level, averaged over all 24 rebalance hours."""
    REPORTS.mkdir(exist_ok=True)
    preds = pd.read_parquet(cfg.PROCESSED_DIR / "predictions_rel.parquet")
    valid, test = preds[preds["split"] == "valid"], preds[preds["split"] == "test"]
    models = [c for c in preds.columns if c.startswith("pred_")]
    costs = {"taker 0.15%/side": COST, "maker-ish 0.05%/side": 0.0005}
    rows = []
    for col, k, span, (cname, c) in itertools.product(models, [5, 10], [1, 3, 7], costs.items()):
        row = {"model": col, "k": k, "span_days": span, "cost": cname}
        for sname, part in (("valid", valid), ("test", test)):
            per_hour = []
            for h in range(24):
                r = daily_rebalance(part, col, k, span, c, h)
                per_hour.append({**_daily_stats(r["net"]), "gross_daily": r["gross"].mean(),
                                 "turnover": r["turnover"].mean()})
            row.update({f"{sname}_{a}": b for a, b in pd.DataFrame(per_hour).mean().items()})
        rows.append(row)
    g = pd.DataFrame(rows)
    g.to_csv(REPORTS / "sim_daily_rebalance.csv", index=False)
    cols = ["model", "k", "span_days", "cost", "valid_sharpe", "test_gross_daily", "test_turnover",
            "test_cagr", "test_sharpe", "test_max_drawdown"]
    print("== daily rebalance, averaged over 24 rebalance hours. Top 8 by VALID Sharpe, with their TEST result")
    print(g.sort_values("valid_sharpe", ascending=False)[cols].head(8).round(4).to_string(index=False))
    print("\nvalid/test Sharpe correlation:", round(g[["valid_sharpe", "test_sharpe"]].corr().iloc[0, 1], 3),
          "| share of configs with TEST Sharpe>0:", round((g["test_sharpe"] > 0).mean(), 3),
          "| share with VALID Sharpe>0:", round((g["valid_sharpe"] > 0).mean(), 3))
    print(g.groupby(["span_days"])[["valid_sharpe", "test_sharpe"]].mean().round(3).to_string())
    bm = benchmarks(test)
    print({k_: {m_: round(float(v_), 3) for m_, v_ in metrics(v).items() if m_ in ("cagr", "sharpe")} for k_, v in bm.items()})

    # ---- best config (chosen on VALID) -> saved for predict.py + equity figure ----
    best = g.sort_values("valid_sharpe", ascending=False).iloc[0]
    col, k, span = best["model"], int(best["k"]), int(best["span_days"])
    (REPORTS / "best_strategy_daily.json").write_text(json.dumps({"model": col, "k": k, "span_days": span}))
    print("\nselected on VALID:", {"model": col, "k": k, "span_days": span})

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for cname, c in costs.items():
        nets = [daily_rebalance(test, col, k, span, c, h)["net"] for h in range(24)]
        avg = pd.concat([n.set_axis(n.index.normalize()) for n in nets], axis=1).mean(axis=1)
        eq = (1 + avg).cumprod()
        eq.plot(ax=ax[0], label=f"neutral daily book, {cname}", lw=2)
        eq.pct_change(30).plot(ax=ax[1], label=f"book, {cname}", lw=1.5)
    for name, v in bm.items():
        d = v.resample("D").last()
        (d / d.iloc[0]).plot(ax=ax[0], label=name, lw=1)
        d.pct_change(30).plot(ax=ax[1], label=name, lw=1)
    ax[0].set_title(f"TEST equity: {col}, k={k}, {span}-day smoothing (avg of 24 rebalance hours)")
    ax[1].set_title("Rolling 30-day return"); ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.savefig(REPORTS / "equity_daily_book.png", dpi=110); plt.close()
    return g


def run_rel() -> dict:
    """Market-neutral long-short book driven by the market-RELATIVE predictions.

    Small grid on purpose (the first grid of 360 configs over-fitted VALID).
    Every configuration is reported on TEST, so the table shows how robust the
    result is, not just the winner picked on VALID.
    """
    REPORTS.mkdir(exist_ok=True)
    preds = pd.read_parquet(cfg.PROCESSED_DIR / "predictions_rel.parquet")
    models = [c for c in preds.columns if c.startswith("pred_")]
    valid, test = preds[preds["split"] == "valid"], preds[preds["split"] == "test"]

    vec = []
    for col in models:
        v = vector_summary(test, col, q=0.9)
        v.insert(0, "model", col)
        vec.append(v)
    vec = pd.concat(vec, ignore_index=True)
    vec.to_csv(REPORTS / "sim_vector_rel_test.csv", index=False)
    print("== vector sims, relative models (test)\n",
          vec[vec["sim"].str.startswith("sim3")][["sim", "trades", "win_rate", "avg_net_trade", "cagr", "sharpe", "max_drawdown"]]
          .round(4).to_string(index=False))

    rows = []
    for col, n, q, sl in itertools.product(models, [4, 10], [0.9, 0.95], [None, 0.05]):
        res = {}
        for name, part in (("valid", valid), ("test", test)):
            hi, lo = valid[col].quantile(q), valid[col].quantile(1 - q)
            eq, tr = exact_sim(part, col, n, hi, lo, True, sl, None, HOLD, False, neutral=True)
            res[name] = exact_summary(eq, tr)
        rows.append({"model": col, "n_slots": n, "q": q, "sl": sl,
                     **{f"valid_{k}": v for k, v in res["valid"].items()},
                     **{f"test_{k}": v for k, v in res["test"].items()}})
    g = pd.DataFrame(rows)
    g.to_csv(REPORTS / "sim_rel_grid.csv", index=False)
    ok = g[g["valid_trades"] >= 50]
    print("\n== robustness (test): share of configs with Sharpe>0:",
          round((ok["test_sharpe"] > 0).mean(), 3), "| valid/test Sharpe corr:",
          round(ok[["valid_sharpe", "test_sharpe"]].corr().iloc[0, 1], 3))
    print(ok.groupby("model")[["valid_sharpe", "test_sharpe", "test_cagr"]].median().round(3).to_string())

    best = ok.sort_values("valid_sharpe", ascending=False).iloc[0]
    col = best["model"]
    sl = None if pd.isna(best["sl"]) else float(best["sl"])
    hi, lo = valid[col].quantile(best["q"]), valid[col].quantile(1 - best["q"])
    eq, tr = exact_sim(test, col, int(best["n_slots"]), hi, lo, True, sl, None, HOLD, False, neutral=True)
    bm = benchmarks(test)
    final = {"strategy": exact_summary(eq, tr), "benchmarks": {k: metrics(v) for k, v in bm.items()},
             "config": {"model": col, "n_slots": int(best["n_slots"]), "q": float(best["q"]),
                        "long_short": True, "sl": sl, "prob_sizing": False, "neutral": True},
             "thr_hi": float(hi), "thr_lo": float(lo)}
    (REPORTS / "best_strategy_rel.json").write_text(json.dumps(final, indent=2, default=float))
    tr.to_csv(REPORTS / "best_strategy_rel_trades.csv", index=False)
    print("\n== best by VALID Sharpe:", final["config"])
    print(pd.DataFrame({"strategy": final["strategy"], **final["benchmarks"]}).T
          [["total_return", "cagr", "sharpe", "max_drawdown"]].round(4).to_string())

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    eq.plot(ax=ax[0], label=f"neutral book ({col})", lw=2)
    for k, v in bm.items():
        (v / v.iloc[0]).plot(ax=ax[0], label=k, lw=1)
    ax[0].set_title("Market-neutral long-short, TEST period"); ax[0].legend()
    eq.pct_change(24 * 30).plot(ax=ax[1], label="strategy")
    for k, v in bm.items():
        v.pct_change(24 * 30).plot(ax=ax[1], label=k, lw=1)
    ax[1].set_title("Rolling 30-day return"); ax[1].legend()
    plt.tight_layout(); plt.savefig(REPORTS / "equity_test_rel.png", dpi=110); plt.close()
    return final


def latest_signals(preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trades the best strategy would open at the most recent hour ('live' rows)."""
    preds = preds if preds is not None else pd.read_parquet(cfg.PROCESSED_DIR / "predictions.parquet")
    best = json.loads((REPORTS / "best_strategy.json").read_text())
    c = best["config"]
    live = preds[preds["split"] == "live"]
    last = live[live["open_time"] == live["open_time"].max()].copy()
    col = c["model"]
    longs = last[last[col] >= best["thr_hi"]].nlargest(int(c["n_slots"]), col).assign(side="long")
    shorts = (last[last[col] <= best["thr_lo"]].nsmallest(int(c["n_slots"]), col).assign(side="short")
              if c["long_short"] else last.iloc[0:0])
    out = pd.concat([longs, shorts])[["open_time", "symbol", "side", col, "close"]].head(int(c["n_slots"]))
    out["stop_price"] = np.where(out["side"] == "long", out["close"] * (1 - (c["sl"] or 0)),
                                 out["close"] * (1 + (c["sl"] or 0))) if c["sl"] else np.nan
    out.to_csv(cfg.PROCESSED_DIR / "latest_signals.csv", index=False)
    return out


if __name__ == "__main__":
    run()
    print("\nLatest signals:\n", latest_signals().to_string(index=False))
