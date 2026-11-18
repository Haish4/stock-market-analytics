"""Train models on the processed dataset and write predictions.

Two targets (``--target``):
    abs   P(24h forward return > 0)                         -> predictions.parquet
    rel   P(24h return beats the average coin that hour)    -> predictions_rel.parquet
          (market-relative, removes the market trend; used by the market-neutral book)

Models:
    lr      LogisticRegression (scaled)            - linear baseline
    dt      DecisionTreeClassifier                 - lecture model   (abs only)
    rf      RandomForestClassifier                 - lecture model
    xgb     XGBClassifier                          - new model
    xgb_reg XGBRegressor on the (relative) return  - regression

Hyperparameters are tuned with a random search that trains on TRAIN and scores
ROC-AUC on the chronological VALID split (no shuffling, no leakage). The best
setting is then evaluated once on TEST.
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from . import config as cfg
from .transform import get_feature_sets

MODELS_DIR = cfg.ROOT / "models"

TARGETS = {
    "abs": {"cls": "y_up", "reg": "ret_fwd_24h", "suffix": "", "models": ["lr", "dt", "rf", "xgb"],
            "out": "predictions.parquet"},
    "rel": {"cls": "y_up_rel", "reg": "ret_fwd_rel_24h", "suffix": "_rel", "models": ["lr", "rf", "xgb"],
            "out": "predictions_rel.parquet"},
}

# Search spaces (sampled randomly)
SPACES = {
    "dt": {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_leaf": [200, 500, 1000, 3000]},
    "rf": {"max_depth": [6, 8, 10, 14], "min_samples_leaf": [100, 300, 1000],
           "max_features": ["sqrt", 0.3, 0.5]},
    "xgb": {"max_depth": [3, 4, 5, 6], "learning_rate": [0.02, 0.05, 0.1],
            "n_estimators": [150, 300, 500], "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0], "min_child_weight": [50, 200, 500]},
    "lr": {"C": [0.001, 0.01, 0.1, 1.0]},
}


def _make(name: str, p: dict):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, **p))
    if name == "dt":
        return DecisionTreeClassifier(random_state=0, **p)
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0, **p)
    if name == "xgb":
        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=0,
                             eval_metric="auc", **p)
    if name == "xgb_reg":
        return XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0, **p)
    raise ValueError(name)


def _score(model, X, y) -> float:
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


def tune(name, Xtr, ytr, Xva, yva, n_iter: int, rng) -> tuple[dict, float]:
    """Random search; returns (best_params, best_valid_auc)."""
    space = SPACES[name]
    best_p, best_s = None, -1.0
    for i in range(n_iter):
        p = {k: v[rng.integers(len(v))] for k, v in space.items()}
        t0 = time.time()
        m = _make(name, p).fit(Xtr, ytr)
        s = _score(m, Xva, yva)
        print(f"  [{name}] try {i + 1}/{n_iter} auc={s:.4f} ({time.time() - t0:.0f}s) {p}")
        if s > best_s:
            best_p, best_s = p, s
    return best_p, best_s


def run(n_iter: int = 6, rf_sample_every: int = 4, seed: int = 0, target: str = "abs") -> pd.DataFrame:
    tc = TARGETS[target]
    TARGET, REG_TARGET, sfx = tc["cls"], tc["reg"], tc["suffix"]
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(cfg.PROCESSED_DIR / "dataset.parquet")
    feats = get_feature_sets(df)["FEATURES"]
    rng = np.random.default_rng(seed)

    tr = df[df["split"] == "train"]
    va = df[df["split"] == "valid"]
    te = df[df["split"] == "test"]
    # Consecutive hours are almost identical samples (overlapping 24h labels),
    # so the slow models train on every k-th hour only.
    tr_small = tr[tr["open_time"].dt.hour % rf_sample_every == 0]

    out = df[df["split"].isin(["valid", "test", "live"])][
        ["symbol", "open_time", "split", "close", "high", "low", "ret_fwd_24h", "ret_fwd_rel_24h",
         "y_up", "y_up_rel", "y_up_1pct", "vol_168h", "mkt_ret_24h", "fear_greed", "btc_ret_24h"]].copy()
    summary = []

    for name in tc["models"]:
        sub = tr_small if name in ("rf", "dt") else tr
        iters = 3 if name == "lr" else n_iter
        print(f"== {name}{sfx}: train rows={len(sub):,}")
        best_p, va_auc = tune(name, sub[feats].values, sub[TARGET].values,
                              va[feats].values, va[TARGET].values, iters, rng)
        model = _make(name, best_p).fit(sub[feats].values, sub[TARGET].values)
        te_auc = _score(model, te[feats].values, te[TARGET].values)
        summary.append({"model": name + sfx, "valid_auc": va_auc, "test_auc": te_auc, "params": best_p})
        print(f"   -> best valid AUC {va_auc:.4f} | TEST AUC {te_auc:.4f}")
        joblib.dump(model, MODELS_DIR / f"{name}{sfx}.joblib")
        out[f"pred_{name}{sfx}"] = model.predict_proba(df.loc[out.index, feats].values)[:, 1]

    # ---- regression on the return itself ----
    print(f"== xgb_reg{sfx}")
    reg = _make("xgb_reg", {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
                            "subsample": 0.8, "colsample_bytree": 0.7,
                            "min_child_weight": 200}).fit(tr[feats].values, tr[REG_TARGET].values)
    joblib.dump(reg, MODELS_DIR / f"xgb_reg{sfx}.joblib")
    out[f"pred_xgb_reg{sfx}"] = reg.predict(df.loc[out.index, feats].values)
    te_mask = out["split"] == "test"
    ic = pd.Series(out.loc[te_mask, f"pred_xgb_reg{sfx}"]).corr(out.loc[te_mask, REG_TARGET], method="spearman")
    print(f"   -> TEST rank-IC (spearman) {ic:.4f}")

    pd.DataFrame(summary).to_csv(cfg.PROCESSED_DIR / f"model_summary{sfx}.csv", index=False)
    out.to_parquet(cfg.PROCESSED_DIR / tc["out"], index=False)
    print(pd.DataFrame(summary)[["model", "valid_auc", "test_auc"]].to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=6)
    ap.add_argument("--target", choices=list(TARGETS), default="abs")
    a = ap.parse_args()
    run(n_iter=a.n_iter, target=a.target)
