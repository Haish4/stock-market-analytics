"""End-to-end workflow entry point (also what the cron job calls).

Regimes
-------
--mode download   fetch the newest data from data.binance.vision (incremental),
                  rebuild the dataset, predict the latest hour.
--mode file       skip downloading and use the files already on disk / a copy
                  restored from a drive (see README); good for quick replication.

Steps (select with --steps, default = daily update):
    download -> transform -> eda -> train -> simulate -> predict

Examples
--------
    python run_pipeline.py                         # daily update: download, transform, predict
    python run_pipeline.py --mode file             # same without the download
    python run_pipeline.py --steps transform train simulate predict   # full research re-run
"""
from __future__ import annotations

import argparse

from src import download, eda, notify, predict, simulate, train, transform

DAILY = ["download", "transform", "predict"]
ALL = ["download", "transform", "eda", "train", "simulate", "train_rel", "simulate_rel",
       "simulate_daily", "predict"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["download", "file"], default="download")
    ap.add_argument("--steps", nargs="*", choices=ALL, default=None)
    a = ap.parse_args()
    steps = a.steps or DAILY
    if a.mode == "file":
        steps = [s for s in steps if s != "download"]

    for s in steps:
        print(f"\n===== {s} =====")
        if s == "download":
            download.run()
        elif s == "transform":
            transform.run()
        elif s == "eda":
            eda.run()
        elif s == "train":
            train.run()
        elif s == "simulate":
            simulate.run()
        elif s == "train_rel":
            train.run(target="rel")
        elif s == "simulate_rel":
            simulate.run_rel()
        elif s == "simulate_daily":
            simulate.run_daily()
        elif s == "predict":
            msg = predict.format_message(predict.predict_live())
            print(msg)
            notify.send(msg)


if __name__ == "__main__":
    main()
