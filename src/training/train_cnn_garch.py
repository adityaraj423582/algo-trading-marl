"""
CNN-GARCH training orchestrator.

Modes
-----
- **QUICK_TEST** (default): 2 tickers, 50 epochs, CPU -- verifies pipeline.
- **FULL_TRAIN**: 20 tickers, 200 epochs, GPU (Param Ganga).

Usage
-----
::

    python -m src.training.train_cnn_garch                  # quick test
    python -m src.training.train_cnn_garch --full           # full training
    python -m src.training.train_cnn_garch --tickers AAPL NVDA  # custom
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.cnn_garch import CNNGARCHHybrid, prepare_sequences
from src.models.cnn_model import set_all_seeds
from src.utils.config import (
    FEATURES_DIR,
    TABLES_DIR,
    RANDOM_SEED,
    get_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _discover_feature_files() -> Dict[str, Path]:
    """Return {ticker_stem: path} for all feature CSVs."""
    out = {}
    for sub in ["nse", "nasdaq"]:
        d = FEATURES_DIR / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*_features.csv")):
            out[p.stem.replace("_features", "")] = p
    return out


def _load_har_baseline() -> Dict[str, float]:
    """Load HAR-RV QLIKE per ticker from Step 3 results."""
    path = TABLES_DIR / "garch_baseline_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    har = df[df["Model"] == "HAR-RV"].set_index("Ticker")
    return har["QLIKE"].to_dict()


def run_training(
    tickers: Optional[List[str]] = None,
    max_epochs: int = 200,
) -> pd.DataFrame:
    """
    Train CNN-GARCH on the specified tickers.

    Returns a results DataFrame.
    """
    set_all_seeds(RANDOM_SEED)
    cfg = get_config().cnn_garch
    cnn_cfg = cfg.cnn

    inventory = _discover_feature_files()
    har_baselines = _load_har_baseline()

    if tickers is None:
        if cfg.quick_test:
            tickers = cfg.quick_test_tickers
            max_epochs = cfg.quick_test_epochs
            logger.info("QUICK TEST MODE: %d tickers, %d epochs", len(tickers), max_epochs)
        else:
            tickers = list(inventory.keys())

    # Filter to available tickers
    tickers = [t for t in tickers if t in inventory]

    logger.info(
        "\n%s\n  CNN-GARCH TRAINING -- %d tickers, %d max epochs\n%s",
        "=" * 65, len(tickers), max_epochs, "=" * 65,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("\n[%d/%d] %s", i, len(tickers), ticker)

        feat_path = inventory[ticker]
        df = pd.read_csv(feat_path, index_col="Datetime", parse_dates=True)

        # Prepare sliding windows
        n_feat = len([c for c in df.columns if c in [
            "Close", "High", "Low", "Open", "Volume",
            "log_return", "rv_daily", "rv_weekly", "rv_monthly",
            "rolling_var_5d", "rolling_var_22d",
            "parkinson_vol", "garman_klass_vol",
            "intraday_range", "volume_ma5", "volume_ratio",
            "momentum_5", "momentum_22", "vol_regime",
            "garch_conditional_vol", "garch_std_resid",
            "target_rv_1d", "target_rv_5d",
        ]])

        X_train, y_train, X_val, y_val, X_test, y_test, scaler = \
            prepare_sequences(df, window_size=cnn_cfg.window_size)

        if len(X_train) < 50 or len(X_test) < 10:
            logger.warning("  Skipping %s -- insufficient data", ticker)
            continue

        hybrid = CNNGARCHHybrid(
            ticker=ticker,
            n_features=X_train.shape[2],
            window_size=cnn_cfg.window_size,
        )

        train_info = hybrid.train(
            X_train, y_train, X_val, y_val,
            max_epochs=max_epochs,
        )

        metrics = hybrid.evaluate(X_test, y_test)
        hybrid.save_scaler(scaler)

        har_qlike = har_baselines.get(ticker, np.nan)
        cnn_qlike = metrics["QLIKE_1d"]
        improvement = (
            (har_qlike - cnn_qlike) / har_qlike * 100.0
            if not np.isnan(har_qlike) and har_qlike > 0 else np.nan
        )
        beat = "YES" if cnn_qlike < har_qlike else "NO"

        row = {
            "Ticker": ticker,
            "HAR_QLIKE": round(har_qlike, 4) if not np.isnan(har_qlike) else np.nan,
            "CNN_QLIKE_1d": round(cnn_qlike, 4),
            "CNN_QLIKE_5d": round(metrics["QLIKE_5d"], 4),
            "CNN_RMSE_1d": round(metrics["RMSE_1d"], 4),
            "CNN_MAE_1d": round(metrics["MAE_1d"], 4),
            "CNN_R2_1d": round(metrics["R2_1d"], 4),
            "Improvement_%": round(improvement, 1) if not np.isnan(improvement) else np.nan,
            "Beat_Baseline": beat,
            "Best_Epoch": train_info["best_epoch"],
            "Train_Time_s": train_info["train_time_s"],
        }
        results.append(row)

        logger.info(
            "  RESULT: QLIKE_1d=%.4f (HAR=%.4f, %s%.1f%%)  R2=%.4f  best_ep=%d",
            cnn_qlike, har_qlike, "+" if improvement < 0 else "-",
            abs(improvement) if not np.isnan(improvement) else 0,
            metrics["R2_1d"], train_info["best_epoch"],
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        logger.error("No results produced.")
        return results_df

    # Save
    out_path = TABLES_DIR / "cnn_garch_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("\nSaved -> %s", out_path)

    # Print summary
    logger.info("\n" + "=" * 95)
    logger.info("  CNN-GARCH RESULTS")
    logger.info("=" * 95)
    logger.info(
        "%-16s %10s %10s %10s %10s %10s %8s",
        "Ticker", "HAR_QLIKE", "CNN_QLIKE", "Improv%", "CNN_R2", "BestEp", "Beat?",
    )
    logger.info("-" * 95)
    for _, r in results_df.iterrows():
        logger.info(
            "%-16s %10.4f %10.4f %10.1f %10.4f %10d %8s",
            r["Ticker"],
            r["HAR_QLIKE"] if not np.isnan(r["HAR_QLIKE"]) else 0,
            r["CNN_QLIKE_1d"],
            r["Improvement_%"] if not np.isnan(r["Improvement_%"]) else 0,
            r["CNN_R2_1d"],
            r["Best_Epoch"],
            r["Beat_Baseline"],
        )

    n_beat = (results_df["Beat_Baseline"] == "YES").sum()
    n_total = len(results_df)
    avg_cnn = results_df["CNN_QLIKE_1d"].mean()
    avg_har = results_df["HAR_QLIKE"].mean()

    logger.info("-" * 95)
    logger.info(
        "  OVERALL: %d/%d beat HAR-RV  |  Avg CNN QLIKE=%.4f  vs  Avg HAR QLIKE=%.4f",
        n_beat, n_total, avg_cnn, avg_har,
    )
    logger.info("=" * 95 + "\n")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-GARCH hybrid.")
    parser.add_argument("--full", action="store_true",
                        help="Full training mode (all tickers, 200 epochs).")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Custom ticker list.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs.")
    args = parser.parse_args()

    tickers = args.tickers
    max_epochs = args.epochs

    if args.full:
        tickers = None
        if max_epochs is None:
            max_epochs = get_config().cnn_garch.cnn.max_epochs
    elif max_epochs is None:
        max_epochs = get_config().cnn_garch.quick_test_epochs

    if tickers is None and not args.full:
        tickers = get_config().cnn_garch.quick_test_tickers

    run_training(tickers=tickers, max_epochs=max_epochs)


if __name__ == "__main__":
    main()
