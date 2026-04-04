"""
GARCH baseline runner — fits all 4 volatility models on all 20 tickers.

Pipeline per ticker:
  1. Load feature CSV
  2. Split into train (<=2022) / val (2023) / test (2024)
  3. Fit GARCH(1,1)-t, EGARCH(1,1)-t, GJR-GARCH(1,1)-t, HAR-RV
  4. Rolling out-of-sample forecast on test set (refit every 22 days)
  5. Evaluate against target_rv_1d
  6. Save conditional variance + residuals back to feature CSV
  7. Save model objects to models/garch/

Results are saved to ``results/tables/garch_baseline_results.csv``.

Usage
-----
::

    python -m src.models.garch_baseline_runner
    python -m src.models.garch_baseline_runner --market nse
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.garch_model import (
    GARCHFamily,
    HARRV,
    evaluate_forecasts,
)
from src.utils.config import (
    FEATURES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    get_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

_TRAIN_END = "2022-12-31"
_VAL_END = "2023-12-31"
_REFIT_EVERY = 22

GARCH_MODELS_DIR = MODELS_DIR / "garch"


# ========================================================================
# Helpers
# ========================================================================

def _discover_feature_files() -> Dict[str, Path]:
    """Return ``{ticker_stem: path}`` for all feature CSVs."""
    inventory = {}
    for subdir in ["nse", "nasdaq"]:
        d = FEATURES_DIR / subdir
        if not d.exists():
            continue
        for p in sorted(d.glob("*_features.csv")):
            ticker = p.stem.replace("_features", "")
            inventory[ticker] = p
    return inventory


def _split_data(df: pd.DataFrame):
    """Split feature DataFrame by date."""
    train = df.loc[:_TRAIN_END].copy()
    val = df.loc[_TRAIN_END:_VAL_END].copy()
    if len(val) and val.index[0] <= pd.Timestamp(_TRAIN_END):
        val = val.iloc[1:]
    test = df.loc[_VAL_END:].copy()
    if len(test) and test.index[0] <= pd.Timestamp(_VAL_END):
        test = test.iloc[1:]
    return train, val, test


# ========================================================================
# Single-ticker pipeline
# ========================================================================

def run_ticker(
    ticker: str,
    feat_path: Path,
) -> List[Dict[str, object]]:
    """
    Fit all 4 models on one ticker and return evaluation rows.

    Also saves:
      - Updated feature CSV with garch_conditional_vol and garch_std_resid
      - Model pickle files to models/garch/
    """
    df = pd.read_csv(feat_path, index_col="Datetime", parse_dates=True)
    train, val, test = _split_data(df)

    if len(test) < 20:
        logger.warning("  %s: test set too small (%d rows), skipping", ticker, len(test))
        return []

    train_returns = train["log_return"]
    test_returns = test["log_return"]
    actual_rv = test["target_rv_1d"]

    # We use train+val combined for final fitting (val used for model selection
    # in a real pipeline, but here we train on train and forecast on test)
    trainval_returns = pd.concat([train_returns, val["log_return"]])
    trainval_df = pd.concat([train, val])

    results: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # GARCH-family models
    # ------------------------------------------------------------------
    garch_specs = [
        ("GARCH",     {"model_type": "GARCH",     "p": 1, "q": 1, "dist": "t"}),
        ("EGARCH",    {"model_type": "EGARCH",     "p": 1, "q": 1, "dist": "t"}),
        ("GJR-GARCH", {"model_type": "GJR-GARCH", "p": 1, "q": 1, "dist": "t"}),
    ]

    best_garch_model = None
    best_garch_qlike = np.inf

    for label, kwargs in garch_specs:
        model = GARCHFamily(**kwargs)
        t0 = time.time()

        # Fit on train+val
        fit_stats = model.fit(trainval_returns)

        # Rolling forecast on test
        forecasts = model.rolling_forecast(trainval_returns, test_returns,
                                           refit_every=_REFIT_EVERY)

        elapsed = time.time() - t0
        metrics = evaluate_forecasts(actual_rv.values, forecasts.values)

        row = {
            "Ticker": ticker,
            "Model": model.name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "QLIKE": metrics["QLIKE"],
            "R2": metrics["R2"],
            "AIC": fit_stats.get("AIC", np.nan),
            "BIC": fit_stats.get("BIC", np.nan),
            "Time_s": round(elapsed, 1),
        }
        results.append(row)

        logger.info(
            "  %-14s  RMSE=%.4f  MAE=%.4f  QLIKE=%.4f  R2=%.4f  (%.1fs)",
            model.name, metrics["RMSE"], metrics["MAE"],
            metrics["QLIKE"], metrics["R2"], elapsed,
        )

        # Track best for saving conditional variance
        if not np.isnan(metrics["QLIKE"]) and metrics["QLIKE"] < best_garch_qlike:
            best_garch_qlike = metrics["QLIKE"]
            best_garch_model = model

        model.save(GARCH_MODELS_DIR / f"{ticker}_{label}.pkl")

    # ------------------------------------------------------------------
    # HAR-RV model
    # ------------------------------------------------------------------
    har = HARRV()
    t0 = time.time()

    har.fit(trainval_df)
    har_forecasts = har.rolling_forecast(trainval_df, test, refit_every=_REFIT_EVERY)

    elapsed = time.time() - t0
    har_metrics = evaluate_forecasts(actual_rv.values, har_forecasts.values)

    row = {
        "Ticker": ticker,
        "Model": har.name,
        "RMSE": har_metrics["RMSE"],
        "MAE": har_metrics["MAE"],
        "QLIKE": har_metrics["QLIKE"],
        "R2": har_metrics["R2"],
        "AIC": np.nan,
        "BIC": np.nan,
        "Time_s": round(elapsed, 1),
    }
    results.append(row)

    logger.info(
        "  %-14s  RMSE=%.4f  MAE=%.4f  QLIKE=%.4f  R2=%.4f  (%.1fs)",
        har.name, har_metrics["RMSE"], har_metrics["MAE"],
        har_metrics["QLIKE"], har_metrics["R2"], elapsed,
    )

    har.save(GARCH_MODELS_DIR / f"{ticker}_HAR-RV.pkl")

    # ------------------------------------------------------------------
    # Update feature CSV with conditional variance from best GARCH
    # ------------------------------------------------------------------
    if best_garch_model is not None and best_garch_model.result_ is not None:
        # Re-fit best model on full data for conditional variance column
        full_model = GARCHFamily(
            model_type=best_garch_model.model_type,
            p=best_garch_model.p, q=best_garch_model.q,
            dist=best_garch_model.dist,
        )
        full_model.fit(df["log_return"])

        if full_model.result_ is not None:
            cond_vol = full_model.get_conditional_variance()
            std_resid = full_model.get_standardized_residuals()

            df["garch_conditional_vol"] = cond_vol
            df["garch_std_resid"] = std_resid
            df.to_csv(feat_path)
            logger.info("  Updated %s with GARCH columns", feat_path.name)

    return results


# ========================================================================
# Main orchestrator
# ========================================================================

def run_baseline(market: Optional[str] = None) -> pd.DataFrame:
    """
    Run the full GARCH baseline on all tickers.

    Returns the results DataFrame.
    """
    cfg = get_config().data
    inventory = _discover_feature_files()

    if market == "nse":
        inventory = {k: v for k, v in inventory.items() if "_NS" in k}
    elif market == "nasdaq":
        inventory = {k: v for k, v in inventory.items() if "_NS" not in k}

    logger.info(
        "\n%s\n  GARCH BASELINE -- %d tickers\n%s",
        "=" * 65, len(inventory), "=" * 65,
    )

    GARCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    for i, (ticker, path) in enumerate(sorted(inventory.items()), 1):
        logger.info("\n[%d/%d] %s", i, len(inventory), ticker)
        rows = run_ticker(ticker, path)
        all_results.extend(rows)

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        logger.error("No results produced.")
        return results_df

    # Mark best model per ticker (lowest QLIKE)
    best_mask = results_df.groupby("Ticker")["QLIKE"].transform("min") == results_df["QLIKE"]
    results_df["Best"] = best_mask.map({True: "*", False: ""})

    # Save
    out_path = TABLES_DIR / "garch_baseline_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("\nSaved results -> %s", out_path)

    # ------------------------------------------------------------------
    # Print formatted summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 100)
    logger.info("  GARCH BASELINE RESULTS")
    logger.info("=" * 100)
    logger.info(
        "%-16s %-18s %8s %8s %8s %8s %10s %10s %s",
        "Ticker", "Model", "RMSE", "MAE", "QLIKE", "R2", "AIC", "BIC", "",
    )
    logger.info("-" * 100)
    for _, r in results_df.iterrows():
        logger.info(
            "%-16s %-18s %8.4f %8.4f %8.4f %8.4f %10.1f %10.1f %s",
            r["Ticker"], r["Model"], r["RMSE"], r["MAE"],
            r["QLIKE"], r["R2"],
            r["AIC"] if not np.isnan(r["AIC"]) else 0,
            r["BIC"] if not np.isnan(r["BIC"]) else 0,
            r["Best"],
        )

    # ------------------------------------------------------------------
    # Model win counts
    # ------------------------------------------------------------------
    best_rows = results_df[results_df["Best"] == "*"]
    win_counts = best_rows["Model"].value_counts()
    logger.info("\n" + "=" * 60)
    logger.info("  MODEL WIN COUNTS (by lowest QLIKE)")
    logger.info("=" * 60)
    for model_name, count in win_counts.items():
        logger.info("  %-20s  %d / %d tickers", model_name, count, len(inventory))

    # ------------------------------------------------------------------
    # Average metrics across all tickers
    # ------------------------------------------------------------------
    avg = results_df.groupby("Model")[["RMSE", "MAE", "QLIKE", "R2"]].mean()
    logger.info("\n" + "=" * 60)
    logger.info("  AVERAGE METRICS ACROSS ALL TICKERS")
    logger.info("=" * 60)
    logger.info(avg.round(4).to_string())
    logger.info("=" * 60 + "\n")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GARCH baseline models.")
    parser.add_argument("--market", choices=["nse", "nasdaq"], default=None)
    args = parser.parse_args()
    run_baseline(market=args.market)


if __name__ == "__main__":
    main()
