"""
Data preprocessor for the Algorithmic Trading MARL project.

Loads raw CSV files produced by the downloader, cleans them, computes
log returns, normalises timezones to UTC, and splits into
train / validation / test sets.

Split strategy
--------------
- **Daily data**: fixed calendar split
    - Train:      2020-01-01  to  2022-12-31
    - Validation: 2023-01-01  to  2023-12-31
    - Test:       2024-01-01  to  2024-12-31

- **Hourly data**: 70 / 15 / 15 chronological split (calendar dates
  vary across tickers because Yahoo provides only the last ~730 days).

Usage
-----
::

    python -m src.data.preprocessor                  # both markets
    python -m src.data.preprocessor --market nse     # NSE only
    python -m src.data.preprocessor --market nasdaq  # NASDAQ only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import (
    PROCESSED_NASDAQ_DIR,
    PROCESSED_NSE_DIR,
    RAW_NASDAQ_DIR,
    RAW_NSE_DIR,
    RANDOM_SEED,
    get_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Calendar split boundaries for daily data ────────────────────────────
_DAILY_TRAIN_END = "2022-12-31"
_DAILY_VAL_END = "2023-12-31"

# ── Chronological split fractions for hourly data ───────────────────────
_HOURLY_TRAIN_FRAC = 0.70
_HOURLY_VAL_FRAC = 0.15
# test = 1 - train - val = 0.15


# ========================================================================
# Loading
# ========================================================================

def load_raw_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a raw CSV saved by the downloader.

    Parses ``Datetime`` as the index and converts to a proper
    ``DatetimeIndex``.  Timezone-aware strings (hourly data) are kept
    as-is; timezone-naive strings (daily data) are left naive until the
    alignment step.
    """
    df = pd.read_csv(filepath, index_col="Datetime", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def discover_raw_files(raw_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan ``raw_dir`` and return a mapping ``{ticker: {interval: path}}``.

    File naming convention from the downloader:
    ``{TICKER}_{interval}.csv``  (e.g. ``RELIANCE_NS_1d.csv``).
    """
    inventory: Dict[str, Dict[str, Path]] = {}
    for p in sorted(raw_dir.glob("*.csv")):
        stem = p.stem                         # e.g. "RELIANCE_NS_1h"
        if "_1d" in stem:
            ticker = stem.replace("_1d", "")
            interval = "1d"
        elif "_1h" in stem:
            ticker = stem.replace("_1h", "")
            interval = "1h"
        else:
            logger.warning("Skipping unrecognised file: %s", p.name)
            continue
        inventory.setdefault(ticker, {})[interval] = p

    return inventory


# ========================================================================
# Cleaning
# ========================================================================

def _count_max_consecutive_nans(series: pd.Series) -> int:
    """Return the length of the longest consecutive NaN streak."""
    is_nan = series.isna()
    if not is_nan.any():
        return 0
    groups = (~is_nan).cumsum()
    return int(is_nan.groupby(groups).sum().max())


def clean_dataframe(
    df: pd.DataFrame,
    ticker: str,
    max_ffill: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Clean a single ticker DataFrame.

    Steps
    -----
    1. Drop rows where Close <= 0 or Volume <= 0.
    2. Remove duplicate timestamps.
    3. Forward-fill gaps of up to ``max_ffill`` consecutive NaNs.
    4. Drop any remaining rows that still contain NaN.
    5. Sort by timestamp ascending.
    6. Compute ``log_return``.

    Returns
    -------
    (cleaned_df, report_dict)
    """
    rows_before = len(df)
    report: Dict[str, object] = {"ticker": ticker, "rows_before": rows_before}

    # 1. Drop bad prices / volumes
    if "Close" in df.columns:
        df = df[df["Close"] > 0]
    if "Volume" in df.columns:
        df = df[df["Volume"] >= 0]

    # 2. Remove duplicate timestamps
    n_dup = df.index.duplicated().sum()
    if n_dup:
        df = df[~df.index.duplicated(keep="first")]
        report["duplicates_removed"] = int(n_dup)

    # 3. Forward-fill small gaps
    df = df.ffill(limit=max_ffill)

    # 4. Drop rows still containing NaN
    n_still_na = df.isna().any(axis=1).sum()
    df = df.dropna()
    report["nan_rows_dropped"] = int(n_still_na)

    # 5. Sort
    df.sort_index(inplace=True)

    # 6. Log returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["log_return"], inplace=True)

    report["rows_after"] = len(df)
    report["rows_removed"] = rows_before - len(df)

    return df, report


# ========================================================================
# Timezone alignment
# ========================================================================

def align_timezone(
    df: pd.DataFrame,
    market: str,
    interval: str,
) -> pd.DataFrame:
    """
    Normalise the index to UTC.

    - **Daily data** is timezone-naive.  For daily bars the exact hour is
      irrelevant, so we simply localise to UTC.
    - **Hourly NSE data** from yfinance already carries ``+05:30`` offsets;
      convert to UTC.
    - **Hourly NASDAQ data** from yfinance already carries UTC offsets;
      ensure UTC.
    """
    idx = df.index

    if idx.tz is not None:
        df.index = idx.tz_convert("UTC")
    else:
        if interval == "1d":
            df.index = idx.tz_localize("UTC")
        elif market == "nse":
            df.index = idx.tz_localize("Asia/Kolkata").tz_convert("UTC")
        else:
            df.index = idx.tz_localize("America/New_York").tz_convert("UTC")

    return df


# ========================================================================
# Train / Validation / Test split
# ========================================================================

def split_daily(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split daily data by fixed calendar boundaries."""
    train = df.loc[:_DAILY_TRAIN_END]
    val = df.loc[_DAILY_TRAIN_END:_DAILY_VAL_END]
    # Exclude the first row of val if it equals _DAILY_TRAIN_END boundary
    if len(val) and val.index[0] <= pd.Timestamp(_DAILY_TRAIN_END, tz=val.index.tz):
        val = val.iloc[1:]
    test = df.loc[_DAILY_VAL_END:]
    if len(test) and test.index[0] <= pd.Timestamp(_DAILY_VAL_END, tz=test.index.tz):
        test = test.iloc[1:]
    return {"train": train, "val": val, "test": test}


def split_hourly(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split hourly data by chronological 70 / 15 / 15 fractions."""
    n = len(df)
    train_end = int(n * _HOURLY_TRAIN_FRAC)
    val_end = int(n * (_HOURLY_TRAIN_FRAC + _HOURLY_VAL_FRAC))
    return {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }


def save_splits(
    splits: Dict[str, pd.DataFrame],
    ticker: str,
    interval: str,
    output_dir: Path,
) -> None:
    """Save each split as ``{ticker}_{interval}_{split}.csv``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        fname = f"{ticker}_{interval}_{split_name}.csv"
        fpath = output_dir / fname
        split_df.index.name = "Datetime"
        split_df.to_csv(fpath)
        logger.debug("  Saved %s  (%d rows)", fpath.name, len(split_df))


# ========================================================================
# Orchestration
# ========================================================================

def process_market(
    raw_dir: Path,
    out_dir: Path,
    market: str,
) -> List[Dict[str, object]]:
    """
    Process all tickers for a single market (NSE or NASDAQ).

    For each ticker and interval found in ``raw_dir``:
      1. Load raw CSV
      2. Clean
      3. Align timezone to UTC
      4. Split into train / val / test
      5. Save to ``out_dir``

    Returns per-ticker-interval summary reports.
    """
    inventory = discover_raw_files(raw_dir)
    reports: List[Dict[str, object]] = []

    logger.info(
        "\n%s\n  Processing %s  --  %d tickers found\n%s",
        "=" * 65, market.upper(), len(inventory), "=" * 65,
    )

    for ticker, intervals in inventory.items():
        for interval, filepath in sorted(intervals.items()):
            logger.info("[%s] %s ...", ticker, interval)

            df = load_raw_csv(filepath)
            df, report = clean_dataframe(df, ticker)

            df = align_timezone(df, market, interval)

            if interval == "1d":
                splits = split_daily(df)
            else:
                splits = split_hourly(df)

            split_sizes = {k: len(v) for k, v in splits.items()}
            report["interval"] = interval
            report["split_sizes"] = split_sizes

            save_splits(splits, ticker, interval, out_dir)
            reports.append(report)

            logger.info(
                "  clean %d -> %d  |  train=%d  val=%d  test=%d",
                report["rows_before"],
                report["rows_after"],
                split_sizes["train"],
                split_sizes["val"],
                split_sizes["test"],
            )

    return reports


def print_preprocessing_summary(
    nse_reports: List[Dict],
    nasdaq_reports: List[Dict],
) -> None:
    """Print a formatted summary of preprocessing results."""

    def _section(title: str, reports: List[Dict]) -> None:
        logger.info("\n" + "=" * 85)
        logger.info("  %s  PREPROCESSING SUMMARY", title)
        logger.info("=" * 85)
        logger.info(
            "%-18s %-5s %8s %8s %8s %7s %7s %7s",
            "Ticker", "Freq", "Before", "After", "Removed",
            "Train", "Val", "Test",
        )
        logger.info("-" * 85)
        for r in reports:
            ss = r.get("split_sizes", {})
            logger.info(
                "%-18s %-5s %8d %8d %8d %7d %7d %7d",
                r["ticker"], r.get("interval", "?"),
                r["rows_before"], r["rows_after"],
                r["rows_removed"],
                ss.get("train", 0), ss.get("val", 0), ss.get("test", 0),
            )

        total_before = sum(r["rows_before"] for r in reports)
        total_after = sum(r["rows_after"] for r in reports)
        logger.info("-" * 85)
        logger.info(
            "%-18s %-5s %8d %8d %8d",
            "TOTAL", "", total_before, total_after,
            total_before - total_after,
        )

    if nse_reports:
        _section("NSE", nse_reports)
    if nasdaq_reports:
        _section("NASDAQ", nasdaq_reports)

    all_r = nse_reports + nasdaq_reports
    total_b = sum(r["rows_before"] for r in all_r)
    total_a = sum(r["rows_after"] for r in all_r)
    logger.info("\n" + "=" * 85)
    logger.info(
        "  OVERALL: %d -> %d rows  (%.1f%% retained)",
        total_b, total_a, 100.0 * total_a / total_b if total_b else 0,
    )
    logger.info("=" * 85 + "\n")


# ========================================================================
# Entry points
# ========================================================================

def run_preprocessor(market: Optional[str] = None) -> None:
    """Run the full preprocessing pipeline."""
    nse_reports: List[Dict] = []
    nasdaq_reports: List[Dict] = []

    if market in (None, "nse"):
        nse_reports = process_market(RAW_NSE_DIR, PROCESSED_NSE_DIR, "nse")

    if market in (None, "nasdaq"):
        nasdaq_reports = process_market(RAW_NASDAQ_DIR, PROCESSED_NASDAQ_DIR, "nasdaq")

    print_preprocessing_summary(nse_reports, nasdaq_reports)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw OHLCV data: clean, align TZ, split.",
    )
    parser.add_argument(
        "--market",
        choices=["nse", "nasdaq"],
        default=None,
        help="Process only one market (default: both).",
    )
    args = parser.parse_args()
    run_preprocessor(market=args.market)


if __name__ == "__main__":
    main()
