"""
Data downloader for NSE and NASDAQ equities via yfinance.

Downloads OHLCV data for the configured tickers at two frequencies:

  - **Daily (1d)**: Full historical range (2020-01-01 → 2024-12-31).
    Used for GARCH modelling, daily volatility analysis, and long-horizon
    backtesting.

  - **Hourly (1h)**: Last ~730 calendar days (Yahoo Finance hard limit).
    Used for CNN feature extraction, intraday signal generation, and
    MARL trading simulation.

Each ticker × interval combination is saved as a separate CSV in
``data/raw/{exchange}/``.

Usage
-----
From the project root::

    python -m src.data.downloader                  # both markets, both intervals
    python -m src.data.downloader --market nse     # NSE only
    python -m src.data.downloader --market nasdaq  # NASDAQ only
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.utils.config import (
    DataConfig,
    RAW_NASDAQ_DIR,
    RAW_NSE_DIR,
    get_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Yahoo Finance caps 1h data to ~730 calendar days from today
_YF_MAX_DAYS_1H: int = 729


def _effective_date_range(
    start: str,
    end: str,
    interval: str,
) -> tuple[str, str]:
    """
    Return the effective (start, end) date range given Yahoo Finance limits.

    For daily data the requested range is used as-is.
    For hourly data the start is clamped to at most 729 days before today.
    """
    if interval == "1h":
        earliest_allowed = datetime.now() - timedelta(days=_YF_MAX_DAYS_1H)
        requested_start = datetime.strptime(start, "%Y-%m-%d")
        effective_start = max(requested_start, earliest_allowed)
        return effective_start.strftime("%Y-%m-%d"), end
    return start, end


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    max_retries: int = 3,
    retry_delay: float = 2.0,
    backoff_factor: float = 2.0,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single ticker with retry logic.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start, end : str
        Date range in ``YYYY-MM-DD`` format.
    interval : str
        Bar interval (``"1d"`` or ``"1h"``).
    max_retries : int
        Maximum retry attempts on failure.
    retry_delay : float
        Initial delay (seconds) between retries.
    backoff_factor : float
        Multiplier applied to delay after each retry.

    Returns
    -------
    pd.DataFrame or None
        Downloaded OHLCV data, or ``None`` if all retries exhausted.
    """
    eff_start, eff_end = _effective_date_range(start, end, interval)
    delay = retry_delay

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=eff_start,
                end=eff_end,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            if df is None or df.empty:
                logger.warning(
                    "  Attempt %d/%d — empty result for %s (%s)",
                    attempt, max_retries, ticker, interval,
                )
            else:
                # Flatten MultiIndex columns produced by newer yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Drop duplicate timestamps
                df = df[~df.index.duplicated(keep="first")]
                df.sort_index(inplace=True)

                logger.info(
                    "  %s [%s]: %d rows  (%s to %s)",
                    ticker, interval, len(df),
                    str(df.index.min())[:19],
                    str(df.index.max())[:19],
                )
                return df

        except Exception as exc:
            logger.error(
                "  Attempt %d/%d — error for %s (%s): %s",
                attempt, max_retries, ticker, interval, exc,
            )

        if attempt < max_retries:
            logger.info("  Retrying in %.1f s …", delay)
            time.sleep(delay)
            delay *= backoff_factor

    logger.error("  All %d attempts failed for %s (%s)", max_retries, ticker, interval)
    return None


def validate_dataframe(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    min_rows: int = 500,
    max_missing_pct: float = 5.0,
) -> Dict[str, object]:
    """
    Run quality checks on downloaded data.

    Checks
    ------
    1. Minimum row count
    2. Percentage of NaN values per column
    3. Zero or negative prices
    4. Duplicate index entries
    """
    report: Dict[str, object] = {
        "ticker": ticker,
        "interval": interval,
        "rows": len(df),
        "start": str(df.index.min())[:19] if len(df) else "N/A",
        "end": str(df.index.max())[:19] if len(df) else "N/A",
        "issues": [],
    }

    if len(df) < min_rows:
        report["issues"].append(f"Row count {len(df)} < minimum {min_rows}")

    for col in df.columns:
        missing = df[col].isna().sum()
        pct = 100.0 * missing / len(df) if len(df) else 0
        if pct > max_missing_pct:
            report["issues"].append(f"{col}: {pct:.1f}% missing")

    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    for col in price_cols:
        n_bad = (df[col] <= 0).sum()
        if n_bad > 0:
            report["issues"].append(f"{col}: {n_bad} zero/negative values")

    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        report["issues"].append(f"{n_dup} duplicate timestamps")

    if not report["issues"]:
        report["issues"].append("PASS")
    else:
        for issue in report["issues"]:
            logger.warning("    [%s %s] %s", ticker, interval, issue)

    return report


def save_ticker_csv(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    output_dir: Path,
) -> Path:
    """
    Save a ticker DataFrame to CSV.

    Naming convention: ``{TICKER}_{interval}.csv``
    (e.g. ``RELIANCE_NS_1d.csv``, ``AAPL_1h.csv``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = ticker.replace(".", "_").replace("^", "")
    filepath = output_dir / f"{safe_name}_{interval}.csv"
    df.index.name = "Datetime"
    df.to_csv(filepath)
    logger.info("  Saved -> %s", filepath.relative_to(output_dir.parents[2]))
    return filepath


def download_market(
    tickers: List[str],
    output_dir: Path,
    cfg: DataConfig,
    market_label: str = "MARKET",
) -> List[Dict[str, object]]:
    """
    Download, validate, and save data for a list of tickers at all
    configured intervals.
    """
    reports: List[Dict[str, object]] = []

    for interval in cfg.intervals:
        min_rows = (
            cfg.min_expected_rows_daily if interval == "1d"
            else cfg.min_expected_rows_hourly
        )

        logger.info(
            "\n%s\n  %s — %s interval — %d tickers\n%s",
            "=" * 65, market_label, interval, len(tickers), "=" * 65,
        )

        for i, ticker in enumerate(tickers, 1):
            logger.info("[%d/%d] %s (%s) …", i, len(tickers), ticker, interval)

            df = download_ticker(
                ticker=ticker,
                start=cfg.start_date,
                end=cfg.end_date,
                interval=interval,
                max_retries=cfg.max_retries,
                retry_delay=cfg.retry_delay_seconds,
                backoff_factor=cfg.retry_backoff_factor,
            )

            if df is None or df.empty:
                reports.append({
                    "ticker": ticker,
                    "interval": interval,
                    "rows": 0,
                    "start": "N/A",
                    "end": "N/A",
                    "issues": ["DOWNLOAD FAILED"],
                })
                continue

            report = validate_dataframe(
                df, ticker, interval,
                min_rows=min_rows,
                max_missing_pct=cfg.max_missing_pct,
            )
            reports.append(report)
            save_ticker_csv(df, ticker, interval, output_dir)

            # Polite delay between tickers
            if i < len(tickers):
                time.sleep(1.0)

    return reports


def print_summary(
    nse_reports: List[Dict],
    nasdaq_reports: List[Dict],
) -> None:
    """Print a formatted summary table of all downloads."""

    def _section(title: str, reports: List[Dict]) -> None:
        logger.info("\n" + "=" * 80)
        logger.info("  %s  DOWNLOAD SUMMARY", title)
        logger.info("=" * 80)
        logger.info(
            "%-18s %-6s %8s  %-19s  %-19s  %s",
            "Ticker", "Freq", "Rows", "Start", "End", "Status",
        )
        logger.info("-" * 80)
        for r in reports:
            status = (
                "OK" if r["issues"] == ["PASS"]
                else "; ".join(str(x) for x in r["issues"])
            )
            logger.info(
                "%-18s %-6s %8s  %-19s  %-19s  %s",
                r["ticker"], r["interval"], r["rows"],
                r["start"], r["end"], status,
            )

        total_rows = sum(r["rows"] for r in reports)
        ok_count = sum(1 for r in reports if r["issues"] == ["PASS"])
        logger.info("-" * 80)
        logger.info(
            "Total: %d downloads | %d rows | %d passed validation",
            len(reports), total_rows, ok_count,
        )

    if nse_reports:
        _section("NSE (NIFTY 50)", nse_reports)
    if nasdaq_reports:
        _section("NASDAQ 100", nasdaq_reports)

    all_reports = nse_reports + nasdaq_reports
    total = len(all_reports)
    ok = sum(1 for r in all_reports if r["issues"] == ["PASS"])
    failed = sum(1 for r in all_reports if r["issues"] == ["DOWNLOAD FAILED"])
    total_rows = sum(r["rows"] for r in all_reports)

    logger.info("\n" + "=" * 80)
    logger.info("  OVERALL: %d / %d passed  |  %d failed  |  %d total rows",
                ok, total, failed, total_rows)
    logger.info("=" * 80 + "\n")


def run_downloader(market: Optional[str] = None) -> None:
    """
    Run the full download pipeline.

    Parameters
    ----------
    market : str or None
        ``"nse"``, ``"nasdaq"``, or ``None`` (both).
    """
    cfg = get_config().data

    nse_reports: List[Dict] = []
    nasdaq_reports: List[Dict] = []

    if market in (None, "nse"):
        nse_reports = download_market(
            tickers=cfg.nse_tickers,
            output_dir=RAW_NSE_DIR,
            cfg=cfg,
            market_label="NSE (NIFTY 50)",
        )

    if market in (None, "nasdaq"):
        nasdaq_reports = download_market(
            tickers=cfg.nasdaq_tickers,
            output_dir=RAW_NASDAQ_DIR,
            cfg=cfg,
            market_label="NASDAQ 100",
        )

    print_summary(nse_reports, nasdaq_reports)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download NSE and NASDAQ equity data via yfinance.",
    )
    parser.add_argument(
        "--market",
        choices=["nse", "nasdaq"],
        default=None,
        help="Download only one market (default: both).",
    )
    args = parser.parse_args()
    run_downloader(market=args.market)


if __name__ == "__main__":
    main()
