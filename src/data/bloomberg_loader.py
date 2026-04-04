"""
Bloomberg Level 2 data loader.

Parses the Bloomberg Excel export (BLMBRG_LVL_2_DATSET__22.xlsx) and
extracts bid/ask prices, spread, volume, and VIX data for integration
into the existing feature engineering pipeline.

Column mapping (by position -- Bloomberg headers are broken #N/A):
    Col 0: Date
    Col 1: Bid price (open-side)
    Col 2: EMPTY (always NaN)
    Col 3: LastPrice / Close
    Col 4: Volume
    Col 5: Bid1 (best bid price)
    Col 6: Ask1 (best ask price)
    Col 7: BidAskSpread (absolute)
    Col 8: BidAskSpread_pct (percentage)
    Col 9: SKIP (partial data, unreliable)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.config import BLOOMBERG_DIR, FEATURES_DIR, PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

BLOOMBERG_FILE_PATH: Path = BLOOMBERG_DIR / "BLMBRG_LVL_2_DATSET__22.xlsx"

SHEET_TO_TICKER: Dict[str, str] = {
    "RELIANCE IN Equity": "RELIANCE_NS",
    "TCS IN Equity":      "TCS_NS",
    "HDFCB IN Equity":    "HDFCBANK_NS",
    "ICICIBC IN Equity":  "ICICIBANK_NS",
    "SBIN IN Equity":     "SBIN_NS",
    "BHARTI IN Equity":   "BHARTIARTL_NS",
    "INFY US Equity":     "INFY_NS",
    "AAPL US Equity":     "AAPL",
    "MSFT US Equity":     "MSFT",
    "GOOGL US Equity":    "GOOGL",
    "AMZN US Equity":     "AMZN",
    "NVDA US Equity":     "NVDA",
    "META US Equity":     "META",
    "TSLA US Equity":     "TSLA",
    "AVGO US Equity":     "AVGO",
    "COST US Equity":     "COST",
    "ASML US Equity":     "ASML",
}

SYNTHETIC_FALLBACK: List[str] = ["KOTAKBANK_NS", "HINDUNILVR_NS", "BAJFINANCE_NS"]

NSE_TICKERS: List[str] = [
    "RELIANCE_NS", "TCS_NS", "HDFCBANK_NS",
    "ICICIBANK_NS", "SBIN_NS", "BHARTIARTL_NS", "INFY_NS",
]
NASDAQ_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AVGO", "COST", "ASML",
]

BLOOMBERG_FEATURE_COLUMNS: List[str] = [
    "bb_bid1", "bb_ask1", "bb_mid_price",
    "bb_spread_abs", "bb_spread_pct", "bb_volume",
    "india_vix", "us_vix", "vix_spread",
]


class BloombergLoader:
    """Loads and processes the Bloomberg Level 2 Excel export."""

    def __init__(self, filepath: Optional[Path] = None) -> None:
        self.filepath = filepath or BLOOMBERG_FILE_PATH
        if not self.filepath.exists():
            raise FileNotFoundError(f"Bloomberg file not found: {self.filepath}")

    def load_stock(self, sheet_name: str, ticker_out: str) -> pd.DataFrame:
        """Load one stock sheet from the Bloomberg Excel file."""
        df = pd.read_excel(
            self.filepath,
            sheet_name=sheet_name,
            header=None,
            skiprows=2,
        )
        df.columns = [
            "Date", "open", "_empty", "close", "volume",
            "bid1", "ask1", "spread_abs", "spread_pct", "col9",
        ]
        df = df.drop(columns=["_empty", "col9"])

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date").sort_index()

        for col in ["open", "close", "volume", "bid1", "ask1",
                     "spread_abs", "spread_pct"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["close"].notna() & (df["close"] > 0)]
        df["mid_price"] = (df["bid1"] + df["ask1"]) / 2.0
        df["realized_spread"] = df["ask1"] - df["bid1"]
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "Datetime"

        return df

    def load_vix(self, sheet_name: str) -> pd.DataFrame:
        """Load VIX or INVIXN sheet (only Date + value columns)."""
        df = pd.read_excel(
            self.filepath,
            sheet_name=sheet_name,
            header=None,
            skiprows=2,
            usecols=[0, 3],
        )
        df.columns = ["Date", "vix_value"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "vix_value"])
        df = df.set_index("Date").sort_index()
        df["vix_value"] = pd.to_numeric(df["vix_value"], errors="coerce")
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "Datetime"
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all 17 stocks + VIX + INVIXN."""
        data: Dict[str, pd.DataFrame] = {}
        success = 0

        for sheet_name, ticker in SHEET_TO_TICKER.items():
            try:
                df = self.load_stock(sheet_name, ticker)
                data[ticker] = df
                success += 1
                logger.info(
                    "  %s: %d rows (%s to %s)",
                    ticker, len(df),
                    df.index[0].date(), df.index[-1].date(),
                )
            except Exception as e:
                logger.error("  %s: Failed -- %s", ticker, e)

        try:
            data["VIX"] = self.load_vix("VIX Index")
            logger.info("  VIX Index: %d rows", len(data["VIX"]))
        except Exception as e:
            logger.error("  VIX Index: Failed -- %s", e)

        try:
            data["INVIXN"] = self.load_vix("INVIXN Index")
            logger.info("  INVIXN Index: %d rows", len(data["INVIXN"]))
        except Exception as e:
            logger.error("  INVIXN Index: Failed -- %s", e)

        logger.info(
            "\nBloomberg data loaded: %d stocks, "
            "3 stocks using synthetic fallback (KOTAKBANK, HINDUNLVR, BAJFINANCE)",
            success,
        )
        return data

    def merge_with_features(self, bloomberg_dict: Dict[str, pd.DataFrame]) -> None:
        """Merge Bloomberg columns into existing feature CSVs."""
        all_tickers = NSE_TICKERS + NASDAQ_TICKERS + SYNTHETIC_FALLBACK

        for ticker in all_tickers:
            sub = "nse" if ticker.endswith("_NS") else "nasdaq"
            feature_csv = FEATURES_DIR / sub / f"{ticker}_features.csv"

            if not feature_csv.exists():
                logger.warning("  Feature CSV not found for %s -- skipping", ticker)
                continue

            feat_df = pd.read_csv(feature_csv, index_col="Datetime", parse_dates=True)

            if ticker in SYNTHETIC_FALLBACK:
                logger.info("  %s: using synthetic LOB (no Bloomberg data)", ticker)
                continue

            if ticker not in bloomberg_dict:
                logger.warning("  %s: not in Bloomberg dict -- skipping", ticker)
                continue

            bb_df = bloomberg_dict[ticker]

            feat_df["bb_bid1"] = bb_df["bid1"]
            feat_df["bb_ask1"] = bb_df["ask1"]
            feat_df["bb_mid_price"] = bb_df["mid_price"]
            feat_df["bb_spread_abs"] = bb_df["spread_abs"]
            feat_df["bb_spread_pct"] = bb_df["spread_pct"]
            feat_df["bb_volume"] = bb_df["volume"]

            bb_cols = [
                "bb_bid1", "bb_ask1", "bb_mid_price",
                "bb_spread_abs", "bb_spread_pct", "bb_volume",
            ]
            feat_df[bb_cols] = feat_df[bb_cols].ffill(limit=3)

            matched = int(feat_df[bb_cols].notna().all(axis=1).sum())
            feat_df.to_csv(feature_csv)
            logger.info("  %s: %d/%d rows matched", ticker, matched, len(feat_df))


def add_vix_features(bloomberg_dict: Dict[str, pd.DataFrame]) -> None:
    """Add India VIX + US VIX + vix_spread to all 20 feature CSVs."""
    invixn = bloomberg_dict.get("INVIXN")
    usvix = bloomberg_dict.get("VIX")

    if invixn is None or usvix is None:
        logger.error("VIX data not available -- skipping VIX feature addition")
        return

    invixn = invixn.rename(columns={"vix_value": "india_vix"})
    usvix = usvix.rename(columns={"vix_value": "us_vix"})

    for sub in ["nse", "nasdaq"]:
        feat_dir = FEATURES_DIR / sub
        if not feat_dir.exists():
            continue

        for csv_file in sorted(feat_dir.glob("*_features.csv")):
            df = pd.read_csv(csv_file, index_col="Datetime", parse_dates=True)

            # Drop existing VIX columns if re-running
            for col in ["india_vix", "us_vix", "vix_spread"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            df = df.join(invixn["india_vix"], how="left")
            df = df.join(usvix["us_vix"], how="left")

            df["india_vix"] = df["india_vix"].ffill(limit=3)
            df["us_vix"] = df["us_vix"].ffill(limit=3)
            df["vix_spread"] = df["us_vix"] - df["india_vix"]

            df.to_csv(csv_file)

    logger.info("  VIX features added to all feature CSVs (india_vix, us_vix, vix_spread)")
