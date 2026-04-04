"""
Level 2 limit order book (LOB) data processor.

Loads Bloomberg-sourced LOB snapshots or tick-by-tick data and computes
15 microstructure features used by the CNN-GARCH volatility model (Stage 1)
and the LOB-aware market-making agent (Stage 2).

Bloomberg data format
---------------------
Each ticker has a CSV with columns:
    Datetime, Bid1..BidN, BidSize1..BidSizeN,
    Ask1..AskN, AskSize1..AskSizeN,
    LastPrice, LastSize, Volume

If Bloomberg data is unavailable, the module generates **synthetic LOB
features** from existing OHLCV data by applying established statistical
relationships (e.g. bid-ask spread ~ f(volatility, volume)).  This allows
the full pipeline to run before Bloomberg access is available.

Feature catalogue (15 features)
-------------------------------
1.  bid_ask_spread        -- best ask - best bid ($)
2.  mid_price             -- (best_bid + best_ask) / 2
3.  microprice            -- bid_size-weighted mid-price
4.  book_imbalance        -- (bid_vol - ask_vol) / total_vol at top level
5.  depth_imbalance_5     -- imbalance across top 5 price levels
6.  total_bid_depth       -- aggregate bid quantity (top N levels)
7.  total_ask_depth       -- aggregate ask quantity (top N levels)
8.  spread_bps            -- spread in basis points
9.  order_flow_imbalance  -- rolling buy-sell volume imbalance
10. trade_intensity       -- number of trades per interval (proxy)
11. vpin                  -- volume-synchronised prob. of informed trading
12. kyle_lambda           -- price impact coefficient (rolling regression)
13. realized_spread       -- post-trade price reversion
14. effective_spread      -- 2 * |trade_price - mid_price|
15. queue_position_est    -- estimated queue position (depth proxy)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.utils.config import (
    BLOOMBERG_DIR,
    BLOOMBERG_NASDAQ_DIR,
    BLOOMBERG_NSE_DIR,
    FEATURES_DIR,
    LOB_DIR,
    LOB_NASDAQ_DIR,
    LOB_NSE_DIR,
    RANDOM_SEED,
    get_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = get_config().lob

np.random.seed(RANDOM_SEED)


# ======================================================================
# Bloomberg LOB loader
# ======================================================================

def load_bloomberg_lob(
    filepath: Path,
    n_levels: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Load a Bloomberg LOB snapshot CSV.

    Expected columns: Datetime, Bid1..Bid{n_levels}, BidSize1..BidSize{n_levels},
    Ask1..Ask{n_levels}, AskSize1..AskSize{n_levels}, LastPrice, LastSize, Volume.
    """
    if not filepath.exists():
        return None

    df = pd.read_csv(filepath, index_col="Datetime", parse_dates=True)
    df.sort_index(inplace=True)

    bid_cols = [f"Bid{i}" for i in range(1, n_levels + 1)]
    ask_cols = [f"Ask{i}" for i in range(1, n_levels + 1)]
    required = bid_cols[:1] + ask_cols[:1]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Bloomberg LOB file %s missing columns: %s", filepath.name, missing)
        return None

    return df


# ======================================================================
# LOB feature computation (from real L2 data)
# ======================================================================

def compute_lob_features_from_l2(
    df: pd.DataFrame,
    n_levels: int = 10,
) -> pd.DataFrame:
    """
    Compute all 15 LOB features from real Level 2 order book snapshots.

    Parameters
    ----------
    df : pd.DataFrame
        Bloomberg LOB data with Bid/Ask price and size columns.
    n_levels : int
        Number of price levels available in the data.
    """
    feat = pd.DataFrame(index=df.index)

    bid_price_cols = [f"Bid{i}" for i in range(1, n_levels + 1) if f"Bid{i}" in df.columns]
    ask_price_cols = [f"Ask{i}" for i in range(1, n_levels + 1) if f"Ask{i}" in df.columns]
    bid_size_cols = [f"BidSize{i}" for i in range(1, n_levels + 1) if f"BidSize{i}" in df.columns]
    ask_size_cols = [f"AskSize{i}" for i in range(1, n_levels + 1) if f"AskSize{i}" in df.columns]

    best_bid = df[bid_price_cols[0]] if bid_price_cols else df.get("Bid1", pd.Series(0, index=df.index))
    best_ask = df[ask_price_cols[0]] if ask_price_cols else df.get("Ask1", pd.Series(0, index=df.index))
    best_bid_size = df[bid_size_cols[0]] if bid_size_cols else pd.Series(1, index=df.index)
    best_ask_size = df[ask_size_cols[0]] if ask_size_cols else pd.Series(1, index=df.index)

    feat["bid_ask_spread"] = best_ask - best_bid
    feat["mid_price"] = (best_bid + best_ask) / 2.0

    total_top = best_bid_size + best_ask_size
    feat["microprice"] = (best_bid * best_ask_size + best_ask * best_bid_size) / total_top.clip(lower=1)
    feat["book_imbalance"] = (best_bid_size - best_ask_size) / total_top.clip(lower=1)

    # Depth imbalance across top 5 levels
    n_depth = min(5, len(bid_size_cols), len(ask_size_cols))
    if n_depth > 0:
        bid_depth = df[bid_size_cols[:n_depth]].sum(axis=1)
        ask_depth = df[ask_size_cols[:n_depth]].sum(axis=1)
        total_depth = bid_depth + ask_depth
        feat["depth_imbalance_5"] = (bid_depth - ask_depth) / total_depth.clip(lower=1)
    else:
        feat["depth_imbalance_5"] = feat["book_imbalance"]

    feat["total_bid_depth"] = df[bid_size_cols].sum(axis=1) if bid_size_cols else best_bid_size
    feat["total_ask_depth"] = df[ask_size_cols].sum(axis=1) if ask_size_cols else best_ask_size

    mid = feat["mid_price"].clip(lower=1e-8)
    feat["spread_bps"] = (feat["bid_ask_spread"] / mid) * 10_000

    # Trade-based features
    last_price = df.get("LastPrice", mid)
    last_size = df.get("LastSize", pd.Series(100, index=df.index))
    volume = df.get("Volume", last_size.rolling(10).sum())

    buy_volume = last_size.where(last_price >= mid, 0)
    sell_volume = last_size.where(last_price < mid, 0)
    feat["order_flow_imbalance"] = (
        buy_volume.rolling(20, min_periods=1).sum()
        - sell_volume.rolling(20, min_periods=1).sum()
    ) / volume.rolling(20, min_periods=1).sum().clip(lower=1)

    feat["trade_intensity"] = last_size.rolling(20, min_periods=1).count() / 20.0

    # VPIN (simplified): |buy_vol - sell_vol| / total_vol over rolling buckets
    abs_imb = (buy_volume - sell_volume).abs()
    feat["vpin"] = abs_imb.rolling(50, min_periods=10).sum() / volume.rolling(50, min_periods=10).sum().clip(lower=1)

    # Kyle's lambda: rolling regression of price change on signed volume
    price_change = mid.diff()
    signed_vol = (buy_volume - sell_volume)
    feat["kyle_lambda"] = _rolling_slope(price_change, signed_vol, window=50)

    # Realized spread: mid_price change 5 periods after trade
    mid_future = mid.shift(-5)
    trade_sign = np.sign(last_price - mid)
    feat["realized_spread"] = 2.0 * trade_sign * (mid_future - mid) / mid.clip(lower=1e-8) * 10_000

    # Effective spread
    feat["effective_spread"] = 2.0 * (last_price - mid).abs() / mid.clip(lower=1e-8) * 10_000

    feat["queue_position_estimate"] = best_bid_size / (best_bid_size + best_ask_size).clip(lower=1)

    return feat


def _rolling_slope(y: pd.Series, x: pd.Series, window: int = 50) -> pd.Series:
    """Compute rolling OLS slope of y on x."""
    result = pd.Series(0.0, index=y.index)
    y_arr = y.fillna(0).values
    x_arr = x.fillna(0).values
    for i in range(window, len(y_arr)):
        xi = x_arr[i - window:i]
        yi = y_arr[i - window:i]
        var_x = np.var(xi)
        if var_x > 1e-12:
            result.iloc[i] = np.cov(xi, yi)[0, 1] / var_x
    return result


# ======================================================================
# Synthetic LOB features (from OHLCV when Bloomberg unavailable)
# ======================================================================

def compute_synthetic_lob_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic synthetic LOB features from OHLCV data.

    Uses established empirical relationships from microstructure literature:
    - Spread ~ f(volatility, 1/volume)  [Roll 1984, Corwin & Schultz 2012]
    - Book imbalance ~ autocorrelation of returns
    - VPIN ~ |returns| / volume
    - Kyle lambda ~ volatility / sqrt(volume)
    """
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].clip(lower=1)
    log_ret = np.log(close / close.shift(1))

    # Corwin-Schultz spread estimator from high-low prices
    hl_ratio = np.log(high / low)
    hl_sum = hl_ratio + hl_ratio.shift(1)
    hl_max = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    hl_min = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    hl_2day = np.log(hl_max / hl_min)
    gamma = hl_sum ** 2 / 2.0
    beta = hl_2day ** 2
    alpha_cs = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    spread_estimate = 2.0 * (np.exp(alpha_cs.clip(lower=0)) - 1.0) / (1.0 + np.exp(alpha_cs.clip(lower=0)))

    feat["bid_ask_spread"] = spread_estimate * close
    feat["mid_price"] = close
    feat["microprice"] = close * (1.0 + 0.001 * log_ret.fillna(0))

    # Synthetic book imbalance: autocorrelation of returns as proxy
    ret_sign = np.sign(log_ret.fillna(0))
    feat["book_imbalance"] = ret_sign.rolling(5, min_periods=1).mean()

    feat["depth_imbalance_5"] = feat["book_imbalance"] * 0.8 + 0.2 * np.random.standard_normal(len(df))

    avg_vol = volume.rolling(22, min_periods=1).mean()
    feat["total_bid_depth"] = avg_vol * 0.5
    feat["total_ask_depth"] = avg_vol * 0.5

    feat["spread_bps"] = (feat["bid_ask_spread"] / close.clip(lower=1e-8)) * 10_000

    buy_proxy = volume.where(log_ret > 0, 0)
    sell_proxy = volume.where(log_ret <= 0, 0)
    feat["order_flow_imbalance"] = (
        buy_proxy.rolling(20, min_periods=1).sum()
        - sell_proxy.rolling(20, min_periods=1).sum()
    ) / volume.rolling(20, min_periods=1).sum().clip(lower=1)

    feat["trade_intensity"] = volume / avg_vol.clip(lower=1)

    # VPIN: |net_volume| / total_volume
    net_vol = (buy_proxy - sell_proxy).abs()
    feat["vpin"] = net_vol.rolling(50, min_periods=10).sum() / volume.rolling(50, min_periods=10).sum().clip(lower=1)

    # Kyle lambda: volatility / sqrt(volume)
    daily_vol = log_ret.abs().rolling(22, min_periods=5).mean() * np.sqrt(252)
    feat["kyle_lambda"] = daily_vol / np.sqrt(volume.rolling(22, min_periods=5).mean().clip(lower=1))

    # Realized spread: proxy from return reversal
    ret_future = log_ret.shift(-5)
    feat["realized_spread"] = -2.0 * ret_sign * ret_future.fillna(0) * 10_000

    # Effective spread: from Corwin-Schultz estimator
    feat["effective_spread"] = feat["spread_bps"] * 0.6

    feat["queue_position_estimate"] = 0.5 + 0.1 * feat["book_imbalance"]

    feat = feat.clip(lower=-100, upper=100)

    return feat


# ======================================================================
# Orchestration
# ======================================================================

def process_lob_features(
    ticker: str,
    raw_dir: Path,
    ohlcv_df: Optional[pd.DataFrame] = None,
    n_levels: int = _CFG.n_price_levels,
) -> pd.DataFrame:
    """
    Compute LOB features for a ticker.  Tries real Bloomberg L2 data
    first; falls back to synthetic features from OHLCV.

    Parameters
    ----------
    ticker : str
        Ticker stem (e.g. ``RELIANCE_NS``).
    raw_dir : Path
        Directory containing Bloomberg LOB CSVs.
    ohlcv_df : pd.DataFrame or None
        OHLCV data for synthetic fallback.
    n_levels : int
        Number of order book levels.

    Returns
    -------
    pd.DataFrame
        15-column LOB feature matrix aligned to the ticker's date index.
    """
    lob_path = raw_dir / f"{ticker}_lob.csv"
    lob_df = load_bloomberg_lob(lob_path, n_levels=n_levels)

    if lob_df is not None:
        logger.info("[%s] Computing LOB features from Bloomberg L2 data (%d snapshots)", ticker, len(lob_df))
        feat = compute_lob_features_from_l2(lob_df, n_levels=n_levels)
    elif ohlcv_df is not None:
        logger.info("[%s] No Bloomberg L2 data -- generating synthetic LOB features from OHLCV", ticker)
        feat = compute_synthetic_lob_features(ohlcv_df)
    else:
        logger.warning("[%s] No data available for LOB features -- returning empty", ticker)
        return pd.DataFrame()

    feat.dropna(inplace=True)
    return feat


def merge_lob_with_features(
    feature_df: pd.DataFrame,
    lob_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge LOB features into the existing feature CSV (left join on date index).

    Columns from ``lob_df`` are prefixed with ``lob_`` to distinguish
    them from the original 23 Level 1 features.
    """
    lob_renamed = lob_df.add_prefix("lob_")
    merged = feature_df.join(lob_renamed, how="left")

    for col in lob_renamed.columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    return merged


def run_lob_processing(market: Optional[str] = None) -> None:
    """
    Process LOB features for all tickers and merge into existing feature CSVs.

    Usage::

        python -m src.data.lob_processor
        python -m src.data.lob_processor --market nse
    """
    from src.data.feature_engineer import _discover_tickers, _load_raw_daily

    markets = []
    if market in (None, "nse"):
        markets.append(("nse", LOB_NSE_DIR, FEATURES_DIR / "nse"))
    if market in (None, "nasdaq"):
        markets.append(("nasdaq", LOB_NASDAQ_DIR, FEATURES_DIR / "nasdaq"))

    for mkt_name, lob_dir, feat_dir in markets:
        from src.utils.config import RAW_NSE_DIR, RAW_NASDAQ_DIR
        raw_dir = RAW_NSE_DIR if mkt_name == "nse" else RAW_NASDAQ_DIR

        tickers = _discover_tickers(raw_dir)
        logger.info("\n%s\n  LOB feature processing -- %s -- %d tickers\n%s",
                     "=" * 65, mkt_name.upper(), len(tickers), "=" * 65)

        for ticker in tickers:
            ohlcv = _load_raw_daily(raw_dir, ticker)

            lob_feat = process_lob_features(
                ticker=ticker,
                raw_dir=lob_dir,
                ohlcv_df=ohlcv,
            )
            if lob_feat.empty:
                continue

            # Load existing feature CSV and merge
            feat_path = feat_dir / f"{ticker}_features.csv"
            if feat_path.exists():
                existing = pd.read_csv(feat_path, index_col="Datetime", parse_dates=True)
                merged = merge_lob_with_features(existing, lob_feat)
                merged.index.name = "Datetime"
                merged.to_csv(feat_path)
                logger.info("  [%s] Merged %d LOB features -> %d total columns",
                            ticker, len(lob_feat.columns), len(merged.columns))
            else:
                logger.warning("  [%s] Feature CSV not found at %s -- skipping merge", ticker, feat_path)

    logger.info("\nLOB processing complete.")


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Process Level 2 LOB features.")
    parser.add_argument("--market", choices=["nse", "nasdaq"], default=None)
    args = parser.parse_args()
    run_lob_processing(market=args.market)


if __name__ == "__main__":
    main()
