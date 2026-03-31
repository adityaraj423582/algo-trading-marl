"""
Feature engineering for the Algorithmic Trading MARL project.

Computes all features required by the CNN-GARCH volatility model (Stage 1)
and the MARL trading agents (Stage 2).

Feature catalogue
-----------------

**Volatility features (Stage 1 -- CNN-GARCH):**

1.  ``rv_daily``       -- annualised close-to-close realised volatility
2.  ``rv_weekly``      -- 5-day rolling mean of ``rv_daily``
3.  ``rv_monthly``     -- 22-day rolling mean of ``rv_daily``
4.  ``rolling_var_5d`` -- 5-day rolling variance of log returns
5.  ``rolling_var_22d``-- 22-day rolling variance of log returns
6.  ``parkinson_vol``  -- Parkinson (High-Low) volatility estimator
7.  ``garman_klass_vol``-- Garman-Klass (OHLC) volatility estimator

**Market microstructure features (Stage 2 -- MARL agents):**

8.  ``intraday_range`` -- (High - Low) / Close
9.  ``volume_ma5``     -- 5-period moving average of Volume
10. ``volume_ratio``   -- Volume / volume_ma5 (relative volume)
11. ``momentum_5``     -- 5-period price momentum
12. ``momentum_22``    -- 22-period price momentum
13. ``vol_regime``     -- binary high-volatility regime indicator

**Forecast targets:**

14. ``target_rv_1d``   -- next-day realised volatility  (rv_daily shifted -1)
15. ``target_rv_5d``   -- next-week realised volatility (rv_weekly shifted -1)

Usage
-----
::

    python -m src.data.feature_engineer                  # both markets
    python -m src.data.feature_engineer --market nse
    python -m src.data.feature_engineer --market nasdaq
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import (
    FEATURES_DIR,
    FIGURES_DIR,
    RAW_NASDAQ_DIR,
    RAW_NSE_DIR,
    RANDOM_SEED,
    get_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ANNUALISATION_FACTOR = np.sqrt(252)


# ========================================================================
# Loading helpers
# ========================================================================

def _load_raw_daily(raw_dir: Path, ticker_stem: str) -> Optional[pd.DataFrame]:
    """Load the raw daily CSV for *ticker_stem* (e.g. ``RELIANCE_NS``)."""
    path = raw_dir / f"{ticker_stem}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Datetime", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def _load_raw_hourly(raw_dir: Path, ticker_stem: str) -> Optional[pd.DataFrame]:
    """Load the raw hourly CSV for *ticker_stem*."""
    path = raw_dir / f"{ticker_stem}_1h.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Datetime", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def _discover_tickers(raw_dir: Path) -> List[str]:
    """Return unique ticker stems found in *raw_dir*."""
    stems = set()
    for p in raw_dir.glob("*_1d.csv"):
        stems.add(p.stem.replace("_1d", ""))
    return sorted(stems)


# ========================================================================
# Volatility features
# ========================================================================

def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute log returns from the Close column."""
    return np.log(df["Close"] / df["Close"].shift(1))


def compute_realised_volatility(
    log_returns: pd.Series,
    window: int = 1,
) -> pd.Series:
    """
    Annualised realised volatility over a rolling window.

    For ``window=1`` this is the absolute daily log return scaled to
    annual, which serves as a proxy for daily RV when we only have
    daily (not tick) data.
    """
    if window == 1:
        return log_returns.abs() * _ANNUALISATION_FACTOR
    return log_returns.rolling(window).std() * _ANNUALISATION_FACTOR


def compute_parkinson_vol(df: pd.DataFrame) -> pd.Series:
    """
    Parkinson (1980) High-Low volatility estimator.

    PV = sqrt( 1/(4*ln2) * ln(H/L)^2 )
    """
    hl = np.log(df["High"] / df["Low"])
    return np.sqrt(hl ** 2 / (4.0 * np.log(2)))


def compute_garman_klass_vol(df: pd.DataFrame) -> pd.Series:
    """
    Garman-Klass (1980) OHLC volatility estimator.

    GK = sqrt( 0.5*ln(H/L)^2 - (2*ln2-1)*ln(C/O)^2 )
    Clipped at zero inside the sqrt to handle edge cases.
    """
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    inside = 0.5 * hl ** 2 - (2.0 * np.log(2) - 1.0) * co ** 2
    return np.sqrt(inside.clip(lower=0.0))


# ========================================================================
# Market microstructure features
# ========================================================================

def compute_intraday_range(df: pd.DataFrame) -> pd.Series:
    """Intraday range normalised by close: (H - L) / C."""
    return (df["High"] - df["Low"]) / df["Close"]


def compute_volume_features(
    df: pd.DataFrame,
    ma_window: int = 5,
) -> pd.DataFrame:
    """Compute volume moving average and relative volume ratio."""
    vol_ma = df["Volume"].rolling(ma_window).mean()
    vol_ratio = df["Volume"] / vol_ma
    return pd.DataFrame({
        "volume_ma5": vol_ma,
        "volume_ratio": vol_ratio,
    }, index=df.index)


def compute_momentum(
    df: pd.DataFrame,
    periods: List[int] = None,
) -> pd.DataFrame:
    """Price momentum: (Close_t / Close_{t-k}) - 1."""
    if periods is None:
        periods = [5, 22]
    result = {}
    for p in periods:
        result[f"momentum_{p}"] = df["Close"] / df["Close"].shift(p) - 1.0
    return pd.DataFrame(result, index=df.index)


def compute_vol_regime(
    rv_daily: pd.Series,
    quantile: float = 0.75,
    lookback: int = 252,
) -> pd.Series:
    """
    Binary volatility regime indicator.

    1 if ``rv_daily`` exceeds the rolling 75th percentile over a
    ``lookback`` window, else 0.  Uses an expanding window at the
    start to avoid look-ahead bias.
    """
    threshold = rv_daily.expanding(min_periods=22).quantile(quantile)
    return (rv_daily > threshold).astype(int)


# ========================================================================
# Master feature builder
# ========================================================================

def build_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Compute all features for a single ticker's daily OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw daily OHLCV with columns ``Open, High, Low, Close, Volume``.
    ticker : str
        Ticker label (for logging only).

    Returns
    -------
    pd.DataFrame
        Original OHLCV columns plus all engineered features and targets.
    """
    feat = df.copy()

    # Log returns
    feat["log_return"] = compute_log_returns(feat)

    # --- Volatility features ---
    feat["rv_daily"] = compute_realised_volatility(feat["log_return"], window=1)
    feat["rv_weekly"] = feat["rv_daily"].rolling(5).mean()
    feat["rv_monthly"] = feat["rv_daily"].rolling(22).mean()
    feat["rolling_var_5d"] = feat["log_return"].rolling(5).var()
    feat["rolling_var_22d"] = feat["log_return"].rolling(22).var()
    feat["parkinson_vol"] = compute_parkinson_vol(feat)
    feat["garman_klass_vol"] = compute_garman_klass_vol(feat)

    # --- Market microstructure ---
    feat["intraday_range"] = compute_intraday_range(feat)
    vol_feats = compute_volume_features(feat)
    feat = pd.concat([feat, vol_feats], axis=1)
    mom_feats = compute_momentum(feat, periods=[5, 22])
    feat = pd.concat([feat, mom_feats], axis=1)
    feat["vol_regime"] = compute_vol_regime(feat["rv_daily"])

    # --- Forecast targets (shift backwards = future value) ---
    feat["target_rv_1d"] = feat["rv_daily"].shift(-1)
    feat["target_rv_5d"] = feat["rv_weekly"].shift(-1)

    # Drop warm-up NaN rows (from rolling windows)
    n_before = len(feat)
    feat.dropna(inplace=True)
    n_after = len(feat)
    logger.info(
        "  %s: %d -> %d rows after feature engineering (%d warm-up dropped)",
        ticker, n_before, n_after, n_before - n_after,
    )

    return feat


# ========================================================================
# Save & summary
# ========================================================================

def save_features(
    feat: pd.DataFrame,
    ticker: str,
    output_dir: Path,
) -> Path:
    """Save the feature matrix as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f"{ticker}_features.csv"
    feat.index.name = "Datetime"
    feat.to_csv(fpath)
    return fpath


def feature_summary(feat: pd.DataFrame, ticker: str) -> Dict[str, object]:
    """Return a compact summary dict for the feature matrix."""
    feature_cols = [
        "log_return", "rv_daily", "rv_weekly", "rv_monthly",
        "rolling_var_5d", "rolling_var_22d",
        "parkinson_vol", "garman_klass_vol",
        "intraday_range", "volume_ma5", "volume_ratio",
        "momentum_5", "momentum_22", "vol_regime",
        "target_rv_1d", "target_rv_5d",
    ]
    present = [c for c in feature_cols if c in feat.columns]
    missing_cols = [c for c in feature_cols if c not in feat.columns]
    nan_counts = {c: int(feat[c].isna().sum()) for c in present}

    return {
        "ticker": ticker,
        "shape": feat.shape,
        "features": len(present),
        "missing_features": missing_cols,
        "nan_per_feature": nan_counts,
    }


def generate_correlation_heatmap(
    feat: pd.DataFrame,
    ticker: str,
    output_dir: Path,
) -> None:
    """
    Save a correlation heatmap of the feature columns.

    Uses matplotlib with the ``Agg`` backend so it works headlessly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_cols = [
        "log_return", "rv_daily", "rv_weekly", "rv_monthly",
        "rolling_var_5d", "rolling_var_22d",
        "parkinson_vol", "garman_klass_vol",
        "intraday_range", "volume_ratio",
        "momentum_5", "momentum_22", "vol_regime",
        "target_rv_1d", "target_rv_5d",
    ]
    cols = [c for c in feature_cols if c in feat.columns]
    corr = feat[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Feature Correlation -- {ticker}", fontsize=14)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f"{ticker}_correlation_heatmap.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    logger.info("  Saved heatmap -> %s", fpath.name)


# ========================================================================
# Orchestration
# ========================================================================

def process_market_features(
    raw_dir: Path,
    output_dir: Path,
    market: str,
) -> List[Dict[str, object]]:
    """
    Build features for every ticker in *raw_dir* using **daily** data.

    We build features on daily bars because:
      - All tickers have 5 years of daily data (sufficient for 22-day
        rolling windows and annual regime detection).
      - The GARCH model and most volatility estimators are designed for
        daily returns.
      - Hourly features will be computed separately at model-training
        time using a sliding window on the hourly processed splits.
    """
    tickers = _discover_tickers(raw_dir)
    reports: List[Dict[str, object]] = []

    logger.info(
        "\n%s\n  Feature engineering -- %s -- %d tickers\n%s",
        "=" * 65, market.upper(), len(tickers), "=" * 65,
    )

    for ticker in tickers:
        logger.info("[%s] building features ...", ticker)

        df = _load_raw_daily(raw_dir, ticker)
        if df is None or df.empty:
            logger.warning("  Skipping %s -- no daily data found", ticker)
            continue

        feat = build_features(df, ticker)
        save_features(feat, ticker, output_dir)

        report = feature_summary(feat, ticker)
        reports.append(report)

        generate_correlation_heatmap(feat, ticker, FIGURES_DIR)

    return reports


def print_feature_summary(
    nse_reports: List[Dict],
    nasdaq_reports: List[Dict],
) -> None:
    """Print a formatted summary of feature engineering results."""

    def _section(title: str, reports: List[Dict]) -> None:
        logger.info("\n" + "=" * 80)
        logger.info("  %s  FEATURE ENGINEERING SUMMARY", title)
        logger.info("=" * 80)
        logger.info(
            "%-18s %12s %10s %10s",
            "Ticker", "Shape", "Features", "NaN cols",
        )
        logger.info("-" * 60)
        for r in reports:
            nan_cols = sum(1 for v in r["nan_per_feature"].values() if v > 0)
            logger.info(
                "%-18s %12s %10d %10d",
                r["ticker"],
                f"{r['shape'][0]}x{r['shape'][1]}",
                r["features"],
                nan_cols,
            )

    if nse_reports:
        _section("NSE", nse_reports)
    if nasdaq_reports:
        _section("NASDAQ", nasdaq_reports)

    all_r = nse_reports + nasdaq_reports
    total_rows = sum(r["shape"][0] for r in all_r)
    logger.info("\n" + "=" * 80)
    logger.info(
        "  OVERALL: %d tickers | %d total feature rows | %d features each",
        len(all_r), total_rows,
        all_r[0]["features"] if all_r else 0,
    )
    logger.info("=" * 80 + "\n")


# ========================================================================
# Entry points
# ========================================================================

def run_feature_engineer(market: Optional[str] = None) -> None:
    """Run the full feature engineering pipeline."""
    nse_reports: List[Dict] = []
    nasdaq_reports: List[Dict] = []

    features_nse_dir = FEATURES_DIR / "nse"
    features_nasdaq_dir = FEATURES_DIR / "nasdaq"

    if market in (None, "nse"):
        nse_reports = process_market_features(
            RAW_NSE_DIR, features_nse_dir, "nse",
        )

    if market in (None, "nasdaq"):
        nasdaq_reports = process_market_features(
            RAW_NASDAQ_DIR, features_nasdaq_dir, "nasdaq",
        )

    print_feature_summary(nse_reports, nasdaq_reports)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute volatility and microstructure features.",
    )
    parser.add_argument(
        "--market",
        choices=["nse", "nasdaq"],
        default=None,
        help="Process only one market (default: both).",
    )
    args = parser.parse_args()
    run_feature_engineer(market=args.market)


if __name__ == "__main__":
    main()
