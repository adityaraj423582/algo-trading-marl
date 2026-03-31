"""
Performance metrics for algorithmic trading evaluation.

Provides all standard risk-adjusted metrics used in quantitative finance
research papers (Sharpe, Sortino, Calmar, VaR, etc.) plus volatility
forecast metrics (QLIKE, Diebold-Mariano).

All return-based metrics assume **daily** returns and annualise using
252 trading days unless otherwise specified.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm


# Risk-free rate defaults
_ANNUAL_RFR = 0.05          # 5 % (India RBI repo approximate)
_DAILY_RFR = _ANNUAL_RFR / 252.0


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = _ANNUAL_RFR,
    annualise: bool = True,
) -> float:
    """Annualised Sharpe ratio."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    daily_rfr = risk_free_rate / 252.0
    excess = r - daily_rfr
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    sr = np.mean(excess) / std
    return float(sr * np.sqrt(252)) if annualise else float(sr)


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough decline (0-1 scale)."""
    v = np.asarray(portfolio_values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return 0.0
    peak = v[0]
    mdd = 0.0
    for val in v:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return float(mdd)


def calmar_ratio(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
) -> float:
    """Annualised return / max drawdown."""
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(portfolio_values)
    if mdd < 1e-12:
        return 0.0
    return float(ann_ret / mdd)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = _ANNUAL_RFR,
) -> float:
    """Sortino ratio (penalises downside volatility only)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    daily_rfr = risk_free_rate / 252.0
    excess = r - daily_rfr
    downside = excess[excess < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-8
    if down_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / down_std * np.sqrt(252))


def win_rate(returns: np.ndarray) -> float:
    """Fraction of days with positive return."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    return float(np.sum(r > 0) / len(r))


def profit_factor(returns: np.ndarray) -> float:
    """Sum of positive returns / |sum of negative returns|."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    gains = np.sum(r[r > 0])
    losses = np.abs(np.sum(r[r < 0]))
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def annualized_return(returns: np.ndarray) -> float:
    """Compound annualised growth rate."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    total = np.prod(1.0 + r)
    n_years = len(r) / 252.0
    if n_years < 1e-8 or total <= 0:
        return 0.0
    return float(total ** (1.0 / n_years) - 1.0)


def annualized_volatility(returns: np.ndarray) -> float:
    """Annualised standard deviation of returns."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(252))


def value_at_risk(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Historical VaR at the given confidence level (positive = loss)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return 0.0
    return float(-np.percentile(r, (1 - confidence) * 100))


def qlike_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Quasi-likelihood loss for volatility forecast evaluation.

    QLIKE = mean( actual / predicted - log(actual / predicted) - 1 )
    """
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    mask = (a > 0) & (p > 0) & np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if len(a) < 5:
        return float("nan")
    ratio = a / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    loss: str = "SE",
) -> Dict[str, object]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Positive DM stat means forecast1 is *worse* than forecast2.
    """
    a = np.asarray(actual, dtype=np.float64)
    p1 = np.asarray(forecast1, dtype=np.float64)
    p2 = np.asarray(forecast2, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(p1) & np.isfinite(p2)
    a, p1, p2 = a[mask], p1[mask], p2[mask]
    if len(a) < 10:
        return {"dm_stat": 0.0, "p_value": 1.0, "significant": False}

    if loss == "SE":
        d = (a - p1) ** 2 - (a - p2) ** 2
    else:
        d = np.abs(a - p1) - np.abs(a - p2)

    n = len(d)
    d_bar = np.mean(d)
    max_lag = max(1, int(n ** (1.0 / 3.0)))
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        gamma_k = np.cov(d[:-k], d[k:])[0, 1]
        gamma_sum += w * gamma_k
    var_d = gamma_0 + 2.0 * gamma_sum

    if var_d <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "significant": False}

    dm_stat = d_bar / np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - sp_norm.cdf(np.abs(dm_stat)))
    return {
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
    }


def rolling_sharpe(
    returns: np.ndarray,
    window: int = 63,
    risk_free_rate: float = _ANNUAL_RFR,
) -> np.ndarray:
    """63-day (quarterly) rolling Sharpe ratio."""
    r = np.asarray(returns, dtype=np.float64)
    daily_rfr = risk_free_rate / 252.0
    out = np.full(len(r), np.nan)
    for i in range(window, len(r)):
        w = r[i - window:i] - daily_rfr
        s = np.std(w, ddof=1)
        out[i] = np.mean(w) / s * np.sqrt(252) if s > 1e-12 else 0.0
    return out


def compute_all_metrics(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    risk_free_rate: float = _ANNUAL_RFR,
) -> Dict[str, float]:
    """Compute all trading performance metrics in one call."""
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(portfolio_values, dtype=np.float64)
    total_ret = (v[-1] / v[0] - 1.0) * 100 if len(v) >= 2 and v[0] > 0 else 0.0
    return {
        "total_return_pct": round(total_ret, 2),
        "annualized_return_pct": round(annualized_return(r) * 100, 2),
        "annualized_vol_pct": round(annualized_volatility(r) * 100, 2),
        "sharpe": round(sharpe_ratio(r, risk_free_rate), 4),
        "sortino": round(sortino_ratio(r, risk_free_rate), 4),
        "max_drawdown_pct": round(max_drawdown(v) * 100, 2),
        "calmar": round(calmar_ratio(r, v), 4),
        "win_rate": round(win_rate(r), 4),
        "profit_factor": round(profit_factor(r), 4),
        "var_95": round(value_at_risk(r, 0.95) * 100, 4),
    }
