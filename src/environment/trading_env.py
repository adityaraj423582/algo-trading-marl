"""
Multi-asset trading environment for MARL agents.

Simulates daily equity trading across *n* stocks, providing each agent
with price data, volatility signals from the CNN-GARCH pipeline, and
portfolio state.  Two agents (market maker and portfolio rebalancer) can
trade simultaneously within the same market.

Design
------
- One ``step`` = one trading day.
- Historical prices loaded from ``data/features/{market}/``.
- Volatility signals computed from the feature window at each step
  using the trained CNN-GARCH ``VolatilitySignalGenerator``, with a
  HAR-RV fallback for tickers without a trained CNN.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.utils.config import FEATURES_DIR, RANDOM_SEED, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CFG = get_config().marl


class MultiAssetTradingEnv(gym.Env):
    """
    Gymnasium environment for multi-asset daily trading.

    Parameters
    ----------
    feature_dfs : dict[str, pd.DataFrame]
        ``{ticker_stem: feature_df}`` with OHLCV + engineered features.
    mode : str
        ``'train'``, ``'val'``, or ``'test'`` -- selects the date split.
    initial_capital : float
        Starting cash for the agent.
    transaction_cost : float
        Proportional cost applied to each trade (e.g. 0.001 = 10 bps).
    max_steps : int
        Maximum steps per episode (default 252 = 1 trading year).
    """

    metadata = {"render_modes": ["human"]}

    # Date boundaries (matching preprocessor splits)
    _SPLIT_BOUNDS = {
        "train": (None, "2022-12-31"),
        "val":   ("2023-01-01", "2023-12-31"),
        "test":  ("2024-01-01", None),
    }

    def __init__(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        mode: str = "train",
        initial_capital: float = _CFG.initial_capital,
        transaction_cost: float = _CFG.transaction_cost,
        max_steps: int = _CFG.max_episode_steps,
        window_size: int = 22,
    ) -> None:
        super().__init__()

        self.tickers = sorted(feature_dfs.keys())
        self.n_stocks = len(self.tickers)
        self.mode = mode
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps
        self.window_size = window_size

        # Slice each feature DF to the requested split
        lo, hi = self._SPLIT_BOUNDS[mode]
        self.feature_dfs: Dict[str, pd.DataFrame] = {}
        for t in self.tickers:
            df = feature_dfs[t].copy()
            if lo is not None:
                df = df.loc[lo:]
            if hi is not None:
                df = df.loc[:hi]
            self.feature_dfs[t] = df

        # Shared date index (intersection of all tickers)
        common_idx = self.feature_dfs[self.tickers[0]].index
        for t in self.tickers[1:]:
            common_idx = common_idx.intersection(self.feature_dfs[t].index)
        self.dates = common_idx.sort_values()
        self.n_dates = len(self.dates)

        if self.n_dates < self.window_size + 10:
            raise ValueError(
                f"Not enough data in '{mode}' split: {self.n_dates} dates "
                f"(need >= {self.window_size + 10})"
            )

        # Pre-extract price and return arrays for speed
        self.close_arr = np.column_stack([
            self.feature_dfs[t].loc[self.dates, "Close"].values
            for t in self.tickers
        ])  # (n_dates, n_stocks)
        self.log_ret_arr = np.column_stack([
            self.feature_dfs[t].loc[self.dates, "log_return"].values
            for t in self.tickers
        ])

        # Volatility features (from feature CSVs, not live CNN)
        self.rv_arr = np.column_stack([
            self.feature_dfs[t].loc[self.dates, "rv_daily"].fillna(0).values
            for t in self.tickers
        ])
        self.vol_regime_arr = np.column_stack([
            self.feature_dfs[t].loc[self.dates, "vol_regime"].fillna(0).values
            for t in self.tickers
        ])
        rv_max = np.nanmax(self.rv_arr, axis=0, keepdims=True)
        rv_max[rv_max < 1e-8] = 1.0
        self.signal_strength_arr = self.rv_arr / rv_max  # 0-1 normalised

        # Observation / action spaces (flat vector for SB3 compatibility)
        # obs = [prices_norm, log_rets, rv, vol_regime, signal_strength,
        #        portfolio_weights, cash_ratio, time_features]
        self.obs_dim = (self.n_stocks * 5   # prices, returns, rv, regime, signal
                        + self.n_stocks     # portfolio weights
                        + 1                 # cash ratio
                        + 3)                # time features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        # Default action space -- overridden by wrappers
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_stocks,), dtype=np.float32,
        )

        # State variables (set in reset)
        self._step_idx = 0
        self.cash = initial_capital
        self.shares = np.zeros(self.n_stocks, dtype=np.float64)
        self.portfolio_values: List[float] = []
        self.initial_prices = np.ones(self.n_stocks)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed or RANDOM_SEED)
        np.random.seed(seed or RANDOM_SEED)

        self._step_idx = self.window_size  # skip warm-up
        self.cash = self.initial_capital
        self.shares = np.zeros(self.n_stocks, dtype=np.float64)
        self.portfolio_values = [self.initial_capital]

        self.initial_prices = self.close_arr[self._step_idx].copy()
        self.initial_prices[self.initial_prices < 1e-8] = 1.0

        obs = self._get_observation()
        info = {"portfolio_value": self.initial_capital, "step": 0}
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one trading day.  Reward computed by wrapper."""
        prev_value = self._portfolio_value()
        prices = self.close_arr[self._step_idx]

        # Default: interpret action as target dollar weights -> trade
        target_weights = self._softmax(action)
        total = self._portfolio_value()
        target_dollars = target_weights * total
        current_dollars = self.shares * prices
        trade_dollars = target_dollars - current_dollars
        cost = np.sum(np.abs(trade_dollars)) * self.transaction_cost

        self.shares += trade_dollars / np.maximum(prices, 1e-8)
        self.cash -= np.sum(trade_dollars) + cost

        # Advance time
        self._step_idx += 1
        new_value = self._portfolio_value()
        self.portfolio_values.append(new_value)

        daily_return = (new_value - prev_value) / max(prev_value, 1e-8)
        reward = float(daily_return)

        terminated = self._step_idx >= self.n_dates - 1
        truncated = (self._step_idx - self.window_size) >= self.max_steps

        obs = self._get_observation()
        info = {
            "portfolio_value": new_value,
            "daily_return": daily_return,
            "total_return": (new_value / self.initial_capital) - 1.0,
            "transaction_cost": cost,
            "step": self._step_idx - self.window_size,
            "date": str(self.dates[min(self._step_idx, self.n_dates - 1)])[:10],
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Build flat observation vector for current step."""
        idx = min(self._step_idx, self.n_dates - 1)
        prices_norm = self.close_arr[idx] / self.initial_prices
        log_rets = self.log_ret_arr[idx]
        rv = self.rv_arr[idx]
        regime = self.vol_regime_arr[idx]
        sig_str = self.signal_strength_arr[idx]

        total = self._portfolio_value()
        stock_values = self.shares * self.close_arr[idx]
        weights = stock_values / max(total, 1e-8)
        cash_ratio = np.array([self.cash / max(total, 1e-8)])

        # Time features: day_of_week/4, month/11, progress
        dt = self.dates[idx]
        day_of_week = dt.dayofweek / 4.0
        month_frac = (dt.month - 1) / 11.0
        progress = (idx - self.window_size) / max(self.n_dates - self.window_size - 1, 1)
        time_feat = np.array([day_of_week, month_frac, progress])

        obs = np.concatenate([
            prices_norm, log_rets, rv, regime, sig_str,
            weights, cash_ratio, time_feat,
        ]).astype(np.float32)

        # Replace any NaN / Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _portfolio_value(self) -> float:
        idx = min(self._step_idx, self.n_dates - 1)
        stock_value = np.sum(self.shares * self.close_arr[idx])
        return float(self.cash + stock_value)

    def _max_drawdown(self) -> float:
        if len(self.portfolio_values) < 2:
            return 0.0
        peak = self.portfolio_values[0]
        mdd = 0.0
        for v in self.portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
        return mdd

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def render(self, mode: str = "human") -> None:
        step = self._step_idx - self.window_size
        if step % 10 == 0:
            val = self._portfolio_value()
            ret = (val / self.initial_capital - 1.0) * 100
            logger.info(
                "  Step %3d | Date %s | Value $%,.0f | Return %+.2f%%",
                step,
                str(self.dates[min(self._step_idx, self.n_dates - 1)])[:10],
                val, ret,
            )


# ======================================================================
# Utility: load feature data for env construction
# ======================================================================

def load_feature_dfs(
    tickers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load feature CSVs for the specified tickers.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker stems (e.g. ``["RELIANCE_NS", "AAPL"]``).
        If None, loads all available.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    out = {}
    for sub in ["nse", "nasdaq"]:
        d = FEATURES_DIR / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*_features.csv")):
            stem = p.stem.replace("_features", "")
            if tickers is not None and stem not in tickers:
                continue
            df = pd.read_csv(p, index_col="Datetime", parse_dates=True)
            out[stem] = df

    logger.info("Loaded feature data for %d tickers", len(out))
    return out
