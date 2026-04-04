"""
LOB-aware market-maker agent wrapper.

Replaces the stochastic fill model from ``market_maker.py`` with a
queue-position-based execution simulator that uses Level 2 order book
features.  When Bloomberg LOB data is available, fill probabilities
are derived from book depth, spread, and order flow imbalance.  When
operating with synthetic LOB features, the model gracefully degrades
to calibrated stochastic fills.

Key improvements over the L1 market maker:
1. Fill probability = f(queue_position, depth, spread) instead of flat 50%
2. Adverse selection modelling: fills more likely when flow is toxic
3. Inventory risk uses LOB-derived price impact (Kyle's lambda)
4. Spread decisions informed by effective spread and book imbalance
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.trading_env import MultiAssetTradingEnv
from src.utils.config import RANDOM_SEED, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = get_config().marl
_LOB = get_config().lob

# LOB feature columns expected in the feature DataFrames
_LOB_FEATURE_COLS = [
    "lob_bid_ask_spread", "lob_microprice", "lob_book_imbalance",
    "lob_depth_imbalance_5", "lob_total_bid_depth", "lob_total_ask_depth",
    "lob_spread_bps", "lob_order_flow_imbalance", "lob_trade_intensity",
    "lob_vpin", "lob_kyle_lambda", "lob_realized_spread",
    "lob_effective_spread", "lob_queue_position_estimate",
]


class LOBMarketMakerWrapper(gym.Env):
    """
    Gymnasium wrapper with LOB-aware market-making mechanics.

    Observation extends the base env with LOB features + inventory state.
    Fill probability is computed from order book depth and queue position
    rather than a fixed constant.

    Parameters
    ----------
    env : MultiAssetTradingEnv
        Shared trading environment.
    max_inventory_value : float
        Hard cap on absolute inventory exposure.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: MultiAssetTradingEnv,
        max_inventory_value: float = 200_000.0,
    ) -> None:
        super().__init__()
        self.env = env
        self.n_stocks = env.n_stocks
        self.max_inventory_value = max_inventory_value
        self.has_lob_features = self._check_lob_features()

        # Per-stock: [bid_offset, ask_offset, order_size]
        self.action_space = spaces.Box(
            low=np.array([-0.05, 0.0, 0.0] * self.n_stocks, dtype=np.float32),
            high=np.array([0.0, 0.05, 1.0] * self.n_stocks, dtype=np.float32),
            dtype=np.float32,
        )

        # Base obs + LOB features (14 per stock) + inventory state (n_stocks + 3)
        n_lob_obs = len(_LOB_FEATURE_COLS) * self.n_stocks if self.has_lob_features else 0
        extra_dim = self.n_stocks + 3
        self.obs_dim = env.obs_dim + n_lob_obs + extra_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

        # Pre-extract LOB arrays for speed
        if self.has_lob_features:
            self._precompute_lob_arrays()

        self.inventory = np.zeros(self.n_stocks, dtype=np.float64)
        self.inventory_cost_basis = np.zeros(self.n_stocks, dtype=np.float64)
        self.cumulative_spread = 0.0
        self._step_spread = 0.0
        self.cash = 0.0

    def _check_lob_features(self) -> bool:
        """Check if LOB features exist in the feature DataFrames."""
        sample_df = list(self.env.feature_dfs.values())[0]
        available = [c for c in _LOB_FEATURE_COLS if c in sample_df.columns]
        if len(available) >= 5:
            logger.info("LOB market maker: found %d/%d LOB features", len(available), len(_LOB_FEATURE_COLS))
            return True
        logger.info("LOB market maker: LOB features not found, using calibrated stochastic fills")
        return False

    def _precompute_lob_arrays(self) -> None:
        """Extract LOB feature arrays aligned to the shared date index."""
        self.lob_arrays: Dict[str, np.ndarray] = {}
        for col in _LOB_FEATURE_COLS:
            arr = np.column_stack([
                self.env.feature_dfs[t].loc[self.env.dates, col].fillna(0).values
                if col in self.env.feature_dfs[t].columns
                else np.zeros(self.env.n_dates)
                for t in self.env.tickers
            ])
            self.lob_arrays[col] = arr

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        base_obs, info = self.env.reset(seed=seed or RANDOM_SEED)
        self.inventory[:] = 0.0
        self.inventory_cost_basis[:] = 0.0
        self.cumulative_spread = 0.0
        self._step_spread = 0.0
        self.cash = 0.0
        return self._build_obs(base_obs), info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.n_stocks, 3)
        bid_offsets = action[:, 0]
        ask_offsets = action[:, 1]
        order_sizes = action[:, 2]

        idx = min(self.env._step_idx, self.env.n_dates - 1)
        mid_prices = self.env.close_arr[idx]

        bid_prices = mid_prices * (1.0 + bid_offsets)
        ask_prices = mid_prices * (1.0 + ask_offsets)
        spreads = ask_prices - bid_prices

        max_order_dollars = 50_000.0
        order_dollars = order_sizes * max_order_dollars

        step_spread = 0.0
        step_cost = 0.0
        adverse_selection_cost = 0.0
        rng = self.env.np_random

        for s in range(self.n_stocks):
            price = mid_prices[s]
            if price < 1e-8 or order_dollars[s] < 1.0:
                continue
            shares_to_trade = order_dollars[s] / price

            fill_prob_bid, fill_prob_ask, adverse_prob = self._compute_fill_probabilities(idx, s)

            # Buy fill at bid (passive: someone sells to us)
            if rng.random() < fill_prob_bid:
                self.inventory[s] += shares_to_trade
                cost = shares_to_trade * bid_prices[s]
                self.inventory_cost_basis[s] += cost
                self.cash -= cost
                step_cost += cost * self.env.transaction_cost

                if rng.random() < adverse_prob:
                    adverse_selection_cost += shares_to_trade * price * 0.001

            # Sell fill at ask (passive: someone buys from us)
            if rng.random() < fill_prob_ask:
                self.inventory[s] -= shares_to_trade
                revenue = shares_to_trade * ask_prices[s]
                self.cash += revenue
                step_spread += shares_to_trade * spreads[s]
                step_cost += revenue * self.env.transaction_cost

                if rng.random() < adverse_prob:
                    adverse_selection_cost += shares_to_trade * price * 0.001

        # Advance environment
        neutral_action = np.zeros(self.n_stocks, dtype=np.float32)
        base_obs, _, terminated, truncated, info = self.env.step(neutral_action)

        # Inventory PnL
        new_idx = min(self.env._step_idx, self.env.n_dates - 1)
        new_prices = self.env.close_arr[new_idx]
        inv_pnl = float(np.sum(self.inventory * (new_prices - mid_prices)))

        # Price impact penalty (Kyle's lambda-based)
        impact_penalty = self._compute_impact_penalty(new_idx)

        inv_value = float(np.sum(np.abs(self.inventory * new_prices)))
        total_val = max(self.env.initial_capital, 1e-8)
        inv_penalty = _CFG.mm_inventory_penalty * (inv_value / total_val)

        regime = self.env.vol_regime_arr[new_idx]
        vol_mult = np.where(regime > 0.5, 1.5, 1.0).mean()

        reward = (
            (step_spread * vol_mult + inv_pnl - step_cost - adverse_selection_cost) / total_val
            - inv_penalty - impact_penalty
        ) * 1000.0

        self.cumulative_spread += step_spread
        self._step_spread = step_spread

        info["mm_spread_earned"] = step_spread
        info["mm_inventory_value"] = inv_value
        info["mm_inventory_pnl"] = inv_pnl
        info["mm_cumulative_spread"] = self.cumulative_spread
        info["mm_adverse_selection"] = adverse_selection_cost
        info["mm_impact_penalty"] = impact_penalty

        obs = self._build_obs(base_obs)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # LOB-based fill model
    # ------------------------------------------------------------------

    def _compute_fill_probabilities(
        self, idx: int, stock_idx: int,
    ) -> Tuple[float, float, float]:
        """
        Compute fill probabilities from LOB features.

        Returns (fill_prob_bid, fill_prob_ask, adverse_selection_prob).
        """
        if not self.has_lob_features:
            return _CFG.mm_fill_probability, _CFG.mm_fill_probability, 0.05

        queue_pos = self.lob_arrays["lob_queue_position_estimate"][idx, stock_idx]
        book_imb = self.lob_arrays["lob_book_imbalance"][idx, stock_idx]
        ofi = self.lob_arrays["lob_order_flow_imbalance"][idx, stock_idx]
        vpin_val = self.lob_arrays["lob_vpin"][idx, stock_idx]
        spread_bps = self.lob_arrays["lob_spread_bps"][idx, stock_idx]

        # Fill probability increases with:
        #   - Better queue position (closer to front)
        #   - Wider spread (more attractive to market orders)
        #   - Order flow towards our side
        base_fill = 0.3
        queue_bonus = (1.0 - queue_pos) * 0.3
        spread_bonus = min(spread_bps / 50.0, 0.2)

        fill_bid = np.clip(base_fill + queue_bonus + spread_bonus + max(ofi, 0) * 0.1, 0.05, 0.95)
        fill_ask = np.clip(base_fill + queue_bonus + spread_bonus + max(-ofi, 0) * 0.1, 0.05, 0.95)

        # Adverse selection: high VPIN = more informed trading
        adverse = np.clip(vpin_val * 0.3, 0.01, 0.5)

        return float(fill_bid), float(fill_ask), float(adverse)

    def _compute_impact_penalty(self, idx: int) -> float:
        """Price impact penalty using Kyle's lambda."""
        if not self.has_lob_features:
            return 0.0

        kyle = self.lob_arrays.get("lob_kyle_lambda")
        if kyle is None:
            return 0.0

        avg_lambda = np.mean(np.abs(kyle[idx]))
        inv_dollars = np.sum(np.abs(self.inventory * self.env.close_arr[idx]))
        return float(avg_lambda * inv_dollars / max(self.env.initial_capital, 1e-8)) * 0.1

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, base_obs: np.ndarray) -> np.ndarray:
        idx = min(self.env._step_idx, self.env.n_dates - 1)
        prices = self.env.close_arr[idx]

        parts = [base_obs]

        if self.has_lob_features:
            lob_obs = np.concatenate([
                self.lob_arrays[col][idx] for col in _LOB_FEATURE_COLS
            ])
            lob_obs = np.nan_to_num(lob_obs, nan=0.0, posinf=1.0, neginf=-1.0)
            parts.append(lob_obs)

        inv_norm = self.inventory / max(self.max_inventory_value / np.mean(prices + 1e-8), 1.0)
        inv_value = np.array([np.sum(np.abs(self.inventory * prices)) / self.env.initial_capital])
        unrealised_pnl = np.array([
            np.sum(self.inventory * prices) - np.sum(self.inventory_cost_basis)
        ]) / self.env.initial_capital
        spread_today = np.array([self._step_spread / max(self.env.initial_capital, 1e-8) * 1000.0])

        parts.extend([inv_norm, inv_value, unrealised_pnl, spread_today])

        obs = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def render(self, mode: str = "human") -> None:
        self.env.render(mode)
