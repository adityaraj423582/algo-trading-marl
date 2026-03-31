"""
Market-maker agent wrapper for the multi-asset trading environment.

The market maker quotes bid/ask prices around the mid-price for each
stock.  It earns the spread when orders are filled and bears inventory
risk when the market moves against its position.

The wrapper overrides ``action_space`` and ``step`` so that the
underlying ``MultiAssetTradingEnv`` can be shared with the portfolio
agent while each sees its own reward semantics.

Action space (per stock):
    [bid_offset, ask_offset, order_size_fraction]
    bid_offset  in [-0.05, 0.0]  -- how far below mid to set bid
    ask_offset  in [ 0.0,  0.05] -- how far above mid to set ask
    order_size  in [ 0.0,  1.0]  -- fraction of max order size

Reward:
    spread_earned * vol_multiplier + inventory_pnl - cost - inventory_penalty
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


class MarketMakerWrapper(gym.Env):
    """
    Gymnasium wrapper that exposes market-maker semantics around a shared
    ``MultiAssetTradingEnv``.

    Parameters
    ----------
    env : MultiAssetTradingEnv
        The shared multi-asset environment instance.
    max_inventory_value : float
        Maximum total dollar inventory (absolute) before hard penalty.
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

        # Per-stock: [bid_offset, ask_offset, order_size]
        self.action_space = spaces.Box(
            low=np.array([-0.05, 0.0, 0.0] * self.n_stocks, dtype=np.float32),
            high=np.array([0.0, 0.05, 1.0] * self.n_stocks, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation = base env obs + inventory info (n_stocks + 3 scalars)
        extra_dim = self.n_stocks + 3  # inventory_shares, inv_value, unreal_pnl, spread_earned_today
        self.obs_dim = env.obs_dim + extra_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

        self.inventory = np.zeros(self.n_stocks, dtype=np.float64)
        self.inventory_cost_basis = np.zeros(self.n_stocks, dtype=np.float64)
        self.cumulative_spread = 0.0
        self._step_spread = 0.0
        self.cash = 0.0

    # ------------------------------------------------------------------
    # Gym API
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
        obs = self._build_obs(base_obs)
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.n_stocks, 3)
        bid_offsets = action[:, 0]   # negative
        ask_offsets = action[:, 1]   # positive
        order_sizes = action[:, 2]   # [0, 1]

        idx = min(self.env._step_idx, self.env.n_dates - 1)
        mid_prices = self.env.close_arr[idx]

        # Compute bid / ask
        bid_prices = mid_prices * (1.0 + bid_offsets)
        ask_prices = mid_prices * (1.0 + ask_offsets)
        spreads = ask_prices - bid_prices

        max_order_dollars = 50_000.0
        order_dollars = order_sizes * max_order_dollars

        step_spread = 0.0
        step_cost = 0.0
        rng = self.env.np_random

        for s in range(self.n_stocks):
            price = mid_prices[s]
            if price < 1e-8 or order_dollars[s] < 1.0:
                continue
            shares_to_trade = order_dollars[s] / price

            # Stochastic fills
            if rng.random() < _CFG.mm_fill_probability:
                # Buy fill at bid
                self.inventory[s] += shares_to_trade
                cost = shares_to_trade * bid_prices[s]
                self.inventory_cost_basis[s] += cost
                self.cash -= cost
                step_cost += cost * self.env.transaction_cost

            if rng.random() < _CFG.mm_fill_probability:
                # Sell fill at ask
                self.inventory[s] -= shares_to_trade
                revenue = shares_to_trade * ask_prices[s]
                self.cash += revenue
                step_spread += shares_to_trade * spreads[s]
                step_cost += revenue * self.env.transaction_cost

        # Advance the base environment (no-op trade: keep portfolio as-is)
        neutral_action = np.zeros(self.n_stocks, dtype=np.float32)
        base_obs, _, terminated, truncated, info = self.env.step(neutral_action)

        # Inventory PnL (mark-to-market change)
        new_idx = min(self.env._step_idx, self.env.n_dates - 1)
        new_prices = self.env.close_arr[new_idx]
        prev_prices = mid_prices
        inv_pnl = float(np.sum(self.inventory * (new_prices - prev_prices)))

        # Inventory penalty
        inv_value = float(np.sum(np.abs(self.inventory * new_prices)))
        total_val = max(self.env.initial_capital, 1e-8)
        inv_penalty = _CFG.mm_inventory_penalty * (inv_value / total_val)

        # Volatility regime multiplier
        regime = self.env.vol_regime_arr[new_idx]
        vol_mult = np.where(regime > 0.5, 1.5, 1.0).mean()

        # Scale reward to be centred around 0, magnitude ~1
        reward = (
            (step_spread * vol_mult + inv_pnl - step_cost) / total_val
            - inv_penalty
        ) * 1000.0  # scale up for PPO

        self.cumulative_spread += step_spread
        self._step_spread = step_spread

        info["mm_spread_earned"] = step_spread
        info["mm_inventory_value"] = inv_value
        info["mm_inventory_pnl"] = inv_pnl
        info["mm_cumulative_spread"] = self.cumulative_spread

        obs = self._build_obs(base_obs)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(self, base_obs: np.ndarray) -> np.ndarray:
        idx = min(self.env._step_idx, self.env.n_dates - 1)
        prices = self.env.close_arr[idx]
        inv_norm = self.inventory / max(self.max_inventory_value / np.mean(prices + 1e-8), 1.0)
        inv_value = np.array([np.sum(np.abs(self.inventory * prices)) / self.env.initial_capital])
        unrealised_pnl = np.array([
            np.sum(self.inventory * prices) - np.sum(self.inventory_cost_basis)
        ]) / self.env.initial_capital
        spread_today = np.array([self._step_spread / max(self.env.initial_capital, 1e-8) * 1000.0])

        extra = np.concatenate([inv_norm, inv_value, unrealised_pnl, spread_today])
        obs = np.concatenate([base_obs, extra]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def render(self, mode: str = "human") -> None:
        self.env.render(mode)
