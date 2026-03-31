"""
Portfolio rebalancing agent wrapper for the multi-asset trading environment.

The portfolio agent decides daily target weights across *n* stocks plus
cash, then the environment executes the rebalancing trades with
transaction costs.

Action space:
    Box(0, 1, shape=(n_stocks + 1,)) -- raw logits passed through softmax
    to produce target weights  [w_stock1, w_stock2, ..., w_cash].

Reward:
    Sharpe-inspired daily reward penalised for variance and turnover.
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


class PortfolioAgentWrapper(gym.Env):
    """
    Gymnasium wrapper that exposes portfolio-rebalancing semantics.

    Parameters
    ----------
    env : MultiAssetTradingEnv
        Shared multi-asset trading environment.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: MultiAssetTradingEnv) -> None:
        super().__init__()
        self.env = env
        self.n_stocks = env.n_stocks

        # Action: logits for (n_stocks + 1) assets (stocks + cash)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_stocks + 1,), dtype=np.float32,
        )

        # Observation: base env obs + portfolio-specific (weights, returns, mdd)
        extra_dim = (self.n_stocks + 1) + 3  # current weights, ret_1d, ret_5d, mdd
        self.obs_dim = env.obs_dim + extra_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )

        self.weights = np.zeros(self.n_stocks + 1, dtype=np.float64)
        self.weights[-1] = 1.0  # start 100% cash
        self.return_history: list = []
        self.prev_weights = self.weights.copy()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        base_obs, info = self.env.reset(seed=seed or RANDOM_SEED)
        self.weights = np.zeros(self.n_stocks + 1, dtype=np.float64)
        self.weights[-1] = 1.0
        self.return_history = []
        self.prev_weights = self.weights.copy()
        obs = self._build_obs(base_obs)
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Normalise action to valid weights via softmax
        target_weights = self._softmax(np.asarray(action, dtype=np.float64))

        # Clamp max single-stock weight
        stock_weights = target_weights[:self.n_stocks]
        stock_weights = np.minimum(stock_weights, _CFG.portfolio_max_weight)
        cash_leftover = 1.0 - stock_weights.sum()
        target_weights = np.append(stock_weights, max(cash_leftover, 0.0))
        target_weights /= target_weights.sum()  # renormalise

        self.prev_weights = self.weights.copy()

        # Convert stock weights to base env action (exclude cash weight)
        base_action = target_weights[:self.n_stocks].astype(np.float32)
        base_obs, _, terminated, truncated, info = self.env.step(base_action)

        # Update tracked weights from actual portfolio
        total = self.env._portfolio_value()
        if total > 1e-8:
            idx = min(self.env._step_idx, self.env.n_dates - 1)
            prices = self.env.close_arr[idx]
            stock_vals = self.env.shares * prices
            self.weights[:self.n_stocks] = stock_vals / total
            self.weights[-1] = self.env.cash / total
        else:
            self.weights[:] = 0.0
            self.weights[-1] = 1.0

        # Reward computation
        daily_ret = info.get("daily_return", 0.0)
        self.return_history.append(daily_ret)

        # Sharpe component (rolling 22-day)
        if len(self.return_history) >= 5:
            window = self.return_history[-min(22, len(self.return_history)):]
            roll_std = float(np.std(window)) + 1e-8
            sharpe_comp = daily_ret / roll_std
        else:
            sharpe_comp = daily_ret * 10.0  # raw return, scaled

        # Turnover penalty
        turnover = float(np.sum(np.abs(self.weights - self.prev_weights)))
        cost_penalty = -self.env.transaction_cost * turnover

        # Variance penalty
        var_penalty = -_CFG.portfolio_variance_penalty * (daily_ret ** 2)

        reward = float(sharpe_comp + cost_penalty + var_penalty)

        info["pf_sharpe_component"] = sharpe_comp
        info["pf_turnover"] = turnover
        info["pf_weights"] = self.weights.copy()
        info["pf_max_drawdown"] = self.env._max_drawdown()

        obs = self._build_obs(base_obs)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(self, base_obs: np.ndarray) -> np.ndarray:
        ret_1d = np.array([self.return_history[-1] if self.return_history else 0.0])
        ret_5d = np.array([
            sum(self.return_history[-5:]) if len(self.return_history) >= 5 else 0.0
        ])
        mdd = np.array([self.env._max_drawdown()])

        extra = np.concatenate([self.weights, ret_1d, ret_5d, mdd])
        obs = np.concatenate([base_obs, extra]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Compute summary metrics for the episode so far."""
        if len(self.return_history) < 2:
            return {"sharpe": 0.0, "total_return": 0.0, "volatility": 0.0, "max_drawdown": 0.0}

        rets = np.array(self.return_history)
        total_ret = float(np.prod(1.0 + rets) - 1.0)
        ann_vol = float(np.std(rets) * np.sqrt(252))
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252))

        return {
            "sharpe": round(sharpe, 4),
            "total_return": round(total_ret * 100, 2),
            "volatility": round(ann_vol * 100, 2),
            "max_drawdown": round(self.env._max_drawdown() * 100, 2),
        }

    def render(self, mode: str = "human") -> None:
        self.env.render(mode)
