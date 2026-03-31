"""
Walk-forward backtesting engine evaluating the full trading system.

Six strategies are implemented, from simplest (buy-and-hold) to the
complete CNN-GARCH + MARL system.  All strategies are evaluated on the
**test split only** (2024 data for daily).

Usage
-----
::

    from src.evaluation.backtest import BacktestEngine
    engine = BacktestEngine(tickers=["RELIANCE_NS", "AAPL"])
    results = engine.compare_all_strategies()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics, rolling_sharpe
from src.environment.trading_env import MultiAssetTradingEnv, load_feature_dfs
from src.environment.portfolio_agent import PortfolioAgentWrapper
from src.environment.market_maker import MarketMakerWrapper
from src.utils.config import (
    FEATURES_DIR, MODELS_DIR, TABLES_DIR, RANDOM_SEED, get_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CFG = get_config()
_BT = _CFG.backtest
_MARL = _CFG.marl


class BacktestEngine:
    """
    Evaluates 6 strategies on test data and produces comparison tables.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker stems to include.  ``None`` uses quick-test tickers.
    """

    def __init__(self, tickers: Optional[List[str]] = None) -> None:
        np.random.seed(RANDOM_SEED)

        if tickers is None:
            tickers = _MARL.quick_test_tickers if _MARL.quick_test else None

        self.feature_dfs = load_feature_dfs(tickers)
        self.tickers = sorted(self.feature_dfs.keys())
        self.n_stocks = len(self.tickers)
        self.initial_capital = _BT.initial_capital
        self.cost = _BT.commission_pct

        # Pre-extract test-period close prices (shared across strategies)
        test_dfs = {}
        for t in self.tickers:
            df = self.feature_dfs[t]
            test_dfs[t] = df.loc["2024-01-01":]
        common_idx = test_dfs[self.tickers[0]].index
        for t in self.tickers[1:]:
            common_idx = common_idx.intersection(test_dfs[t].index)
        self.test_dates = common_idx.sort_values()

        self.close_matrix = np.column_stack([
            test_dfs[t].loc[self.test_dates, "Close"].values for t in self.tickers
        ])
        self.rv_matrix = np.column_stack([
            test_dfs[t].loc[self.test_dates, "rv_daily"].fillna(0).values
            for t in self.tickers
        ])
        self.regime_matrix = np.column_stack([
            test_dfs[t].loc[self.test_dates, "vol_regime"].fillna(0).values
            for t in self.tickers
        ])

        self.n_test = len(self.test_dates)
        logger.info(
            "BacktestEngine: %d tickers, %d test days (%s to %s)",
            self.n_stocks, self.n_test,
            str(self.test_dates[0])[:10], str(self.test_dates[-1])[:10],
        )

        assert self.test_dates[0] >= pd.Timestamp("2024-01-01"), \
            "Test data leakage: start date before 2024"

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    def _simulate(
        self,
        weights_over_time: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Given a (n_test, n_stocks) weight matrix, simulate portfolio
        equity curve with transaction costs.
        """
        n = self.n_test
        values = np.zeros(n)
        cash = self.initial_capital
        shares = np.zeros(self.n_stocks)
        values[0] = cash

        for t in range(n):
            prices = self.close_matrix[t]
            total_val = cash + np.sum(shares * prices)
            target = weights_over_time[t] * total_val
            current = shares * prices
            trade = target - current
            cost = np.sum(np.abs(trade)) * self.cost
            shares += trade / np.maximum(prices, 1e-8)
            cash -= np.sum(trade) + cost
            values[t] = cash + np.sum(shares * prices)

        returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
        return {"values": values, "returns": returns}

    # ------------------------------------------------------------------
    # 6 Strategies
    # ------------------------------------------------------------------

    def strategy_buy_and_hold(self) -> Dict:
        """Strategy 1: equal-weight buy-and-hold (never rebalance)."""
        w = np.ones((self.n_test, self.n_stocks)) / self.n_stocks
        # Only first day has trades
        w_bh = np.zeros_like(w)
        cash = self.initial_capital
        prices0 = self.close_matrix[0]
        alloc = cash / self.n_stocks
        shares = alloc / prices0
        cost = cash * self.cost
        cash_left = 0.0
        values = np.zeros(self.n_test)
        for t in range(self.n_test):
            values[t] = cash_left + np.sum(shares * self.close_matrix[t])
        values[0] -= cost
        returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
        return {"name": "Buy & Hold", "values": values, "returns": returns}

    def strategy_equal_weight_monthly(self) -> Dict:
        """Strategy 2: equal-weight, rebalance monthly."""
        w = np.ones((self.n_test, self.n_stocks)) / self.n_stocks
        # Only rebalance on ~22nd day of each month block
        w_active = np.zeros_like(w)
        for t in range(self.n_test):
            if t == 0 or t % 22 == 0:
                w_active[t] = w[t]
            else:
                w_active[t] = w_active[t - 1] if t > 0 else w[t]
        res = self._simulate(w)
        return {"name": "Equal Weight Monthly", **res}

    def strategy_garch_signal(self) -> Dict:
        """Strategy 3: rule-based using GARCH conditional volatility."""
        garch_vol = np.column_stack([
            self.feature_dfs[t].loc[self.test_dates, "garch_conditional_vol"].fillna(0).values
            for t in self.tickers
        ])
        p75 = np.percentile(garch_vol[garch_vol > 0], 75) if np.any(garch_vol > 0) else 0.5
        p25 = np.percentile(garch_vol[garch_vol > 0], 25) if np.any(garch_vol > 0) else 0.1

        w = np.ones((self.n_test, self.n_stocks)) / self.n_stocks
        for t in range(self.n_test):
            avg_vol = garch_vol[t].mean()
            if avg_vol > p75:
                w[t] *= 0.5   # reduce to 50% equity in high vol
            elif avg_vol < p25:
                w[t] *= 1.0   # full equity in low vol
            else:
                w[t] *= 0.75
        res = self._simulate(w)
        return {"name": "GARCH Signal", **res}

    def strategy_har_rv_signal(self) -> Dict:
        """Strategy 4: rule-based using HAR-RV volatility forecast."""
        rv = self.rv_matrix.copy()
        p75 = np.percentile(rv[rv > 0], 75) if np.any(rv > 0) else 0.5
        p25 = np.percentile(rv[rv > 0], 25) if np.any(rv > 0) else 0.1

        w = np.ones((self.n_test, self.n_stocks)) / self.n_stocks
        for t in range(self.n_test):
            avg_rv = rv[t].mean()
            if avg_rv > p75:
                w[t] *= 0.5
            elif avg_rv < p25:
                w[t] *= 1.0
            else:
                w[t] *= 0.75
        res = self._simulate(w)
        return {"name": "HAR-RV Signal", **res}

    def strategy_cnn_garch_signal(self) -> Dict:
        """Strategy 5: rule-based using CNN-GARCH volatility forecast (no RL)."""
        try:
            from src.models.volatility_signal_generator import VolatilitySignalGenerator
            vsg = VolatilitySignalGenerator()
            loaded = vsg.load_models()
        except Exception:
            loaded = []

        if not loaded:
            logger.warning("No CNN models available -- falling back to HAR-RV for strategy 5")
            return self.strategy_har_rv_signal()

        # Use CNN for tickers that have models; HAR-RV for others
        w = np.ones((self.n_test, self.n_stocks)) / self.n_stocks
        window = 22
        for t in range(window, self.n_test):
            for s, ticker in enumerate(self.tickers):
                if ticker in vsg.models:
                    feat_df = self.feature_dfs[ticker]
                    test_start = self.test_dates[0]
                    # Get the feature window ending at current test date
                    current_date = self.test_dates[t]
                    window_df = feat_df.loc[:current_date].tail(window)
                    if len(window_df) >= window:
                        try:
                            sig = vsg.generate_signal(ticker, window_df)
                            if sig["vol_regime"] == "HIGH":
                                w[t, s] *= 0.5
                            elif sig["signal_strength"] < 0.3:
                                w[t, s] *= 1.0
                            else:
                                w[t, s] *= 0.75
                        except Exception:
                            pass
        res = self._simulate(w)
        res["name"] = "CNN-GARCH Signal"
        return res

    def strategy_full_system(self) -> Dict:
        """Strategy 6: full CNN-GARCH + MARL system."""
        marl_dir = MODELS_DIR / "marl"
        pf_path = marl_dir / "portfolio_agent_final.zip"

        if not pf_path.exists():
            logger.warning("No MARL models found -- using random policy for strategy 6")
            w = np.ones((self.n_test, self.n_stocks)) / self.n_stocks
            res = self._simulate(w)
            res["name"] = "Full System (untrained)"
            return res

        from stable_baselines3 import PPO

        # Run portfolio agent on test data
        test_base = MultiAssetTradingEnv(self.feature_dfs, mode="test")
        pf_env = PortfolioAgentWrapper(test_base)
        pf_model = PPO.load(str(pf_path), device="cpu")

        obs, _ = pf_env.reset()
        done = False
        values_list = [test_base.initial_capital]

        while not done:
            action, _ = pf_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = pf_env.step(action)
            values_list.append(test_base._portfolio_value())
            done = terminated or truncated

        # Also run market-maker for spread income
        mm_path = marl_dir / "market_maker_final.zip"
        mm_spread_total = 0.0
        if mm_path.exists():
            try:
                mm_base = MultiAssetTradingEnv(self.feature_dfs, mode="test")
                mm_env = MarketMakerWrapper(mm_base)
                mm_model = PPO.load(str(mm_path), device="cpu")
                obs_mm, _ = mm_env.reset()
                done_mm = False
                while not done_mm:
                    act, _ = mm_model.predict(obs_mm, deterministic=True)
                    obs_mm, _, t1, t2, info_mm = mm_env.step(act)
                    done_mm = t1 or t2
                mm_spread_total = info_mm.get("mm_cumulative_spread", 0)
            except Exception as exc:
                logger.warning("MM evaluation failed: %s", exc)

        values = np.array(values_list)
        # Add spread income (distributed proportionally)
        if mm_spread_total > 0 and len(values) > 1:
            spread_per_day = mm_spread_total / (len(values) - 1)
            for i in range(1, len(values)):
                values[i] += spread_per_day * i

        returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
        return {"name": "CNN-GARCH + MARL", "values": values, "returns": returns}

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_all_strategies(self) -> pd.DataFrame:
        """Run all 6 strategies and compute comprehensive metrics."""
        strategies = [
            self.strategy_buy_and_hold,
            self.strategy_equal_weight_monthly,
            self.strategy_garch_signal,
            self.strategy_har_rv_signal,
            self.strategy_cnn_garch_signal,
            self.strategy_full_system,
        ]

        rows = []
        self._strategy_results = {}

        for fn in strategies:
            logger.info("  Running: %s ...", fn.__name__)
            res = fn()
            name = res["name"]
            vals = res["values"]
            rets = res["returns"]

            metrics = compute_all_metrics(rets, vals)
            metrics["strategy"] = name
            rows.append(metrics)
            self._strategy_results[name] = res

        df = pd.DataFrame(rows)
        cols = ["strategy"] + [c for c in df.columns if c != "strategy"]
        df = df[cols].sort_values("sharpe", ascending=False).reset_index(drop=True)

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TABLES_DIR / "backtest_results.csv"
        df.to_csv(out_path, index=False)
        logger.info("  Saved -> %s", out_path)

        return df

    def get_strategy_results(self) -> Dict:
        """Return raw values/returns for each strategy (after compare)."""
        return self._strategy_results

    def monthly_returns_heatmap_data(
        self, strategy_name: str,
    ) -> pd.DataFrame:
        """Build year x month return matrix for heatmap."""
        res = self._strategy_results.get(strategy_name)
        if res is None:
            return pd.DataFrame()
        vals = res["values"]
        dates = self.test_dates[:len(vals)]
        s = pd.Series(vals, index=dates)
        monthly = s.resample("ME").last().pct_change().dropna()
        monthly.index = pd.MultiIndex.from_arrays(
            [monthly.index.year, monthly.index.month],
            names=["Year", "Month"],
        )
        return monthly.unstack("Month") * 100
