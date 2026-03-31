"""
MARL training orchestrator using alternating PPO (MAPPO-style).

Two agents share the same market but have independent policy networks:
  1. **MarketMaker** -- earns spread, manages inventory
  2. **PortfolioAgent** -- rebalances weights, maximises risk-adjusted return

Training follows an *alternating* schedule: in each round the market
maker is trained for ``T/N`` timesteps while the portfolio agent acts
with its *current* policy (and vice versa).  This approximates
centralised-training--decentralised-execution (CTDE).

Usage
-----
::

    python -m src.training.train_marl            # quick test (10k steps)
    python -m src.training.train_marl --full     # full training (1M steps)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.trading_env import MultiAssetTradingEnv, load_feature_dfs
from src.environment.market_maker import MarketMakerWrapper
from src.environment.portfolio_agent import PortfolioAgentWrapper
from src.utils.config import MODELS_DIR, TABLES_DIR, RANDOM_SEED, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = get_config().marl


# ======================================================================
# Logging callback
# ======================================================================

class MARLLogCallback(BaseCallback):
    """Lightweight callback that records episode metrics to a list."""

    def __init__(self, agent_name: str, log_list: list, verbose: int = 0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.log_list = log_list

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.log_list.append({
                    "agent": self.agent_name,
                    "timestep": self.num_timesteps,
                    "ep_reward": info["episode"]["r"],
                    "ep_length": info["episode"]["l"],
                })
        return True


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_agent(
    agent_model: PPO,
    env: object,
    n_episodes: int = 1,
) -> Dict[str, float]:
    """Run deterministic evaluation episodes and return mean metrics."""
    rewards_all, lengths_all = [], []
    infos_last = {}

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action, _ = agent_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated
        rewards_all.append(ep_reward)
        lengths_all.append(ep_len)
        infos_last = info

    return {
        "mean_reward": float(np.mean(rewards_all)),
        "mean_length": float(np.mean(lengths_all)),
        **{k: v for k, v in infos_last.items() if isinstance(v, (int, float))},
    }


# ======================================================================
# Main training loop
# ======================================================================

def train_marl(
    tickers: Optional[List[str]] = None,
    total_timesteps: Optional[int] = None,
    n_rounds: Optional[int] = None,
) -> pd.DataFrame:
    """
    Train market-maker and portfolio agents in alternating rounds.

    Returns training log DataFrame.
    """
    np.random.seed(RANDOM_SEED)
    cfg = _CFG

    if tickers is None:
        tickers = cfg.quick_test_tickers if cfg.quick_test else None
    if total_timesteps is None:
        total_timesteps = cfg.total_timesteps_quick if cfg.quick_test else cfg.total_timesteps_full
    if n_rounds is None:
        n_rounds = cfg.n_alternating_rounds

    logger.info("\n" + "=" * 70)
    logger.info("  MARL TRAINING -- %s mode", "QUICK_TEST" if cfg.quick_test else "FULL")
    logger.info("  Tickers: %s", tickers)
    logger.info("  Total timesteps: %d  |  Alternating rounds: %d", total_timesteps, n_rounds)
    logger.info("=" * 70 + "\n")

    # Load data
    feature_dfs = load_feature_dfs(tickers)
    if not feature_dfs:
        raise RuntimeError("No feature data found")

    # Create environments
    train_base_mm = MultiAssetTradingEnv(feature_dfs, mode="train")
    train_base_pf = MultiAssetTradingEnv(feature_dfs, mode="train")

    mm_env = MarketMakerWrapper(train_base_mm)
    pf_env = PortfolioAgentWrapper(train_base_pf)

    logger.info(
        "  Environment created: %d stocks, %d train dates",
        train_base_mm.n_stocks, train_base_mm.n_dates,
    )
    logger.info("  MM obs_dim=%d  action_dim=%d", mm_env.obs_dim, mm_env.action_space.shape[0])
    logger.info("  PF obs_dim=%d  action_dim=%d", pf_env.obs_dim, pf_env.action_space.shape[0])

    # Build PPO agents
    steps_per_round = max(total_timesteps // n_rounds, 256)

    mm_model = PPO(
        "MlpPolicy", mm_env,
        learning_rate=cfg.learning_rate,
        n_steps=min(cfg.n_steps, steps_per_round),
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.entropy_coef,
        vf_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        seed=RANDOM_SEED,
        verbose=0,
    )

    pf_model = PPO(
        "MlpPolicy", pf_env,
        learning_rate=cfg.learning_rate,
        n_steps=min(cfg.n_steps, steps_per_round),
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.entropy_coef,
        vf_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        seed=RANDOM_SEED,
        verbose=0,
    )

    # Training logs
    mm_logs: list = []
    pf_logs: list = []
    all_logs: list = []

    t0 = time.time()

    for rd in range(1, n_rounds + 1):
        logger.info("  Round %d/%d -- training market maker (%d steps) ...",
                     rd, n_rounds, steps_per_round)
        mm_model.learn(
            total_timesteps=steps_per_round,
            callback=MARLLogCallback("market_maker", mm_logs),
            reset_num_timesteps=False,
            progress_bar=False,
        )

        logger.info("  Round %d/%d -- training portfolio agent (%d steps) ...",
                     rd, n_rounds, steps_per_round)
        pf_model.learn(
            total_timesteps=steps_per_round,
            callback=MARLLogCallback("portfolio", pf_logs),
            reset_num_timesteps=False,
            progress_bar=False,
        )

        # Quick evaluation after each round
        mm_eval = evaluate_agent(mm_model, mm_env)
        pf_eval = evaluate_agent(pf_model, pf_env)

        all_logs.append({
            "round": rd,
            "mm_reward": mm_eval["mean_reward"],
            "mm_ep_len": mm_eval["mean_length"],
            "mm_spread": mm_eval.get("mm_cumulative_spread", 0),
            "pf_reward": pf_eval["mean_reward"],
            "pf_ep_len": pf_eval["mean_length"],
            "pf_total_return": pf_eval.get("total_return", 0),
            "pf_max_drawdown": pf_eval.get("pf_max_drawdown", 0),
        })

        logger.info(
            "  Round %d: MM_reward=%.2f  PF_reward=%.2f  PF_ret=%.2f%%  elapsed=%.0fs",
            rd, mm_eval["mean_reward"], pf_eval["mean_reward"],
            pf_eval.get("total_return", 0) * 100,
            time.time() - t0,
        )

    elapsed = time.time() - t0

    # Save models
    marl_dir = MODELS_DIR / "marl"
    marl_dir.mkdir(parents=True, exist_ok=True)
    mm_model.save(str(marl_dir / "market_maker_final"))
    pf_model.save(str(marl_dir / "portfolio_agent_final"))
    logger.info("  Models saved -> %s", marl_dir)

    # Save training log
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(all_logs)
    log_path = TABLES_DIR / "marl_training_log.csv"
    log_df.to_csv(log_path, index=False)
    logger.info("  Training log saved -> %s", log_path)

    # Detailed episode logs
    ep_df = pd.DataFrame(mm_logs + pf_logs)
    if not ep_df.empty:
        ep_path = TABLES_DIR / "marl_episode_log.csv"
        ep_df.to_csv(ep_path, index=False)
        logger.info("  Episode log saved -> %s", ep_path)

    # ---- Validation evaluation ----
    logger.info("\n  Evaluating on VALIDATION split ...")
    try:
        val_base_mm = MultiAssetTradingEnv(feature_dfs, mode="val")
        val_base_pf = MultiAssetTradingEnv(feature_dfs, mode="val")
        val_mm = MarketMakerWrapper(val_base_mm)
        val_pf = PortfolioAgentWrapper(val_base_pf)

        val_mm_res = evaluate_agent(mm_model, val_mm)
        val_pf_res = evaluate_agent(pf_model, val_pf)

        # Portfolio metrics
        pf_metrics = val_pf.calculate_portfolio_metrics()

        logger.info("  Val MM:  reward=%.2f  spread=$%.0f",
                     val_mm_res["mean_reward"],
                     val_mm_res.get("mm_cumulative_spread", 0))
        logger.info("  Val PF:  reward=%.2f  sharpe=%.2f  return=%.1f%%  mdd=%.1f%%",
                     val_pf_res["mean_reward"],
                     pf_metrics["sharpe"],
                     pf_metrics["total_return"],
                     pf_metrics["max_drawdown"])
    except Exception as exc:
        logger.warning("  Validation evaluation failed: %s", exc)
        pf_metrics = {}

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("  MARL TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("  Total time:        %.1f seconds", elapsed)
    logger.info("  Total timesteps:   %d", total_timesteps)
    logger.info("  Rounds:            %d", n_rounds)
    logger.info("  Stocks:            %d", train_base_mm.n_stocks)

    if not log_df.empty:
        logger.info("\n  Round-by-round results:")
        logger.info("  %6s %12s %12s %12s %14s",
                     "Round", "MM_Reward", "PF_Reward", "PF_Ret(%)", "MM_Spread($)")
        logger.info("  " + "-" * 60)
        for _, r in log_df.iterrows():
            logger.info("  %6d %12.2f %12.2f %12.2f %14.0f",
                         r["round"], r["mm_reward"], r["pf_reward"],
                         r["pf_total_return"] * 100, r["mm_spread"])

    logger.info("=" * 70 + "\n")
    return log_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MARL agents.")
    parser.add_argument("--full", action="store_true",
                        help="Full training mode (all tickers, 1M steps).")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    tickers = args.tickers
    total = args.steps
    n_rounds = args.rounds

    if args.full:
        if total is None:
            total = _CFG.total_timesteps_full
        tickers = None  # all

    train_marl(tickers=tickers, total_timesteps=total, n_rounds=n_rounds)


if __name__ == "__main__":
    main()
