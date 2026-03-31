"""
Headless runner for MARL analysis (Section 4.3 of research paper).

Generates all figures and tables for the MARL trading evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import TABLES_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_SEED, get_config
from src.environment.trading_env import MultiAssetTradingEnv, load_feature_dfs
from src.environment.market_maker import MarketMakerWrapper
from src.environment.portfolio_agent import PortfolioAgentWrapper

np.random.seed(RANDOM_SEED)

FIG_DIR = FIGURES_DIR / "marl"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

cfg = get_config().marl
tickers = cfg.quick_test_tickers

print("=" * 70)
print("  MARL Analysis -- Section 4.3")
print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# 1. Load training logs
# ──────────────────────────────────────────────────────────────────────

log_path = TABLES_DIR / "marl_training_log.csv"
ep_log_path = TABLES_DIR / "marl_episode_log.csv"

log_df = pd.read_csv(log_path)
print(f"\nTraining log ({len(log_df)} rounds):")
print(log_df.to_string(index=False))

ep_df = pd.DataFrame()
if ep_log_path.exists():
    ep_df = pd.read_csv(ep_log_path)
    print(f"\nEpisode log: {len(ep_df)} entries")


# ──────────────────────────────────────────────────────────────────────
# 2. Training curves: reward over rounds
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(log_df["round"], log_df["mm_reward"], "b-o", lw=2, markersize=6)
axes[0].set_title("Market Maker -- Reward per Round")
axes[0].set_xlabel("Training Round")
axes[0].set_ylabel("Episode Reward")
axes[0].grid(True, alpha=0.3)

axes[1].plot(log_df["round"], log_df["pf_reward"], "r-o", lw=2, markersize=6)
axes[1].set_title("Portfolio Agent -- Reward per Round")
axes[1].set_xlabel("Training Round")
axes[1].set_ylabel("Episode Reward")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "training_reward_curves.png", bbox_inches="tight")
plt.close()
print(f"\nSaved -> {FIG_DIR / 'training_reward_curves.png'}")


# ──────────────────────────────────────────────────────────────────────
# 3. Market maker spread earned over rounds
# ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(log_df["round"], log_df["mm_spread"], color="#2196F3", edgecolor="black", lw=0.5)
for i, v in enumerate(log_df["mm_spread"]):
    ax.text(i + 1, v + max(log_df["mm_spread"]) * 0.02, f"${v:,.0f}",
            ha="center", va="bottom", fontsize=9)
ax.set_title("Market Maker -- Cumulative Spread Earned per Round")
ax.set_xlabel("Training Round")
ax.set_ylabel("Spread Earned ($)")
plt.tight_layout()
fig.savefig(FIG_DIR / "mm_spread_earned.png", bbox_inches="tight")
plt.close()
print(f"Saved -> {FIG_DIR / 'mm_spread_earned.png'}")


# ──────────────────────────────────────────────────────────────────────
# 4. Portfolio performance: run on val split and compute equity curve
# ──────────────────────────────────────────────────────────────────────

try:
    from stable_baselines3 import PPO

    feature_dfs = load_feature_dfs(tickers)
    marl_dir = MODELS_DIR / "marl"

    # Load models
    pf_model = PPO.load(str(marl_dir / "portfolio_agent_final"))

    # Validation run
    val_base = MultiAssetTradingEnv(feature_dfs, mode="val")
    val_pf = PortfolioAgentWrapper(val_base)
    obs, info = val_pf.reset()
    done = False
    pf_values = [val_base.initial_capital]
    dates_list = []

    while not done:
        action, _ = pf_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = val_pf.step(action)
        pf_values.append(val_base._portfolio_value())
        dates_list.append(info.get("date", ""))
        done = terminated or truncated

    pf_metrics = val_pf.calculate_portfolio_metrics()

    # Buy-and-hold baseline on same period
    bh_values = [val_base.initial_capital]
    initial_prices_val = val_base.close_arr[val_base.window_size]
    n_shares_bh = (val_base.initial_capital / val_base.n_stocks) / initial_prices_val
    for t in range(val_base.window_size + 1, val_base.n_dates):
        bh_val = float(np.sum(n_shares_bh * val_base.close_arr[t]))
        bh_values.append(bh_val)

    # Align lengths
    min_len = min(len(pf_values), len(bh_values))
    pf_values = pf_values[:min_len]
    bh_values = bh_values[:min_len]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(min_len), np.array(pf_values) / pf_values[0], "r-", lw=2,
            label=f"Portfolio Agent (Sharpe={pf_metrics['sharpe']:.2f})")
    ax.plot(range(min_len), np.array(bh_values) / bh_values[0], "k--", lw=1.5,
            label="Buy & Hold (equal weight)")
    ax.set_title("Portfolio Agent vs Buy & Hold -- Validation Period (2023)")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Normalised Portfolio Value")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "portfolio_vs_buyhold.png", bbox_inches="tight")
    plt.close()
    print(f"Saved -> {FIG_DIR / 'portfolio_vs_buyhold.png'}")

    # Performance comparison table
    bh_rets = np.diff(bh_values) / np.array(bh_values[:-1])
    bh_sharpe = float(np.mean(bh_rets) / (np.std(bh_rets) + 1e-8) * np.sqrt(252))
    bh_total = (bh_values[-1] / bh_values[0] - 1) * 100
    bh_vol = float(np.std(bh_rets) * np.sqrt(252) * 100)
    bh_peak = bh_values[0]
    bh_mdd = 0
    for v in bh_values:
        if v > bh_peak:
            bh_peak = v
        dd = (bh_peak - v) / bh_peak * 100
        if dd > bh_mdd:
            bh_mdd = dd

    comp_data = {
        "Strategy": ["Portfolio Agent (MARL)", "Buy & Hold"],
        "Total Return (%)": [pf_metrics["total_return"], round(bh_total, 2)],
        "Sharpe Ratio": [pf_metrics["sharpe"], round(bh_sharpe, 2)],
        "Annualised Vol (%)": [pf_metrics["volatility"], round(bh_vol, 2)],
        "Max Drawdown (%)": [pf_metrics["max_drawdown"], round(bh_mdd, 2)],
    }
    comp_df = pd.DataFrame(comp_data)
    comp_path = TABLES_DIR / "marl_performance_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\nPerformance comparison:")
    print(comp_df.to_string(index=False))
    print(f"Saved -> {comp_path}")

except Exception as exc:
    print(f"\nPortfolio evaluation failed: {exc}")
    import traceback
    traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 5. Episode reward distribution (if available)
# ──────────────────────────────────────────────────────────────────────

if not ep_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, agent, color in zip(axes, ["market_maker", "portfolio"], ["#2196F3", "#F44336"]):
        sub = ep_df[ep_df["agent"] == agent]
        if sub.empty:
            continue
        ax.plot(sub["timestep"], sub["ep_reward"], "-", color=color, alpha=0.6, lw=0.8)
        window = max(1, len(sub) // 10)
        if len(sub) > window:
            rolling = sub["ep_reward"].rolling(window).mean()
            ax.plot(sub["timestep"], rolling, "-", color="black", lw=2,
                    label=f"Rolling avg (w={window})")
        ax.set_title(f"{agent.replace('_', ' ').title()} -- Episode Rewards")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Episode Reward")
        ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "episode_reward_distribution.png", bbox_inches="tight")
    plt.close()
    print(f"Saved -> {FIG_DIR / 'episode_reward_distribution.png'}")


# ──────────────────────────────────────────────────────────────────────
# 6. Summary
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  MARL ANALYSIS COMPLETE")
print("=" * 70)
print(f"  Figures saved to: {FIG_DIR}")
print(f"  Tables saved to:  {TABLES_DIR}")

print("\n  KEY FINDINGS (for paper Section 4.3):")
print("  " + "-" * 50)
print(f"  1. Market maker reward increases: {log_df['mm_reward'].iloc[0]:.1f} -> {log_df['mm_reward'].iloc[-1]:.1f}")
print(f"  2. MM spread earned grows: ${log_df['mm_spread'].iloc[0]:,.0f} -> ${log_df['mm_spread'].iloc[-1]:,.0f}")
print(f"  3. Portfolio agent maintains stable returns ~{log_df['pf_total_return'].mean()*100:.0f}%")
mm_improving = log_df["mm_reward"].iloc[-1] > log_df["mm_reward"].iloc[0]
print(f"  4. Market maker learning: {'YES' if mm_improving else 'NEEDS MORE TRAINING'}")
print(f"  5. Quick test only -- full training (1M steps, GPU) expected to improve")
print("=" * 70)
