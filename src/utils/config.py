"""
Configuration module for the Algorithmic Trading MARL project.

Contains all hyperparameters, file paths, ticker lists, and settings
used across the project. Centralizing configuration here ensures
reproducibility and easy experiment management.

Author: Aditya Raj Singh
Project: Algorithmic Trading Using Time Series Volatility Signals and MARL
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Project paths (all relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"

RAW_NSE_DIR: Path = RAW_DATA_DIR / "nse"
RAW_NASDAQ_DIR: Path = RAW_DATA_DIR / "nasdaq"
PROCESSED_NSE_DIR: Path = PROCESSED_DATA_DIR / "nse"
PROCESSED_NASDAQ_DIR: Path = PROCESSED_DATA_DIR / "nasdaq"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"

LOG_DIR: Path = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Data download settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Configuration for data download and preprocessing."""

    # NSE (NIFTY 50) — top 10 constituents by market cap
    nse_tickers: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "HINDUNILVR.NS",
        "SBIN.NS",
        "BAJFINANCE.NS",
        "BHARTIARTL.NS",
        "KOTAKBANK.NS",
    ])

    # NASDAQ 100 — top 10 constituents by market cap
    nasdaq_tickers: List[str] = field(default_factory=lambda: [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AVGO",
        "COST",
        "ASML",
    ])

    # Download period
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"

    # Yahoo Finance limits:
    #   - 1h interval: last 730 calendar days only
    #   - 1d interval: full history available
    # Strategy: download daily for full 5-year range, hourly for last ~2 years
    intervals: List[str] = field(default_factory=lambda: ["1d", "1h"])

    # Rate-limit handling
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    retry_backoff_factor: float = 2.0

    # Validation thresholds
    max_missing_pct: float = 5.0      # warn if > 5 % rows missing
    min_expected_rows_daily: int = 500    # ~250 trading days/year × 2+ years
    min_expected_rows_hourly: int = 1000  # ~7 bars/day × 250 days

    @property
    def all_tickers(self) -> List[str]:
        return self.nse_tickers + self.nasdaq_tickers


# ---------------------------------------------------------------------------
# GARCH model settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GARCHConfig:
    """Hyperparameters for GARCH-family volatility models."""

    model_type: str = "GARCH"         # GARCH, EGARCH, GJR-GARCH
    p: int = 1
    q: int = 1
    o: int = 0                        # leverage order (for GJR)
    distribution: str = "t"           # Student-t for fat tails
    mean_model: str = "AR"
    mean_lags: int = 1
    forecast_horizon: int = 1


# ---------------------------------------------------------------------------
# CNN feature extractor settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CNNConfig:
    """Hyperparameters for the 1-D CNN volatility feature extractor."""

    window_size: int = 22             # lookback window (22 trading days ~ 1 month)
    n_features: int = 23             # number of input features per timestep
    n_filters_1: int = 64
    n_filters_2: int = 128
    n_filters_3: int = 64
    kernel_1: int = 3
    kernel_2: int = 5
    kernel_3: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20                # early-stopping patience
    loss_alpha: float = 0.5          # weight of MSE in combined MSE+QLIKE loss


# ---------------------------------------------------------------------------
# CNN-GARCH hybrid settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CNNGARCHConfig:
    """Settings for the combined CNN-GARCH hybrid model."""

    cnn: CNNConfig = field(default_factory=CNNConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)

    # Training mode switches
    quick_test: bool = True           # True  = 2 tickers, 50 epochs, CPU
    quick_test_tickers: List[str] = field(default_factory=lambda: [
        "RELIANCE_NS", "AAPL",
    ])
    quick_test_epochs: int = 50


# ---------------------------------------------------------------------------
# MARL training settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MARLConfig:
    """Hyperparameters for multi-agent reinforcement learning."""

    algorithm: str = "MAPPO"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    reward_scale: float = 1.0

    # Timestep budgets
    total_timesteps_quick: int = 10_000
    total_timesteps_full: int = 1_000_000
    n_alternating_rounds: int = 5

    # Quick test mode
    quick_test: bool = True
    quick_test_tickers: List[str] = field(default_factory=lambda: [
        "RELIANCE_NS", "AAPL",
    ])

    # Trading environment
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001       # 0.1% per trade
    max_episode_steps: int = 252          # 1 trading year

    # Market-maker agent
    mm_spread_range: tuple = (0.0001, 0.05)
    mm_fill_probability: float = 0.5
    mm_inventory_penalty: float = 0.001

    # Portfolio agent
    portfolio_max_weight: float = 0.4
    portfolio_variance_penalty: float = 0.1


# ---------------------------------------------------------------------------
# Backtesting settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestConfig:
    """Settings for the backtesting engine."""

    initial_capital: float = 1_000_000.0
    commission_pct: float = 0.001        # 10 bps
    slippage_pct: float = 0.0005         # 5 bps
    risk_free_rate: float = 0.05         # annualised
    benchmark_ticker: str = "^NSEI"      # NIFTY 50 index


# ---------------------------------------------------------------------------
# Experiment tracking (Weights & Biases)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases experiment tracking settings."""

    project: str = "algo-trading-marl"
    entity: Optional[str] = None
    enabled: bool = False                # flip to True when ready


# ---------------------------------------------------------------------------
# Master config aggregator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating all sub-configs."""

    seed: int = RANDOM_SEED
    data: DataConfig = field(default_factory=DataConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    cnn_garch: CNNGARCHConfig = field(default_factory=CNNGARCHConfig)
    marl: MARLConfig = field(default_factory=MARLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def get_config() -> Config:
    """Return the default project configuration."""
    return Config()
