# Algorithmic Trading Using Time Series Volatility Signals and Multi-Agent Reinforcement Learning

**Evidence from NSE and NASDAQ**

## Overview

This project implements a two-stage algorithmic trading system that combines advanced volatility forecasting with multi-agent reinforcement learning (MARL):

1. **Stage 1 — Volatility Forecasting:** A CNN-GARCH hybrid model processes high-frequency time series data from NSE (India) and NASDAQ (US) to forecast short-term (5-minute ahead) and medium-term (1-day ahead) realized volatility.

2. **Stage 2 — Multi-Agent RL Trading:** Volatility forecasts serve as signals for two cooperative MARL agents:
   - **Market Making Agent** — quotes bid/ask prices, earns the spread, and manages inventory risk.
   - **Portfolio Rebalancing Agent** — dynamically allocates weights across a basket of equities.

Both agents are trained to maximize risk-adjusted PnL using Proximal Policy Optimization (PPO).

## Project Structure

```
algo_trading_marl/
├── data/               # Raw, processed, and engineered feature data
├── models/             # Saved model checkpoints (GARCH, CNN, MARL)
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/
│   ├── data/           # Data download, cleaning, feature engineering
│   ├── models/         # GARCH, CNN, CNN-GARCH model definitions
│   ├── environment/    # Gym trading environments and agent logic
│   ├── training/       # Training scripts for both stages
│   ├── evaluation/     # Backtesting engine and performance metrics
│   └── utils/          # Config, logging, and helper utilities
├── results/            # Figures, tables, and checkpoints
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd algo_trading_marl
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Copy `.env` and fill in your API keys:

```bash
cp .env .env.local
```

### 4. Download data

```bash
python -m src.data.downloader                # both markets
python -m src.data.downloader --market nse   # NSE only
python -m src.data.downloader --market nasdaq # NASDAQ only
```

## Data

| Market  | Tickers | Interval | Period |
|---------|---------|----------|--------|
| NSE     | Top 10 NIFTY 50 constituents | 1 hour | 2020–2024 |
| NASDAQ  | Top 10 NASDAQ 100 constituents | 1 hour | 2020–2024 |

## Methodology

### Volatility Forecasting (Stage 1)

- **GARCH(1,1)** with Student-t innovations as baseline
- **1-D CNN** feature extractor on windowed returns
- **CNN-GARCH hybrid** combining learned features with conditional variance

### MARL Trading (Stage 2)

- Custom OpenAI Gymnasium environment with realistic market microstructure
- Two cooperative agents trained via PPO (Stable-Baselines3)
- Reward: risk-adjusted PnL with transaction cost penalties

## Evaluation

- Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio
- RMSE / MAE for volatility forecasts
- Comparison against buy-and-hold, GARCH-only, and single-agent baselines

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{singh2026algotrading,
  author = {Singh, Aditya Raj},
  title  = {Algorithmic Trading Using Time Series Volatility Signals
            and Multi-Agent Reinforcement Learning},
  year   = {2026},
}
```

## License

This project is for academic research purposes. See LICENSE for details.
