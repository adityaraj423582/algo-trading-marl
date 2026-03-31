<p align="center">
  <h1 align="center">Algorithmic Trading with CNN-GARCH & Multi-Agent RL</h1>
  <p align="center">
    <em>Evidence from NSE (India) and NASDAQ (US) Equity Markets</em>
  </p>
  <p align="center">
    <a href="#key-results"><img src="https://img.shields.io/badge/Sharpe_Ratio-1.06-brightgreen?style=for-the-badge" alt="Sharpe"></a>
    <a href="#key-results"><img src="https://img.shields.io/badge/Return-18.37%25-blue?style=for-the-badge" alt="Return"></a>
    <a href="#key-results"><img src="https://img.shields.io/badge/Max_Drawdown-9.15%25-orange?style=for-the-badge" alt="MDD"></a>
    <a href="#key-results"><img src="https://img.shields.io/badge/Beats_Buy_%26_Hold-%E2%9C%94-success?style=for-the-badge" alt="Beat"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-4B8BBE?logo=openai&logoColor=white" alt="SB3">
    <img src="https://img.shields.io/badge/arch-GARCH-9B59B6" alt="arch">
    <img src="https://img.shields.io/badge/License-Academic-lightgrey" alt="License">
  </p>
</p>

---

## What is this?

A **two-stage algorithmic trading system** that first forecasts market volatility using deep learning, then uses those forecasts to trade profitably with reinforcement learning agents.

```
  Historical Data ──> CNN-GARCH Model ──> Volatility Signals ──> MARL Agents ──> Trading Decisions
       (NSE + NASDAQ)     (Stage 1)          (rv, regime)          (Stage 2)       (buy/sell/hold)
```

| Stage | What it does | How |
|:---:|---|---|
| **Stage 1** | Forecasts next-day & next-week realized volatility | CNN-GARCH hybrid (PyTorch) trained on 23 engineered features |
| **Stage 2** | Trades stocks using volatility signals | Two RL agents (market maker + portfolio manager) via PPO |

> **Target:** PhD admission -- IIT Bombay Citadel Securities Quantitative Research Lab (2026)

---

## Key Results

### Trading Performance (Test Period: 2024)

| Strategy | Return | Sharpe | Max Drawdown | Calmar |
|:---|:---:|:---:|:---:|:---:|
| **CNN-GARCH + MARL (ours)** | **18.37%** | **1.062** | **9.15%** | **2.415** |
| Buy & Hold | 16.21% | 0.767 | 12.55% | 1.393 |
| Equal Weight Monthly | 15.19% | 0.705 | 13.10% | 1.250 |
| HAR-RV Signal | 11.55% | 0.638 | 9.65% | 1.288 |
| GARCH Signal | 5.74% | 0.142 | 10.57% | 0.584 |

**Our system outperforms buy-and-hold by +38.5% on Sharpe ratio with 27% less drawdown.**

### Volatility Forecasting (QLIKE Loss -- lower is better)

| Model | QLIKE | vs Best Baseline |
|:---|:---:|:---:|
| HAR-RV | 0.4701 | -- |
| CNN-GARCH | 0.4966 | +5.6% (quick test*) |
| EGARCH | 0.5611 | +19.4% |
| GARCH(1,1) | 0.5704 | +21.3% |
| GJR-GARCH | 0.5758 | +22.5% |

*\*Quick test on 2 tickers / 50 epochs. Full training (20 tickers / 200 epochs on GPU) expected to beat HAR-RV.*

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │           STAGE 1: FORECASTING          │
                          │                                         │
  20 Stocks ──> Features  │  ┌──────────┐    ┌──────────────────┐  │
  (NSE+NASDAQ)  (23 cols) │  │ GARCH    │───>│ Conditional Vol  │  │
                          │  │ Family   │    │ + Std Residuals  │  │
                          │  └──────────┘    └────────┬─────────┘  │
                          │                           │            │
                          │  ┌──────────┐    ┌────────▼─────────┐  │
                          │  │ 1D-CNN   │───>│ rv_1d, rv_5d     │──│──> Volatility
                          │  │ (PyTorch)│    │ forecasts        │  │    Signals
                          │  └──────────┘    └──────────────────┘  │
                          └─────────────────────────────────────────┘
                                              │
                          ┌───────────────────▼─────────────────────┐
                          │           STAGE 2: TRADING               │
                          │                                         │
                          │  ┌──────────────┐  ┌─────────────────┐  │
                          │  │ Market Maker │  │ Portfolio Agent  │  │
                          │  │ (bid/ask)    │  │ (weights)        │  │
                          │  │ PPO Agent    │  │ PPO Agent        │  │
                          │  └──────┬───────┘  └────────┬────────┘  │
                          │         └──────────┬────────┘           │
                          │                    ▼                    │
                          │            Trading Decisions            │
                          └─────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/adityaraj423582/algo-trading-marl.git
cd algo-trading-marl

# Install
pip install -r requirements.txt

# Run the full pipeline (quick test mode: 2 stocks, ~10 min total)
python -m src.data.downloader                    # 1. Download data
python -m src.data.preprocessor                  # 2. Clean & engineer features
python -m src.models.garch_baseline_runner       # 3. Fit GARCH baselines
python -m src.training.train_cnn_garch           # 4. Train CNN-GARCH
python -m src.training.train_marl                # 5. Train MARL agents
python -m src.evaluation.run_full_backtest       # 6. Backtest everything
```

---

## Project Structure

```
algo_trading_marl/
├── src/
│   ├── data/                # Download, preprocess, feature engineering
│   ├── models/              # GARCH family, CNN, CNN-GARCH hybrid, signal generator
│   ├── environment/         # Gymnasium trading env, market maker, portfolio agent
│   ├── training/            # CNN-GARCH & MARL training orchestrators
│   ├── evaluation/          # Backtesting engine, 10 metrics, 6 strategies
│   └── utils/               # Config (all hyperparams), logging
├── notebooks/               # Analysis scripts generating paper figures
├── results/
│   ├── figures/             # 45 plots (EDA, GARCH, CNN, MARL, final)
│   └── tables/              # 15 CSV result tables (paper-ready)
├── requirements.txt
├── PROJECT_JOURNAL.md       # Detailed step-by-step research log
└── PROGRESS.md              # Technical progress tracker
```

---

## Data

| Market | Stocks | Period | Source |
|:---|:---|:---|:---|
| **NSE** (India) | RELIANCE, TCS, HDFC Bank, Infosys, ICICI Bank, HUL, SBI, Bajaj Finance, Bharti Airtel, Kotak Bank | 2020--2024 | yfinance |
| **NASDAQ** (US) | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO, COST, ASML | 2020--2024 | yfinance |

**50,860 rows** of OHLCV data across 40 files. **23 engineered features** per ticker including HAR-RV, Parkinson & Garman-Klass volatility, momentum, and regime indicators.

---

## Features & Methods

<table>
<tr><td width="50%">

### Volatility Forecasting
- GARCH(1,1)-t, EGARCH, GJR-GARCH
- HAR-RV (Corsi 2009) baseline
- 1D-CNN (3 conv layers, 72K params)
- Custom loss: 0.5 MSE + 0.5 QLIKE
- Walk-forward rolling evaluation
- Diebold-Mariano significance tests

</td><td width="50%">

### Multi-Agent Trading
- Gymnasium environment (daily steps)
- Market maker: bid/ask quoting, inventory mgmt
- Portfolio agent: weight allocation via softmax
- MAPPO: alternating PPO training rounds
- Sharpe-inspired reward with turnover penalty
- 6 baselines for fair comparison

</td></tr>
</table>

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Deep Learning | PyTorch 2.x |
| GARCH Models | `arch` library |
| Reinforcement Learning | Stable-Baselines3 (PPO) |
| Environment | Gymnasium |
| Data | yfinance, pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Statistical Tests | SciPy, statsmodels |

---

## Publication Targets

- **ICAIF 2026** -- ACM International Conference on AI in Finance
- **NeurIPS 2026** -- Finance Workshop
- **Quantitative Finance** -- Taylor & Francis Journal

---

## Author

**Aditya Raj Singh**
M.Tech, Applied Mathematics & Scientific Computing -- IIT Roorkee

---

## Citation

```bibtex
@misc{singh2026algotrading,
  author = {Singh, Aditya Raj},
  title  = {Algorithmic Trading Using Time Series Volatility Signals
            and Multi-Agent Reinforcement Learning: Evidence from NSE and NASDAQ},
  year   = {2026},
  url    = {https://github.com/adityaraj423582/algo-trading-marl}
}
```

---

<p align="center">
  <sub>Built with research rigor and production-quality code. All results are reproducible with seed=42.</sub>
</p>
