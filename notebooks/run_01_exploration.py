"""
Script equivalent of notebooks/01_data_exploration.ipynb.

Generates all exploratory figures and the descriptive statistics table.
Run from the project root:

    python notebooks/run_01_exploration.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAW_NSE_DIR, RAW_NASDAQ_DIR, FEATURES_DIR, FIGURES_DIR,
    TABLES_DIR, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TICKERS = {
    "RELIANCE_NS": {"raw_dir": RAW_NSE_DIR, "feat_dir": FEATURES_DIR / "nse", "label": "Reliance (NSE)"},
    "TCS_NS":      {"raw_dir": RAW_NSE_DIR, "feat_dir": FEATURES_DIR / "nse", "label": "TCS (NSE)"},
    "AAPL":        {"raw_dir": RAW_NASDAQ_DIR, "feat_dir": FEATURES_DIR / "nasdaq", "label": "Apple (NASDAQ)"},
    "NVDA":        {"raw_dir": RAW_NASDAQ_DIR, "feat_dir": FEATURES_DIR / "nasdaq", "label": "NVIDIA (NASDAQ)"},
}

print("Loading data ...")
raw_data, feat_data = {}, {}
for ticker, info in SAMPLE_TICKERS.items():
    raw_data[ticker] = pd.read_csv(info["raw_dir"] / f"{ticker}_1d.csv", index_col="Datetime", parse_dates=True)
    feat_data[ticker] = pd.read_csv(info["feat_dir"] / f"{ticker}_features.csv", index_col="Datetime", parse_dates=True)
    print(f"  {info['label']:20s}  raw={raw_data[ticker].shape}  features={feat_data[ticker].shape}")

# ── 1. Normalised price series ──────────────────────────────────────────
print("\n[1/7] Price series ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ticker in ["RELIANCE_NS", "TCS_NS"]:
    c = raw_data[ticker]["Close"]; axes[0].plot(c / c.iloc[0] * 100, label=SAMPLE_TICKERS[ticker]["label"], lw=1)
axes[0].set_title("NSE -- Normalised Close (base=100)"); axes[0].set_ylabel("Price Index"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
for ticker in ["AAPL", "NVDA"]:
    c = raw_data[ticker]["Close"]; axes[1].plot(c / c.iloc[0] * 100, label=SAMPLE_TICKERS[ticker]["label"], lw=1)
axes[1].set_title("NASDAQ -- Normalised Close (base=100)"); axes[1].set_ylabel("Price Index"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(FIGURES_DIR / "price_series_normalised.png"); plt.close(fig)
print("  Saved price_series_normalised.png")

# ── 2. Log returns ──────────────────────────────────────────────────────
print("[2/7] Log returns ...")
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)
for i, (ticker, info) in enumerate(SAMPLE_TICKERS.items()):
    lr = feat_data[ticker]["log_return"]
    axes[i].bar(lr.index, lr, width=1.5, color="steelblue", alpha=0.7, linewidth=0)
    axes[i].axhline(0, color="black", linewidth=0.5)
    axes[i].set_ylabel("Log Return"); axes[i].set_title(f"{info['label']} -- Daily Log Returns"); axes[i].grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(FIGURES_DIR / "log_returns_daily.png"); plt.close(fig)
print("  Saved log_returns_daily.png")

# ── 3. ACF / PACF of squared returns ────────────────────────────────────
print("[3/7] ACF/PACF of squared returns ...")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
for i, (ticker, info) in enumerate(SAMPLE_TICKERS.items()):
    sq = feat_data[ticker]["log_return"] ** 2
    plot_acf(sq.dropna(), lags=40, ax=axes[i, 0], title=f"{info['label']} -- ACF (r^2)")
    plot_pacf(sq.dropna(), lags=40, ax=axes[i, 1], title=f"{info['label']} -- PACF (r^2)", method="ywm")
plt.tight_layout(); fig.savefig(FIGURES_DIR / "acf_pacf_squared_returns.png"); plt.close(fig)
print("  Saved acf_pacf_squared_returns.png")

# ── 4. Realised volatility (HAR-RV) ─────────────────────────────────────
print("[4/7] Realised volatility ...")
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)
for i, (ticker, info) in enumerate(SAMPLE_TICKERS.items()):
    df = feat_data[ticker]
    axes[i].plot(df.index, df["rv_daily"], alpha=0.4, lw=0.6, label="RV daily", color="steelblue")
    axes[i].plot(df.index, df["rv_weekly"], lw=1.2, label="RV weekly", color="darkorange")
    axes[i].plot(df.index, df["rv_monthly"], lw=1.5, label="RV monthly", color="darkred")
    axes[i].set_title(f"{info['label']} -- Realised Volatility (annualised)"); axes[i].set_ylabel("Vol"); axes[i].legend(loc="upper right", fontsize=9); axes[i].grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(FIGURES_DIR / "realised_volatility_har.png"); plt.close(fig)
print("  Saved realised_volatility_har.png")

# ── 5. Return distribution with fat tails ────────────────────────────────
print("[5/7] Return distributions ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (ticker, info) in enumerate(SAMPLE_TICKERS.items()):
    ax = axes[idx // 2, idx % 2]; lr = feat_data[ticker]["log_return"].dropna()
    ax.hist(lr, bins=80, density=True, alpha=0.6, color="steelblue", edgecolor="white", label="Empirical")
    x = np.linspace(lr.min(), lr.max(), 300)
    ax.plot(x, stats.norm.pdf(x, lr.mean(), lr.std()), "r-", lw=1.5, label="Normal")
    skew, kurt = lr.skew(), lr.kurtosis()
    jb_stat, jb_p = stats.jarque_bera(lr)
    ax.annotate(f"Skew={skew:.3f}\nKurt={kurt:.2f}\nJB={jb_stat:.1f} (p={jb_p:.4f})",
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.set_title(f"{info['label']}"); ax.set_xlabel("Log Return"); ax.legend(fontsize=9)
plt.tight_layout(); fig.savefig(FIGURES_DIR / "return_distribution_fat_tails.png"); plt.close(fig)
print("  Saved return_distribution_fat_tails.png")

# ── 6. Correlation heatmaps ──────────────────────────────────────────────
print("[6/7] Feature correlations ...")
feature_cols = ["log_return", "rv_daily", "rv_weekly", "rv_monthly", "rolling_var_5d", "rolling_var_22d",
                "parkinson_vol", "garman_klass_vol", "intraday_range", "volume_ratio",
                "momentum_5", "momentum_22", "vol_regime", "target_rv_1d", "target_rv_5d"]
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
for idx, (ticker, info) in enumerate(SAMPLE_TICKERS.items()):
    ax = axes[idx // 2, idx % 2]; corr = feat_data[ticker][feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.6}, annot_kws={"size": 7})
    ax.set_title(info["label"], fontsize=12)
plt.suptitle("Feature Correlation Matrices", fontsize=16, y=1.01)
plt.tight_layout(); fig.savefig(FIGURES_DIR / "feature_correlation_all_stocks.png", bbox_inches="tight"); plt.close(fig)
print("  Saved feature_correlation_all_stocks.png")

# ── 7. Descriptive statistics table ──────────────────────────────────────
print("[7/7] Descriptive statistics ...")
rows = []
for ticker, info in SAMPLE_TICKERS.items():
    df = feat_data[ticker]; lr = df["log_return"]
    jb_stat, jb_p = stats.jarque_bera(lr)
    rows.append({
        "Stock": info["label"], "Obs": len(df),
        "Start": str(df.index.min().date()), "End": str(df.index.max().date()),
        "Ann. Return": f"{lr.mean() * 252:.2%}", "Ann. Vol": f"{lr.std() * np.sqrt(252):.2%}",
        "Skewness": f"{lr.skew():.3f}", "Kurtosis": f"{lr.kurtosis():.2f}",
        "Min Daily": f"{lr.min():.4f}", "Max Daily": f"{lr.max():.4f}",
        "Avg RV": f"{df['rv_daily'].mean():.4f}", "Vol Regime %": f"{df['vol_regime'].mean():.1%}",
    })
summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(TABLES_DIR / "descriptive_statistics.csv", index=False)
print(f"\n  Saved descriptive_statistics.csv")

# ── Final ────────────────────────────────────────────────────────────────
all_figs = list(FIGURES_DIR.glob("*.png"))
print(f"\n{'='*60}")
print(f"  Exploration complete: {len(all_figs)} figures in results/figures/")
print(f"{'='*60}")
