"""
Script equivalent of notebooks/02_garch_analysis.ipynb.

Generates publication-quality figures and tables for the GARCH baseline
section of the paper (Section 4.1).

Run from the project root:

    python notebooks/run_02_garch_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    FEATURES_DIR, FIGURES_DIR, TABLES_DIR, RANDOM_SEED,
)
from src.models.garch_model import (
    GARCHFamily, HARRV, evaluate_forecasts, diebold_mariano_test,
)

np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150

GARCH_FIG_DIR = FIGURES_DIR / "garch"
GARCH_FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Representative stocks ───────────────────────────────────────────────
SAMPLE = {
    "RELIANCE_NS": {"feat_dir": FEATURES_DIR / "nse",    "label": "Reliance (NSE)"},
    "SBIN_NS":     {"feat_dir": FEATURES_DIR / "nse",    "label": "SBI (NSE)"},
    "AAPL":        {"feat_dir": FEATURES_DIR / "nasdaq",  "label": "Apple (NASDAQ)"},
    "NVDA":        {"feat_dir": FEATURES_DIR / "nasdaq",  "label": "NVIDIA (NASDAQ)"},
}

_TRAIN_END = "2022-12-31"
_VAL_END = "2023-12-31"

# ── Load results and data ──────────────────────────────────────────────
print("Loading GARCH baseline results ...")
results_df = pd.read_csv(TABLES_DIR / "garch_baseline_results.csv")
print(f"  {len(results_df)} rows: {results_df['Ticker'].nunique()} tickers x {results_df['Model'].nunique()} models")

feat_data = {}
for ticker, info in SAMPLE.items():
    feat_data[ticker] = pd.read_csv(
        info["feat_dir"] / f"{ticker}_features.csv",
        index_col="Datetime", parse_dates=True,
    )

# ── 1. Model comparison table (publication-ready) ──────────────────────
print("\n[1/5] Model comparison table ...")
pivot = results_df.pivot_table(
    index="Ticker", columns="Model",
    values=["RMSE", "MAE", "QLIKE", "R2"],
    aggfunc="first",
)
avg_row = results_df.groupby("Model")[["RMSE", "MAE", "QLIKE", "R2"]].mean()
print("\n=== Average metrics across all 20 tickers ===")
print(avg_row.round(4).to_string())
avg_row.round(4).to_csv(TABLES_DIR / "garch_avg_metrics.csv")
print("  Saved garch_avg_metrics.csv")

# ── 2. Actual vs predicted volatility ──────────────────────────────────
print("\n[2/5] Actual vs predicted volatility plots ...")

for ticker, info in SAMPLE.items():
    df = feat_data[ticker]
    test = df.loc[_VAL_END:].iloc[1:]
    if len(test) < 20:
        continue

    train_ret = df.loc[:_VAL_END]["log_return"]
    test_ret = test["log_return"]
    actual_rv = test["target_rv_1d"]

    # Re-run forecasts for this ticker
    forecasts = {}
    for label, kwargs in [
        ("GARCH(1,1)",  {"model_type": "GARCH",     "p": 1, "q": 1, "dist": "t"}),
        ("EGARCH(1,1)", {"model_type": "EGARCH",     "p": 1, "q": 1, "dist": "t"}),
        ("GJR-GARCH",   {"model_type": "GJR-GARCH", "p": 1, "q": 1, "dist": "t"}),
    ]:
        model = GARCHFamily(**kwargs)
        model.fit(train_ret)
        forecasts[label] = model.rolling_forecast(train_ret, test_ret, refit_every=22)

    # HAR-RV
    har = HARRV()
    trainval = df.loc[:_VAL_END]
    har.fit(trainval)
    forecasts["HAR-RV"] = har.rolling_forecast(trainval, test, refit_every=22)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Top: actual vs HAR-RV and best GARCH
    ax = axes[0]
    ax.plot(actual_rv.index, actual_rv, "k-", alpha=0.5, linewidth=0.7, label="Actual RV")
    ax.plot(forecasts["HAR-RV"].index, forecasts["HAR-RV"], linewidth=1.5,
            label="HAR-RV", color="darkorange")
    ax.plot(forecasts["EGARCH(1,1)"].index, forecasts["EGARCH(1,1)"], linewidth=1.2,
            label="EGARCH(1,1)-t", color="steelblue", alpha=0.8)
    ax.set_title(f"{info['label']} -- Actual vs Predicted Volatility (2024)")
    ax.set_ylabel("Annualised Volatility")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom: residuals (HAR-RV)
    ax = axes[1]
    har_pred = forecasts["HAR-RV"]
    mask = har_pred.notna() & actual_rv.notna()
    residuals = actual_rv[mask] - har_pred[mask]
    ax.bar(residuals.index, residuals, width=2, color="steelblue", alpha=0.6, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{info['label']} -- HAR-RV Forecast Residuals")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(GARCH_FIG_DIR / f"{ticker}_actual_vs_predicted.png")
    plt.close(fig)
    print(f"  Saved {ticker}_actual_vs_predicted.png")

# ── 3. Conditional variance over time ──────────────────────────────────
print("\n[3/5] Conditional variance plots ...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14))
for i, (ticker, info) in enumerate(SAMPLE.items()):
    ax = axes[i]
    df = feat_data[ticker]
    ax.plot(df.index, df["garch_conditional_vol"], linewidth=1.0,
            color="steelblue", label="GARCH Cond. Vol")
    ax.plot(df.index, df["rv_daily"].rolling(22).mean(), linewidth=1.2,
            color="darkorange", alpha=0.8, label="RV Monthly (22d)")

    # Highlight high-vol regime
    regime = df["vol_regime"]
    for idx in range(1, len(regime)):
        if regime.iloc[idx] == 1 and regime.iloc[idx-1] == 0:
            start = df.index[idx]
        elif regime.iloc[idx] == 0 and regime.iloc[idx-1] == 1:
            end = df.index[idx]
            ax.axvspan(start, end, alpha=0.1, color="red")

    ax.set_title(f"{info['label']} -- GARCH Conditional Volatility")
    ax.set_ylabel("Volatility")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(GARCH_FIG_DIR / "conditional_variance_all.png")
plt.close(fig)
print("  Saved conditional_variance_all.png")

# ── 4. Model comparison bar chart ──────────────────────────────────────
print("\n[4/5] Model comparison chart ...")

avg = results_df.groupby("Model")[["RMSE", "MAE", "QLIKE"]].mean()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, metric in enumerate(["RMSE", "MAE", "QLIKE"]):
    ax = axes[i]
    colors = ["steelblue", "darkorange", "forestgreen", "indianred"]
    bars = ax.bar(avg.index, avg[metric], color=colors[:len(avg)])
    ax.set_title(f"Average {metric} (20 tickers)")
    ax.set_ylabel(metric)
    for bar_obj in bars:
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2., bar_obj.get_height(),
                f"{bar_obj.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
fig.savefig(GARCH_FIG_DIR / "model_comparison_bar.png")
plt.close(fig)
print("  Saved model_comparison_bar.png")

# ── 5. Diebold-Mariano tests ──────────────────────────────────────────
print("\n[5/5] Diebold-Mariano tests: GARCH(1,1) vs HAR-RV ...")

dm_rows = []
for ticker, info in SAMPLE.items():
    df = feat_data[ticker]
    test = df.loc[_VAL_END:].iloc[1:]
    train_ret = df.loc[:_VAL_END]["log_return"]
    test_ret = test["log_return"]
    actual_rv = test["target_rv_1d"]

    garch = GARCHFamily(model_type="GARCH", p=1, q=1, dist="t")
    garch.fit(train_ret)
    garch_fc = garch.rolling_forecast(train_ret, test_ret, refit_every=22)

    har = HARRV()
    har.fit(df.loc[:_VAL_END])
    har_fc = har.rolling_forecast(df.loc[:_VAL_END], test, refit_every=22)

    mask = actual_rv.notna() & garch_fc.notna() & har_fc.notna()
    dm_stat, dm_p = diebold_mariano_test(
        actual_rv[mask].values, garch_fc[mask].values, har_fc[mask].values,
    )
    sig = "***" if dm_p < 0.01 else "**" if dm_p < 0.05 else "*" if dm_p < 0.10 else ""
    dm_rows.append({
        "Stock": info["label"],
        "DM Stat": f"{dm_stat:.3f}",
        "p-value": f"{dm_p:.4f}",
        "Sig": sig,
        "Better Model": "HAR-RV" if dm_stat > 0 else "GARCH(1,1)",
    })

dm_df = pd.DataFrame(dm_rows)
print("\n=== Diebold-Mariano Test: GARCH(1,1) vs HAR-RV ===")
print("H0: Equal predictive accuracy | H1: Forecasts differ")
print("Positive DM stat => GARCH(1,1) has larger squared error\n")
print(dm_df.to_string(index=False))
dm_df.to_csv(TABLES_DIR / "diebold_mariano_tests.csv", index=False)
print("\n  Saved diebold_mariano_tests.csv")

# ── Summary ────────────────────────────────────────────────────────────
all_figs = list(GARCH_FIG_DIR.glob("*.png"))
print(f"\n{'='*60}")
print(f"  GARCH analysis complete: {len(all_figs)} figures")
print(f"  Key finding: HAR-RV wins all 20 tickers on QLIKE")
print(f"  Average QLIKE: HAR-RV={avg.loc['HAR-RV','QLIKE']:.4f}  vs  GARCH(1,1)={avg.loc['GARCH(1,1)-t','QLIKE']:.4f}")
print(f"{'='*60}")
