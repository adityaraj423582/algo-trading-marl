"""
Master evaluation script.  Runs the complete pipeline:

1. Table 1 -- Volatility model comparison (Stage 1)
2. Table 2 -- Trading strategy comparison (Stage 2)
3. Table 3 -- Statistical significance tests

Usage
-----
::

    python -m src.evaluation.run_full_backtest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import BacktestEngine
from src.evaluation.volatility_backtest import VolatilityBacktest
from src.evaluation.metrics import diebold_mariano_test
from src.utils.config import TABLES_DIR, RANDOM_SEED, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
np.random.seed(RANDOM_SEED)

_CFG = get_config()


def run_full_evaluation(tickers=None):
    """Execute all three evaluation tables and print results."""

    if tickers is None:
        tickers = _CFG.marl.quick_test_tickers if _CFG.marl.quick_test else None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("  FULL SYSTEM EVALUATION")
    print("=" * 75)

    # ------------------------------------------------------------------
    # Table 1: Volatility model comparison
    # ------------------------------------------------------------------
    print("\n[1/3] Volatility model comparison ...")
    vol_bt = VolatilityBacktest(tickers=tickers)
    vol_results = vol_bt.compare_volatility_models()

    t1_path = TABLES_DIR / "table1_volatility_comparison.csv"
    vol_results.to_csv(t1_path, index=False)

    # Show averages only
    avg = vol_results[vol_results["Ticker"] == "AVERAGE"].copy()
    avg = avg.sort_values("QLIKE").reset_index(drop=True)
    print("\n  Table 1: Volatility Model Comparison (Average across tickers)")
    print("  " + "-" * 65)
    print("  {:18s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Model", "RMSE", "MAE", "QLIKE", "R2"))
    print("  " + "-" * 65)
    for _, r in avg.iterrows():
        print("  {:18s} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(
            r["Model"], r["RMSE"], r["MAE"], r["QLIKE"], r["R2"]))
    print("  " + "-" * 65)
    print(f"  Saved -> {t1_path}")

    best_vol_model = avg.iloc[0]["Model"]
    best_vol_qlike = avg.iloc[0]["QLIKE"]
    har_qlike = avg[avg["Model"] == "HAR-RV"]["QLIKE"].values
    har_qlike = har_qlike[0] if len(har_qlike) else np.nan

    # ------------------------------------------------------------------
    # Table 2: Trading strategy comparison
    # ------------------------------------------------------------------
    print("\n[2/3] Trading strategy backtesting ...")
    bt = BacktestEngine(tickers=tickers)
    strat_results = bt.compare_all_strategies()

    t2_path = TABLES_DIR / "table2_strategy_comparison.csv"
    strat_results.to_csv(t2_path, index=False)

    print("\n  Table 2: Trading Strategy Comparison (Test Period 2024)")
    print("  " + "-" * 100)
    print("  {:25s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Strategy", "Return%", "Sharpe", "Sortino", "MaxDD%", "Calmar", "WinRate", "PF"))
    print("  " + "-" * 100)
    for _, r in strat_results.iterrows():
        print("  {:25s} {:8.2f} {:8.4f} {:8.4f} {:8.2f} {:8.4f} {:8.4f} {:8.4f}".format(
            r["strategy"],
            r["total_return_pct"],
            r["sharpe"],
            r["sortino"],
            r["max_drawdown_pct"],
            r["calmar"],
            r["win_rate"],
            r["profit_factor"],
        ))
    print("  " + "-" * 100)
    print(f"  Saved -> {t2_path}")

    best_strategy = strat_results.iloc[0]["strategy"]
    our_sharpe = strat_results[strat_results["strategy"] == "CNN-GARCH + MARL"]["sharpe"].values
    our_sharpe = our_sharpe[0] if len(our_sharpe) else 0
    bh_sharpe = strat_results[strat_results["strategy"] == "Buy & Hold"]["sharpe"].values
    bh_sharpe = bh_sharpe[0] if len(bh_sharpe) else 0

    our_ret = strat_results[strat_results["strategy"] == "CNN-GARCH + MARL"]["total_return_pct"].values
    our_ret = our_ret[0] if len(our_ret) else 0
    bh_ret = strat_results[strat_results["strategy"] == "Buy & Hold"]["total_return_pct"].values
    bh_ret = bh_ret[0] if len(bh_ret) else 0

    # ------------------------------------------------------------------
    # Table 3: Significance tests
    # ------------------------------------------------------------------
    print("\n[3/3] Statistical significance tests ...")
    sig_rows = []

    # DM tests from volatility backtest
    dm_results = vol_bt.dm_tests_all_models()
    if not dm_results.empty:
        dm_path = TABLES_DIR / "table3_significance_tests.csv"
        dm_results.to_csv(dm_path, index=False)

        print("\n  Table 3: Diebold-Mariano Tests (vs HAR-RV)")
        print("  " + "-" * 70)
        print("  {:16s} {:16s} {:>10s} {:>10s} {:>8s}".format(
            "Ticker", "Model", "DM_stat", "p_value", "Sig?"))
        print("  " + "-" * 70)
        for _, r in dm_results.iterrows():
            sig_str = "YES *" if r.get("significant", False) else "NO"
            print("  {:16s} {:16s} {:10.4f} {:10.4f} {:>8s}".format(
                r["Ticker"], r["Model_1"], r["dm_stat"], r["p_value"], sig_str))
        print("  " + "-" * 70)
        print(f"  Saved -> {dm_path}")

    # Returns significance: our system vs buy-and-hold
    results_dict = bt.get_strategy_results()
    if "CNN-GARCH + MARL" in results_dict and "Buy & Hold" in results_dict:
        our_rets = results_dict["CNN-GARCH + MARL"]["returns"]
        bh_rets = results_dict["Buy & Hold"]["returns"]
        ml = min(len(our_rets), len(bh_rets))
        if ml >= 20:
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(our_rets[:ml], bh_rets[:ml])
            sig_rows.append({
                "Test": "t-test (returns)",
                "Comparison": "CNN-GARCH+MARL vs Buy&Hold",
                "Statistic": round(t_stat, 4),
                "p_value": round(p_val, 4),
                "Significant": "YES" if p_val < 0.05 else "NO",
            })
            print(f"\n  Returns t-test: stat={t_stat:.4f}  p={p_val:.4f}  "
                  f"{'SIGNIFICANT' if p_val < 0.05 else 'not significant'}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    cnn_beat_har = "YES" if (not np.isnan(best_vol_qlike) and
                             not np.isnan(har_qlike) and
                             best_vol_model == "CNN-GARCH") else "NO"
    system_beat_bh = "YES" if our_sharpe > bh_sharpe else "NO"

    print("\n" + "=" * 75)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 75)
    print(f"  Best volatility model:       {best_vol_model} (QLIKE={best_vol_qlike:.4f})")
    if not np.isnan(har_qlike) and har_qlike > 0 and best_vol_qlike < har_qlike:
        pct = (har_qlike - best_vol_qlike) / har_qlike * 100
        print(f"  CNN-GARCH vs HAR-RV:         {pct:+.1f}% improvement")
    else:
        print(f"  CNN-GARCH vs HAR-RV:         Quick test only (full training pending)")
    print(f"  CNN beat HAR-RV?             {cnn_beat_har}")
    print()
    print(f"  Best trading strategy:       {best_strategy}")
    print(f"  Our system return:           {our_ret:.2f}%")
    print(f"  Buy-and-hold return:         {bh_ret:.2f}%")
    print(f"  Our system Sharpe:           {our_sharpe:.4f}")
    print(f"  Buy-and-hold Sharpe:         {bh_sharpe:.4f}")
    print(f"  System beat buy-and-hold?    {system_beat_bh}")
    print("=" * 75 + "\n")

    return {
        "vol_results": vol_results,
        "strat_results": strat_results,
        "dm_results": dm_results if not dm_results.empty else None,
    }


if __name__ == "__main__":
    run_full_evaluation()
