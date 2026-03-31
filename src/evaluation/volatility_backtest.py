"""
Stage 1 volatility forecasting evaluation.

Compares all five volatility models (GARCH, EGARCH, GJR-GARCH, HAR-RV,
CNN-GARCH) on the 2024 test set using RMSE, MAE, QLIKE, and R-squared.
Produces Table 1 of the research paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import qlike_loss, diebold_mariano_test
from src.models.garch_model import (
    GARCHFamily, HARRV, evaluate_forecasts,
)
from src.utils.config import FEATURES_DIR, TABLES_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRAIN_END = "2022-12-31"
_VAL_END = "2023-12-31"


class VolatilityBacktest:
    """
    Evaluate all volatility models on the out-of-sample test period.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker stems.  ``None`` loads all available.
    """

    def __init__(self, tickers: Optional[List[str]] = None) -> None:
        np.random.seed(RANDOM_SEED)
        self.feature_dfs: Dict[str, pd.DataFrame] = {}

        for sub in ["nse", "nasdaq"]:
            d = FEATURES_DIR / sub
            if not d.exists():
                continue
            for p in sorted(d.glob("*_features.csv")):
                stem = p.stem.replace("_features", "")
                if tickers is not None and stem not in tickers:
                    continue
                self.feature_dfs[stem] = pd.read_csv(
                    p, index_col="Datetime", parse_dates=True,
                )

        self.tickers = sorted(self.feature_dfs.keys())
        logger.info("VolatilityBacktest: %d tickers loaded", len(self.tickers))

    def compare_volatility_models(self) -> pd.DataFrame:
        """
        Compare GARCH, EGARCH, GJR-GARCH, HAR-RV, and CNN-GARCH on the
        test set.  Returns per-ticker + average rows.
        """
        rows: List[Dict] = []

        for ticker in self.tickers:
            df = self.feature_dfs[ticker]
            train_df = df.loc[:_TRAIN_END]
            test_df = df.loc[_VAL_END:]
            if len(test_df) > 0 and test_df.index[0] <= pd.Timestamp(_VAL_END):
                test_df = test_df.iloc[1:]
            actual = test_df["target_rv_1d"].dropna().values

            if len(actual) < 20:
                continue

            # ---- GARCH family rolling forecasts ----
            for model_type in ["GARCH", "EGARCH", "GJR-GARCH"]:
                try:
                    model = GARCHFamily(model_type=model_type, p=1, q=1, dist="t")
                    preds = model.rolling_forecast(
                        train_df["log_return"], test_df["log_return"], refit_every=22,
                    )
                    pred_vals = preds.dropna().values
                    min_len = min(len(actual), len(pred_vals))
                    if min_len < 20:
                        continue
                    m = evaluate_forecasts(actual[:min_len], pred_vals[:min_len])
                    rows.append({
                        "Ticker": ticker, "Model": model_type,
                        "RMSE": m["RMSE"], "MAE": m["MAE"],
                        "QLIKE": m["QLIKE"], "R2": m["R2"],
                    })
                except Exception as exc:
                    logger.warning("  %s %s failed: %s", ticker, model_type, exc)

            # ---- HAR-RV ----
            try:
                har = HARRV()
                har.fit(train_df)
                har_preds = har.rolling_forecast(train_df, test_df, refit_every=22)
                pred_vals = har_preds.dropna().values
                min_len = min(len(actual), len(pred_vals))
                if min_len >= 20:
                    m = evaluate_forecasts(actual[:min_len], pred_vals[:min_len])
                    rows.append({
                        "Ticker": ticker, "Model": "HAR-RV",
                        "RMSE": m["RMSE"], "MAE": m["MAE"],
                        "QLIKE": m["QLIKE"], "R2": m["R2"],
                    })
            except Exception as exc:
                logger.warning("  %s HAR-RV failed: %s", ticker, exc)

            # ---- CNN-GARCH ----
            try:
                from src.models.cnn_garch import CNNGARCHHybrid, prepare_sequences
                _, _, _, _, X_test, y_test, _ = prepare_sequences(df, window_size=22)
                if len(X_test) >= 20:
                    hybrid = CNNGARCHHybrid(
                        ticker=ticker,
                        n_features=X_test.shape[2],
                        window_size=22,
                    )
                    hybrid.load_checkpoint()
                    preds = hybrid.predict(X_test)
                    m = evaluate_forecasts(y_test[:, 0], preds[:, 0])
                    rows.append({
                        "Ticker": ticker, "Model": "CNN-GARCH",
                        "RMSE": m["RMSE"], "MAE": m["MAE"],
                        "QLIKE": m["QLIKE"], "R2": m["R2"],
                    })
            except Exception as exc:
                logger.debug("  %s CNN-GARCH not available: %s", ticker, exc)

        df_out = pd.DataFrame(rows)
        if df_out.empty:
            logger.error("No volatility results produced")
            return df_out

        # Average row per model
        avg_rows = []
        for model in df_out["Model"].unique():
            sub = df_out[df_out["Model"] == model]
            avg_rows.append({
                "Ticker": "AVERAGE",
                "Model": model,
                "RMSE": sub["RMSE"].mean(),
                "MAE": sub["MAE"].mean(),
                "QLIKE": sub["QLIKE"].mean(),
                "R2": sub["R2"].mean(),
            })

        result = pd.concat([df_out, pd.DataFrame(avg_rows)], ignore_index=True)
        return result

    def cross_market_comparison(self) -> pd.DataFrame:
        """Compare average metrics for NSE vs NASDAQ tickers."""
        full = self.compare_volatility_models()
        if full.empty:
            return full

        full_per_ticker = full[full["Ticker"] != "AVERAGE"]
        nse_tickers = [t for t in self.tickers if t.endswith("_NS")]
        nasdaq_tickers = [t for t in self.tickers if not t.endswith("_NS")]

        rows = []
        for market, tlist in [("NSE", nse_tickers), ("NASDAQ", nasdaq_tickers)]:
            sub = full_per_ticker[full_per_ticker["Ticker"].isin(tlist)]
            for model in sub["Model"].unique():
                m = sub[sub["Model"] == model]
                rows.append({
                    "Market": market, "Model": model,
                    "QLIKE": m["QLIKE"].mean(),
                    "RMSE": m["RMSE"].mean(),
                    "R2": m["R2"].mean(),
                    "N_tickers": len(m),
                })
        return pd.DataFrame(rows)

    def dm_tests_all_models(self) -> pd.DataFrame:
        """Run DM tests comparing each model against HAR-RV baseline."""
        rows = []
        for ticker in self.tickers:
            df = self.feature_dfs[ticker]
            train_df = df.loc[:_TRAIN_END]
            test_df = df.loc[_VAL_END:]
            if len(test_df) > 0 and test_df.index[0] <= pd.Timestamp(_VAL_END):
                test_df = test_df.iloc[1:]
            actual = test_df["target_rv_1d"].dropna().values

            if len(actual) < 30:
                continue

            # HAR-RV baseline forecast
            try:
                har = HARRV()
                har.fit(train_df)
                har_preds = har.rolling_forecast(train_df, test_df, refit_every=22).dropna().values
            except Exception:
                continue

            min_len_har = min(len(actual), len(har_preds))
            if min_len_har < 30:
                continue

            # Test GARCH variants vs HAR-RV
            for mtype in ["GARCH", "EGARCH", "GJR-GARCH"]:
                try:
                    model = GARCHFamily(model_type=mtype, p=1, q=1, dist="t")
                    preds = model.rolling_forecast(
                        train_df["log_return"], test_df["log_return"], refit_every=22,
                    ).dropna().values
                    ml = min(min_len_har, len(preds))
                    if ml < 30:
                        continue
                    dm = diebold_mariano_test(actual[:ml], preds[:ml], har_preds[:ml])
                    rows.append({
                        "Ticker": ticker,
                        "Model_1": mtype,
                        "Model_2": "HAR-RV",
                        **dm,
                    })
                except Exception:
                    pass

            # Test CNN-GARCH vs HAR-RV
            try:
                from src.models.cnn_garch import CNNGARCHHybrid, prepare_sequences
                _, _, _, _, X_test, y_test, _ = prepare_sequences(df, window_size=22)
                if len(X_test) >= 30:
                    hybrid = CNNGARCHHybrid(
                        ticker=ticker, n_features=X_test.shape[2], window_size=22,
                    )
                    hybrid.load_checkpoint()
                    cnn_preds = hybrid.predict(X_test)[:, 0]
                    ml = min(len(cnn_preds), min_len_har)
                    if ml >= 30:
                        dm = diebold_mariano_test(actual[:ml], cnn_preds[:ml], har_preds[:ml])
                        rows.append({
                            "Ticker": ticker,
                            "Model_1": "CNN-GARCH",
                            "Model_2": "HAR-RV",
                            **dm,
                        })
            except Exception:
                pass

        return pd.DataFrame(rows)
