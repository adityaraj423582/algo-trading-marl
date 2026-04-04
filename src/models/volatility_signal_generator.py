"""
Volatility Signal Generator -- bridge between Stage 1 (CNN-GARCH) and Stage 2 (MARL).

Loads trained CNN-GARCH checkpoints and feature scalers, then generates
real-time volatility signals that MARL agents consume as observations.

Usage
-----
::

    from src.models.volatility_signal_generator import VolatilitySignalGenerator

    vsg = VolatilitySignalGenerator()
    vsg.load_models()

    # Single ticker signal
    signal = vsg.generate_signal("RELIANCE_NS", last_22_days_df)

    # All tickers at once
    signals_df = vsg.generate_all_signals({"RELIANCE_NS": df1, "AAPL": df2})
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.models.cnn_model import VolatilityCNN, get_device
from src.utils.config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

CNN_DIR = MODELS_DIR / "cnn"

# Same feature columns used during training
_FEATURE_COLS = [
    "Close", "High", "Low", "Open", "Volume",
    "log_return",
    "rv_daily", "rv_weekly", "rv_monthly",
    "rolling_var_5d", "rolling_var_22d",
    "parkinson_vol", "garman_klass_vol",
    "intraday_range", "volume_ma5", "volume_ratio",
    "momentum_5", "momentum_22",
    "vol_regime",
    "garch_conditional_vol", "garch_std_resid",
    "target_rv_1d", "target_rv_5d",
]


class VolatilitySignalGenerator:
    """
    Loads trained CNN-GARCH models and generates volatility signals.

    Each signal contains:
      - rv_1d_forecast: next-day predicted realised volatility
      - rv_5d_forecast: next-week predicted realised volatility
      - vol_regime: 'HIGH' or 'LOW' (above/below 75th percentile)
      - signal_strength: 0-1 normalised forecast intensity
    """

    def __init__(self, model_dir: Path = CNN_DIR) -> None:
        self.model_dir = model_dir
        self.device = get_device()
        self.models: Dict[str, VolatilityCNN] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.rv_history: Dict[str, List[float]] = {}

    def load_models(self) -> List[str]:
        """
        Scan model_dir for checkpoints and load all available models.

        Returns list of loaded ticker names.
        """
        loaded = []
        for ckpt in sorted(self.model_dir.glob("*_best.pt")):
            ticker = ckpt.stem.replace("_best", "")
            scaler_path = self.model_dir / f"{ticker}_scaler.pkl"

            if not scaler_path.exists():
                logger.warning("Scaler not found for %s, skipping", ticker)
                continue

            # Load scaler
            with open(scaler_path, "rb") as f:
                self.scalers[ticker] = pickle.load(f)

            # Load CNN model
            n_features = len(self.scalers[ticker].mean_)
            model = VolatilityCNN(n_features=n_features, window_size=22)
            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )
            model.to(self.device)
            model.eval()
            self.models[ticker] = model
            loaded.append(ticker)

        logger.info("Loaded %d CNN-GARCH models: %s", len(loaded), loaded)
        return loaded

    def generate_signal(
        self,
        ticker: str,
        feature_window_df: pd.DataFrame,
    ) -> Dict[str, object]:
        """
        Generate a volatility signal for one ticker.

        Parameters
        ----------
        ticker : str
            Ticker identifier matching the checkpoint name.
        feature_window_df : pd.DataFrame
            Last ``window_size`` rows of the feature DataFrame.

        Returns
        -------
        dict with keys: ticker, rv_1d_forecast, rv_5d_forecast,
                        vol_regime, signal_strength
        """
        if ticker not in self.models:
            raise KeyError(f"No model loaded for {ticker}")

        cols = [c for c in _FEATURE_COLS if c in feature_window_df.columns]
        raw = feature_window_df[cols].values

        scaled = self.scalers[ticker].transform(raw)
        x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.models[ticker](x).cpu().numpy()[0]

        rv_1d = float(max(pred[0], 1e-8))
        rv_5d = float(max(pred[1], 1e-8))

        # Track history for regime detection
        self.rv_history.setdefault(ticker, []).append(rv_1d)
        hist = self.rv_history[ticker]
        p75 = np.percentile(hist, 75) if len(hist) >= 20 else rv_1d * 1.5
        regime = "HIGH" if rv_1d > p75 else "LOW"

        # Normalised signal strength (0-1 based on historical range)
        if len(hist) >= 20:
            lo, hi = np.percentile(hist, 5), np.percentile(hist, 95)
            strength = float(np.clip((rv_1d - lo) / max(hi - lo, 1e-8), 0, 1))
        else:
            strength = 0.5

        return {
            "ticker": ticker,
            "rv_1d_forecast": rv_1d,
            "rv_5d_forecast": rv_5d,
            "vol_regime": regime,
            "signal_strength": round(strength, 4),
        }

    def generate_all_signals(
        self,
        feature_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate signals for all tickers in feature_dict.

        Parameters
        ----------
        feature_dict : dict
            ``{ticker: last_22_rows_df}`` for each ticker.

        Returns
        -------
        pd.DataFrame
            One row per ticker with all signal columns.
        """
        rows = []
        for ticker, window_df in feature_dict.items():
            if ticker not in self.models:
                continue
            try:
                sig = self.generate_signal(ticker, window_df)
                rows.append(sig)
            except Exception as exc:
                logger.warning("Signal generation failed for %s: %s", ticker, exc)

        return pd.DataFrame(rows)

    def save_signals(
        self,
        signals_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Append signals to a CSV file (accumulates across trading steps)."""
        if output_path is None:
            output_path = FEATURES_DIR / "volatility_signals.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        signals_df["timestamp"] = pd.Timestamp.now()

        if output_path.exists():
            signals_df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            signals_df.to_csv(output_path, index=False)

        return output_path
