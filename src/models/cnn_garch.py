"""
CNN-GARCH hybrid model for volatility forecasting.

Combines the learned non-linear features from a 1-D CNN with GARCH
conditional variance (already stored in the feature CSV from Step 3).

This module handles:
  - Sliding-window sequence preparation with train-only scaling
  - Custom MSE + QLIKE combined loss
  - Training loop with early stopping, LR scheduling, and checkpointing
  - Out-of-sample evaluation
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_model import VolatilityCNN, get_device, set_all_seeds
from src.utils.config import MODELS_DIR, RANDOM_SEED, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRAIN_END = "2022-12-31"
_VAL_END = "2023-12-31"

# Columns used as CNN input features (all numeric, excluding raw targets)
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

_TARGET_COLS = ["target_rv_1d", "target_rv_5d"]


# ========================================================================
# Loss function
# ========================================================================

class CombinedVolLoss(nn.Module):
    """
    alpha * MSE  +  (1 - alpha) * QLIKE

    QLIKE = mean( actual / pred - log(actual / pred) - 1 )
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        pred_c = pred.clamp(min=1e-8)
        actual_c = actual.clamp(min=1e-8)

        mse_loss = self.mse(pred_c, actual_c)

        ratio = actual_c / pred_c
        qlike_loss = (ratio - torch.log(ratio) - 1.0).mean()

        return self.alpha * mse_loss + (1.0 - self.alpha) * qlike_loss


# ========================================================================
# Sequence preparation
# ========================================================================

def prepare_sequences(
    df: pd.DataFrame,
    window_size: int = 22,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    StandardScaler,
]:
    """
    Build sliding-window sequences and split into train / val / test.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame (1,214 rows x 23 cols typical).
    window_size : int
        Lookback window length.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    target_cols = [c for c in _TARGET_COLS if c in df.columns]

    # Split by date
    train_df = df.loc[:_TRAIN_END]
    val_df = df.loc[_TRAIN_END:_VAL_END]
    if len(val_df) and val_df.index[0] <= pd.Timestamp(_TRAIN_END):
        val_df = val_df.iloc[1:]
    test_df = df.loc[_VAL_END:]
    if len(test_df) and test_df.index[0] <= pd.Timestamp(_VAL_END):
        test_df = test_df.iloc[1:]

    # Fit scaler on train features only
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    def _make_windows(split_df: pd.DataFrame):
        feat_scaled = scaler.transform(split_df[feature_cols].values)
        targets = split_df[target_cols].values
        X_list, y_list = [], []
        for i in range(window_size, len(feat_scaled)):
            X_list.append(feat_scaled[i - window_size:i])
            y_list.append(targets[i])
        if not X_list:
            return np.empty((0, window_size, len(feature_cols))), np.empty((0, len(target_cols)))
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    X_train, y_train = _make_windows(train_df)
    X_val, y_val = _make_windows(val_df)
    X_test, y_test = _make_windows(test_df)

    logger.info(
        "  Sequences: train=%d  val=%d  test=%d  window=%d  features=%d",
        len(X_train), len(X_val), len(X_test), window_size, len(feature_cols),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ========================================================================
# CNNGARCHHybrid
# ========================================================================

class CNNGARCHHybrid:
    """
    End-to-end trainer and evaluator for the CNN-GARCH hybrid.

    Parameters
    ----------
    ticker : str
        Ticker identifier (for logging / saving).
    n_features : int
        Number of input channels to the CNN.
    window_size : int
        Lookback window.
    """

    def __init__(
        self,
        ticker: str,
        n_features: int = 23,
        window_size: int = 22,
    ) -> None:
        set_all_seeds(RANDOM_SEED)
        self.ticker = ticker
        self.device = get_device()
        self.model = VolatilityCNN(
            n_features=n_features,
            window_size=window_size,
        ).to(self.device)

        cfg = get_config().cnn_garch.cnn
        self.lr = cfg.learning_rate
        self.weight_decay = cfg.weight_decay
        self.batch_size = cfg.batch_size
        self.alpha = cfg.loss_alpha
        self.patience = cfg.patience

        self.save_dir = MODELS_DIR / "cnn"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses_: list = []
        self.val_losses_: list = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_epochs: int = 200,
    ) -> Dict[str, object]:
        """
        Train the CNN with early stopping.

        Returns
        -------
        dict with best_epoch, best_val_loss, train_time_s
        """
        set_all_seeds(RANDOM_SEED)

        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val),
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                              drop_last=False)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        criterion = CombinedVolLoss(alpha=self.alpha)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5,
        )

        best_val = float("inf")
        best_epoch = 0
        patience_counter = 0
        ckpt_path = self.save_dir / f"{self.ticker}_best.pt"

        t0 = time.time()

        for epoch in range(1, max_epochs + 1):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(train_ds)

            # --- Validate ---
            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_loss_sum += criterion(pred, yb).item() * len(xb)
            val_loss = val_loss_sum / len(val_ds)

            self.train_losses_.append(train_loss)
            self.val_losses_.append(val_loss)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if epoch % 10 == 0 or epoch <= 3 or epoch == max_epochs:
                logger.info(
                    "  Epoch %3d/%d  train=%.5f  val=%.5f  lr=%.1e",
                    epoch, max_epochs, train_loss, val_loss, current_lr,
                )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        "  Early stopping at epoch %d (best=%d, val=%.5f)",
                        epoch, best_epoch, best_val,
                    )
                    break

        elapsed = time.time() - t0
        logger.info(
            "  Training complete: best_epoch=%d  best_val=%.5f  time=%.1fs",
            best_epoch, best_val, elapsed,
        )

        # Reload best checkpoint
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device,
                                               weights_only=True))

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "train_time_s": round(elapsed, 1),
            "final_train_loss": self.train_losses_[-1],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference and return predictions as numpy array of shape (n, 2)."""
        self.model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).to(self.device)
            pred = self.model(xt).cpu().numpy()
        return np.clip(pred, 1e-8, None)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate on test set.

        Returns dict with RMSE_1d, MAE_1d, QLIKE_1d, R2_1d (and _5d variants).
        """
        from src.models.garch_model import evaluate_forecasts

        pred = self.predict(X_test)
        actual_1d, actual_5d = y_test[:, 0], y_test[:, 1]
        pred_1d, pred_5d = pred[:, 0], pred[:, 1]

        m1 = evaluate_forecasts(actual_1d, pred_1d)
        m5 = evaluate_forecasts(actual_5d, pred_5d)

        return {
            "RMSE_1d": m1["RMSE"], "MAE_1d": m1["MAE"],
            "QLIKE_1d": m1["QLIKE"], "R2_1d": m1["R2"],
            "RMSE_5d": m5["RMSE"], "MAE_5d": m5["MAE"],
            "QLIKE_5d": m5["QLIKE"], "R2_5d": m5["R2"],
        }

    def save_scaler(self, scaler: StandardScaler) -> None:
        """Save the feature scaler alongside the model checkpoint."""
        path = self.save_dir / f"{self.ticker}_scaler.pkl"
        with open(path, "wb") as f:
            pickle.dump(scaler, f)

    def load_checkpoint(self) -> None:
        """Load the best checkpoint from disk."""
        ckpt = self.save_dir / f"{self.ticker}_best.pt"
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device,
                                               weights_only=True))
        self.model.eval()
