"""
1-D Convolutional Neural Network for volatility forecasting.

Architecture
------------
Input: (batch, window=22, features=25)  -- note: Conv1d wants (batch, channels, length)
       so we transpose to (batch, 25, 22) internally.

Conv block 1:  Conv1d(25 -> 64,  k=3, pad=1) + BN + ReLU + Dropout(0.2)
Conv block 2:  Conv1d(64 -> 128, k=5, pad=2) + BN + ReLU + Dropout(0.2)
Conv block 3:  Conv1d(128 -> 64, k=3, pad=1) + BN + ReLU
AdaptiveAvgPool1d(1) -> squeeze -> (batch, 64)
Dense: 64 -> 32 (ReLU) -> 2
Output: [forecast_rv_1d, forecast_rv_5d]
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import RANDOM_SEED


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VolatilityCNN(nn.Module):
    """
    1-D CNN for predicting next-day and next-week realised volatility.

    Parameters
    ----------
    n_features : int
        Number of input features per time step (channel dimension).
    window_size : int
        Length of the lookback window (time dimension).
    dropout : float
        Dropout probability applied after the first two conv blocks.
    """

    def __init__(
        self,
        n_features: int = 25,
        window_size: int = 22,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        assert n_features >= 16, "Minimum 16 features required"
        self.n_features = n_features
        self.window_size = window_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VolatilityCNN  |  params: {total:,} total, {trainable:,} trainable")

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, window_size, n_features)

        Returns
        -------
        Tensor of shape (batch, 2) -- [rv_1d_forecast, rv_5d_forecast]
        """
        # Conv1d expects (batch, channels, length) = (batch, features, time)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1)   # (batch, 64)
        return self.head(x)
