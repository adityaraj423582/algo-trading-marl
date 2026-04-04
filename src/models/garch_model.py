"""
GARCH-family volatility models and HAR-RV baseline.

Implements four volatility forecasting models:

1. **GARCH(1,1)** — standard symmetric volatility clustering model
2. **EGARCH(1,1)** — Nelson (1991) exponential GARCH with leverage
3. **GJR-GARCH(1,1)** — Glosten-Jagannathan-Runkle with asymmetric shocks
4. **HAR-RV** — Corsi (2009) heterogeneous autoregressive model for
   realised volatility (OLS regression, not GARCH)

All GARCH variants are fitted via the ``arch`` library with Student-t
innovations (justified by excess kurtosis found in Step 2).

Usage
-----
::

    from src.models.garch_model import GARCHFamily, HARRV, evaluate_forecasts

    model = GARCHFamily("GARCH", p=1, q=1, dist="t")
    result = model.fit(train_returns)
    forecasts = model.rolling_forecast(train_returns, test_returns)
    metrics = evaluate_forecasts(actual_rv, forecasts)
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from src.utils.config import MODELS_DIR, RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress arch convergence warnings during rolling forecasts
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arch")
warnings.filterwarnings("ignore", message=".*convergence.*", category=UserWarning)


# ========================================================================
# Evaluation metrics
# ========================================================================

def evaluate_forecasts(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard volatility forecasting metrics.

    Parameters
    ----------
    actual : array-like
        Realised volatility (must be > 0).
    predicted : array-like
        Model-forecasted volatility (must be > 0).

    Returns
    -------
    dict with keys: RMSE, MAE, QLIKE, R2

    Notes
    -----
    QLIKE (quasi-likelihood loss) is the standard loss function for
    volatility forecast evaluation (Patton, 2011). Lower is better.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Guard against zero / negative values
    mask = (actual > 0) & (predicted > 0) & np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]

    if len(a) < 10:
        return {"RMSE": np.nan, "MAE": np.nan, "QLIKE": np.nan, "R2": np.nan}

    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae = np.mean(np.abs(a - p))

    # QLIKE: mean(actual/predicted - log(actual/predicted) - 1)
    ratio = a / p
    qlike = np.mean(ratio - np.log(ratio) - 1.0)

    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"RMSE": rmse, "MAE": mae, "QLIKE": qlike, "R2": r2}


def diebold_mariano_test(
    actual: np.ndarray,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    loss: str = "SE",
) -> Tuple[float, float]:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: both forecasts have equal expected loss.

    Parameters
    ----------
    actual, pred_1, pred_2 : array-like
    loss : str
        ``"SE"`` for squared error, ``"AE"`` for absolute error.

    Returns
    -------
    (DM_statistic, p_value)
    """
    from scipy.stats import norm as sp_norm

    a = np.asarray(actual, dtype=np.float64)
    p1 = np.asarray(pred_1, dtype=np.float64)
    p2 = np.asarray(pred_2, dtype=np.float64)

    if loss == "SE":
        d = (a - p1) ** 2 - (a - p2) ** 2
    else:
        d = np.abs(a - p1) - np.abs(a - p2)

    d_bar = np.mean(d)
    n = len(d)

    # Newey-West variance estimate (lag = int(n^(1/3)))
    max_lag = max(1, int(n ** (1.0 / 3.0)))
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        gamma_k = np.cov(d[:-k], d[k:])[0, 1]
        gamma_sum += w * gamma_k
    var_d = gamma_0 + 2.0 * gamma_sum

    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = d_bar / np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - sp_norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)


# ========================================================================
# GARCH Family
# ========================================================================

class GARCHFamily:
    """
    Unified interface for GARCH, EGARCH, and GJR-GARCH models.

    Parameters
    ----------
    model_type : str
        One of ``"GARCH"``, ``"EGARCH"``, ``"GJR-GARCH"``.
    p, q : int
        Lag orders for the conditional variance equation.
    dist : str
        Innovation distribution: ``"t"`` (Student-t) or ``"normal"``.
    mean : str
        Mean model: ``"Constant"``, ``"AR"``, or ``"Zero"``.
    """

    VALID_TYPES = ("GARCH", "EGARCH", "GJR-GARCH")

    def __init__(
        self,
        model_type: str = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "t",
        mean: str = "Constant",
    ) -> None:
        if model_type not in self.VALID_TYPES:
            raise ValueError(f"model_type must be one of {self.VALID_TYPES}")
        self.model_type = model_type
        self.p = p
        self.q = q
        self.o = 1 if model_type == "GJR-GARCH" else 0
        self.dist = dist
        self.mean = mean
        self.result_ = None
        self._name = f"{model_type}({p},{q})-{dist}"

    @property
    def name(self) -> str:
        return self._name

    def _build_model(self, returns: pd.Series):
        """Construct an ``arch_model`` object from parameters."""
        vol_type = "EGARCH" if self.model_type == "EGARCH" else "GARCH"
        return arch_model(
            returns * 100,   # arch expects percentage returns
            mean=self.mean,
            vol=vol_type,
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
        )

    def fit(
        self,
        returns: pd.Series,
        show_summary: bool = False,
    ) -> Dict[str, float]:
        """
        Fit the model on a return series.

        Returns
        -------
        dict with AIC, BIC, loglikelihood
        """
        model = self._build_model(returns)
        try:
            self.result_ = model.fit(
                disp="off",
                show_warning=False,
                options={"maxiter": 500},
            )
        except Exception as exc:
            logger.warning("  %s fit failed: %s", self.name, exc)
            return {"AIC": np.nan, "BIC": np.nan, "loglikelihood": np.nan}

        if show_summary:
            print(self.result_.summary())

        return {
            "AIC": self.result_.aic,
            "BIC": self.result_.bic,
            "loglikelihood": self.result_.loglikelihood,
        }

    def get_conditional_variance(self) -> pd.Series:
        """
        Return the in-sample conditional variance (annualised vol scale).

        The ``arch`` library works in percentage-return space, so
        conditional variance is in (%-return)^2. We convert back to
        decimal-return scale then annualise:

            sigma_annual = sqrt(cond_var / 10000 * 252)
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted yet.")
        cv = self.result_.conditional_volatility / 100.0 * np.sqrt(252)
        cv.name = "garch_conditional_vol"
        return cv

    def get_standardized_residuals(self) -> pd.Series:
        """Return standardised residuals (epsilon_t / sigma_t)."""
        if self.result_ is None:
            raise RuntimeError("Model not fitted yet.")
        resid = self.result_.std_resid
        resid.name = "garch_std_resid"
        return resid

    def rolling_forecast(
        self,
        train_returns: pd.Series,
        test_returns: pd.Series,
        refit_every: int = 22,
    ) -> pd.Series:
        """
        Walk-forward rolling one-step-ahead volatility forecast.

        The model is re-estimated every ``refit_every`` steps.  Between
        re-fits, we extend the data window and use ``forecast(horizon=1)``
        from the last fit.

        Returns
        -------
        pd.Series
            Annualised volatility forecasts aligned to the test index.
        """
        combined = pd.concat([train_returns, test_returns])
        test_start_idx = len(train_returns)
        n_test = len(test_returns)

        forecasts = np.full(n_test, np.nan)
        last_fit_end = test_start_idx

        for i in range(n_test):
            current_end = test_start_idx + i

            # Re-fit periodically
            if i == 0 or (i % refit_every == 0):
                window = combined.iloc[:current_end]
                model = self._build_model(window)
                try:
                    res = model.fit(disp="off", show_warning=False,
                                    options={"maxiter": 500})
                except Exception:
                    continue

            # One-step-ahead forecast
            try:
                fcast = res.forecast(horizon=1, reindex=False)
                var_forecast = fcast.variance.values[-1, 0]
                # Convert from (%-return)^2 to annualised vol
                forecasts[i] = np.sqrt(var_forecast / 10000.0 * 252.0)
            except Exception:
                pass

        return pd.Series(forecasts, index=test_returns.index, name=f"{self.name}_forecast")

    def save(self, filepath: Path) -> None:
        """Pickle the fitted result to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"params": self.result_.params if self.result_ else None,
                         "model_type": self.model_type, "p": self.p, "q": self.q,
                         "dist": self.dist}, f)
        logger.debug("  Saved %s -> %s", self.name, filepath.name)


# ========================================================================
# HAR-RV Model
# ========================================================================

class HARRV:
    """
    Heterogeneous Autoregressive model for Realised Volatility
    (Corsi, 2009).

    RV_{t+1} = alpha + beta_d * RV_t + beta_w * RV_weekly_t
                     + beta_m * RV_monthly_t + epsilon_{t+1}

    Fitted via OLS (statsmodels).
    """

    def __init__(self) -> None:
        self.result_: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self._name = "HAR-RV"

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build the HAR feature matrix X and target y from a feature DataFrame.

        Requires columns: rv_daily, rv_weekly, rv_monthly, target_rv_1d.
        """
        required = ["rv_daily", "rv_weekly", "rv_monthly", "target_rv_1d"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing column: {col}")

        X = df[["rv_daily", "rv_weekly", "rv_monthly"]].copy()
        X = sm.add_constant(X)
        y = df["target_rv_1d"]

        mask = X.notna().all(axis=1) & y.notna()
        return X.loc[mask], y.loc[mask]

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit HAR-RV model on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with rv_daily, rv_weekly, rv_monthly, target_rv_1d.

        Returns
        -------
        dict with AIC, BIC, R2
        """
        X, y = self._prepare_features(df)
        self.result_ = sm.OLS(y, X).fit()
        return {
            "AIC": self.result_.aic,
            "BIC": self.result_.bic,
            "loglikelihood": self.result_.llf,
            "R2_in_sample": self.result_.rsquared,
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict RV using fitted coefficients."""
        if self.result_ is None:
            raise RuntimeError("Model not fitted yet.")
        X, _ = self._prepare_features(df)
        preds = self.result_.predict(X)
        preds.name = "HAR-RV_forecast"
        return preds.clip(lower=1e-8)

    def rolling_forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        refit_every: int = 22,
    ) -> pd.Series:
        """
        Walk-forward rolling forecast for HAR-RV.

        Re-estimates OLS every ``refit_every`` steps.
        """
        combined = pd.concat([train_df, test_df])
        test_start = len(train_df)
        n_test = len(test_df)
        forecasts = np.full(n_test, np.nan)

        for i in range(n_test):
            current_end = test_start + i

            if i == 0 or (i % refit_every == 0):
                window = combined.iloc[:current_end]
                try:
                    X, y = self._prepare_features(window)
                    res = sm.OLS(y, X).fit()
                except Exception:
                    continue

            # One-step prediction using current row features
            row = combined.iloc[current_end:current_end + 1]
            try:
                X_row = row[["rv_daily", "rv_weekly", "rv_monthly"]]
                X_row = sm.add_constant(X_row, has_constant="add")
                forecasts[i] = max(res.predict(X_row).values[0], 1e-8)
            except Exception:
                pass

        return pd.Series(forecasts, index=test_df.index, name="HAR-RV_forecast")

    def save(self, filepath: Path) -> None:
        """Save fitted OLS result."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"params": self.result_.params.to_dict() if self.result_ else None,
                         "model": "HAR-RV"}, f)
