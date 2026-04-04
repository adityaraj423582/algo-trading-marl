"""Tests for project configuration."""

from src.utils.config import (
    Config,
    N_INPUT_FEATURES,
    BLOOMBERG_FILE_PATH,
    BLOOMBERG_NSE_TICKERS,
    BLOOMBERG_NASDAQ_TICKERS,
    BLOOMBERG_SYNTHETIC_FALLBACK,
    BLOOMBERG_FEATURE_COLUMNS,
    USE_BLOOMBERG,
    get_config,
)


def test_n_input_features_is_correct():
    assert N_INPUT_FEATURES == 25


def test_bloomberg_enabled():
    assert USE_BLOOMBERG is True


def test_bloomberg_nse_tickers():
    assert len(BLOOMBERG_NSE_TICKERS) == 7
    assert "RELIANCE_NS" in BLOOMBERG_NSE_TICKERS


def test_bloomberg_nasdaq_tickers():
    assert len(BLOOMBERG_NASDAQ_TICKERS) == 10
    assert "AAPL" in BLOOMBERG_NASDAQ_TICKERS


def test_synthetic_fallback_tickers():
    assert len(BLOOMBERG_SYNTHETIC_FALLBACK) == 3
    assert "KOTAKBANK_NS" in BLOOMBERG_SYNTHETIC_FALLBACK


def test_bloomberg_feature_columns():
    assert len(BLOOMBERG_FEATURE_COLUMNS) == 9
    assert "india_vix" in BLOOMBERG_FEATURE_COLUMNS
    assert "bb_bid1" in BLOOMBERG_FEATURE_COLUMNS


def test_cnn_config_features():
    cfg = get_config()
    assert cfg.cnn.n_features == 25


def test_bloomberg_file_path_string():
    assert "BLMBRG_LVL_2_DATSET__22.xlsx" in str(BLOOMBERG_FILE_PATH)
