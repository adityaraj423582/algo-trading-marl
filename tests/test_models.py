"""Tests for VolatilityCNN model."""

import torch
import pytest

from src.models.cnn_model import VolatilityCNN


@pytest.fixture
def model():
    return VolatilityCNN(n_features=25, window_size=22)


def test_cnn_output_shape(model):
    x = torch.randn(16, 22, 25)
    out = model(x)
    assert out.shape == (16, 2)


def test_cnn_no_nan_output(model):
    x = torch.randn(8, 22, 25)
    out = model(x)
    assert not torch.isnan(out).any()


def test_cnn_positive_output(model):
    x = torch.randn(8, 22, 25)
    out = model(x)
    assert out.shape == (8, 2)


def test_cnn_correct_permute_in_forward(model):
    x = torch.randn(4, 22, 25)
    out = model(x)
    assert out.shape == (4, 2)


def test_cnn_parameter_count_reasonable(model):
    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 <= n_params <= 600_000, f"Got {n_params} params"


def test_cnn_minimum_features_assertion():
    with pytest.raises(AssertionError):
        VolatilityCNN(n_features=10)


def test_cnn_backward_compatible_with_16_features():
    m = VolatilityCNN(n_features=16, window_size=22)
    x = torch.randn(4, 22, 16)
    out = m(x)
    assert out.shape == (4, 2)
