import numpy as np

from demand_forecasting.metrics import (
    forecast_bias,
    regression_metrics,
    smape,
    wape,
)


def test_wape():
    y_true = np.array([10, 20])
    y_pred = np.array([8, 22])

    assert abs(wape(y_true, y_pred) - 4 / 30) < 1e-12


def test_regression_metrics_contains_expected_metrics():
    y_true = np.array([10, 20])
    y_pred = np.array([8, 22])

    metrics = regression_metrics(y_true, y_pred)

    assert set(metrics) == {
        "wape",
        "mae",
        "rmse",
        "smape",
        "bias",
    }

    assert abs(metrics["wape"] - 4 / 30) < 1e-12
    assert abs(metrics["mae"] - 2.0) < 1e-12
    assert abs(metrics["rmse"] - 2.0) < 1e-12


def test_forecast_bias_sign():
    y_true = np.array([10, 20, 30])

    overforecast = np.array([12, 22, 32])
    underforecast = np.array([8, 18, 28])

    assert forecast_bias(y_true, overforecast) > 0
    assert forecast_bias(y_true, underforecast) < 0


def test_smape_is_zero_for_perfect_forecast():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])

    assert smape(y_true, y_pred) == 0.0


def test_metrics_are_finite_for_valid_inputs():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([11, 18, 31])

    metrics = regression_metrics(y_true, y_pred)

    assert all(np.isfinite(value) for value in metrics.values())