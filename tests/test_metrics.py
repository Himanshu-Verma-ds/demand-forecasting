import numpy as np

from demand_forecasting.metrics import (
    forecast_bias,
    mape,
    mape_coverage,
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
        "mape",
        "mape_coverage",
        "mae",
        "rmse",
        "smape",
        "bias",
    }

    assert abs(metrics["wape"] - 4 / 30) < 1e-12
    assert abs(metrics["mae"] - 2.0) < 1e-12
    assert abs(metrics["rmse"] - 2.0) < 1e-12


def test_mape_matches_hand_calculation():
    y_true = np.array([100.0, 50.0])
    y_pred = np.array([90.0, 60.0])

    # |10|/100 = 0.1 and |10|/50 = 0.2 -> mean 0.15
    assert abs(mape(y_true, y_pred) - 0.15) < 1e-12


def test_mape_excludes_zero_actuals_and_reports_coverage():
    y_true = np.array([100.0, 0.0, 50.0, 0.0])
    y_pred = np.array([90.0, 7.0, 60.0, 3.0])

    # Only the two non-zero actuals are scored; the zeros would be a division by zero.
    assert abs(mape(y_true, y_pred) - 0.15) < 1e-12
    assert np.isfinite(mape(y_true, y_pred))
    assert abs(mape_coverage(y_true) - 0.5) < 1e-12


def test_mape_is_nan_when_every_actual_is_zero():
    y_true = np.zeros(3)
    y_pred = np.array([1.0, 2.0, 3.0])

    assert np.isnan(mape(y_true, y_pred))


def test_mape_explodes_on_small_actuals_but_wape_does_not():
    """Documents why WAPE is the default selection metric on low-demand rows."""
    y_true = np.array([1000.0, 1.0])
    y_pred = np.array([1000.0, 3.0])

    # The same 2-unit miss is 0.2% of total volume but a 200% error on that one row.
    assert abs(wape(y_true, y_pred) - 2 / 1001) < 1e-12
    assert abs(mape(y_true, y_pred) - 1.0) < 1e-12


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