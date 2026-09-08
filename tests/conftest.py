import pytest


@pytest.fixture
def cfg():
    """Minimal configuration shared across unit tests."""
    return {
        "project": {
            "random_seed": 42,
        },
        "data": {
            "date_col": "date",
            "target_col": "units_sold",
            "series_cols": ["store_id", "sku_id"],
            "horizon": 14,
            "lookback": 56,
        },
        "split": {
            "validation_days": 14,
            "test_days": 14,
        },
        "features": {
            "demand_lags": [1, 7, 14, 28],
            "demand_roll_windows": [7, 14, 28],
            "stockout_lags": [1, 7, 14],
            "stock_roll_window": 28,
            "promo_known_future": True,
            "price_known_future": False,
            "weather_known_future": False,
            "stockout_target_mode": "none",
            "stockout_imputation_window": 56,
            "stockout_imputation_min_periods": 7,
            "stockout_percentage_uplift": 0.50,
            "stockout_sample_weight": 0.50,
        },
    }