import pandas as pd

from demand_forecasting.features import build_causal_features


def make_test_frame():
    dates = pd.date_range("2024-01-01", periods=35)

    return pd.DataFrame({
        "date": dates,
        "store_id": "S1",
        "sku_id": "K1",
        "demand_target": range(35),
        "units_sold": range(35),
        "stock_out_flag": 0,
        "stock_on_hand": 100,
        "list_price": 10.0,
        "discount_pct": 0.0,
        "promo_flag": 0,
        "temperature": 10.0,
        "rain_mm": 0.0,
        "is_holiday": 0,
        "country": "X",
        "city": "Y",
        "channel": "Store",
        "category": "C",
        "subcategory": "SC",
        "brand": "B",
        "purchase_cost": 5.0,
        "margin_pct": 0.5,
    })


def test_demand_lags_are_causal(cfg):
    df = make_test_frame()

    features = build_causal_features(
        df,
        cfg,
        demand_col="demand_target",
    )

    assert features.iloc[10]["demand_lag_1"] == 9
    assert features.iloc[10]["demand_lag_7"] == 3


def test_rolling_features_do_not_include_current_target(cfg):
    df = make_test_frame()

    features = build_causal_features(
        df,
        cfg,
        demand_col="demand_target",
    )

    # At index 10, a shifted 7-day rolling window must use values 3..9.
    assert features.iloc[10]["demand_roll_mean_7"] == 6.0


def test_mutating_current_target_does_not_change_current_features(cfg):
    df = make_test_frame()

    before = build_causal_features(
        df,
        cfg,
        demand_col="demand_target",
    )

    causal_columns = [
        "demand_lag_1",
        "demand_lag_7",
        "demand_roll_mean_7",
        "demand_roll_std_7",
    ]

    expected = before.loc[10, causal_columns].copy()

    # A forecasting feature for date t must not depend on y_t.
    df.loc[10, "demand_target"] = 9999

    after = build_causal_features(
        df,
        cfg,
        demand_col="demand_target",
    )

    pd.testing.assert_series_equal(
        expected,
        after.loc[10, causal_columns],
    )


def test_future_target_does_not_change_past_features(cfg):
    df = make_test_frame()

    before = build_causal_features(
        df,
        cfg,
        demand_col="demand_target",
    )

    df.loc[20:, "demand_target"] = 9999

    after = build_causal_features(
        df,
        cfg,
        demand_col="demand_target",
    )

    columns = [
        "demand_lag_1",
        "demand_lag_7",
        "demand_roll_mean_7",
    ]

    pd.testing.assert_frame_equal(
        before.loc[:19, columns],
        after.loc[:19, columns],
    )