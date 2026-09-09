import numpy as np
import pandas as pd
import pytest

from demand_forecasting.hierarchy import (
    aggregate_bottom_up,
    middle_out_allocate,
    trailing_shares,
)


def make_history(days: int = 60):
    dates = pd.date_range("2023-01-01", periods=days)
    rows = []

    for store in ["S1", "S2"]:
        for sku, sub in [("K1", "SubA"), ("K2", "SubA")]:
            for i, d in enumerate(dates):
                rows.append({
                    "date": d,
                    "store_id": store,
                    "sku_id": sku,
                    "subcategory": sub,
                    "units_sold": 10 + i % 5 + (5 if sku == "K1" else 0),
                    "stock_out_flag": 0,
                })

    return pd.DataFrame(rows)


def make_forecast(horizon: int):
    dates = pd.date_range("2023-03-02", periods=horizon)
    rows = []

    for store in ["S1", "S2"]:
        for sku, sub in [("K1", "SubA"), ("K2", "SubA")]:
            for d in dates:
                rows.append({
                    "date": d,
                    "store_id": store,
                    "sku_id": sku,
                    "subcategory": sub,
                    "prediction": 12.0 if sku == "K1" else 8.0,
                })

    return pd.DataFrame(rows)


def test_bottom_up_preserves_total(cfg):
    fc = make_forecast(horizon=14)
    up = aggregate_bottom_up(fc, ["store_id"], cfg)

    assert np.isclose(up["prediction"].sum(), fc["prediction"].sum())


def test_trailing_shares_sum_to_one_per_parent(cfg):
    shares = trailing_shares(
        make_history(),
        ["store_id", "subcategory", "sku_id"],
        ["store_id", "subcategory"],
        cfg,
    )

    totals = shares.groupby(["store_id", "subcategory"])["share"].sum()

    assert np.allclose(totals.to_numpy(float), 1.0)


@pytest.mark.parametrize("horizon", [1, 7, 14])
def test_middle_out_allocates_across_a_multi_day_horizon(cfg, horizon):
    """Regression: the parent keys repeat once per horizon day, which must not break the merge."""
    fc = make_forecast(horizon=horizon)
    parent = aggregate_bottom_up(fc, ["store_id", "subcategory"], cfg)

    shares = trailing_shares(
        make_history(),
        ["store_id", "subcategory", "sku_id"],
        ["store_id", "subcategory"],
        cfg,
    )

    allocated = middle_out_allocate(
        parent,
        shares,
        ["store_id", "subcategory"],
        ["store_id", "subcategory", "sku_id"],
        cfg,
    )

    assert len(allocated) == horizon * 4
    assert np.isclose(allocated["prediction"].sum(), parent["prediction"].sum())

    # Every parent total must be preserved on every individual date, not just in aggregate.
    got = allocated.groupby(["date", "store_id", "subcategory"])["prediction"].sum()
    want = parent.set_index(["date", "store_id", "subcategory"])["prediction"]

    pd.testing.assert_series_equal(got, want.reindex(got.index), check_names=False)


def test_middle_out_rejects_duplicate_shares(cfg):
    fc = make_forecast(horizon=3)
    parent = aggregate_bottom_up(fc, ["store_id", "subcategory"], cfg)

    shares = trailing_shares(
        make_history(),
        ["store_id", "subcategory", "sku_id"],
        ["store_id", "subcategory"],
        cfg,
    )

    with pytest.raises(ValueError, match="one row per child"):
        middle_out_allocate(
            parent,
            pd.concat([shares, shares], ignore_index=True),
            ["store_id", "subcategory"],
            ["store_id", "subcategory", "sku_id"],
            cfg,
        )
