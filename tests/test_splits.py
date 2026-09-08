import pandas as pd

from demand_forecasting.splits import label_split, make_temporal_split


def make_test_frame():
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", "2023-12-31")
    })


def test_configured_validation_and_test_windows(cfg):
    df = make_test_frame()

    split = make_temporal_split(df, cfg)

    assert (split.val_end - split.val_start).days + 1 == 14
    assert (split.test_end - split.test_start).days + 1 == 14


def test_temporal_splits_are_contiguous(cfg):
    df = make_test_frame()

    split = make_temporal_split(df, cfg)

    assert split.train_end + pd.Timedelta(days=1) == split.val_start
    assert split.val_end + pd.Timedelta(days=1) == split.test_start


def test_test_split_ends_on_last_dataset_date(cfg):
    df = make_test_frame()

    split = make_temporal_split(df, cfg)

    assert split.test_end == pd.Timestamp("2023-12-31")


def test_expected_split_boundaries(cfg):
    df = make_test_frame()

    split = make_temporal_split(df, cfg)

    assert split.train_end == pd.Timestamp("2023-12-03")
    assert split.val_start == pd.Timestamp("2023-12-04")
    assert split.val_end == pd.Timestamp("2023-12-17")
    assert split.test_start == pd.Timestamp("2023-12-18")
    assert split.test_end == pd.Timestamp("2023-12-31")


def test_label_split_assigns_correct_periods(cfg):
    df = make_test_frame()

    split = make_temporal_split(df, cfg)
    labels = label_split(df, split, cfg)

    assert labels.loc[df["date"] == "2023-12-03"].iloc[0] == "train"
    assert labels.loc[df["date"] == "2023-12-04"].iloc[0] == "validation"
    assert labels.loc[df["date"] == "2023-12-17"].iloc[0] == "validation"
    assert labels.loc[df["date"] == "2023-12-18"].iloc[0] == "test"
    assert labels.loc[df["date"] == "2023-12-31"].iloc[0] == "test"


def test_config_controls_split_size(cfg):
    df = make_test_frame()

    cfg["split"]["validation_days"] = 21
    cfg["split"]["test_days"] = 7

    split = make_temporal_split(df, cfg)

    assert (split.val_end - split.val_start).days + 1 == 21
    assert (split.test_end - split.test_start).days + 1 == 7