import numpy as np
import pandas as pd
import pytest

from bikeshare import build_features, clean_hourly, load_raw
from bikeshare.features import history_features


@pytest.fixture(scope="module")
def frame():
    return clean_hourly(load_raw())[0]


def test_lag_168_is_exactly_one_week_earlier(frame):
    X, y, _ = build_features(frame, 24)
    t = pd.Timestamp("2012-06-20 08:00")
    assert X.loc[t, "lag_168"] == y.loc[t - pd.Timedelta(hours=168)]
    assert X.loc[t, "lag_24"] == y.loc[t - pd.Timedelta(hours=24)]


@pytest.mark.parametrize("horizon", [1, 24])
def test_no_future_leakage_features_ignore_data_newer_than_horizon(frame, horizon):
    """Corrupt every observation newer than t - horizon; the features at t must not change."""
    t = pd.Timestamp("2012-03-14 17:00")
    base, _, groups = build_features(frame, horizon)
    corrupted = frame.copy()
    newer = corrupted.index > t - pd.Timedelta(hours=horizon)
    corrupted.loc[newer, "cnt"] = np.random.default_rng(0).integers(0, 1000, newer.sum())
    X2, _, _ = build_features(corrupted, horizon)
    cols = groups["history"]
    pd.testing.assert_series_equal(base.loc[t, cols], X2.loc[t, cols])


def test_day_ahead_has_no_lag_below_24_hours(frame):
    X, _, groups = build_features(frame, 24)
    lags = [int(c.split("_")[1]) for c in groups["history"] if c.startswith("lag_")]
    assert min(lags) == 24
    X1, _, g1 = build_features(frame, 1)
    assert "lag_1" in X1.columns and "lag_1" not in X.columns


def test_groups_partition_the_columns_and_weather_describes_hour_t(frame):
    X, _, groups = build_features(frame, 24)
    allcols = sum(groups.values(), [])
    assert sorted(allcols) == sorted(X.columns) and len(set(allcols)) == len(allcols)
    assert (X["temp_c"] == frame["temp_c"]).all()


def test_cyclical_encodings_are_bounded_and_midnight_equals_hour_24(frame):
    X, _, _ = build_features(frame, 24)
    assert X[["hour_sin", "hour_cos", "doy_sin", "doy_cos"]].abs().max().max() <= 1 + 1e-12
    assert X.loc[X["hr"] == 0, "hour_cos"].eq(1).all()


def test_irregular_index_is_rejected(frame):
    with pytest.raises(ValueError):
        build_features(frame.drop(frame.index[100]), 24)


def test_invalid_horizon_is_rejected(frame):
    with pytest.raises(ValueError):
        history_features(frame, 0)


def test_history_is_nan_only_at_the_start_of_the_series(frame):
    X, _, groups = build_features(frame, 24)
    h = X[groups["history"]]
    assert h.iloc[800:].isna().sum().sum() == 0              # complete once a full window of history exists
