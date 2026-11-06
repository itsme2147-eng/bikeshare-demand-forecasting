import numpy as np
import pandas as pd
import pytest

from bikeshare import clean_hourly, load_raw


def tiny_raw():
    """Two days of hourly data (Fri 2011-01-07, Sat 2011-01-08) with some hours removed and quirks injected."""
    ts = pd.date_range("2011-01-07", periods=48, freq="h")
    df = pd.DataFrame({"ts": ts, "dteday": ts.normalize(), "hr": ts.hour, "instant": np.arange(1, 49)})
    df["season"] = 1; df["yr"] = 0; df["mnth"] = 1
    df["holiday"] = 0; df["weekday"] = (ts.dayofweek + 1) % 7; df["workingday"] = (ts.dayofweek < 5).astype(int)
    df["weathersit"] = 1; df["temp"] = 0.4; df["atemp"] = 0.4; df["hum"] = 0.5; df["windspeed"] = 0.1
    df["casual"] = 3; df["registered"] = 7; df["cnt"] = 10
    df.loc[5, "hum"] = 0.0                                   # humidity sensor glitch
    df.loc[10, "weathersit"] = 4                             # rare category
    df.loc[20, "windspeed"] = 0.0                            # calm reading
    return df.drop(index=[2, 3, 30]).reset_index(drop=True)  # three absent hours


def test_absent_hours_become_zero_demand_and_index_is_regular():
    raw = tiny_raw()
    out, audit = clean_hourly(raw)
    assert len(out) == 48 and audit["missing_hours"] == 3
    assert out["missing_hour"].sum() == 3
    assert (out.loc[out["missing_hour"] == 1, "cnt"] == 0).all()
    assert out["cnt"].sum() == raw["cnt"].sum()               # no demand is created or lost
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_humidity_glitch_is_interpolated_not_zero():
    out, audit = clean_hourly(tiny_raw())
    assert audit["humidity_zero_rows"] == 1
    assert out["hum_pct"].min() > 0 and abs(out["hum_pct"].iloc[5] - 50) < 1e-9


def test_rare_weather_category_is_merged_and_units_converted():
    out, audit = clean_hourly(tiny_raw())
    assert out["weathersit"].max() == 3 and audit["weathersit4_rows"] == 1
    assert out["temp_c"].iloc[0] == pytest.approx(0.4 * 47 - 8)
    assert out["wind_calm"].sum() == 1


def test_calendar_is_complete_and_weekday_zero_is_sunday():
    out, _ = clean_hourly(tiny_raw())
    assert out["weekday"].iloc[0] == 5 and out["weekday"].iloc[-1] == 6     # Friday = 5, Saturday = 6
    assert out["workingday"].iloc[:24].eq(1).all() and out["workingday"].iloc[24:].eq(0).all()
    assert not out.isna().any().any()


def test_duplicate_timestamps_are_rejected():
    raw = tiny_raw()
    with pytest.raises(ValueError):
        clean_hourly(pd.concat([raw, raw.iloc[[0]]]))


def test_real_data_cleaning_preserves_total_demand():
    raw = load_raw()
    out, audit = clean_hourly(raw)
    assert len(raw) == 17379 and len(out) == 17544 and audit["missing_hours"] == 165
    assert out["cnt"].sum() == raw["cnt"].sum()
    assert audit["humidity_zero_rows"] == 22 and not out.isna().any().any()
    assert (out["casual"] + out["registered"] == out["cnt"]).all()
