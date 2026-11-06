"""Leakage-safe feature engineering for a forecast made ``horizon`` hours ahead.

Every history feature at row t is computed only from demand observed at or before t - horizon.
Weather and calendar features describe hour t itself and are assumed to come from a forecast / calendar.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GROUP_NAMES = ("calendar", "weather", "history")


def _assert_regular(df: pd.DataFrame) -> None:
    steps = df.index.to_series().diff().dropna()
    if not df.index.is_monotonic_increasing or not (steps == pd.Timedelta("1h")).all():
        raise ValueError("frame must have a regular, sorted hourly index")


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["hr"] = df["hr"]
    out["weekday"] = df["weekday"]
    out["workingday"] = df["workingday"]
    out["holiday"] = df["holiday"]
    out["mnth"] = df["mnth"]
    out["season"] = df["season"]
    out["yr"] = df["yr"]
    out["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)
    doy = df.index.dayofyear.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    return out


def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["temp_c"] = df["temp_c"]
    out["hum_pct"] = df["hum_pct"]
    out["wind_kmh"] = df["wind_kmh"]
    out["wind_calm"] = df["wind_calm"]
    out["weathersit"] = df["weathersit"]
    bad = (df["weathersit"] >= 3).astype(int)
    out["bad_weather"] = bad
    out["rain_last3h"] = bad.rolling(3, min_periods=1).max()
    out["temp_delta_3h"] = (df["temp_c"] - df["temp_c"].shift(3)).fillna(0.0)
    return out


def history_features(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Lagged demand and rolling means; the most recent observation used is cnt[t - horizon]."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    y = df["cnt"].astype(float)
    out = pd.DataFrame(index=df.index)
    lags = {horizon, horizon + 24, 168, 336}
    if horizon < 24:
        lags |= {horizon + 1, horizon + 2}
    for L in sorted(lags):
        out[f"lag_{L}"] = y.shift(L)
    weekly = pd.concat([y.shift(L) for L in (168, 336, 504, 672)], axis=1)
    out["same_hour_4w_mean"] = weekly.mean(axis=1, skipna=True)
    known = y.shift(horizon)                                         # nothing newer than t - horizon is ever used
    out["recent_6h_mean"] = known.rolling(6, min_periods=3).mean()
    out["roll_7d_mean"] = known.rolling(168, min_periods=48).mean()
    out["roll_28d_mean"] = known.rolling(672, min_periods=168).mean()
    out["level_ratio"] = out["roll_7d_mean"] / out["roll_28d_mean"].replace(0, np.nan)
    return out


def build_features(df: pd.DataFrame, horizon: int = 24) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """Return X, y and a dict mapping feature-group name -> column list."""
    _assert_regular(df)
    cal, wea, his = calendar_features(df), weather_features(df), history_features(df, horizon)
    X = pd.concat([cal, wea, his], axis=1)
    groups = {"calendar": list(cal.columns), "weather": list(wea.columns), "history": list(his.columns)}
    return X, df["cnt"].astype(float), groups
