"""Loading and cleaning the UCI Bike Sharing hourly data.

Key facts established during the data audit (see notebook 01):
* The raw file contains no hours with zero rentals (minimum count is 1), 165 hours are absent, 67% of them
  overnight and the rest clustered on storm days. Absent hours are therefore treated as **zero rentals**.
* Humidity is exactly 0 for 22 hours on 2011-03-10 (sensor failure); 12.5% of wind-speed readings are exactly 0
  (smallest positive reading is 6 km/h, i.e. a measurement floor); weathersit 4 occurs only 3 times.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "hour.csv"
COUNT_COLS = ["casual", "registered", "cnt"]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and add a proper hourly timestamp column ``ts``."""
    df = pd.read_csv(path, parse_dates=["dteday"])
    df["ts"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df


def clean_hourly(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a complete, regular hourly frame (index ``ts``) and an audit dictionary of what was changed."""
    raw = raw.sort_values("ts")
    if not raw["ts"].is_unique:
        raise ValueError("duplicate timestamps in raw data")
    audit: dict = {"raw_rows": int(len(raw))}

    idx = pd.date_range(raw["ts"].min(), raw["ts"].max(), freq="h", name="ts")
    df = raw.set_index("ts").reindex(idx)
    df["missing_hour"] = df["instant"].isna().astype(int)
    absent = df["missing_hour"] == 1
    audit.update(expected_hours=int(len(idx)), missing_hours=int(absent.sum()),
                 missing_overnight_share=float((df.index[absent].hour <= 5).mean()) if absent.any() else 0.0)

    # calendar fields are recomputed from the timestamp so that absent hours are complete (UCI weekday: 0 = Sunday)
    df["dteday"] = df.index.normalize()
    df["hr"] = df.index.hour
    df["yr"] = df.index.year - 2011
    df["mnth"] = df.index.month
    df["weekday"] = (df.index.dayofweek + 1) % 7
    for c in ["holiday", "workingday", "season"]:
        df[c] = df.groupby("dteday")[c].transform("first")          # same-day value from the hours that do exist
    if df[["holiday", "workingday", "season"]].isna().any().any():
        raise ValueError("a whole day is missing; day-level fields cannot be recovered")
    df[["holiday", "workingday", "season"]] = df[["holiday", "workingday", "season"]].astype(int)

    # absent hours = no rentals
    df[COUNT_COLS] = df[COUNT_COLS].fillna(0).astype(int)

    # weather: humidity sensor failure -> missing; interpolate through time for absent hours and the glitch
    glitch = df["hum"] == 0
    audit["humidity_zero_rows"] = int(glitch.sum())
    df.loc[glitch, "hum"] = np.nan
    for c in ["temp", "atemp", "hum", "windspeed"]:
        df[c] = df[c].interpolate(method="time", limit=72, limit_direction="both")
    df["weathersit"] = df["weathersit"].ffill().bfill().astype(int)
    audit["weathersit4_rows"] = int((df["weathersit"] == 4).sum())
    df["weathersit"] = df["weathersit"].clip(upper=3)                # merge the 3 rows of heavy rain/snow into category 3
    audit["wind_zero_share"] = float((df["windspeed"] == 0).mean())

    # physical units
    df["temp_c"] = df["temp"] * 47 - 8
    df["atemp_c"] = df["atemp"] * 66 - 16
    df["hum_pct"] = df["hum"] * 100
    df["wind_kmh"] = df["windspeed"] * 67
    df["wind_calm"] = (df["windspeed"] == 0).astype(int)

    keep = ["yr", "mnth", "hr", "weekday", "season", "holiday", "workingday", "weathersit", "temp_c", "atemp_c",
            "hum_pct", "wind_kmh", "wind_calm", "casual", "registered", "cnt", "missing_hour"]
    out = df[keep]
    if out.isna().any().any():
        raise ValueError("NaNs remain after cleaning")
    audit["clean_rows"] = int(len(out))
    return out, audit
