"""Cleaning and feature engineering shared by training and the dashboard."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_COLUMNS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "temperature", "humidity", "wind_speed"]
FEATURES = RAW_COLUMNS + ["season", "day_of_week"]


def season_from_month(month):
    # Indian seasonal grouping: winter, summer, monsoon, post-monsoon.
    return np.select([month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8, 9])], [0, 1, 2], default=3)


def prepare_data(path=ROOT / "data" / "aqi_data.csv"):
    data = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    data[RAW_COLUMNS] = data[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data[RAW_COLUMNS] = data[RAW_COLUMNS].interpolate().bfill().ffill()
    # Winsorise only predictor columns: robust against accidental sensor spikes.
    for col in RAW_COLUMNS:
        low, high = data[col].quantile([.01, .99])
        data[col] = data[col].clip(low, high)
    data["season"] = season_from_month(data["date"].dt.month).astype(int)
    data["day_of_week"] = data["date"].dt.dayofweek.astype(int)
    data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce").interpolate().bfill().ffill()
    return data


def feature_row(values, when=None):
    """Turn dashboard inputs into the same feature order used for training."""
    when = pd.Timestamp.today() if when is None else pd.Timestamp(when)
    row = {name: float(values[name]) for name in RAW_COLUMNS}
    row.update(season=int(season_from_month(pd.Series([when.month]))[0]), day_of_week=when.dayofweek)
    return pd.DataFrame([row], columns=FEATURES)
