"""Create a reproducible, synthetic two-year daily AQI dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("aqi_data.csv")
RNG = np.random.default_rng(42)

# CPCB concentration bands and their matching AQI bands (CO is mg/m3).
BANDS = {
    "PM2.5": ([0, 30, 60, 90, 120, 250, 350], [0, 50, 100, 200, 300, 400, 500]),
    "PM10": ([0, 50, 100, 250, 350, 430, 500], [0, 50, 100, 200, 300, 400, 500]),
    "NO2": ([0, 40, 80, 180, 280, 400, 500], [0, 50, 100, 200, 300, 400, 500]),
    # The final upper edge represents the open-ended CPCB "severe" band.
    "SO2": ([0, 40, 80, 380, 800, 1600, 2000], [0, 50, 100, 200, 300, 400, 500]),
    "CO": ([0, 1, 2, 10, 17, 34, 50], [0, 50, 100, 200, 300, 400, 500]),
}


def sub_index(value, pollutant):
    """Linearly interpolate an individual CPCB pollutant sub-index."""
    bp, iaqi = BANDS[pollutant]
    value = float(np.clip(value, bp[0], bp[-1]))
    i = min(np.searchsorted(bp, value, side="right") - 1, len(bp) - 2)
    return round(((iaqi[i + 1] - iaqi[i]) / (bp[i + 1] - bp[i])) *
                 (value - bp[i]) + iaqi[i])


def calculate_aqi(row):
    """CPCB AQI is the highest individual pollutant sub-index."""
    return max(sub_index(row[name], name) for name in BANDS)


def main():
    # A rolling history keeps the dashboard relevant whenever it is regenerated.
    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end_date, periods=730, freq="D")
    day = np.arange(len(dates))
    winter = (1 + np.cos(2 * np.pi * (day - 15) / 365.25)) / 2
    rain = (1 - np.cos(2 * np.pi * (day - 200) / 365.25)) / 2
    event = RNG.gamma(1.3, 8, len(dates))  # occasional local pollution episodes

    frame = pd.DataFrame({"date": dates})
    frame["temperature"] = np.clip(19 + 13 * np.sin(2 * np.pi * (day - 105) / 365.25) + RNG.normal(0, 2.5, len(day)), 4, 45)
    frame["humidity"] = np.clip(48 + 30 * rain + RNG.normal(0, 9, len(day)), 20, 98)
    frame["wind_speed"] = np.clip(2.8 + 2.3 * rain + RNG.normal(0, 0.9, len(day)), 0.2, 12)
    dispersion = frame["wind_speed"] * 5 + (100 - frame["humidity"]) * .12
    frame["PM2.5"] = np.clip(35 + 80 * winter + event - dispersion + RNG.normal(0, 12, len(day)), 5, 340)
    frame["PM10"] = np.clip(frame["PM2.5"] * 1.45 + RNG.normal(12, 18, len(day)), 10, 490)
    frame["NO2"] = np.clip(22 + 52 * winter + event * .35 - dispersion * .35 + RNG.normal(0, 8, len(day)), 4, 460)
    frame["SO2"] = np.clip(7 + 17 * winter + event * .15 + RNG.normal(0, 4, len(day)), 2, 150)
    frame["CO"] = np.clip(.35 + 1.05 * winter + event * .012 - frame["wind_speed"] * .04 + RNG.normal(0, .12, len(day)), .1, 8)
    frame["AQI"] = frame.apply(calculate_aqi, axis=1)
    frame = frame[["date", "PM2.5", "PM10", "NO2", "SO2", "CO", "temperature", "humidity", "wind_speed", "AQI"]].round(2)
    OUT.parent.mkdir(exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Saved {len(frame)} synthetic daily records to {OUT}")


if __name__ == "__main__":
    main()
