"""Interactive Streamlit dashboard for the synthetic AQI project."""
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from urllib.parse import urlencode
from urllib.request import urlopen
from preprocess import FEATURES, RAW_COLUMNS, feature_row, prepare_data

ROOT = Path(__file__).resolve().parent
GEO_LOCATOR = components.declare_component("geo_locator", path=str(ROOT / "components" / "geo_locator"))
st.set_page_config(page_title="AirLens | AQI Intelligence", page_icon="🌿", layout="wide")


def aqi_band(value):
    bands = [(50, "Good", "Outdoor activity is safe."), (100, "Satisfactory", "Sensitive people can limit prolonged exertion."),
             (200, "Moderate", "Consider reducing long outdoor activity."), (300, "Poor", "Limit outdoor exertion; sensitive groups should stay indoors."),
             (400, "Very Poor", "Avoid outdoor exertion and keep windows closed."), (float("inf"), "Severe", "Stay indoors when possible; use clean indoor air.")]
    return next((name, advice) for limit, name, advice in bands if value <= limit)


def us_aqi_band(value):
    bands = [(50, "Good", "Outdoor activity is safe."), (100, "Moderate", "Unusually sensitive people can reduce prolonged exertion."),
             (150, "Unhealthy for sensitive groups", "Children, older adults, and people with respiratory illness should reduce outdoor exertion."),
             (200, "Unhealthy", "Reduce outdoor activity and close windows if air quality worsens."), (300, "Very unhealthy", "Avoid outdoor exertion."),
             (float("inf"), "Hazardous", "Stay indoors where possible and use clean indoor air.")]
    return next((name, advice) for limit, name, advice in bands if value <= limit)


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(place):
    """Resolve a city/postcode using Open-Meteo's free geocoding service."""
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode({"name": place, "count": 1, "language": "en", "format": "json"})
    with urlopen(url, timeout=12) as response:
        result = json.load(response).get("results", [])
    if not result:
        raise ValueError("Location not found. Try a city, state, country, or postcode.")
    return result[0]


@st.cache_data(ttl=1800, show_spinner=False)
def live_air_quality(latitude, longitude, timezone):
    """Get free live conditions and up to seven days of location-specific AQI."""
    variables = "us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide"
    query = urlencode({"latitude": latitude, "longitude": longitude, "current": variables, "hourly": variables, "forecast_days": 7, "timezone": timezone})
    with urlopen("https://air-quality-api.open-meteo.com/v1/air-quality?" + query, timeout=12) as response:
        payload = json.load(response)
    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    daily = hourly.set_index("time").resample("D").agg({"us_aqi": "max", "pm2_5": "mean", "pm10": "mean", "nitrogen_dioxide": "mean"}).reset_index()
    return payload["current"], daily


@st.cache_data(ttl=1800, show_spinner=False)
def live_weather(latitude, longitude, timezone):
    """Get free location-specific weather used by the live dashboard and assistant."""
    query = urlencode({"latitude": latitude, "longitude": longitude, "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code", "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "forecast_days": 7, "timezone": timezone})
    with urlopen("https://api.open-meteo.com/v1/forecast?" + query, timeout=12) as response:
        return json.load(response)


def assistant_reply(question, location_name, current_air, weather, forecast):
    """A free rule-based assistant grounded only in the dashboard's visible data."""
    question = question.lower()
    if current_air is None:
        return "Choose a location first, then I can explain its live weather and air quality."
    if any(word in question for word in ["weather", "temperature", "rain", "wind", "humidity"]):
        current = weather["current"]
        return (f"In {location_name}, the current temperature is {current['temperature_2m']:.1f}°C, humidity is "
                f"{current['relative_humidity_2m']:.0f}%, and wind speed is {current['wind_speed_10m']:.1f} km/h.")
    if any(word in question for word in ["pm", "pollution", "no2", "no₂", "co", "so2", "so₂"]):
        return (f"Current pollution in {location_name}: PM2.5 {current_air.get('pm2_5', 0):.1f} µg/m³, "
                f"PM10 {current_air.get('pm10', 0):.1f} µg/m³, NO₂ {current_air.get('nitrogen_dioxide', 0):.1f} µg/m³, "
                f"and CO {current_air.get('carbon_monoxide', 0):.0f} µg/m³.")
    if any(word in question for word in ["forecast", "tomorrow", "next", "week"]):
        tomorrow = forecast.iloc[min(1, len(forecast) - 1)]
        return f"Tomorrow’s location forecast has a maximum US AQI of {tomorrow['us_aqi']:.0f}. The chart above shows the available daily outlook."
    if "aqi" in question or "air" in question or "safe" in question:
        value = float(current_air.get("us_aqi", 0))
        band, advice = us_aqi_band(value)
        return f"The live US AQI in {location_name} is {value:.0f} ({band}). {advice}"
    return "Ask me about the current AQI, PM2.5/PM10/NO₂ pollution, weather, or the location forecast shown on this dashboard."


@st.cache_data
def load_data(_version):
    return prepare_data()


@st.cache_resource
def load_saved_model(_version):
    path = ROOT / "models" / "best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def predict(saved, row):
    if saved["name"] != "LSTM":
        return float(saved["model"].predict(row)[0])
    from tensorflow import keras
    model = keras.models.load_model(ROOT / "models" / "best_lstm.keras")
    history = pd.concat([data[FEATURES].tail(saved["lookback"] - 1), row])
    sequence = saved["scaler"].transform(history).reshape(1, saved["lookback"], -1)
    return float(model.predict(sequence, verbose=0)[0, 0])


def forecast_next_week(saved, history):
    """Forecast from the recent 14-day average; no external API is required."""
    dates = pd.date_range(history.date.iloc[-1] + pd.Timedelta(days=1), periods=7, freq="D")
    baseline = history[RAW_COLUMNS].tail(14).mean().to_dict()
    rows = pd.concat([feature_row(baseline, date) for date in dates], ignore_index=True)
    if saved is None:
        return pd.DataFrame({"date": dates, "Forecast AQI": np.nan})
    if saved["name"] != "LSTM":
        values = saved["model"].predict(rows)
    else:
        from tensorflow import keras
        model = keras.models.load_model(ROOT / "models" / "best_lstm.keras")
        feature_history, values = history[FEATURES].tail(saved["lookback"] - 1).copy(), []
        for _, row in rows.iterrows():
            sequence = pd.concat([feature_history, row.to_frame().T])
            scaled = saved["scaler"].transform(sequence).reshape(1, saved["lookback"], -1)
            values.append(model.predict(scaled, verbose=0)[0, 0])
            feature_history = sequence.tail(saved["lookback"] - 1)
    return pd.DataFrame({"date": dates, "Forecast AQI": np.maximum(values, 0)})


data_file, model_file = ROOT / "data" / "aqi_data.csv", ROOT / "models" / "best_model.pkl"
data = load_data(data_file.stat().st_mtime_ns)
saved = load_saved_model(model_file.stat().st_mtime_ns if model_file.exists() else 0)
week_forecast = forecast_next_week(saved, data)
latest = float(data.iloc[-1]["AQI"])
label, advice = aqi_band(latest)
st.markdown("""<style>
.hero {padding: 1.2rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#0b3d2e,#176b4d); color:white; margin-bottom:1rem}
.hero h1 {margin:0; font-size:2.1rem}.hero p {margin:.35rem 0 0; opacity:.85}
</style>""", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>🌿 AirLens</h1><p>Explore air quality, understand exposure, and estimate tomorrow's AQI.</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📍 Live location")
    with st.form("location_form"):
        place = st.text_input("City, state, country, or postcode", value=st.session_state.get("place", "New Delhi, India"))
        location_submitted = st.form_submit_button("Update location", type="primary")
    if location_submitted:
        st.session_state["place"] = place
        st.session_state["location_source"] = "manual"
    st.caption("Or use your device location. Chrome will ask for permission after you click the button.")
    geo_response = GEO_LOCATOR(key="geo_locator", default={})

if isinstance(geo_response, dict) and geo_response.get("request_id") != st.session_state.get("last_geo_request"):
    st.session_state["last_geo_request"] = geo_response.get("request_id")
    if geo_response.get("action") == "location":
        st.session_state["device_coordinates"] = (geo_response["latitude"], geo_response["longitude"])
        st.session_state["location_source"] = "device"
    elif geo_response.get("action") == "error":
        st.session_state["geo_error"] = geo_response.get("message", "Browser location was unavailable.")

location_query = st.session_state.get("place", "New Delhi, India")
try:
    if st.session_state.get("location_source") == "device" and "device_coordinates" in st.session_state:
        latitude, longitude = st.session_state["device_coordinates"]
        location = {"name": "Current device location", "latitude": latitude, "longitude": longitude, "timezone": "auto"}
    else:
        location = geocode(location_query)
    current_air, local_daily = live_air_quality(location["latitude"], location["longitude"], location.get("timezone", "auto"))
    local_weather = live_weather(location["latitude"], location["longitude"], location.get("timezone", "auto"))
    location_error = None
except Exception as error:
    location, current_air, local_daily, local_weather, location_error = None, None, None, None, str(error)

tab_live, tab_overview, tab_predict, tab_models, tab_data = st.tabs(["Live location", "Overview", "AQI predictor", "Model lab", "Data explorer"])
with tab_live:
    st.subheader("Location-specific air quality")
    if location_error:
        st.warning(f"Live location data is unavailable: {location_error}")
    else:
        place_label = ", ".join(filter(None, [location.get("name"), location.get("admin1"), location.get("country")]))
        st.caption(f"{place_label} · {location['latitude']:.3f}, {location['longitude']:.3f} · Free Open-Meteo data")
        local_aqi = float(current_air.get("us_aqi", 0))
        live_band, live_advice = us_aqi_band(local_aqi)
        a, b, c, d = st.columns(4)
        a.metric("Live US AQI", f"{local_aqi:.0f}", live_band)
        b.metric("PM2.5", f"{current_air.get('pm2_5', 0):.1f} µg/m³")
        c.metric("PM10", f"{current_air.get('pm10', 0):.1f} µg/m³")
        d.metric("NO₂", f"{current_air.get('nitrogen_dioxide', 0):.1f} µg/m³")
        weather_now = local_weather["current"]
        w1, w2, w3 = st.columns(3)
        w1.metric("Temperature", f"{weather_now['temperature_2m']:.1f} °C")
        w2.metric("Humidity", f"{weather_now['relative_humidity_2m']:.0f}%")
        w3.metric("Wind speed", f"{weather_now['wind_speed_10m']:.1f} km/h")
        st.info(f"**Health guidance — {live_band}:** {live_advice}")
        local_chart = local_daily.rename(columns={"time": "date", "us_aqi": "Location AQI"})[["date", "Location AQI"]].set_index("date")
        st.subheader("Location AQI forecast")
        st.line_chart(local_chart, color="#38bdf8")
        display_daily = local_daily.rename(columns={"time": "Date", "us_aqi": "US AQI", "pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO₂"})
        st.dataframe(display_daily.style.format({"US AQI": "{:.0f}", "PM2.5": "{:.1f}", "PM10": "{:.1f}", "NO₂": "{:.1f}"}), hide_index=True, use_container_width=True)
        st.caption(f"Updated: {current_air.get('time', 'unknown local time')}. Live AQI uses the US AQI scale and is separate from this project’s synthetic CPCB-style training dataset.")
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest AQI", f"{latest:.0f}", label)
    c2.metric("7-day average", f"{data.AQI.tail(7).mean():.0f}")
    c3.metric("Peak AQI", f"{data.AQI.max():.0f}")
    c4.metric("Cleanest day", f"{data.AQI.min():.0f}")
    st.info(f"**Today’s guidance — {label}:** {advice}")
    st.subheader("7-day AQI outlook")
    if saved is None:
        st.warning("Train a model to enable the seven-day forecast.")
    else:
        f1, f2, f3 = st.columns(3)
        f1.metric("Tomorrow", f"{week_forecast.iloc[0, 1]:.0f}")
        f2.metric("7-day average", f"{week_forecast.iloc[:, 1].mean():.0f}")
        f3.metric("Forecast model", saved["name"])
        st.line_chart(week_forecast.set_index("date"), color="#f59e0b")
        forecast_display = week_forecast.copy()
        forecast_display["Air-quality band"] = forecast_display["Forecast AQI"].map(lambda value: aqi_band(value)[0])
        st.dataframe(forecast_display.style.format({"Forecast AQI": "{:.0f}"}), hide_index=True, use_container_width=True)
        st.caption("Forecast uses the trained model and the last 14 days of readings as a baseline. It is a planning estimate, not a live sensor feed.")
    start, end = st.columns(2)
    begin = start.date_input("From", value=data.date.iloc[0].date(), min_value=data.date.iloc[0].date(), max_value=data.date.iloc[-1].date())
    finish = end.date_input("To", value=data.date.iloc[-1].date(), min_value=data.date.iloc[0].date(), max_value=data.date.iloc[-1].date())
    window = data[data.date.between(pd.Timestamp(begin), pd.Timestamp(finish))]
    st.area_chart(window.set_index("date")[["AQI"]], color="#22a06b")
    st.caption(f"Historical window: {data.date.iloc[0]:%d %b %Y} to {data.date.iloc[-1]:%d %b %Y}. Use the date range to focus on a season or episode.")
    pollutant = st.selectbox("Compare AQI with", ["PM2.5", "PM10", "NO2", "SO2", "CO"])
    st.line_chart(window.set_index("date")[["AQI", pollutant]])

with tab_predict:
    st.subheader("Personal AQI scenario")
    st.caption("Enter expected daily readings. Weather defaults to the historical median.")
    defaults = data[RAW_COLUMNS].median()
    with st.form("prediction_form"):
        left, right = st.columns(2)
        values = {}
        for i, name in enumerate(RAW_COLUMNS):
            target = left if i % 2 == 0 else right
            values[name] = target.number_input(name, min_value=0.0, value=float(defaults[name]), step=0.1)
        submitted = st.form_submit_button("Estimate AQI", type="primary")
    if submitted:
        if saved is None:
            st.warning("No trained model found. Run `python train.py`, then refresh this page.")
        else:
            result = max(0, predict(saved, feature_row(values)))
            name, guidance = aqi_band(result)
            dominant = max(["PM2.5", "PM10", "NO2", "SO2", "CO"], key=lambda item: values[item] / max(float(defaults[item]), .1))
            a, b, c = st.columns(3)
            a.metric("Estimated AQI", f"{result:.0f}")
            b.metric("Air-quality band", name)
            c.metric("Likely driver", dominant)
            st.success(guidance)
            # Simple seven-day planning view with the scenario's weather and pollutant values.
            future = pd.concat([feature_row(values, data.date.iloc[-1] + pd.Timedelta(days=i)) for i in range(1, 8)], ignore_index=True)
            if saved["name"] != "LSTM":
                forecast = saved["model"].predict(future)
                st.line_chart(pd.DataFrame({"date": pd.date_range(data.date.iloc[-1] + pd.Timedelta(days=1), periods=7), "Estimated AQI": forecast}).set_index("date"))
            else:
                st.caption("LSTM prediction is shown above; its multi-day forecast needs fresh daily measurements.")

with tab_models:
    st.subheader("Model comparison")
    metrics_path = ROOT / "models" / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics.style.format({"RMSE": "{:.2f}", "R2": "{:.3f}"}), hide_index=True, use_container_width=True)
        winner = metrics.loc[metrics.Status == "ready"].iloc[0]
        st.success(f"Active model: **{winner.Model}** — lower RMSE and higher R² indicate a better held-out fit.")
    else:
        st.warning("Train the models to view evaluation results.")
    st.code("python train.py\npython visualize.py", language="powershell")

with tab_data:
    st.subheader("Explore the generated dataset")
    st.download_button("Download CSV", data.to_csv(index=False).encode("utf-8"), "aqi_data_cleaned.csv", "text/csv")
    st.dataframe(data.sort_values("date", ascending=False), hide_index=True, use_container_width=True)

# A compact floating assistant, grounded in information currently visible in the app.
st.markdown("""<style>
div[data-testid="stPopover"] {position:fixed; right:1.2rem; bottom:1.2rem; z-index:99999;}
</style>""", unsafe_allow_html=True)
with st.popover("💬 Ask AirLens"):
    st.caption("Answers use the live location data displayed above.")
    for role, message in st.session_state.get("assistant_messages", [])[-4:]:
        with st.chat_message(role):
            st.write(message)
    with st.form("assistant_form", clear_on_submit=True):
        question = st.text_input("Ask about AQI, pollution, weather, or forecast")
        asked = st.form_submit_button("Ask")
    if asked and question:
        label = location.get("name", "your location") if location else "your location"
        answer = assistant_reply(question, label, current_air, local_weather, local_daily)
        st.session_state.setdefault("assistant_messages", []).extend([("user", question), ("assistant", answer)])
        st.rerun()
