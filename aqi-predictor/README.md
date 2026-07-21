# AQI Predictor

An end-to-end, synthetic daily air-quality project. The generator simulates the most recent rolling two years of weather and pollutant readings and calculates AQI as the maximum CPCB pollutant sub-index.

## Setup

```powershell
cd aqi-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### TensorFlow note (Windows / Python 3.13)

The dashboard and the two scikit-learn models work without TensorFlow. The LSTM is trained automatically only when TensorFlow imports correctly. The `No module named 'tensorflow.python'` error indicates a broken or incompatible TensorFlow installation, not an AQI-project error.

For the LSTM, use a clean Python 3.11 virtual environment, then reinstall dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, run the project as-is with Python 3.13: `train.py` will train Linear Regression and Random Forest and clearly mark LSTM as unavailable.

## Run

Run these commands in order:

```powershell
python data/generate_data.py
python train.py
python visualize.py
streamlit run app.py
```

Outputs:

- `data/aqi_data.csv` — 730 synthetic daily readings.
- `models/metrics.csv`, `models/predictions.csv`, and the saved best model.
- `plots/aqi_trend.png`, `plots/correlation_heatmap.png`, and `plots/actual_vs_predicted.png`.

The upgraded dashboard includes a location-specific live AQI tab, the full previous two-year history, date-range exploration, pollutant comparison, AQI health guidance, CSV download, model diagnostics, a scenario predictor, and an automatic seven-day planning forecast. In the sidebar, click **Use my current location** and allow Chrome's location prompt, or enter a city, state, country, or postcode manually. The floating **Ask AirLens** button answers questions from the live AQI, pollution, weather, and forecast shown on the page. It uses Open-Meteo's free geocoding, weather, and air-quality services, with no API key or paid service. The project model's forecast uses the trained local model with a recent 14-day baseline. The model uses pollutant concentrations, weather, season, and weekday. For reproducibility, data generation and model seeds are fixed.
