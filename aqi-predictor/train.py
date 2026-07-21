"""Train AQI models. The LSTM is optional so the project works without TensorFlow."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from preprocess import FEATURES, prepare_data

try:  # TensorFlow wheels are not available for every Python version.
    from tensorflow import keras
    TF_ERROR = None
except Exception as error:  # Keep useful sklearn models available.
    keras, TF_ERROR = None, str(error)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def score(actual, predicted):
    return {"RMSE": mean_squared_error(actual, predicted) ** .5, "R2": r2_score(actual, predicted)}


def main():
    data = prepare_data()
    split, lookback = int(len(data) * .8), 7
    X, y = data[FEATURES], data["AQI"]
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    fitted, records, predictions = {}, [], {}

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=350, min_samples_leaf=2, max_features=.85, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        fitted[name], predictions[name] = model, prediction
        records.append({"Model": name, "Status": "ready", **score(y_test, prediction)})

    if keras is not None:
        scaler = StandardScaler().fit(X_train)
        scaled = scaler.transform(X)
        # Each sequence includes the current day's measurements plus six prior days.
        sequence = lambda indices: np.array([scaled[i - lookback + 1:i + 1] for i in indices])
        train_idx, test_idx = np.arange(lookback - 1, split), np.arange(split, len(data))
        keras.utils.set_random_seed(42)
        lstm = keras.Sequential([
            keras.Input((lookback, len(FEATURES))), keras.layers.LSTM(24, dropout=.1), keras.layers.Dense(12, activation="relu"), keras.layers.Dense(1),
        ])
        lstm.compile(optimizer="adam", loss="mse")
        lstm.fit(sequence(train_idx), y.iloc[train_idx], validation_split=.15, epochs=40, batch_size=32, verbose=0)
        prediction = lstm.predict(sequence(test_idx), verbose=0).ravel()
        records.append({"Model": "LSTM", "Status": "ready", **score(y_test, prediction)})
        predictions["LSTM"] = prediction
    else:
        records.append({"Model": "LSTM", "Status": "TensorFlow unavailable", "RMSE": np.nan, "R2": np.nan})
        print(f"LSTM skipped: {TF_ERROR}")

    results = pd.DataFrame(records).sort_values("RMSE", na_position="last").reset_index(drop=True)
    results.to_csv(MODELS / "metrics.csv", index=False)
    pd.DataFrame({"date": data.loc[X_test.index, "date"].to_numpy(), "actual": y_test.to_numpy(), **predictions}).to_csv(MODELS / "predictions.csv", index=False)
    best = results.loc[results["Status"] == "ready"].iloc[0]["Model"]
    if best == "LSTM":
        lstm.save(MODELS / "best_lstm.keras")
        payload = {"name": best, "scaler": scaler, "lookback": lookback}
    else:
        payload = {"name": best, "model": fitted[best], "lookback": lookback}
    with open(MODELS / "best_model.pkl", "wb") as file:
        pickle.dump(payload, file)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved best model: {best}")


if __name__ == "__main__":
    main()
