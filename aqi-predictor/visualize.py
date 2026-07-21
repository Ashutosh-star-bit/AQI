"""Create standalone charts; it also works before model training."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocess import prepare_data

ROOT, PLOTS = Path(__file__).resolve().parent, Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)


def main():
    sns.set_theme(style="whitegrid", palette="deep")
    data = prepare_data()
    plt.figure(figsize=(11, 4)); sns.lineplot(data=data, x="date", y="AQI", color="#d95f02")
    plt.axhline(100, color="#777", ls="--", lw=1); plt.title("Daily AQI trend"); plt.tight_layout(); plt.savefig(PLOTS / "aqi_trend.png", dpi=150); plt.close()
    plt.figure(figsize=(9, 7)); sns.heatmap(data.select_dtypes("number").corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Feature correlation"); plt.tight_layout(); plt.savefig(PLOTS / "correlation_heatmap.png", dpi=150); plt.close()
    metrics, prediction_path = ROOT / "models" / "metrics.csv", ROOT / "models" / "predictions.csv"
    if not (metrics.exists() and prediction_path.exists()):
        print("Saved trend and correlation plots. Run train.py to create actual_vs_predicted.png.")
        return
    pred, best = pd.read_csv(prediction_path), pd.read_csv(metrics).query("Status == 'ready'").iloc[0]["Model"]
    plt.figure(figsize=(6, 6)); sns.scatterplot(data=pred, x="actual", y=best, alpha=.7)
    limit = [min(pred["actual"].min(), pred[best].min()), max(pred["actual"].max(), pred[best].max())]
    plt.plot(limit, limit, "k--"); plt.xlabel("Actual AQI"); plt.ylabel("Predicted AQI"); plt.title(f"Actual vs predicted ({best})")
    plt.tight_layout(); plt.savefig(PLOTS / "actual_vs_predicted.png", dpi=150); plt.close()
    print(f"Saved plots to {PLOTS}")


if __name__ == "__main__":
    main()
