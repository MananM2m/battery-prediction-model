"""Generate plots and a project summary from evaluation metrics."""
import os
import glob
import pandas as pd
from battery_prediction.visualization.plots import plot_model_comparison
from battery_prediction.features.engineering import feature_columns


def latest_metrics_csv():
    files = glob.glob(os.path.join("results", "metrics", "*.csv"))
    if not files:
        raise FileNotFoundError("No metrics CSV found in results/metrics")
    return sorted(files)[-1]


def generate_summary():
    metrics_path = latest_metrics_csv()
    df = pd.read_csv(metrics_path)
    os.makedirs("results/plots", exist_ok=True)
    plot_model_comparison(df.to_dict(orient="records"), os.path.join("results", "plots", "rmse_comparison.png"))

    summary_path = os.path.join("results", "project_summary.md")
    ds_size = 0
    if os.path.exists("data/telemetry.csv"):
        ds_size = sum(1 for _ in open("data/telemetry.csv")) - 1

    with open(summary_path, "w") as fh:
        fh.write("# Project Summary\n\n")
        fh.write(f"**Dataset size:** {ds_size} rows\n\n")
        fh.write(f"**Number of features:** {len(feature_columns())}\n\n")
        fh.write("**Targets:** power, minutes_left\n\n")
        fh.write("## Best models by target\n\n")
        for target in ["power", "minutes"]:
            sub = df[df.target == target]
            if sub.empty:
                continue
            best = sub.loc[sub.rmse.idxmin()]
            fh.write(f"- **{target}**: {best['model']} (MAE={best['mae']:.3f}, RMSE={best['rmse']:.3f}, R²={best['r2']:.3f})\n")
        fh.write("\nPlots saved in `results/plots/` and metrics in `results/metrics/`.\n")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    generate_summary()
