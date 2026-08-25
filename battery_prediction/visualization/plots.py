import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_actual_vs_predicted(y_true, y_pred, out_path, title="Actual vs Predicted"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=10)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_model_comparison(metrics_df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(metrics_df)
    # pivot for RMSE comparison
    pivot = df.pivot(index="model", columns="target", values="rmse")
    pivot.plot(kind="bar")
    plt.title("RMSE by model and target")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
