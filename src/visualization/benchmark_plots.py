from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FILE = Path("results/metrics/benchmark_results.csv")
OUTPUT_DIR = Path("results/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)


def save_metric_plot(metric, title, filename, ylabel=None):
    pivot = df.pivot(
        index="dataset",
        columns="model",
        values=metric,
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Model")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


# Predictive performance
save_metric_plot(
    "accuracy",
    "Model Accuracy Across Datasets",
    "accuracy_comparison.png",
    "Accuracy",
)

save_metric_plot(
    "f1",
    "F1 Score Across Datasets",
    "f1_comparison.png",
    "F1 Score",
)

save_metric_plot(
    "roc_auc",
    "ROC-AUC Across Datasets",
    "roc_auc_comparison.png",
    "ROC-AUC",
)


# Computational performance
save_metric_plot(
    "training_time",
    "Training Time Across Datasets",
    "training_time_comparison.png",
    "Training Time (seconds)",
)

save_metric_plot(
    "prediction_time",
    "Prediction Time Across Datasets",
    "prediction_time_comparison.png",
    "Prediction Time (seconds)",
)


print("Visualization generation complete.")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"Created: {file}")
