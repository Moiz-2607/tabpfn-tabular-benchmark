import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_FILE = "results/metrics/benchmark_results.csv"
OUTPUT_FILE = "results/plots/f1_heatmap.png"


def main():
    df = pd.read_csv(INPUT_FILE)

    pivot = df.pivot(
        index="dataset",
        columns="model",
        values="f1",
    )

    # Keep the benchmark order consistent.
    dataset_order = [
        "Breast Cancer",
        "Wine",
        "Adult Income",
    ]

    model_order = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "TabPFN",
    ]

    pivot = pivot.reindex(
        index=dataset_order,
        columns=model_order,
    )

    plt.figure(figsize=(10, 5))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "F1 Score"},
    )

    plt.title("F1 Score Comparison Across Models and Datasets")
    plt.xlabel("Model")
    plt.ylabel("Dataset")

    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close()

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
