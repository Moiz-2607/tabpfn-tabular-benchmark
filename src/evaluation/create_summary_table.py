import os

import pandas as pd


INPUT_FILE = "results/metrics/benchmark_results.csv"
OUTPUT_FILE = "results/tables/benchmark_summary.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    columns = [
        "dataset",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "training_time",
        "prediction_time",
    ]

    summary = df[columns].copy()

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    time_columns = [
        "training_time",
        "prediction_time",
    ]

    summary[metric_columns] = summary[metric_columns].round(4)
    summary[time_columns] = summary[time_columns].round(4)

    os.makedirs("results/tables", exist_ok=True)

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(summary.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
