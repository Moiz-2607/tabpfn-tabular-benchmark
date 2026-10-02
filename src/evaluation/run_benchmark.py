import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd


MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "TabPFN",
]

DATASETS = [
    "Breast Cancer",
    "Wine",
    "Adult Income",
]


def run_model(model_name, dataset_name):
    print(f"\n{'=' * 70}", flush=True)
    print(f"Dataset: {dataset_name} | Model: {model_name}", flush=True)
    print(f"{'=' * 70}", flush=True)

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.model_worker",
            model_name,
            dataset_name,
        ],
        capture_output=True,
        text=True,
    )

    print(process.stdout, end="", flush=True)

    if process.returncode != 0:
        print(
            f"ERROR: {model_name} on {dataset_name} "
            f"exited with code {process.returncode}",
            flush=True,
        )

        if process.stderr:
            print(process.stderr, flush=True)

        return None

    for line in process.stdout.splitlines():
        if line.startswith("RESULT:"):
            return ast.literal_eval(
                line[len("RESULT:"):].strip()
            )

    print("Could not find benchmark result.", flush=True)
    return None


def main():
    results = []

    for dataset_name in DATASETS:
        for model_name in MODELS:
            result = run_model(model_name, dataset_name)

            if result is not None:
                results.append(result)
                print(
                    f"Completed: {dataset_name} | {model_name}",
                    flush=True,
                )

    if not results:
        print("No benchmark results were produced.")
        return

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("FINAL MULTI-DATASET BENCHMARK")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_results.csv"

    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
