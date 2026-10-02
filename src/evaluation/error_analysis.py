import subprocess
import sys
import ast
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


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

OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_worker(model_name, dataset_name):

    worker_code = r'''
import sys

from sklearn.model_selection import train_test_split

from src.data.datasets import get_dataset
from src.data.adult_experiment import get_adult_experiment_data

model_name = sys.argv[1]
dataset_name = sys.argv[2]

if model_name == "Logistic Regression":
    from src.models.classical import (
        create_logistic_regression,
        create_adult_logistic_regression,
    )
    create_model = (
        create_adult_logistic_regression
        if dataset_name == "Adult Income"
        else create_logistic_regression
    )

elif model_name == "Random Forest":
    from src.models.classical import (
        create_random_forest,
        create_adult_random_forest,
    )
    create_model = (
        create_adult_random_forest
        if dataset_name == "Adult Income"
        else create_random_forest
    )

elif model_name == "XGBoost":
    from src.models.classical import (
        create_xgboost,
        create_adult_xgboost,
    )
    create_model = (
        create_adult_xgboost
        if dataset_name == "Adult Income"
        else create_xgboost
    )

elif model_name == "TabPFN":
    from src.models.tabpfn_model import create_tabpfn
    create_model = create_tabpfn

else:
    raise ValueError(model_name)


if dataset_name == "Adult Income":
    X_train, X_test, y_train, y_test = get_adult_experiment_data()

else:
    data = get_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        data["X"],
        data["y"],
        test_size=0.2,
        random_state=42,
        stratify=data["y"],
    )


model = create_model()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("TARGETS:", y_test.tolist())
print("PREDICTIONS:", predictions.tolist())
'''

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            worker_code,
            model_name,
            dataset_name,
        ],
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        print(process.stderr)
        raise RuntimeError(
            f"Worker failed: {model_name} | {dataset_name}"
        )

    targets = None
    predictions = None

    for line in process.stdout.splitlines():

        if line.startswith("TARGETS:"):
            targets = ast.literal_eval(
                line[len("TARGETS:"):].strip()
            )

        elif line.startswith("PREDICTIONS:"):
            predictions = ast.literal_eval(
                line[len("PREDICTIONS:"):].strip()
            )

    return targets, predictions


all_results = []

for dataset_name in DATASETS:

    for model_name in MODELS:

        print(
            f"Analyzing errors: "
            f"{dataset_name} | {model_name}",
            flush=True,
        )

        y_true, y_pred = run_worker(
            model_name,
            dataset_name,
        )

        cm = confusion_matrix(y_true, y_pred)

        total = cm.sum()

        errors = total - cm.trace()

        error_rate = errors / total

        result = {
            "dataset": dataset_name,
            "model": model_name,
            "total_samples": total,
            "errors": errors,
            "error_rate": error_rate,
        }

        if cm.shape == (2, 2):

            result["true_negatives"] = cm[0, 0]
            result["false_positives"] = cm[0, 1]
            result["false_negatives"] = cm[1, 0]
            result["true_positives"] = cm[1, 1]

        else:

            result["true_negatives"] = None
            result["false_positives"] = None
            result["false_negatives"] = None
            result["true_positives"] = None

        all_results.append(result)


df = pd.DataFrame(all_results)

output_file = OUTPUT_DIR / "error_analysis.csv"

df.to_csv(output_file, index=False)

print()
print("Error analysis completed successfully.")
print(f"Saved: {output_file}")
print()
print(df.to_string(index=False))
