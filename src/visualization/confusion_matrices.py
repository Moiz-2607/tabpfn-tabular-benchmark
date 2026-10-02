import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


RESULTS_DIR = "results/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = [
    "Breast Cancer",
    "Wine",
    "Adult Income",
]

MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "TabPFN",
]


def run_model(dataset, model):
    """Train one model and return true/predicted labels."""

    worker_code = f"""
import json

from src.evaluation.model_worker import (
    dataset_name,
    model_name,
    X_test,
    y_test,
    model,
)

predictions = model.predict(X_test)

print("Y_TEST:", json.dumps(y_test.tolist()), flush=True)
print("PREDICTIONS:", json.dumps(predictions.tolist()), flush=True)
"""

    # We cannot import model_worker directly because it expects command-line
    # arguments. Instead, execute it with the correct arguments and append
    # prediction extraction in the same process.
    code = f"""
import json
import runpy
import sys

sys.argv = [
    "model_worker.py",
    {model!r},
    {dataset!r},
]

namespace = runpy.run_module(
    "src.evaluation.model_worker",
    run_name="__main__"
)

model = namespace["model"]
X_test = namespace["X_test"]
y_test = namespace["y_test"]

predictions = model.predict(X_test)

print("CONFUSION_Y_TEST:", json.dumps(y_test.tolist()), flush=True)
print("CONFUSION_PREDICTIONS:", json.dumps(predictions.tolist()), flush=True)
"""

    process = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        print(process.stderr)
        raise RuntimeError(f"Failed: {model} on {dataset}")

    y_true = None
    y_pred = None

    for line in process.stdout.splitlines():
        if line.startswith("CONFUSION_Y_TEST:"):
            y_true = json.loads(
                line[len("CONFUSION_Y_TEST:"):].strip()
            )

        elif line.startswith("CONFUSION_PREDICTIONS:"):
            y_pred = json.loads(
                line[len("CONFUSION_PREDICTIONS:"):].strip()
            )

    if y_true is None or y_pred is None:
        print(process.stdout)
        raise RuntimeError(
            f"Could not retrieve predictions for {model} on {dataset}"
        )

    return y_true, y_pred


def main():
    for dataset in DATASETS:
        for model in MODELS:

            print(
                f"Generating confusion matrix: "
                f"{dataset} - {model}"
            )

            y_true, y_pred = run_model(dataset, model)

            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(6, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
            )

            plt.title(f"{model} — {dataset}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            plt.tight_layout()

            filename = (
                dataset.lower().replace(" ", "_")
                + "_"
                + model.lower().replace(" ", "_")
                + "_confusion_matrix.png"
            )

            path = os.path.join(RESULTS_DIR, filename)

            plt.savefig(path, dpi=200)
            plt.close()

            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
