import subprocess
import sys
import ast
from pathlib import Path

import pandas as pd
import json


MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "TabPFN",
]

OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


worker_code = r'''
import sys

from src.data.adult_experiment import get_adult_experiment_data

model_name = sys.argv[1]

if model_name == "Logistic Regression":
    from src.models.classical import create_adult_logistic_regression
    create_model = create_adult_logistic_regression

elif model_name == "Random Forest":
    from src.models.classical import create_adult_random_forest
    create_model = create_adult_random_forest

elif model_name == "XGBoost":
    from src.models.classical import create_adult_xgboost
    create_model = create_adult_xgboost

elif model_name == "TabPFN":
    from src.models.tabpfn_model import create_tabpfn
    create_model = create_tabpfn

else:
    raise ValueError(model_name)


X_train, X_test, y_train, y_test = get_adult_experiment_data()

model = create_model()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("X_TEST:", X_test.to_json(orient="records"))
print("TARGETS:", y_test.tolist())
print("PREDICTIONS:", predictions.tolist())
'''


def run_model(model_name):

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            worker_code,
            model_name,
        ],
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        print(process.stderr)
        raise RuntimeError(
            f"Worker failed: {model_name}"
        )

    X_test = None
    y_true = None
    y_pred = None

    for line in process.stdout.splitlines():

        if line.startswith("X_TEST:"):
            X_test = pd.DataFrame(
                json.loads(line[len("X_TEST:"):].strip())
            )

        elif line.startswith("TARGETS:"):
            y_true = ast.literal_eval(
                line[len("TARGETS:"):].strip()
            )

        elif line.startswith("PREDICTIONS:"):
            y_pred = ast.literal_eval(
                line[len("PREDICTIONS:"):].strip()
            )

    return X_test, pd.Series(y_true), pd.Series(y_pred)


all_summaries = []


for model_name in MODELS:

    print(
        f"\nAnalyzing Adult Income errors: {model_name}",
        flush=True,
    )

    X_test, y_true, y_pred = run_model(model_name)

    X_test = X_test.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    X_test["actual"] = y_true
    X_test["predicted"] = y_pred
    X_test["correct"] = (
        X_test["actual"] == X_test["predicted"]
    )

    correct = X_test[X_test["correct"]].copy()
    incorrect = X_test[~X_test["correct"]].copy()

    numeric_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    for column in numeric_columns:

        all_summaries.append({
            "model": model_name,
            "feature": column,
            "correct_mean": correct[column].mean(),
            "incorrect_mean": incorrect[column].mean(),
            "difference": (
                incorrect[column].mean()
                - correct[column].mean()
            ),
        })


summary_df = pd.DataFrame(all_summaries)

summary_file = (
    OUTPUT_DIR / "adult_numeric_error_analysis.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
)

print("\nNumeric feature error analysis:")
print(summary_df.to_string(index=False))

print(f"\nSaved: {summary_file}")


# ---------------------------------------------------------
# Categorical feature analysis
# ---------------------------------------------------------

categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


categorical_rows = []


for model_name in MODELS:

    print(
        f"\nCategorical analysis: {model_name}",
        flush=True,
    )

    X_test, y_true, y_pred = run_model(model_name)

    X_test = X_test.reset_index(drop=True)
    X_test["actual"] = y_true.reset_index(drop=True)
    X_test["predicted"] = y_pred.reset_index(drop=True)
    X_test["correct"] = (
        X_test["actual"] == X_test["predicted"]
    )

    correct = X_test[X_test["correct"]]
    incorrect = X_test[~X_test["correct"]]

    for column in categorical_columns:

        correct_distribution = (
            correct[column]
            .value_counts(normalize=True)
            .to_dict()
        )

        incorrect_distribution = (
            incorrect[column]
            .value_counts(normalize=True)
            .to_dict()
        )

        categories = set(correct_distribution) | set(
            incorrect_distribution
        )

        for category in categories:

            categorical_rows.append({
                "model": model_name,
                "feature": column,
                "category": category,
                "correct_fraction": correct_distribution.get(
                    category, 0
                ),
                "incorrect_fraction": incorrect_distribution.get(
                    category, 0
                ),
            })


categorical_df = pd.DataFrame(categorical_rows)

categorical_file = (
    OUTPUT_DIR / "adult_categorical_error_analysis.csv"
)

categorical_df.to_csv(
    categorical_file,
    index=False,
)

print(
    f"\nSaved: {categorical_file}"
)

print("\nAdult feature-level error analysis completed.")
