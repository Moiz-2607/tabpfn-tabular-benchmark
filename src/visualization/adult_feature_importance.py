import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data.adult_experiment import get_adult_experiment_data
from src.models.classical import (
    create_adult_logistic_regression,
    create_adult_random_forest,
    create_adult_xgboost,
)


OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_analysis(model_name, model):
    print(f"Analyzing feature importance: {model_name}", flush=True)

    X_train, X_test, y_train, y_test = get_adult_experiment_data()

    model.fit(X_train, y_train)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1",
        n_jobs=1,
    )

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print("\nTop features:", flush=True)
    print(importance.head(10).to_string(index=False), flush=True)

    plt.figure(figsize=(9, 6))

    top = importance.head(10).sort_values("importance")

    plt.barh(
        top["feature"],
        top["importance"],
    )

    plt.xlabel("Mean decrease in F1 after permutation")
    plt.ylabel("Feature")
    plt.title(f"Adult Income — {model_name} Permutation Importance")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_adult_permutation_importance.png"
    )

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}", flush=True)


def main():
    models = {
        "Logistic Regression": create_adult_logistic_regression(),
        "Random Forest": create_adult_random_forest(),
        "XGBoost": create_adult_xgboost(),
    }

    for name, model in models.items():
        run_analysis(name, model)


if __name__ == "__main__":
    main()
