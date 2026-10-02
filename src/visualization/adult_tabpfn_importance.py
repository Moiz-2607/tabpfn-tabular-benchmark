import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data.adult_experiment import get_adult_experiment_data
from src.models.tabpfn_model import create_tabpfn


OUTPUT_DIR = "results/plots"
OUTPUT_TABLE = "results/tables/adult_tabpfn_permutation_importance.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


def main():
    print("Loading Adult Income experiment data...", flush=True)

    X_train, X_test, y_train, y_test = get_adult_experiment_data()

    print("Creating TabPFN...", flush=True)

    model = create_tabpfn()

    print("Training TabPFN...", flush=True)

    model.fit(X_train, y_train)

    print("Calculating permutation importance...", flush=True)

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
        "std": result.importances_std,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print("\nTabPFN Top Features:")
    print(importance.head(10).to_string(index=False))

    importance.to_csv(
        OUTPUT_TABLE,
        index=False,
    )

    top = importance.head(10).sort_values("importance")

    plt.figure(figsize=(9, 6))

    plt.barh(
        top["feature"],
        top["importance"],
        xerr=top["std"],
    )

    plt.xlabel("Mean decrease in F1 after permutation")
    plt.ylabel("Feature")
    plt.title("Adult Income — TabPFN Permutation Importance")

    plt.tight_layout()

    output_plot = os.path.join(
        OUTPUT_DIR,
        "tabpfn_adult_permutation_importance.png",
    )

    plt.savefig(output_plot, dpi=200)
    plt.close()

    print(f"\nSaved: {OUTPUT_TABLE}")
    print(f"Saved: {output_plot}")


if __name__ == "__main__":
    main()
