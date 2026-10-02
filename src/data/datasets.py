from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml


def load_breast_cancer_dataset():
    data = load_breast_cancer()

    return {
        "name": "Breast Cancer",
        "X": data.data,
        "y": data.target,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
    }


def load_wine_dataset():
    data = load_wine()

    return {
        "name": "Wine",
        "X": data.data,
        "y": data.target,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
    }


def load_adult_income_dataset():
    data = fetch_openml(
        "adult",
        version=2,
        as_frame=True,
    )

    X = data.data.copy()
    y = data.target.copy()
    y = y.map({'<=50K': 0, '>50K': 1})

    return {
        "name": "Adult Income",
        "X": X,
        "y": y,
        "feature_names": X.columns.tolist(),
        "target_names": y.unique().tolist(),
    }


DATASETS = {
    "Breast Cancer": load_breast_cancer_dataset,
    "Wine": load_wine_dataset,
    "Adult Income": load_adult_income_dataset,
}


def get_dataset(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASETS.keys())}"
        )

    return DATASETS[dataset_name]()
