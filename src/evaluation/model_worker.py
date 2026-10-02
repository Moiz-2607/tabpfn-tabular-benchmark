import sys

from sklearn.model_selection import train_test_split

from src.data.datasets import get_dataset
from src.data.adult_experiment import get_adult_experiment_data
from src.evaluation.benchmark import benchmark_model


model_name = sys.argv[1]
dataset_name = sys.argv[2]


# Import only the requested model.
# This keeps XGBoost and TabPFN in separate processes
# and avoids the macOS OpenMP conflict we encountered.
if model_name == "Logistic Regression":
    from src.models.classical import create_logistic_regression, create_adult_logistic_regression
    create_model = create_adult_logistic_regression if dataset_name == 'Adult Income' else create_logistic_regression

elif model_name == "Random Forest":
    from src.models.classical import create_random_forest, create_adult_random_forest
    create_model = create_adult_random_forest if dataset_name == 'Adult Income' else create_random_forest

elif model_name == "XGBoost":
    from src.models.classical import create_xgboost, create_adult_xgboost
    create_model = create_adult_xgboost if dataset_name == 'Adult Income' else create_xgboost

elif model_name == "TabPFN":
    from src.models.tabpfn_model import create_tabpfn
    create_model = create_tabpfn

else:
    raise ValueError(f"Unknown model: {model_name}")


# Adult Income uses the controlled 5,000-row experiment.
if dataset_name == "Adult Income":

    X_train, X_test, y_train, y_test = get_adult_experiment_data()

else:

    data = get_dataset(dataset_name)

    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


print(
    f"Starting worker: {model_name} | Dataset: {dataset_name}",
    flush=True,
)


model = create_model()


metrics = benchmark_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
)


result = {
    "dataset": dataset_name,
    "model": model_name,
    **metrics,
}


print("RESULT:", result, flush=True)

