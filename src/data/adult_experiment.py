from sklearn.model_selection import train_test_split

from src.data.datasets import get_dataset


RANDOM_STATE = 42
ADULT_SAMPLE_SIZE = 5000


def get_adult_experiment_data():
    data = get_dataset("Adult Income")

    X = data["X"]
    y = data["y"]

    # Stratified sample so the income-class distribution is preserved.
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=ADULT_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fixed train/test split used by every model.
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_sample,
    )

    return X_train, X_test, y_train, y_test
