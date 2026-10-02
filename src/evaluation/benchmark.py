import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def benchmark_model(model, X_train, X_test, y_train, y_test):
    print("  Training...", flush=True)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    print(f"  Training complete: {training_time:.4f}s", flush=True)

    print("  Predicting...", flush=True)

    start_time = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_time = time.perf_counter() - start_time

    print(f"  Prediction complete: {prediction_time:.4f}s", flush=True)

    # Basic classification metrics
    accuracy = accuracy_score(y_test, predictions)

    number_of_classes = len(set(y_test))
    average_method = "binary" if number_of_classes == 2 else "weighted"

    precision = precision_score(
        y_test,
        predictions,
        average=average_method,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average=average_method,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average=average_method,
        zero_division=0,
    )

    # ROC-AUC
    roc_auc = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)

        if number_of_classes == 2:
            roc_auc = roc_auc_score(
                y_test,
                probabilities[:, 1],
            )
        else:
            roc_auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "training_time": training_time,
        "prediction_time": prediction_time,
    }
