import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_results_exist():
    path = PROJECT_ROOT / "results" / "metrics" / "benchmark_results.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 12
    assert set(df["dataset"]) == {
        "Breast Cancer",
        "Wine",
        "Adult Income",
    }
    assert set(df["model"]) == {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "TabPFN",
    }


def test_benchmark_metrics_are_valid():
    path = PROJECT_ROOT / "results" / "metrics" / "benchmark_results.csv"
    df = pd.read_csv(path)

    for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert df[column].notna().all()
        assert ((df[column] >= 0) & (df[column] <= 1)).all()

    assert (df["training_time"] > 0).all()
    assert (df["prediction_time"] > 0).all()


def test_expected_visualizations_exist():
    plots = [
        "accuracy_comparison.png",
        "f1_comparison.png",
        "f1_heatmap.png",
        "prediction_time_comparison.png",
        "roc_auc_comparison.png",
        "training_time_comparison.png",
    ]

    for plot in plots:
        path = PROJECT_ROOT / "results" / "plots" / plot
        assert path.exists()
        assert path.stat().st_size > 0


def test_adult_experiment_split():
    command = [
        sys.executable,
        "-c",
        (
            "from src.data.adult_experiment import get_adult_experiment_data; "
            "X_train, X_test, y_train, y_test = get_adult_experiment_data(); "
            "assert X_train.shape == (4000, 14); "
            "assert X_test.shape == (1000, 14); "
            "assert len(y_train) == 4000; "
            "assert len(y_test) == 1000"
        ),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
