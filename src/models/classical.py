from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def create_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )


def create_xgboost():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )


def create_adult_logistic_regression():
    from src.data.preprocessing import create_adult_preprocessor

    return Pipeline([
        (
            "preprocessor",
            create_adult_preprocessor(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def create_adult_random_forest():
    from src.data.preprocessing import create_adult_preprocessor

    return Pipeline([
        (
            "preprocessor",
            create_adult_preprocessor(),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def create_adult_xgboost():
    from src.data.preprocessing import create_adult_preprocessor

    return Pipeline([
        (
            "preprocessor",
            create_adult_preprocessor(),
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        ),
    ])
