from src.models.classical import (
    create_logistic_regression,
    create_random_forest,
    create_xgboost,
)

from src.models.tabpfn_model import create_tabpfn


MODELS = {
    "Logistic Regression": create_logistic_regression,
    "Random Forest": create_random_forest,
    "XGBoost": create_xgboost,
    "TabPFN": create_tabpfn,
}
