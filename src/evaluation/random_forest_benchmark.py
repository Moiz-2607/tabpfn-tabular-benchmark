import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.models.classical import create_random_forest


# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# 2. Create the SAME train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 3. Create Random Forest
model = create_random_forest()


# 4. Train
start_time = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time


# 5. Predict
start_time = time.perf_counter()

predictions = model.predict(X_test)

prediction_time = time.perf_counter() - start_time


# 6. Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


# 7. Display results
print("\n===== Random Forest Benchmark =====")
print(f"Dataset samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Precision:       {precision:.4f}")
print(f"Recall:          {recall:.4f}")
print(f"F1 Score:        {f1:.4f}")
print(f"Training time:   {training_time:.4f} seconds")
print(f"Prediction time: {prediction_time:.4f} seconds")
