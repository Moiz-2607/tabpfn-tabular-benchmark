# Tabular ML Benchmark: Classical Models vs TabPFN

A reproducible benchmark comparing classical machine-learning models with TabPFN across multiple tabular classification datasets.

The project evaluates predictive performance, computational cost, error patterns, and feature-level behavior using a consistent experimental workflow.

---

## Research Question

How does TabPFN compare with commonly used classical tabular machine-learning models across datasets with different characteristics?

The benchmark compares:

- Logistic Regression
- Random Forest
- XGBoost
- TabPFN

---

## Datasets

Three classification datasets are evaluated:

### 1. Breast Cancer

The Breast Cancer Wisconsin dataset available through scikit-learn.

- Samples: 569
- Features: 30
- Classes: 2

### 2. Wine

The Wine dataset available through scikit-learn.

- Samples: 178
- Features: 13
- Classes: 3

### 3. Adult Income

The Adult Income dataset obtained through OpenML.

The target represents whether annual income is:

- `<=50K` → 0
- `>50K` → 1

For the controlled benchmark experiment, a stratified subset of 5,000 samples was used.

- Training samples: 4,000
- Test samples: 1,000
- Features: 14
- Classes: 2

---

## Models

### Logistic Regression

A linear classification baseline.

### Random Forest

An ensemble of decision trees using randomized feature and sample selection.

### XGBoost

A gradient-boosted tree model configured for tabular classification.

### TabPFN

A pretrained tabular foundation model designed for classification tasks.

---

## Experimental Methodology

For Breast Cancer and Wine:

- 80/20 stratified train-test split
- Random state: 42

For Adult Income:

- 5,000-row stratified subset
- 80/20 stratified train-test split
- Random state: 42

The Adult experiment uses a controlled subset so that the benchmark remains practical on the local Apple Silicon environment.

### Adult preprocessing

Numerical features use:

- Median imputation
- Standardization

Categorical features use:

- Most-frequent imputation
- One-hot encoding
- Unknown-category handling

---

## Evaluation Metrics

The benchmark reports:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Training Time
- Prediction Time

For binary classification, precision, recall and F1 use the positive class.

For the multiclass Wine dataset, weighted averaging is used.

---

## Benchmark Results

The complete results are available in:

```text
results/tables/benchmark_summary.csv
EOF
