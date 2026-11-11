
# ============================================================
# 1. IMPORTS NECESSARY LIBRARIES
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# >>> COMPLEMENT
# Additional libraries for project directory management
# and preparation report generation.

from pathlib import Path
import json

# <<< FIN COMPLEMENT


# ============================================================
# 2. PROJECT DIRECTORY CONFIGURATION
# ============================================================

# >>> COMPLEMENT
# Define the directories used to organize project files.

PROJECT_DIR = Path("/content/tourism_project")
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model_building"
REPORT_DIR = MODEL_DIR / "reports"

# Create the reports directory if it does not exist.
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# <<< FIN COMPLEMENT


# ============================================================
# 3. DATASET LOADING
# ============================================================

df = pd.read_csv(
    "/content/tourism_project/data/tourism.csv"
)  # complete the code: path to the registered tourism.csv inside the data folder


# ============================================================
# 4. DATA PREPARATION
# ============================================================

df.drop(
    columns=["CustomerID"],
    inplace=True
)  # complete the code: drop the customer identifier column, it is not a predictive feature


# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.


# Define the target variable.

target = "ProdTaken"  # complete the code to set the name of the column to predict (whether customer purchased the package), 1 if the customer purchased the package, else 0

X = df.drop(columns=[target])
y = df[target]


# >>> COMPLEMENT
# Validate the target variable before splitting the dataset.

if y.isna().any():
    raise ValueError("Target variable contains missing values.")

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must contain only binary values 0 and 1.")

if len(X) != len(y):
    raise ValueError("Features and target have inconsistent dimensions.")

# <<< FIN COMPLEMENT


# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================

# stratify keeps the (imbalanced) purchase ratio consistent across splits

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # complete the code: which variable should stay balanced across the splits?
)


# >>> COMPLEMENT
# Verify that both classes are represented in the training
# and testing datasets.

assert set(ytrain.unique()) == {0, 1}, "Missing class in training data."
assert set(ytest.unique()) == {0, 1}, "Missing class in testing data."

# Verify that no observations were lost during the split.

assert len(Xtrain) + len(Xtest) == len(df)

# <<< FIN COMPLEMENT


# ============================================================
# 6. SAVING PREPARED DATA
# ============================================================

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)


# >>> COMPLEMENT
# Calculate the percentage distribution of the target classes
# in the training and testing datasets.

train_distribution = (
    ytrain.value_counts(normalize=True).sort_index() * 100
)

test_distribution = (
    ytest.value_counts(normalize=True).sort_index() * 100
)

# <<< FIN COMPLEMENT


# ============================================================
# 7. GENERATING DATA PREPARATION REPORT
# ============================================================

# >>> COMPLEMENT
# Create a structured JSON report for MLOps traceability.

report = {
    "dataset": "tourism.csv",
    "total_observations": int(len(df)),
    "number_of_features": int(X.shape[1]),
    "target": target,
    "test_size": 0.2,
    "random_state": 42,
    "train_observations": int(len(Xtrain)),
    "test_observations": int(len(Xtest)),
    "train_target_distribution_percent": {
        str(k): round(float(v), 2)
        for k, v in train_distribution.items()
    },
    "test_target_distribution_percent": {
        str(k): round(float(v), 2)
        for k, v in test_distribution.items()
    },
    "stratification": "PASS"
}

report_path = REPORT_DIR / "data_preparation_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

# <<< FIN COMPLEMENT


# ============================================================
# 8. DISPLAYING PREPARED DATA
# ============================================================

print("Data prepared: train/test splits written.")

print("ProdTaken distribution in train:")
print(ytrain.value_counts())


# >>> COMPLEMENT
# Additional monitoring information.

print("\nProdTaken distribution in test:")
print(ytest.value_counts())

print("\nTraining dataset dimensions:", Xtrain.shape)
print("Testing dataset dimensions:", Xtest.shape)

print("\nTraining target distribution (%):")
print(train_distribution.round(2))

print("\nTesting target distribution (%):")
print(test_distribution.round(2))

print("\nDATA PREPARATION: SUCCESS")
print("Preparation report:", report_path)

# >>> COMPLEMENT
# Remove the technical index column from the dataset.
df.drop(columns=["Unnamed: 0"], inplace=True)
# <<< FIN COMPLEMENT

# <<< FIN COMPLEMENT
