
# ============================================================
# 1. IMPORTS NECESSARY LIBRARIES
# ============================================================

# for data manipulation
import os
import pandas as pd

# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# for model serialization and experiment tracking
import joblib
import mlflow

# >>> COMPLEMENT
# Resolve project paths independently of the current working directory.
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

# <<< FIN COMPLEMENT


# ============================================================
# 2. MLFLOW CONFIGURATION
# ============================================================

# >>> COMPLEMENT
# Use the tracking URI provided by the execution environment.
# GitHub Actions uses its local MLflow server.
# A local fallback is provided for development environments.

TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(TRACKING_URI)

print("MLflow tracking URI:", mlflow.get_tracking_uri())

# <<< FIN COMPLEMENT

mlflow.set_experiment("Tourism_Package_Prediction")     # complete the code to set the MLflow experiment name (same as the dev experimentation cell)


# ============================================================
# 3. LOADING TRAINING AND TEST DATA
# ============================================================

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact

Xtrain = pd.read_csv(PROJECT_DIR / "Xtrain.csv")
Xtest = pd.read_csv(PROJECT_DIR / "Xtest.csv")
ytrain = pd.read_csv(PROJECT_DIR / "ytrain.csv").squeeze()
ytest = pd.read_csv(PROJECT_DIR / "ytest.csv").squeeze()


# >>> COMPLEMENT
# Validate the availability and integrity of the datasets.

if Xtrain.empty or Xtest.empty:
    raise ValueError("Training or test features dataset is empty.")

if ytrain.empty or ytest.empty:
    raise ValueError("Training or test target dataset is empty.")

if ytrain.isna().any() or ytest.isna().any():
    raise ValueError("Missing values detected in target variables.")

if set(ytrain.unique()) != {0, 1}:
    raise ValueError("Training target must contain both classes 0 and 1.")

if set(ytest.unique()) != {0, 1}:
    raise ValueError("Test target must contain both classes 0 and 1.")

if len(Xtrain) != len(ytrain):
    raise ValueError("Training features and target dimensions do not match.")

if len(Xtest) != len(ytest):
    raise ValueError("Test features and target dimensions do not match.")

# <<< FIN COMPLEMENT


# ============================================================
# 4. FEATURE DEFINITION
# ============================================================

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]   # complete the code to list all numerical feature names (same as in prep.py)

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]   # complete the code to list all categorical feature names (same as in prep.py)


# >>> COMPLEMENT
# Validate the training and test feature schemas.

assert list(Xtrain.columns) == list(Xtest.columns), (
    "Training and test feature columns do not match."
)

assert set(numeric_features + categorical_features) == set(Xtrain.columns), (
    "Feature mismatch: check prep.py and the training dataset."
)

assert len(numeric_features) == 12
assert len(categorical_features) == 6
assert Xtrain.shape[1] == 18

assert not any(
    col in Xtrain.columns
    for col in ["CustomerID", "Unnamed: 0"]
), "Technical identifier found in predictive features."

print("\nDATA VALIDATION: PASS")
print("Training observations:", len(Xtrain))
print("Test observations:", len(Xtest))
print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("Predictive features:", Xtrain.shape[1])

# <<< FIN COMPLEMENT


# ============================================================
# 5. MODEL PREPROCESSING
# ============================================================

# Set the class weight to handle class imbalance

class_weight = (
    ytrain.value_counts()[0] / ytrain.value_counts()[1]
)

# >>> COMPLEMENT
# Display the class imbalance ratio.

print("\nClass distribution in training data:")
print(ytrain.value_counts())

print("Positive class weight:", round(class_weight, 4))

# <<< FIN COMPLEMENT


# Define the preprocessing steps

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)


# ============================================================
# 6. XGBOOST MODEL DEFINITION
# ============================================================

# Define base XGBoost model

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=1
)


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

# Define hyperparameter grid
# Fill in suitable values for each parameter based on your understanding of XGBoost tuning.

param_grid = {
    'xgbclassifier__n_estimators': [100, 200],        # Number of boosting trees. More trees can improve performance but increase training time.
    'xgbclassifier__max_depth': [3, 5],           # Maximum depth of each tree. Higher values increase model complexity and risk of overfitting.
    'xgbclassifier__colsample_bytree': [0.8, 1.0],    # Fraction of features sampled when building each tree.
    'xgbclassifier__colsample_bylevel': [0.8, 1.0],   # Fraction of features sampled at each tree level.
    'xgbclassifier__learning_rate': [0.05, 0.1],       # Step size used during boosting. Smaller values may improve generalization but require more trees.
    'xgbclassifier__reg_lambda': [1, 5],          # L2 regularization strength. Higher values help reduce overfitting.
}


# Model pipeline

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)   # complete the code to build the model pipeline by chaining preprocessor and xgb_model


# >>> COMPLEMENT
# Calculate the number of hyperparameter combinations.

number_of_combinations = 1

for values in param_grid.values():
    number_of_combinations *= len(values)

print("\nMODEL CONFIGURATION")
print("Hyperparameter combinations:", number_of_combinations)
print("Cross-validation folds: 5")
print("Total expected model fits:", number_of_combinations * 5)

# <<< FIN COMPLEMENT


# ============================================================
# 8. MODEL TRAINING AND EXPERIMENT TRACKING
# ============================================================

# Start MLflow run

with mlflow.start_run():

    # Hyperparameter tuning with GridSearchCV

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # >>> COMPLEMENT
    # Display the best model configuration.

    print("\nBEST MODEL CONFIGURATION")
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", round(grid_search.best_score_, 4))

    # Log the best cross-validation score.
    mlflow.log_metric(
        "best_cv_score",
        float(grid_search.best_score_)
    )

    mlflow.log_metric(
        "number_of_hyperparameter_combinations",
        number_of_combinations
    )

    # <<< FIN COMPLEMENT


    # Log every parameter combination tried during the search as a nested run,
    # so all experiments can be compared side by side in the MLflow UI

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(nested=True):

            mlflow.log_params(results["params"][i])

            mlflow.log_metric(
                "mean_test_score",
                float(results["mean_test_score"][i])
            )

            mlflow.log_metric(
                "std_test_score",
                float(results["std_test_score"][i])
            )


    # Log the best hyperparameters in the main run

    mlflow.log_params(grid_search.best_params_)


    # Store the best model

    best_model = grid_search.best_estimator_


    # ========================================================
    # 9. MODEL EVALUATION
    # ========================================================

    # Set classification threshold

    classification_threshold = 0.5    # Choose a classification threshold between 0 and 1. Lower thresholds typically increase recall and decrease precision and vice versa. Experiment with different values to find the best trade-off.


    # Make predictions on the training and test data

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)


    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)


    # Evaluation

    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True,
        zero_division=0
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True,
        zero_division=0
    )


    # >>> COMPLEMENT
    # Display classification reports.

    print("\nTRAINING CLASSIFICATION REPORT")
    print(
        classification_report(
            ytrain,
            y_pred_train,
            zero_division=0
        )
    )

    print("\nTEST CLASSIFICATION REPORT")
    print(
        classification_report(
            ytest,
            y_pred_test,
            zero_division=0
        )
    )

    print("Classification threshold:", classification_threshold)

    # <<< FIN COMPLEMENT


    # ========================================================
    # 10. MLFLOW METRICS LOGGING
    # ========================================================

    # Log metrics

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })


    # >>> COMPLEMENT
    # Log the classification threshold and dataset dimensions.

    mlflow.log_metric(
        "classification_threshold",
        classification_threshold
    )

    mlflow.log_metric(
        "training_observations",
        len(Xtrain)
    )

    mlflow.log_metric(
        "test_observations",
        len(Xtest)
    )

    print("\nMODEL PERFORMANCE")
    print("Training accuracy:", round(train_report["accuracy"], 4))
    print("Test accuracy:", round(test_report["accuracy"], 4))
    print("Test precision:", round(test_report["1"]["precision"], 4))
    print("Test recall:", round(test_report["1"]["recall"], 4))
    print("Test F1-score:", round(test_report["1"]["f1-score"], 4))

    # <<< FIN COMPLEMENT


    # ========================================================
    # 11. MODEL SERIALIZATION
    # ========================================================

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability

    model_path = str(
        DEPLOYMENT_DIR / "model.pkl"
    )   # Specify the local file path (inside tourism_project/deployment/) where the trained model should be saved.

    joblib.dump(
        best_model,
        model_path
    )  # complete the code to save the model

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(f"\nModel saved to {model_path}")


# ============================================================
# 12. FINAL EXECUTION VALIDATION
# ============================================================

# >>> COMPLEMENT
# Verify that the serialized model exists and is not empty.

saved_model_path = Path(model_path)

assert saved_model_path.exists(), (
    "Trained model file was not created."
)

assert saved_model_path.stat().st_size > 0, (
    "Trained model file is empty."
)

print("\nMODEL TRAINING: SUCCESS")
print("Model file size:", saved_model_path.stat().st_size, "bytes")
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# <<< FIN COMPLEMENT
