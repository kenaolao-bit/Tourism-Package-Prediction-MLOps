
import json
import hashlib
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/content/tourism_project")
RAW_PATH = PROJECT_DIR / "data" / "tourism.csv"
REPORT_DIR = PROJECT_DIR / "model_building" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]


# ============================================================
# 2. DATA LOADING
# ============================================================

if not RAW_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

if df.empty:
    raise ValueError("Dataset is empty.")

print("=" * 60)
print("TOURISM DATA REGISTRATION")
print("=" * 60)


# ============================================================
# 3. SCHEMA VALIDATION
# ============================================================

if df.columns.duplicated().any():
    raise ValueError("Duplicate column names detected.")

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("SCHEMA VALIDATION: PASS")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 4. DATA QUALITY ASSESSMENT
# ============================================================

missing_values = df.isnull().sum()
duplicate_rows = int(df.duplicated().sum())

target = df["ProdTaken"]

if target.isnull().any():
    raise ValueError("Target ProdTaken contains missing values.")

if not set(target.unique()).issubset({0, 1}):
    raise ValueError("Unexpected values detected in ProdTaken.")

class_distribution = (
    target.value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_dict()
)

print("\nDATA QUALITY")
print("Total missing values:", int(missing_values.sum()))
print("Duplicate rows:", duplicate_rows)
print("Target distribution (%):", class_distribution)


# ============================================================
# 5. DATA FINGERPRINT
# ============================================================

file_hash = hashlib.sha256(
    RAW_PATH.read_bytes()
).hexdigest()


# ============================================================
# 6. REGISTRATION REPORT
# ============================================================

report = {
    "dataset": RAW_PATH.name,
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_values": int(missing_values.sum()),
    "duplicate_rows": duplicate_rows,
    "target": "ProdTaken",
    "target_distribution_percent": class_distribution,
    "sha256": file_hash,
    "schema_validation": "PASS"
}

report_path = REPORT_DIR / "data_registration_report.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\nREGISTRATION: SUCCESS")
print("SHA256:", file_hash)
print("Report:", report_path)
print("=" * 60)
