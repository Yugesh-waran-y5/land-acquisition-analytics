"""
Train a risk-category model from project records.

Usage (run inside your project folder):
    python train_model.py                      # auto-finds a CSV in this folder
    python train_model.py --data my_file.csv   # use a specific CSV
    python train_model.py --make-demo          # create sample_projects.csv (synthetic) and train on it
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent          # works from any folder
DEFAULT_DATA_FILE = BASE_DIR / "sample_projects.csv"
MODEL_FILE = BASE_DIR / "land_delay_model.joblib"

# CSV names the dashboard / earlier steps may have produced.
CANDIDATE_FILES = [
    "sample_projects.csv",
    "land_acquisition_records.csv",
    "land_acquisition_demo.csv",
    "sample_valid_projects.csv",
]

FEATURES = [
    "Year",
    "District",
    "Documents_Complete",
    "Compensation_Paid",
    "Legal_Dispute",
    "Approval_Pending",
]
TARGET = "Risk"
BOOLEAN_COLUMNS = [
    "Documents_Complete",
    "Compensation_Paid",
    "Legal_Dispute",
    "Approval_Pending",
]
RISK_LEVELS = ["Low", "Medium", "High"]
BOOL_MAP = {
    "true": 1, "false": 0,
    "yes": 1, "no": 0,
    "1": 1, "0": 0,
}


def make_demo_csv(path: Path, n: int = 250) -> None:
    """Write SYNTHETIC records (same generator as the dashboard demo)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Project_ID": [f"LA-{i:03d}" for i in range(1, n + 1)],
        "Year": rng.integers(2022, 2027, n),
        "District": rng.choice(
            ["Chennai", "Salem", "Madurai", "Coimbatore", "Trichy"], n
        ),
        "Documents_Complete": rng.choice([True, False], n, p=[0.7, 0.3]),
        "Compensation_Paid": rng.choice([True, False], n, p=[0.6, 0.4]),
        "Legal_Dispute": rng.choice([True, False], n, p=[0.2, 0.8]),
        "Approval_Pending": rng.choice([True, False], n, p=[0.3, 0.7]),
    })
    df["Delay_Days"] = (
        (~df["Documents_Complete"]) * rng.integers(15, 70, n)
        + (~df["Compensation_Paid"]) * rng.integers(20, 90, n)
        + df["Legal_Dispute"] * rng.integers(30, 120, n)
        + df["Approval_Pending"] * rng.integers(10, 60, n)
        + rng.integers(0, 16, n)
    )
    df["Risk"] = pd.cut(
        df["Delay_Days"],
        bins=[-1, 40, 100, float("inf")],
        labels=RISK_LEVELS,
    ).astype(str)
    df.to_csv(path, index=False)
    print(f"Created synthetic demo file: {path}")


def find_data_file(cli_value: str | None) -> Path:
    # 1) A file given with --data
    if cli_value:
        path = Path(cli_value)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
        sys.exit(f"ERROR: File not found: {path}")

    # 2) Known file names in the script's folder, then the current folder
    for folder in (BASE_DIR, Path.cwd()):
        for name in CANDIDATE_FILES:
            candidate = folder / name
            if candidate.exists():
                return candidate

    sys.exit(
        "ERROR: No project CSV found.\n"
        f"Looked in: {BASE_DIR}\n"
        "Fix it in one of these ways:\n"
        "  1) Copy your CSV into the project folder and run:\n"
        "       python train_model.py --data your_file.csv\n"
        "  2) Create a synthetic test file (demo only) with:\n"
        "       python train_model.py --make-demo"
    )


def load_and_clean(data_file: Path):
    df = pd.read_csv(data_file)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: Missing columns in {data_file.name}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df[required].dropna().copy()
    if df.empty:
        sys.exit("ERROR: The CSV has no usable rows after removing blanks.")

    # Yes/No style columns -> 1/0 (works for bool and text values)
    for col in BOOLEAN_COLUMNS:
        df[col] = (
            df[col].astype(str).str.strip().str.lower().map(BOOL_MAP)
        )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["District"] = df["District"].astype(str).str.strip()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()

    bad_cols = [c for c in BOOLEAN_COLUMNS + ["Year"] if df[c].isna().any()]
    if bad_cols:
        sys.exit(
            f"ERROR: Could not read values in: {bad_cols}. "
            "Use True/False (or Yes/No) for issue columns and numbers for Year."
        )

    unknown = sorted(set(df[TARGET]) - set(RISK_LEVELS))
    if unknown:
        sys.exit(f"ERROR: Risk must be Low, Medium or High. Found: {unknown}")

    if df[TARGET].nunique() < 2:
        sys.exit("ERROR: Training needs at least two different Risk classes.")

    if len(df) < 20:
        print(f"WARNING: Only {len(df)} records. Results will be unreliable.")

    X = df[FEATURES].copy()
    X[BOOLEAN_COLUMNS] = X[BOOLEAN_COLUMNS].astype(int)
    X["Year"] = X["Year"].astype(int)
    y = df[TARGET]
    return X, y


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("district", OneHotEncoder(handle_unknown="ignore"), ["District"]),
            ("numeric", "passthrough", ["Year"] + BOOLEAN_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            )),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the risk model.")
    parser.add_argument("--data", help="Path to the project records CSV")
    parser.add_argument(
        "--make-demo", action="store_true",
        help="Create a synthetic sample_projects.csv and train on it",
    )
    args = parser.parse_args()

    if args.make_demo:
        make_demo_csv(DEFAULT_DATA_FILE)
        data_file = DEFAULT_DATA_FILE
    else:
        data_file = find_data_file(args.data)

    print(f"Using data file: {data_file}")
    X, y = load_and_clean(data_file)

    # Stratify only if every class has enough records for it.
    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel evaluation")
    print("----------------")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    print(
        "\nNOTE: If this CSV is synthetic demo data, the accuracy above only "
        "shows the code works. It is not evidence of real-world prediction."
    )


if __name__ == "__main__":
    main()