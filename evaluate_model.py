
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from train_model import load_and_clean, build_model

# Load the existing dataset without changing it.
data_file = Path(__file__).resolve().parent / "sample_projects.csv"
X, y = load_and_clean(data_file)

# Keep a separate test set that the model never sees during training.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Train a temporary model for evaluation only.
model = build_model()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n=== MODEL EVALUATION ===")
print("Dataset: sample_projects.csv (synthetic demo data)")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"],
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted)")
print("Class order: Low, Medium, High")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"],
))

print("\nNOTE: These results test the code on synthetic data.")
print("They do not establish real-world prediction accuracy.")
print("The existing land_delay_model.joblib was not changed.")
