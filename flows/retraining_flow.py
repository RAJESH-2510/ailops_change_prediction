import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from prefect import flow, task

# Add the project root to sys.path to import your pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_models import EnhancedTrainingPipeline

# Define the exact feature columns expected by the model (including 'risk_score')
FEATURE_COLS = [
    "lines_changed",
    "total_lines",
    "code_churn_ratio",
    "author_success_rate",
    "service_failure_rate_7d",
    "is_hotfix",
    "touches_critical_path",
    "test_coverage",
    "build_duration_sec",
    "risk_score"
]

# -----------------------------
# Utility Functions
# -----------------------------

def preprocess_dataframe(df: pd.DataFrame):
    """
    Converts non-numeric columns (like IDs, timestamps, booleans)
    into numeric encodings suitable for model training.
    Drops irrelevant columns.
    """
    df = df.copy()

    # Drop unused columns if present
    for col in ["labels", "deployment_id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Convert explicit boolean columns to int if they exist
    for col in ["is_hotfix", "touches_critical_path"]:
        if col in df.columns and df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # Convert timestamp columns to numeric (epoch time)
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce").astype(np.int64) // 10**9
            except Exception:
                pass

    # Label encode string columns
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

# -----------------------------
# Tasks
# -----------------------------

@task(log_prints=True)
def load_training_data():
    """Load training data for model retraining."""
    csv_path = "data/training_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Training data not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded training data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

@task(log_prints=True)
def train_model_task(df: pd.DataFrame):
    """Train the model using EnhancedTrainingPipeline and evaluate."""
    target_col = "failure_label"

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in training data.")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocess: encode categorical, timestamps, booleans → numeric
    X_encoded, encoders = preprocess_dataframe(X)

    # Add missing features as zero columns instead of error
    for f in FEATURE_COLS:
        if f not in X_encoded.columns:
            print(f"⚠️ Warning: Missing expected feature '{f}' in data, filling with zeros.")
            X_encoded[f] = 0

    X_selected = X_encoded[FEATURE_COLS]

    print(f"📊 Encoded & selected training data shape: {X_selected.shape}")
    print(f"🔤 Selected columns: {list(X_selected.columns)}")

    # Split train/test for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"🔀 Split data: train {X_train.shape[0]} rows, val {X_val.shape[0]} rows")

    # Train model using EnhancedTrainingPipeline
    pipeline = EnhancedTrainingPipeline()
    models = pipeline.train_model(X_train, y_train)
    print("✅ Model training completed successfully.")

    # Predict and evaluate using classifier model
    classifier = models.get("classifier", None)
    if classifier is None:
        raise ValueError("❌ Classifier model not found in models dictionary.")

    y_pred = classifier.predict(X_val)

    # Evaluation metrics
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    print(f"📈 Validation Accuracy: {acc:.4f}")
    print(f"📊 Classification Report:\n{report}")

    return models

# -----------------------------
# Flow
# -----------------------------
@flow(name="Model Retraining Flow", log_prints=True)
def retraining_flow():
    df = load_training_data()
    train_model_task(df)

if __name__ == "__main__":
    retraining_flow()
