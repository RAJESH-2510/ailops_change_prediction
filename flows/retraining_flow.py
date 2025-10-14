import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prefect import flow, task

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_models import EnhancedTrainingPipeline

# -----------------------------
# Utility Functions
# -----------------------------
def preprocess_dataframe(df: pd.DataFrame):
    """
    Converts non-numeric columns (like IDs, timestamps, booleans)
    into numeric encodings suitable for model training.
    """
    df = df.copy()

    # Convert boolean columns to int
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # Convert timestamp columns to numeric (epoch time)
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce').astype(np.int64) // 10**9
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
    """Train the model using EnhancedTrainingPipeline."""
    target_col = "failure_label"

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in training data.")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocess: encode categorical, timestamps, booleans → numeric
    X_encoded, encoders = preprocess_dataframe(X)

    print(f"📊 Encoded training data shape: {X_encoded.shape}")
    print(f"🔤 Encoded columns: {list(X_encoded.columns)}")

    # Train model
    pipeline = EnhancedTrainingPipeline()
    models = pipeline.train_model(X_encoded, y)

    print("✅ Model training completed and saved successfully.")
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
