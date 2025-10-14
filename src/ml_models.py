import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# -----------------------------
# Enhanced Training Pipeline
# -----------------------------
class EnhancedTrainingPipeline:
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.scaler = StandardScaler()
        self.encoders = {}  # store label encoders for categorical columns

        # Define the exact features to use for model training
        self.feature_cols = [
            'lines_changed', 'total_lines', 'code_churn_ratio',
            'author_success_rate', 'service_failure_rate_7d',
            'is_hotfix', 'touches_critical_path',
            'test_coverage', 'build_duration_sec'
        ]

    def preprocess(self, df: pd.DataFrame):
        """Preprocesses the dataset (encoding, conversions, etc.)."""
        df = df.copy()

        # Convert boolean columns to int
        for col in df.select_dtypes(include=["bool"]).columns:
            df[col] = df[col].astype(int)

        # Convert timestamps to epoch time (seconds)
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce").astype(np.int64) // 10**9
                except Exception:
                    pass

        # Encode categorical columns
        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        return df

    def hyperparameter_tuning(self, X, y):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        return grid_search.best_estimator_

    def evaluate_model(self, model, X, y):
        scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        return {
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
            "min_auc": scores.min(),
            "max_auc": scores.max(),
            "confidence_interval": (
                scores.mean() - 2 * scores.std(),
                scores.mean() + 2 * scores.std(),
            ),
        }

    def train_model(self, X, y):
        print("🚀 Starting model retraining...")

        # Apply preprocessing
        X = self.preprocess(X)

        # Filter to training features only
        X = X[self.feature_cols]

        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        classifier = self.hyperparameter_tuning(X_scaled, y)
        self.models["classifier"] = classifier

        # Train anomaly detector
        anomaly_model = IsolationForest(contamination=0.1, random_state=42).fit(X_scaled)
        self.models["anomaly_detector"] = anomaly_model

        # Save models, scaler, encoders, and feature list
        joblib.dump(classifier, "models/rf_model.joblib")
        joblib.dump(anomaly_model, "models/anomaly_model.joblib")
        joblib.dump(self.scaler, "models/scaler.joblib")
        joblib.dump(self.encoders, "models/encoders.joblib")
        joblib.dump(self.feature_cols, "models/feature_columns.joblib")

        print("✅ Models, scaler, and encoders saved to 'models/' folder.")
        return self.models

# -----------------------------
# Real-Time Inference
# -----------------------------
class RealTimeInference:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.prediction_cache = {}

        # Load encoders and feature list
        self.encoders = joblib.load("models/encoders.joblib") if os.path.exists("models/encoders.joblib") else {}
        self.feature_cols = joblib.load("models/feature_columns.joblib") if os.path.exists("models/feature_columns.joblib") else []

    def predict_with_explanation(self, deployment_data):
        """Run inference with consistent preprocessing."""
        df = pd.DataFrame([deployment_data])

        # Convert booleans to int
        for col in df.select_dtypes(include=["bool"]).columns:
            df[col] = df[col].astype(int)

        # Convert timestamps to epoch time
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce").astype(np.int64) // 10**9
                except Exception:
                    pass

        # Encode categorical columns
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str)) if set(df[col]).issubset(encoder.classes_) \
                    else encoder.transform([str(x) if x in encoder.classes_ else encoder.classes_[0] for x in df[col]])

        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_cols]

        # Scale features
        X_scaled = self.scaler.transform(df)

        # Predict probabilities
        prediction_proba = self.model.predict_proba(X_scaled)[0]

        # Mock feature importance
        feature_contributions = [
            ("code_churn_ratio", 0.25),
            ("test_coverage", 0.20),
            ("author_success_rate", 0.15),
            ("is_hotfix", 0.12),
            ("touches_critical_path", 0.10),
        ]

        result = {
            "deployment_id": deployment_data.get("deployment_id"),
            "timestamp": datetime.now(),
            "prediction": {
                "failure_probability": float(prediction_proba[1]),
                "success_probability": float(prediction_proba[0]),
                "confidence": float(max(prediction_proba)),
            },
            "explanation": {
                "top_risk_factors": feature_contributions[:3],
                "top_safe_factors": feature_contributions[-2:],
            },
            "recommendations": self._generate_recommendations(prediction_proba[1]),
        }

        self.prediction_cache[deployment_data.get("deployment_id")] = result
        print("✅ Inference completed successfully.")
        return result

    def _generate_recommendations(self, failure_prob):
        if failure_prob > 0.7:
            return [
                "⚠️ HIGH RISK: Require manual approval from senior engineer",
                "📊 Run extended test suite in staging environment",
            ]
        elif failure_prob > 0.4:
            return [
                "⚡ MEDIUM RISK: Run additional integration tests",
                "🔍 Monitor closely for first 30 minutes post-deployment",
            ]
        else:
            return [
                "✅ LOW RISK: Proceed with standard deployment process",
                "📈 Monitor standard metrics and alerts",
            ]
