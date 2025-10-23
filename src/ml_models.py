import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.makedirs("models", exist_ok=True)

class EnhancedTrainingPipeline:
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.scaler = StandardScaler()
        self.encoders = {}

        self.feature_cols = [
            'lines_changed', 'total_lines', 'code_churn_ratio',
            'author_success_rate', 'service_failure_rate_7d',
            'is_hotfix', 'touches_critical_path',
            'test_coverage', 'build_duration_sec',
            'risk_score'
        ]

        # Define candidate classifiers and their parameter grids
        self.classifiers = {
            "RandomForest": (
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                },
            ),
            "LogisticRegression": (
                LogisticRegression(random_state=42, max_iter=500),
                {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                },
            ),
        }

    def preprocess(self, df: pd.DataFrame):
        df = df.copy()
        drop_cols = ['labels', 'deployment_id']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        for col in ['is_hotfix', 'touches_critical_path']:
            if col in df.columns and df[col].dtype == 'bool':
                df[col] = df[col].astype(int)

        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce").astype(np.int64) // 10**9
                except Exception:
                    pass

        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        return df

    def hyperparameter_tuning(self, model, param_grid, X, y):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

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

        X = self.preprocess(X)
        missing_features = [f for f in self.feature_cols if f not in X.columns]
        for mf in missing_features:
            X[mf] = 0
        X = X[self.feature_cols]

        X_scaled = self.scaler.fit_transform(X)

        best_model_name = None
        best_model = None
        best_score = -np.inf
        best_params = None

        # Train and tune all candidate classifiers
        for name, (clf, param_grid) in self.classifiers.items():
            print(f"🔍 Tuning hyperparameters for {name}...")
            tuned_model, score, params = self.hyperparameter_tuning(clf, param_grid, X_scaled, y)
            print(f"  {name} best AUC: {score:.4f} with params {params}")

            if score > best_score:
                best_score = score
                best_model = tuned_model
                best_model_name = name
                best_params = params

        print(f"🏆 Best model: {best_model_name} with AUC: {best_score:.4f}")

        self.models["classifier"] = best_model
        self.best_params = best_params

        # Train anomaly detector on scaled data
        anomaly_model = IsolationForest(contamination=0.1, random_state=42).fit(X_scaled)
        self.models["anomaly_detector"] = anomaly_model

        # Save best model and artifacts
        joblib.dump(best_model, f"models/{best_model_name}_model.joblib")
        joblib.dump(anomaly_model, "models/anomaly_model.joblib")
        joblib.dump(self.scaler, "models/scaler.joblib")
        joblib.dump(self.encoders, "models/encoders.joblib")
        joblib.dump(self.feature_cols, "models/feature_columns.joblib")

        print("✅ Models, scaler, and encoders saved to 'models/' folder.")
        return self.models


class RealTimeInference:
    def __init__(self, model, scaler, encoders=None, feature_cols=None):
        self.model = model
        self.scaler = scaler
        self.encoders = encoders or {}
        self.feature_cols = feature_cols or [
            'lines_changed', 'total_lines', 'code_churn_ratio',
            'author_success_rate', 'service_failure_rate_7d',
            'is_hotfix', 'touches_critical_path',
            'test_coverage', 'build_duration_sec',
            'risk_score'
        ]

    def preprocess(self, raw_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([raw_data])

        # Convert boolean columns if needed
        for col in ['is_hotfix', 'touches_critical_path']:
            if col in df.columns and df[col].dtype == 'bool':
                df[col] = df[col].astype(int)

        # Apply label encoding for categorical columns if encoders are available
        for col, encoder in self.encoders.items():
            if col in df.columns:
                # If new/unseen category, transform will fail; handle gracefully:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except Exception:
                    # For unseen labels, assign a default encoding (e.g., 0)
                    df[col] = 0

        # Fill missing features with zeros
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Select only expected features
        df = df[self.feature_cols]
        return df

    def predict_with_explanation(self, raw_data: dict):
        df = self.preprocess(raw_data)
        X_scaled = self.scaler.transform(df)

        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0][1] if hasattr(self.model, "predict_proba") else None

        # Provide feature importances as explanation (if available)
        explanation = {}
        if hasattr(self.model, "feature_importances_"):
            explanation = dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))

        return {
            "prediction": prediction,
            "probability": proba,
            "explanation": explanation,
        }
