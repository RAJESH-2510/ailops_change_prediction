from prefect import task
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


class EnhancedTrainingPipeline:
    def __init__(self):
        self.models = {}
        self.best_params = {}

    @task(log_prints=True)
    def hyperparameter_tuning(self, X, y):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )

        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        return grid_search.best_estimator_

    @task(log_prints=True)
    def evaluate_model(self, model, X, y):
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        return {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'min_auc': scores.min(),
            'max_auc': scores.max(),
            'confidence_interval': (
                scores.mean() - 2 * scores.std(),
                scores.mean() + 2 * scores.std()
            )
        }

    @task(log_prints=True)
    def train_model(self, X, y):
        classifier = self.hyperparameter_tuning(X, y)
        anomaly_model = IsolationForest(contamination=0.1, random_state=42).fit(X)
        joblib.dump(classifier, 'models/rf_model.joblib')
        return {"classifier": classifier, "anomaly_detector": anomaly_model}


class RealTimeInference:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler or StandardScaler()
        self.prediction_cache = {}
        self.performance_metrics = []

    @task(log_prints=True)
    def predict_with_explanation(self, deployment_data):
        features = {
            'lines_changed': deployment_data.get('lines_changed', 0),
            'total_lines': deployment_data.get('total_lines', 1),
            'code_churn_ratio': deployment_data.get('code_churn_ratio', 0),
            'author_success_rate': deployment_data.get('author_success_rate', 0.8),
            'service_failure_rate_7d': deployment_data.get('service_failure_rate_7d', 0.05),
            'is_hotfix': 1 if deployment_data.get('is_hotfix', False) else 0,
            'touches_critical_path': 1 if deployment_data.get('touches_critical_path', False) else 0,
            'test_coverage': deployment_data.get('test_coverage', 70),
            'build_duration_sec': deployment_data.get('build_duration_sec', 180)
        }

        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.fit_transform(features_df)
        prediction_proba = self.model.predict_proba(features_scaled)[0]

        feature_contributions = [
            ('code_churn_ratio', 0.25),
            ('test_coverage', 0.20),
            ('author_success_rate', 0.15),
            ('is_hotfix', 0.12),
            ('touches_critical_path', 0.10)
        ]

        result = {
            'deployment_id': deployment_data.get('deployment_id'),
            'timestamp': datetime.now(),
            'prediction': {
                'failure_probability': prediction_proba[1],
                'success_probability': prediction_proba[0],
                'confidence': max(prediction_proba)
            },
            'explanation': {
                'top_risk_factors': feature_contributions[:3],
                'top_safe_factors': feature_contributions[-2:]
            },
            'recommendations': self._generate_recommendations(prediction_proba[1])
        }

        self.prediction_cache[deployment_data.get('deployment_id')] = result
        return result

    def _generate_recommendations(self, failure_prob):
        if failure_prob > 0.7:
            return [
                "⚠️ HIGH RISK: Require manual approval from senior engineer",
                "📊 Run extended test suite in staging environment"
            ]
        elif failure_prob > 0.4:
            return [
                "⚡ MEDIUM RISK: Run additional integration tests",
                "🔍 Monitor closely for first 30 minutes post-deployment"
            ]
        else:
            return [
                "✅ LOW RISK: Proceed with standard deployment process",
                "📈 Monitor standard metrics and alerts"
            ]
