from prefect import task
from datetime import datetime
import pandas as pd


class FeatureEngineering:
    @task(log_prints=True)
    def create_deployment_features(self, raw_data):
        features = {
            'lines_changed': raw_data.get('lines_changed', 0),
            'total_lines': raw_data.get('total_lines', 1),
            'code_churn_ratio': raw_data.get('code_churn_ratio', 0),
            'author_success_rate': raw_data.get('author_success_rate', 0.8),
            'service_failure_rate_7d': raw_data.get('service_failure_rate_7d', 0.05),
            'is_hotfix': raw_data.get('is_hotfix', False),
            'touches_critical_path': raw_data.get('touches_critical_path', False),
            'test_coverage': raw_data.get('test_coverage', 70),
            'build_duration_sec': raw_data.get('build_duration_sec', 180)
        }
        return features


class FeatureStore:
    def __init__(self):
        self.features = {}
        self.feature_metadata = {}

    @task(log_prints=True)
    def register_feature(self, name, description, computation_logic):
        self.feature_metadata[name] = {
            'description': description,
            'created_at': datetime.now(),
            'version': '1.0',
            'computation': computation_logic,
            'usage_count': 0
        }

    @task(log_prints=True)
    def compute_feature_importance(self):
        return {
            'code_churn_ratio': 0.25,
            'test_coverage': 0.20,
            'author_success_rate': 0.15,
            'is_hotfix': 0.12,
            'touches_critical_path': 0.10,
            'service_failure_rate_7d': 0.07,
            'build_duration_sec': 0.06,
            'lines_changed': 0.05
        }
