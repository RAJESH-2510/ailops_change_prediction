from typing import Dict, Optional, List
from prefect import task
from datetime import datetime
import pandas as pd


def default_for(feature: str):
    """Default values for features."""
    defaults = {
        'lines_changed': 0,
        'total_lines': 1,
        'code_churn_ratio': 0,
        'author_success_rate': 0.8,
        'service_failure_rate_7d': 0.05,
        'is_hotfix': False,
        'touches_critical_path': False,
        'test_coverage': 70,
        'build_duration_sec': 180,
        'risk_score': 0
    }
    return defaults.get(feature, None)


class FeatureEngineering:
    @staticmethod
    @task(log_prints=True, retries=3, retry_delay_seconds=10)
    def create_deployment_features(raw_data: Dict, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extracts deployment features from raw data.

        Args:
            raw_data (Dict): Input deployment data.
            feature_list (Optional[List[str]]): List of features to extract. Defaults to pre-defined list.

        Returns:
            pd.DataFrame: Single-row DataFrame with extracted features.
        """
        if feature_list is None:
            feature_list = [
                'lines_changed', 'total_lines', 'code_churn_ratio',
                'author_success_rate', 'service_failure_rate_7d',
                'is_hotfix', 'touches_critical_path',
                'test_coverage', 'build_duration_sec', 'risk_score'
            ]
        features = {f: raw_data.get(f, default_for(f)) for f in feature_list}
        return pd.DataFrame([features])


class FeatureStore:
    def __init__(self):
        self.features = {}  # Store computed features if needed
        self.feature_metadata = {}

    @task(log_prints=True)
    def register_feature(self, name: str, description: str, computation_logic: str) -> None:
        """
        Registers a feature with metadata.

        Args:
            name (str): Feature name.
            description (str): Feature description.
            computation_logic (str): Description or code of how feature is computed.
        """
        self.feature_metadata[name] = {
            'description': description,
            'created_at': datetime.now(),
            'version': '1.0',
            'computation': computation_logic,
            'usage_count': 0
        }
        # TODO: persist metadata to disk or DB here

    @task(log_prints=True)
    def increment_usage(self, name: str) -> None:
        """
        Increment the usage count of a registered feature.

        Args:
            name (str): Feature name.
        """
        if name in self.feature_metadata:
            self.feature_metadata[name]['usage_count'] += 1
            # TODO: persist updated usage count

    @task(log_prints=True)
    def compute_feature_importance(self) -> Dict[str, float]:
        """
        Returns static feature importance scores.
        Replace with ML-driven importance as needed.

        Returns:
            Dict[str, float]: Feature importance scores.
        """
        return {
            'code_churn_ratio': 0.25,
            'test_coverage': 0.20,
            'author_success_rate': 0.15,
            'is_hotfix': 0.12,
            'touches_critical_path': 0.10,
            'service_failure_rate_7d': 0.07,
            'build_duration_sec': 0.06,
            'lines_changed': 0.05,
            'risk_score': 0.05
        }
