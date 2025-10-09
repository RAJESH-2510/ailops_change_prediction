from prefect import task
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
import random
from datetime import datetime


class AIOpsDashboard:
    def __init__(self):
        self.metrics = {}

    @task(log_prints=True)
    def generate_dashboard_data(self):
        data = {
            'real_time_metrics': {
                'current_risk_score': random.uniform(0, 1),
                'active_deployments': random.randint(1, 5),
                'predictions_last_hour': random.randint(10, 50),
                'prevented_failures': random.randint(0, 3),
                'system_health': random.choice(['GOOD', 'WARNING', 'CRITICAL'])
            },
            'historical_trends': {
                'daily_failure_rate': [random.uniform(0.01, 0.06) for _ in range(7)],
                'prediction_accuracy': [random.uniform(0.84, 0.89) for _ in range(7)],
                'deployment_volume': [random.randint(45, 55) for _ in range(7)]
            },
            'model_performance': {
                'precision': 0.88,
                'recall': 0.82,
                'f1_score': 0.85,
                'auc_roc': 0.91,
                'last_retrained': datetime.now().strftime('%Y-%m-%d'),
                'data_drift_detected': random.choice([True, False])
            },
            'top_risk_factors': [
                {'factor': 'Code Churn', 'impact': 0.25, 'trend': 'increasing'},
                {'factor': 'Test Coverage', 'impact': 0.20, 'trend': 'stable'},
                {'factor': 'Deploy Time', 'impact': 0.15, 'trend': 'decreasing'}
            ]
        }

        self._push_metrics(data)
        return data

    def _push_metrics(self, data):
        registry = CollectorRegistry()
        gauge = Gauge('risk_score', 'Deployment Risk Score', registry=registry)
        gauge.set(data['real_time_metrics']['current_risk_score'])
        push_to_gateway('localhost:9090', job='aiops', registry=registry)
