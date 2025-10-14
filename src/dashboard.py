# src/dashboard.py
from prefect import task
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
from datetime import datetime

class AIOpsDashboard:
    def __init__(self, pushgateway_url='http://localhost:9091'):
        self.pushgateway_url = pushgateway_url
        self.metrics_registry = CollectorRegistry()

    @task(log_prints=True)
    def generate_dashboard_data(
        self,
        real_time_metrics,
        historical_trends=None,
        model_performance=None,
        top_risk_factors=None
    ):
        """
        Generates dashboard data and pushes metrics to Prometheus Pushgateway.
        real_time_metrics: dict with keys:
            - current_risk_score
            - active_deployments
            - predictions_last_hour
            - prevented_failures
            - system_health ('GOOD', 'WARNING', 'CRITICAL')
        historical_trends: optional dict of historical metrics
        model_performance: optional dict of precision, recall, etc.
        top_risk_factors: optional list of dicts with factor info
        """
        # Validate input to prevent crashes
        if not isinstance(real_time_metrics, dict):
            raise ValueError("real_time_metrics must be a dictionary")

        data = {
            'real_time_metrics': real_time_metrics,
            'historical_trends': historical_trends or {},
            'model_performance': model_performance or {},
            'top_risk_factors': top_risk_factors or []
        }

        # Push metrics safely
        try:
            self._push_metrics(data)
        except Exception as e:
            print(f"[Warning] Failed to push metrics: {e}")

        return data

    def _push_metrics(self, data):
        self.metrics_registry = CollectorRegistry()  # Reset registry

        metrics = data['real_time_metrics']

        # Numeric metrics
        Gauge('current_risk_score', 'Deployment Risk Score', registry=self.metrics_registry).set(
            float(metrics.get('current_risk_score', 0))
        )
        Gauge('active_deployments', 'Number of Active Deployments', registry=self.metrics_registry).set(
            int(metrics.get('active_deployments', 0))
        )
        Gauge('predictions_last_hour', 'Predictions in the Last Hour', registry=self.metrics_registry).set(
            int(metrics.get('predictions_last_hour', 0))
        )
        Gauge('prevented_failures', 'Prevented Failures', registry=self.metrics_registry).set(
            int(metrics.get('prevented_failures', 0))
        )

        # Map system health to numeric
        system_health_map = {'GOOD': 0, 'WARNING': 1, 'CRITICAL': 2}
        Gauge('system_health', 'System Health Status', registry=self.metrics_registry).set(
            system_health_map.get(str(metrics.get('system_health', 'GOOD')).upper(), 0)
        )

        # Push metrics to Prometheus Pushgateway
        push_to_gateway(self.pushgateway_url, job='aiops_pipeline', registry=self.metrics_registry)
        print("[Info] Metrics successfully pushed to Pushgateway.")
