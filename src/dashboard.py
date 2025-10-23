from prometheus_client import Gauge, CollectorRegistry, push_to_gateway


class AIOpsDashboard:
    def __init__(self, pushgateway_url='http://localhost:9091', env='prod', service='all'):
        self.pushgateway_url = pushgateway_url
        self.env = env
        self.service = service
        self.metrics_registry = CollectorRegistry()

    def generate_dashboard_data(self, real_time_metrics):
        """
        Pushes the provided real_time_metrics to Prometheus Pushgateway.
        Expected structure:
            - change_failure_rate_change_percentage
            - model_accuracy_percentage
            - estimated_annual_savings_usd
            - prevented_incidents_total
            - mttr_reduction_percentage
            - ml_pipeline_status
            - active_deployments_count
            - average_risk_score_percentage
            - current_test_coverage_percentage
            - build_success_rate_percentage
            - current_overall_risk_score
            - overall_risk_score
            - your_deployment_metric: dict of custom submetrics
        """
        if not isinstance(real_time_metrics, dict):
            raise ValueError("real_time_metrics must be a dictionary")

        try:
            self._push_metrics(real_time_metrics)
        except Exception as e:
            print(f"[Warning] Failed to push metrics: {e}")

        return real_time_metrics

    def _push_metrics(self, metrics):
        self.metrics_registry = CollectorRegistry()  # Reset registry
        labels = {'env': self.env, 'service': self.service}

        def push_metric(name, description, value, labelnames, labelvalues):
            Gauge(
                name,
                description,
                labelnames=labelnames,
                registry=self.metrics_registry
            ).labels(*labelvalues).set(float(value))

        # Define known metric mappings
        metric_map = {
            'change_failure_rate_change_percentage': 'Change Failure Rate Improvement (%)',
            'model_accuracy_percentage': 'Model Accuracy (%)',
            'estimated_annual_savings_usd': 'Estimated Annual Savings (USD)',
            'prevented_incidents_total': 'Total Prevented Incidents',
            'mttr_reduction_percentage': 'MTTR Reduction (%)',
            'ml_pipeline_status': 'ML Pipeline Status (1=Active, 0=Inactive)',
            'active_deployments_count': 'Active Deployments Count',
            'average_risk_score_percentage': 'Average Risk Score (%)',
            'current_test_coverage_percentage': 'Current Test Coverage (%)',
            'build_success_rate_percentage': 'Build Success Rate (%)',
            'current_overall_risk_score': 'Current Overall Risk Score',
            'overall_risk_score': 'Overall Risk Score',
        }

        # Push all known metrics
        for key, description in metric_map.items():
            if key in metrics:
                push_metric(
                    key,
                    description,
                    metrics[key],
                    ['env', 'service'],
                    [self.env, self.service]
                )

        # Push nested custom metric(s) if present
        if 'your_deployment_metric' in metrics and isinstance(metrics['your_deployment_metric'], dict):
            gauge = Gauge(
                'your_deployment_metric',
                'Custom Deployment Metric',
                labelnames=['env', 'service', 'metric'],
                registry=self.metrics_registry
            )
            for sub_key, value in metrics['your_deployment_metric'].items():
                gauge.labels(env=self.env, service=self.service, metric=sub_key).set(float(value))

        # Push all to Pushgateway
        push_to_gateway(self.pushgateway_url, job='aiops_pipeline', registry=self.metrics_registry)
        print("[Info] Metrics successfully pushed to Pushgateway.")
