from prefect import task
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

class ROICalculator:
    def __init__(self, infrastructure_cost=5000, pushgateway_url="http://localhost:9091"):
        self.costs = {
            'average_incident_cost': 25000,
            'engineer_hourly_rate': 150,
            'downtime_cost_per_hour': 100000
        }
        self.infrastructure_cost = infrastructure_cost
        self.pushgateway_url = pushgateway_url

    @task(log_prints=True)
    def calculate_savings(self, baseline_cfr, current_cfr, deployments_per_month, actual_prevented_failures=None):
        """
        Calculate ROI and savings from prevented incidents.

        :param baseline_cfr: Change Failure Rate before AIops (float)
        :param current_cfr: Change Failure Rate after AIops (float)
        :param deployments_per_month: Total deployments per month (int)
        :param actual_prevented_failures: Optional actual prevented failures (float or int)
        :return: Dictionary with savings and ROI metrics
        """
        if actual_prevented_failures is not None:
            prevented_failures = actual_prevented_failures
        else:
            prevented_failures = deployments_per_month * (baseline_cfr - current_cfr)

        incident_cost_savings = prevented_failures * self.costs['average_incident_cost']
        productivity_savings = prevented_failures * 8 * self.costs['engineer_hourly_rate']
        downtime_savings = prevented_failures * 0.5 * self.costs['downtime_cost_per_hour']

        total_savings = incident_cost_savings + productivity_savings + downtime_savings
        roi = (total_savings - self.infrastructure_cost) / self.infrastructure_cost

        savings = {
            'prevented_incidents': prevented_failures,
            'incident_cost_savings': incident_cost_savings,
            'productivity_savings': productivity_savings,
            'downtime_savings': downtime_savings,
            'total_monthly_savings': total_savings,
            'roi': roi
        }

        # 🔁 Push dynamic ROI metrics to Prometheus Pushgateway
        self._push_to_prometheus(savings)

        return savings

    def _push_to_prometheus(self, savings: dict, job_name="aiops_roi_calculator"):
        """
        Internal method to push ROI metrics to Prometheus Pushgateway.
        """
        registry = CollectorRegistry()

        for key, value in savings.items():
            metric_name = f"roi_{key}".lower().replace(" ", "_")
            try:
                g = Gauge(metric_name, f"ROI metric for {key}", registry=registry)
                g.set(value)
            except Exception as e:
                print(f"❌ Error setting Prometheus metric '{metric_name}': {e}")

        try:
            push_to_gateway(self.pushgateway_url, job=job_name, registry=registry)
            print(f"✅ ROI metrics pushed to Prometheus Pushgateway at {self.pushgateway_url}")
        except Exception as e:
            print(f"❌ Failed to push metrics to Pushgateway: {e}")
