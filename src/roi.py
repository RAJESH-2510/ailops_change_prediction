from prefect import task

class ROICalculator:
    def __init__(self, infrastructure_cost=5000):
        self.costs = {
            'average_incident_cost': 25000,
            'engineer_hourly_rate': 150,
            'downtime_cost_per_hour': 100000
        }
        self.infrastructure_cost = infrastructure_cost

    @task(log_prints=True)
    def calculate_savings(self, baseline_cfr, current_cfr, deployments_per_month, actual_prevented_failures=None):
        """
        :param baseline_cfr: Change Failure Rate before AIops
        :param current_cfr: Change Failure Rate after AIops
        :param deployments_per_month: Total deployments per month
        :param actual_prevented_failures: Optional actual prevented failures
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

        return savings
