from prefect import task

class ROICalculator:
    def __init__(self):
        self.costs = {
            'average_incident_cost': 25000,
            'engineer_hourly_rate': 150,
            'downtime_cost_per_hour': 100000
        }

    @task(log_prints=True)
    def calculate_savings(self, baseline_cfr, current_cfr, deployments_per_month):
        prevented_failures = deployments_per_month * (baseline_cfr - current_cfr)
        savings = {
            'prevented_incidents': prevented_failures,
            'incident_cost_savings': prevented_failures * self.costs['average_incident_cost'],
            'productivity_savings': prevented_failures * 8 * self.costs['engineer_hourly_rate'],
            'downtime_savings': prevented_failures * 0.5 * self.costs['downtime_cost_per_hour'],
            'total_monthly_savings': 0
        }
        savings['total_monthly_savings'] = sum([
            savings['incident_cost_savings'],
            savings['productivity_savings'],
            savings['downtime_savings']
        ])
        return savings
