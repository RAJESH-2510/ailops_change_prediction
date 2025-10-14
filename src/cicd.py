# src/cicd.py
from prefect import task, flow
import pandas as pd
from datetime import datetime
from src.dashboard import AIOpsDashboard

class CICDIntegration:
    def __init__(self, predictor):
        self.predictor = predictor
        self.dashboard = AIOpsDashboard()

    @task(log_prints=True)
    def jenkins_plugin(self, build_data):
        """
        Evaluate deployment from Jenkins build.
        """
        risk_assessment = self.predictor.predict_with_explanation(build_data)
        failure_prob = risk_assessment['prediction']['failure_probability']

        if failure_prob > 0.7:
            action = 'BLOCK'
            message = 'High risk deployment detected'
        elif failure_prob > 0.4:
            action = 'CAUTION'
            message = 'Medium risk deployment: extra checks recommended'
        else:
            action = 'PROCEED'
            message = 'Low risk deployment: safe to proceed'

        print(f"[{datetime.now()}] Deployment {build_data.get('deployment_id')} - Action: {action}")
        return {
            'deployment_id': build_data.get('deployment_id'),
            'action': action,
            'message': message,
            'details': risk_assessment
        }

    @task(log_prints=True)
    def github_actions_check(self, pr_data):
        """
        Evaluate risk for GitHub PR.
        """
        # Standardize PR data for prediction
        risk_factors = {
            'lines_changed': pr_data.get('lines_changed', 0),
            'total_lines': pr_data.get('total_lines', 1),
            'code_churn_ratio': pr_data.get('code_churn_ratio', 0),
            'author_success_rate': pr_data.get('author_success_rate', 0.8),
            'service_failure_rate_7d': pr_data.get('service_failure_rate_7d', 0.05),
            'is_hotfix': pr_data.get('is_hotfix', False),
            'touches_critical_path': pr_data.get('touches_critical_path', False),
            'test_coverage': pr_data.get('test_coverage', 70),
            'build_duration_sec': pr_data.get('build_duration_sec', 180)
        }

        risk_df = pd.DataFrame([risk_factors])
        prediction = self.predictor.predict_with_explanation(risk_df.to_dict(orient='records')[0])
        return prediction

    @task(log_prints=True)
    def push_metrics_to_dashboard(self, deployment_data):
        """
        Push the deployment metrics to Prometheus dashboard.
        """
        dashboard_data = self.dashboard.generate_dashboard_data(deployment_data)
        print(f"[{datetime.now()}] Dashboard metrics updated.")
        return dashboard_data


# Prefect flow to tie everything together
@flow(name="CICD Deployment Flow")
def cicd_flow(predictor, deployment_data):
    cicd = CICDIntegration(predictor)
    result = cicd.jenkins_plugin(deployment_data)
    cicd.push_metrics_to_dashboard(deployment_data)
    return result


# Example usage
if __name__ == "__main__":
    from src.ml_models import RealTimeInference
    import joblib

    classifier = joblib.load('models/rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    predictor = RealTimeInference(classifier, scaler)

    example_data = {
        'deployment_id': 'deploy_101',
        'lines_changed': 120,
        'total_lines': 5000,
        'code_churn_ratio': 0.024,
        'author_success_rate': 0.85,
        'service_failure_rate_7d': 0.03,
        'is_hotfix': True,
        'touches_critical_path': False,
        'test_coverage': 75,
        'build_duration_sec': 200
    }

    cicd_flow(predictor, example_data)
