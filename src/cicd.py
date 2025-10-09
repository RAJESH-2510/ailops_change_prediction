from prefect import task
import pandas as pd


class CICDIntegration:
    def __init__(self, predictor):
        self.predictor = predictor
        self.webhook_endpoints = []

    @task(log_prints=True)
    def jenkins_plugin(self, build_data):
        risk_assessment = self.predictor.predict_with_explanation(build_data)
        if risk_assessment['prediction']['failure_probability'] > 0.7:
            return {
                'action': 'BLOCK',
                'message': 'High risk deployment detected',
                'details': risk_assessment
            }
        return {'action': 'PROCEED', 'details': risk_assessment}

    @task(log_prints=True)
    def github_actions_check(self, pr_data):
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
        return self.predictor.predict_with_explanation(risk_df.to_dict(orient='records')[0])
