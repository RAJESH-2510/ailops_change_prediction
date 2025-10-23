import pandas as pd
from datetime import datetime
from src.dashboard import AIOpsDashboard
import numpy as np


class CICDIntegration:
    def __init__(self, predictor):
        """
        Handles CI/CD risk evaluation and dashboard integration.
        """
        self.predictor = predictor
        self.dashboard = AIOpsDashboard()

    def jenkins_plugin(self, build_data: dict) -> dict:
        """
        Evaluate deployment risk using Jenkins build data.
        """
        # Run prediction
        risk_assessment = self.predictor.predict_with_explanation(build_data)

        print(f"🔍 Prediction value: {risk_assessment}")
        print(f"🔎 Type of prediction['prediction']: {type(risk_assessment.get('prediction'))}")

        # --- Extract predicted risk safely ---
        predicted_risk = None

        if isinstance(risk_assessment, dict):
            # Try multiple keys that may contain risk
            predicted_risk = (
                risk_assessment.get('probability')
                or risk_assessment.get('predicted_risk')
                or (risk_assessment.get('prediction', {}).get('failure_probability')
                    if isinstance(risk_assessment.get('prediction'), dict) else None)
            )

            # Try inside explanation dict as fallback
            if predicted_risk is None and 'explanation' in risk_assessment:
                predicted_risk = risk_assessment['explanation'].get('risk_score')

        # Default fallback
        if predicted_risk is None:
            predicted_risk = 0.0

        # Convert numpy datatypes to native Python types
        if isinstance(predicted_risk, (np.generic,)):
            predicted_risk = predicted_risk.item()

        actual_failure = int(build_data.get('failure_label', 0))

        # --- Determine prediction correctness ---
        prediction_correct = (actual_failure == 0 and predicted_risk < 0.5) or \
                             (actual_failure == 1 and predicted_risk >= 0.5)

        # --- Determine deployment action ---
        if predicted_risk > 0.7:
            action = 'BLOCK'
            message = '🚫 High risk deployment detected'
        elif predicted_risk > 0.4:
            action = 'CAUTION'
            message = '⚠️ Medium risk deployment: extra checks recommended'
        else:
            action = 'PROCEED'
            message = '✅ Low risk deployment: safe to proceed'

        deployment_id = build_data.get('deployment_id', f"build-{datetime.now().strftime('%H%M%S')}")

        print(f"[{datetime.now()}] Deployment {deployment_id} - Action: {action}")

        # --- Return standardized result ---
        return {
            'deployment_id': deployment_id,
            'action': action,
            'message': message,
            'predicted_risk': predicted_risk,
            'actual_failure': actual_failure,
            'prediction_correct': prediction_correct,
            'details': risk_assessment
        }

    def github_actions_check(self, pr_data: dict) -> dict:
        """
        Evaluate risk for GitHub Pull Request.
        """
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

        print(f"📊 GitHub Actions risk analysis complete: {prediction}")
        return prediction

    def push_metrics_to_dashboard(self, deployment_data: dict) -> dict:
        """
        Push deployment metrics to Prometheus dashboard.
        """
        dashboard_data = self.dashboard.generate_dashboard_data(deployment_data)
        print(f"[{datetime.now()}] 📈 Dashboard metrics updated for deployment: {deployment_data.get('deployment_id')}")
        return dashboard_data
