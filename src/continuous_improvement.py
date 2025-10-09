from prefect import task
from datetime import datetime

class ContinuousImprovement:
    def __init__(self):
        self.feedback_loop = []
        self.improvement_metrics = {}

    @task(log_prints=True)
    def collect_feedback(self, prediction, actual_outcome):
        feedback = {
            'timestamp': datetime.now(),
            'predicted_risk': prediction['prediction']['failure_probability'],
            'actual_failure': actual_outcome,
            'prediction_correct': (prediction['prediction']['failure_probability'] > 0.5) == actual_outcome
        }
        self.feedback_loop.append(feedback)

    @task(log_prints=True)
    def analyze_performance(self):
        if len(self.feedback_loop) < 100:
            return "Insufficient data for analysis"

        correct_predictions = sum(1 for f in self.feedback_loop if f['prediction_correct'])
        accuracy = correct_predictions / len(self.feedback_loop)

        return {
            'accuracy': accuracy,
            'total_predictions': len(self.feedback_loop),
            'improvement_needed': accuracy < 0.85,
            'retrain_recommended': accuracy < 0.85
        }
