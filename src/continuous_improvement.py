from datetime import datetime
from prefect import task

class ContinuousImprovement:
    def __init__(self, predictor=None, retraining_callback=None):
        """
        :param predictor: RealTimeInference instance
        :param retraining_callback: function to call when retraining is needed
        """
        self.predictor = predictor
        self.retraining_callback = retraining_callback
        self.feedback_loop = []
        self.improvement_metrics = {}

    @task(log_prints=True)
    def collect_feedback(self, prediction, actual_outcome):
        feedback = {
            'timestamp': datetime.now(),
            'deployment_id': prediction.get('deployment_id'),
            'predicted_risk': prediction['prediction']['failure_probability'],
            'actual_failure': actual_outcome,
            'prediction_correct': (prediction['prediction']['failure_probability'] > 0.5) == actual_outcome
        }
        self.feedback_loop.append(feedback)
        print("Feedback collected:", feedback)
        return feedback

    @task(log_prints=True)
    def analyze_performance(self):
        if len(self.feedback_loop) < 50:
            print("Insufficient data for analysis")
            return {"status": "insufficient_data", "total_predictions": len(self.feedback_loop)}

        correct_predictions = sum(1 for f in self.feedback_loop if f['prediction_correct'])
        accuracy = correct_predictions / len(self.feedback_loop)

        result = {
            'accuracy': accuracy,
            'total_predictions': len(self.feedback_loop),
            'improvement_needed': accuracy < 0.85,
            'retrain_recommended': accuracy < 0.85
        }

        print("Performance analysis:", result)

        if result['retrain_recommended'] and self.retraining_callback:
            print("Triggering retraining...")
            self.retraining_callback()

        return result
