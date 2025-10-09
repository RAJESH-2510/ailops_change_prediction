import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prefect import flow
from src.ml_models import EnhancedTrainingPipeline
from src.continuous_improvement import ContinuousImprovement
import pandas as pd

@flow(name="Model Retraining Flow", log_prints=True)
def retraining_flow():
    ci = ContinuousImprovement()
    performance = ci.analyze_performance()

    if isinstance(performance, dict) and performance.get('retrain_recommended', False):
        df = pd.read_csv('data/training_data.csv')
        features = [
            'lines_changed', 'total_lines', 'code_churn_ratio',
            'author_success_rate', 'service_failure_rate_7d',
            'is_hotfix', 'touches_critical_path', 'test_coverage',
            'build_duration_sec'
        ]
        X = df[features]
        y = df['failure_label']

        training_pipeline = EnhancedTrainingPipeline()
        models = training_pipeline.train_model(X, y)
        return {"status": "Retrained", "models": models}

    return {"status": "No retrain needed"}


if __name__ == "__main__":
    retraining_flow()
