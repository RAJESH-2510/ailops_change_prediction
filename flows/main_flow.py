import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prefect import flow, task
from src.pipeline import AIOpsPipeline
from src.features import FeatureEngineering
from src.ml_models import EnhancedTrainingPipeline, RealTimeInference
from src.dashboard import AIOpsDashboard
from src.cicd import CICDIntegration
from src.continuous_improvement import ContinuousImprovement
import pandas as pd
import random
import joblib

@flow(name="Change Failure Prediction Pipeline", log_prints=True)
def main_flow():
    pipeline = AIOpsPipeline()
    ingested = pipeline.ingest_data(topic='deployment-metrics')

    if ingested is None:
        return {"status": "No new data"}

    fe = FeatureEngineering()
    features = fe.create_deployment_features(ingested['data'])
    features_df = pd.DataFrame([features])

    # Load trained model
    classifier = joblib.load('models/rf_model.joblib')
    inference = RealTimeInference(classifier, None)
    prediction = inference.predict_with_explanation(ingested['data'])

    dashboard = AIOpsDashboard()
    dashboard_data = dashboard.generate_dashboard_data()

    cicd = CICDIntegration(inference)
    cicd_result = cicd.jenkins_plugin(ingested['data'])

    ci = ContinuousImprovement()
    actual_outcome = ingested['data'].get('failure_label', random.choice([True, False]))  # Use label from dataset
    ci.collect_feedback(prediction, actual_outcome)
    performance = ci.analyze_performance()

    return {
        'ingested_data': ingested,
        'features': features,
        'prediction': prediction,
        'dashboard': dashboard_data,
        'cicd': cicd_result,
        'performance': performance
    }

if __name__ == "__main__":
    main_flow()
