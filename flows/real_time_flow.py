import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prefect import flow
from src.pipeline import AIOpsPipeline
from src.features import FeatureEngineering
from src.ml_models import RealTimeInference
import pandas as pd
import joblib

@flow(name="Real-Time Inference Flow", log_prints=True)
def real_time_flow():
    pipeline = AIOpsPipeline()
    ingested = pipeline.ingest_data(topic='deployment-metrics')

    if ingested is None:
        return {"status": "No new data"}

    fe = FeatureEngineering()
    features = fe.create_deployment_features(ingested['data'])
    features_df = pd.DataFrame([features])

    classifier = joblib.load('models/rf_model.joblib')
    inference = RealTimeInference(classifier, None)
    prediction = inference.predict_with_explanation(ingested['data'])

    return prediction

if __name__ == "__main__":
    real_time_flow()
