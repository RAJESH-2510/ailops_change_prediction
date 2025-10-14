import sys
import os
import pandas as pd
import joblib

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prefect import flow
from src.pipeline import AIOpsPipeline
from src.features import FeatureEngineering
from src.ml_models import RealTimeInference

@flow(name="Real-Time Inference Flow", log_prints=True)
def real_time_flow():
    pipeline = AIOpsPipeline()
    ingested = pipeline.ingest_data(topic='deployment-metrics')

    if ingested is None:
        return {"status": "No new data"}

    # Feature engineering
    fe = FeatureEngineering()
    features = fe.create_deployment_features(ingested['data'])

    # Load RF model and scaler
    if not os.path.exists('models/rf_model.joblib') or not os.path.exists('models/scaler.joblib'):
        raise FileNotFoundError("Model or scaler not found. Run retraining_flow first.")

    classifier = joblib.load('models/rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')

    # Real-time inference
    inference = RealTimeInference(classifier, scaler)
    prediction = inference.predict_with_explanation(ingested['data'])

    return prediction

if __name__ == "__main__":
    result = real_time_flow()
    print("Prediction result:", result)
