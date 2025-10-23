import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import joblib
from prefect import flow, task, get_run_logger

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import AIOpsPipeline
from src.features import FeatureEngineering
from src.ml_models import RealTimeInference


@task(log_prints=True)
def ingest_data_task(topic: str) -> Optional[Dict[str, Any]]:
    pipeline = AIOpsPipeline()
    try:
        records = pipeline.ingest_data(topic=topic, timeout=10, max_records=1)
        return records[0] if records else None
    finally:
        pipeline.close()


@task(log_prints=True)
def feature_engineering_task(raw_data: Dict[str, Any]) -> Any:
    fe = FeatureEngineering()
    return fe.create_deployment_features(raw_data=raw_data)


@task(log_prints=True)
def model_inference_task(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    model_dir = Path(__file__).resolve().parent.parent / "models"
    rf_model_path = model_dir / "RandomForest_model.joblib"
    scaler_path = model_dir / "scaler.joblib"

    if not rf_model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler not found. Please run the retraining pipeline first.")

    classifier = joblib.load(rf_model_path)
    scaler = joblib.load(scaler_path)
    predictor = RealTimeInference(classifier, scaler)
    return predictor.predict_with_explanation(raw_data)


@flow(name="Real-Time Inference Flow", log_prints=True)
def real_time_flow() -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting real-time inference flow...")

    ingested_record = ingest_data_task(topic="deployment-metrics")
    if not ingested_record:
        logger.info("No new data found in ingestion step.")
        return {"status": "No new data"}

    raw_data = ingested_record.get("data")
    if not raw_data:
        logger.warning("Ingested record missing 'data' field.")
        return {"status": "Invalid data"}

    logger.info(f"Processing deployment ID: {raw_data.get('deployment_id', 'N/A')}")

    features_df = feature_engineering_task(raw_data=raw_data)
    prediction = model_inference_task(raw_data=raw_data)

    logger.info(f"Prediction completed: {prediction}")

    return {
        "features": features_df.to_dict(orient="records") if hasattr(features_df, "to_dict") else features_df,
        "prediction": prediction,
    }


if __name__ == "__main__":
    result = real_time_flow()
    print("Prediction result:", result)
