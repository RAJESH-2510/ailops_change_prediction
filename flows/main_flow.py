import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import joblib
from prefect import flow, task, get_run_logger

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Paths to models
MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
RF_MODEL_PATH = MODEL_DIR / 'rf_model.joblib'
SCALER_PATH = MODEL_DIR / 'scaler.joblib'

from src.pipeline import AIOpsPipeline
from src.features import FeatureEngineering
from src.ml_models import RealTimeInference
from src.dashboard import AIOpsDashboard

# Import CICD task wrappers
from src.cicd import CICDIntegration

# -----------------------------
# Tasks
# -----------------------------

@task(log_prints=True)
def ingest_data_task(topic: str) -> List[Dict[str, Any]]:
    pipeline = AIOpsPipeline()
    records = pipeline.ingest_data(topic=topic, timeout=10, max_records=50)
    pipeline.close()
    return records or []

@task(log_prints=True)
def feature_engineering_task(raw_data: Dict[str, Any]) -> pd.DataFrame:
    fe = FeatureEngineering()
    return fe.create_deployment_features(raw_data=raw_data)

@task(log_prints=True)
def model_inference_task(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    logger = get_run_logger()
    if not RF_MODEL_PATH.exists() or not SCALER_PATH.exists():
        logger.error(f"Model files not found at {RF_MODEL_PATH} or {SCALER_PATH}")
        raise FileNotFoundError("Required model files are missing.")
    classifier = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    inference = RealTimeInference(classifier, scaler)
    return inference.predict_with_explanation(deployment_data=raw_data)

@task(log_prints=True)
def dashboard_task(real_time_metrics: Dict[str, Any]) -> Dict[str, Any]:
    dashboard = AIOpsDashboard()
    return dashboard.generate_dashboard_data(real_time_metrics=real_time_metrics)

# CICD tasks wrapping CICDIntegration methods

@task(log_prints=True)
def run_jenkins_plugin_task(predictor, build_data):
    cicd = CICDIntegration(predictor)
    return cicd.jenkins_plugin(build_data)

@task(log_prints=True)
def push_metrics_task(deployment_data):
    dashboard = AIOpsDashboard()
    return dashboard.generate_dashboard_data(deployment_data)

@task(log_prints=True)
def ci_task(prediction: Dict[str, Any], raw_data: Dict[str, Any]) -> Any:
    from src.continuous_improvement import ContinuousImprovement
    ci = ContinuousImprovement()
    actual_outcome = raw_data.get('failure_label', False)
    ci.collect_feedback(prediction, actual_outcome)
    return ci.analyze_performance()

# -----------------------------
# Flow
# -----------------------------

@flow(name="Change Failure Prediction Pipeline", log_prints=True)
def main_flow() -> Dict[str, Any]:
    logger = get_run_logger()

    # Load models once
    if not RF_MODEL_PATH.exists() or not SCALER_PATH.exists():
        logger.error(f"Model files not found at {RF_MODEL_PATH} or {SCALER_PATH}")
        return {"status": "Model files missing"}

    classifier = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    predictor = RealTimeInference(classifier, scaler)

    ingested_records = ingest_data_task(topic='deployment-metrics')

    if not ingested_records:
        logger.info("No new data")
        return {"status": "No new data"}

    results = []

    for record in ingested_records:
        try:
            raw_data = record.get('data')
            if not raw_data:
                logger.warning("Skipping empty record")
                continue

            logger.info(f"Processing deployment: {raw_data.get('deployment_id')}")

            # Feature engineering
            features_df = feature_engineering_task(raw_data=raw_data)

            # Model inference
            prediction = model_inference_task(raw_data=raw_data)

            # Prepare metrics for dashboard
            real_time_metrics = {
                'current_risk_score': raw_data.get('risk_score', 0),
                'active_deployments': 1,  # Could be dynamic if you track it
                'predictions_last_hour': 10,  # Placeholder, implement as needed
                'prevented_failures': 0,  # Placeholder, implement as needed
                'system_health': 'GOOD'  # Could come from monitoring logic
            }

            # Dashboard update
            dashboard_data = dashboard_task(real_time_metrics=real_time_metrics)

            # CICD integration - use wrapped tasks with predictor object
            cicd_result = run_jenkins_plugin_task(predictor, raw_data)

            # Push metrics to dashboard from CICD (optional, here dashboard_task already does)
            push_metrics_task(raw_data)

            # Continuous improvement
            performance = ci_task(prediction=prediction, raw_data=raw_data)

            results.append({
                'raw_data': raw_data,
                'features': features_df,
                'prediction': prediction,
                'dashboard': dashboard_data,
                'cicd': cicd_result,
                'performance': performance
            })

        except Exception as e:
            logger.error(f"Error processing record: {e}")

    logger.info(f"✅ Processed {len(results)} deployment(s)")
    return {"results": results}


if __name__ == "__main__":
    main_flow()
