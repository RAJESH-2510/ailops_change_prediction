import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import joblib
from prefect import flow, task, get_run_logger

from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

# Add parent directory to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Paths to models (updated to match new naming in previous code)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
RF_MODEL_PATH = MODEL_DIR / "RandomForest_model.joblib"  # updated filename per previous pipeline save
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
ENCODERS_PATH = MODEL_DIR / "encoders.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"

from src.pipeline import AIOpsPipeline
from src.features import FeatureEngineering
from src.ml_models import RealTimeInference
from src.dashboard import AIOpsDashboard
from src.cicd import CICDIntegration


class ROICalculator:
    def __init__(self, infrastructure_cost: float = 5000, pushgateway_url: str = "http://localhost:9091") -> None:
        self.costs = {
            "average_incident_cost": 25000,
            "engineer_hourly_rate": 150,
            "downtime_cost_per_hour": 100000,
        }
        self.infrastructure_cost = infrastructure_cost
        self.pushgateway_url = pushgateway_url

    @task(log_prints=True)
    def calculate_savings(
        self,
        baseline_cfr: float,
        current_cfr: float,
        deployments_per_month: int,
        actual_prevented_failures: Optional[float] = None,
    ) -> Dict[str, float]:
        prevented_failures = actual_prevented_failures if actual_prevented_failures is not None else deployments_per_month * (baseline_cfr - current_cfr)

        incident_cost_savings = prevented_failures * self.costs["average_incident_cost"]
        productivity_savings = prevented_failures * 8 * self.costs["engineer_hourly_rate"]
        downtime_savings = prevented_failures * 0.5 * self.costs["downtime_cost_per_hour"]

        total_savings = incident_cost_savings + productivity_savings + downtime_savings
        roi = (total_savings - self.infrastructure_cost) / self.infrastructure_cost

        savings = {
            "prevented_incidents": prevented_failures,
            "incident_cost_savings": incident_cost_savings,
            "productivity_savings": productivity_savings,
            "downtime_savings": downtime_savings,
            "total_monthly_savings": total_savings,
            "roi": roi,
        }

        self._push_to_prometheus(savings)
        return savings

    def _push_to_prometheus(self, savings: Dict[str, float], job_name: str = "aiops_roi_calculator") -> None:
        registry = CollectorRegistry()
        for key, value in savings.items():
            metric_name = f"roi_{key}".lower()
            try:
                gauge = Gauge(metric_name, f"ROI metric for {key}", registry=registry)
                gauge.set(value)
            except Exception as e:
                print(f"❌ Error setting Prometheus metric '{metric_name}': {e}")
        try:
            push_to_gateway(self.pushgateway_url, job=job_name, registry=registry)
            print(f"✅ ROI metrics pushed to Prometheus Pushgateway at {self.pushgateway_url}")
        except Exception as e:
            print(f"❌ Failed to push metrics to Pushgateway: {e}")


@task(log_prints=True)
def ingest_data_task(topic: str) -> List[Dict[str, Any]]:
    pipeline = AIOpsPipeline()
    try:
        records = pipeline.ingest_data(topic=topic, timeout=10, max_records=50)
    finally:
        pipeline.close()
    return records or []


@task(log_prints=True)
def feature_engineering_task(raw_data: Dict[str, Any]) -> pd.DataFrame:
    fe = FeatureEngineering()
    return fe.create_deployment_features(raw_data=raw_data)


@task(log_prints=True)
def model_inference_task(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    classifier = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    predictor = RealTimeInference(classifier, scaler)
    return predictor.predict_with_explanation(raw_data)


@task(log_prints=True)
def dashboard_task(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    dashboard = AIOpsDashboard()
    return dashboard.generate_dashboard_data(metrics_data)


@task(log_prints=True)
def run_jenkins_plugin_task(predictor: RealTimeInference, build_data: Dict[str, Any]) -> Dict[str, Any]:
    cicd = CICDIntegration(predictor)
    return cicd.jenkins_plugin(build_data)


@task(log_prints=True)
def ci_task(prediction: Dict[str, Any], raw_data: Dict[str, Any]) -> Any:
    from src.continuous_improvement import ContinuousImprovement

    ci = ContinuousImprovement()
    actual_outcome = raw_data.get("failure_label", False)
    ci.collect_feedback(prediction, actual_outcome)
    return ci.analyze_performance()


roi_calculator = ROICalculator()


@flow(name="Change Failure Prediction Pipeline with ROI", log_prints=True)
def main_flow() -> Dict[str, Any]:
    logger = get_run_logger()

    if not RF_MODEL_PATH.exists() or not SCALER_PATH.exists():
        logger.error("❌ Model files missing.")
        return {"status": "Model files missing"}

    classifier = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    predictor = RealTimeInference(classifier, scaler)

    ingested_records = ingest_data_task(topic="deployment-metrics")
    if not ingested_records:
        logger.info("⚠️ No new data found.")
        return {"status": "No new data"}

    results = []
    baseline_cfr = 0.3
    current_cfr = 0.1
    deployments_per_month = 100
    total_prevented_failures = 0
    active_deployments_count = 0

    for record in ingested_records:
        raw_data = record.get("data")
        if not raw_data:
            logger.warning("⚠️ Skipping empty record.")
            continue

        logger.info(f"📦 Processing deployment: {raw_data.get('deployment_id')}")
        active_deployments_count += 1

        features_df = feature_engineering_task(raw_data=raw_data)

        # Run inference
        prediction = model_inference_task(raw_data=raw_data)

        # Jenkins + CI/CD
        cicd_result = run_jenkins_plugin_task(predictor, raw_data)
        is_correct = cicd_result.get("prediction_correct", False)
        if is_correct:
            total_prevented_failures += 1

        # Dashboard metrics
        real_time_metrics = {
            "change_failure_rate_change_percentage": ((baseline_cfr - current_cfr) / baseline_cfr) * 100,
            "model_accuracy_percentage": 98.5,
            "estimated_annual_savings_usd": 123456.78,
            "prevented_incidents_total": total_prevented_failures,
            "mttr_reduction_percentage": 12.3,
            "ml_pipeline_status": 1,
            "active_deployments_count": active_deployments_count,
            "average_risk_score_percentage": raw_data.get("risk_score", 20),
            "current_test_coverage_percentage": 85.0,
            "build_success_rate_percentage": 99.1,
            "current_overall_risk_score": raw_data.get("risk_score", 20),
            "overall_risk_score": raw_data.get("risk_score", 20),
            "your_deployment_metric": {"ANALYZE": 1},
        }

        dashboard_data = dashboard_task(real_time_metrics)

        # Continuous improvement
        performance = ci_task(prediction=prediction, raw_data=raw_data)

        results.append(
            {
                "raw_data": raw_data,
                "features": features_df,
                "prediction": prediction,
                "dashboard": dashboard_data,
                "cicd": cicd_result,
                "performance": performance,
            }
        )

    savings = roi_calculator.calculate_savings(
        baseline_cfr, current_cfr, deployments_per_month, actual_prevented_failures=total_prevented_failures
    )

    if savings.get("roi", 0) < 0:
        logger.warning("⚠️ ROI is negative. Consider reviewing model accuracy or incident cost assumptions.")

    logger.info(f"📊 ROI Calculation: {savings}")
    logger.info(f"✅ Processed {len(results)} deployment(s)")

    return {"results": results, "roi": savings}


if __name__ == "__main__":
    main_flow()
