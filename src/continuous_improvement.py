from datetime import datetime
from prefect import task
from typing import Callable, Optional, Dict, Any
import numpy as np
import json
import os


class ContinuousImprovement:
    def __init__(
        self,
        predictor: Optional[Any] = None,
        retraining_callback: Optional[Callable] = None,
        feedback_file: str = "feedback_data.json",
    ):
        """
        Continuous Improvement engine that tracks model performance
        and suggests retraining if performance drops.

        :param predictor: RealTimeInference instance or similar predictor
        :param retraining_callback: Function to call when retraining is needed
        :param feedback_file: Path to JSON file used to persist feedback data
        """
        self.predictor = predictor
        self.retraining_callback = retraining_callback
        self.feedback_file = feedback_file

        # Load persisted feedback data if it exists
        self.feedback_loop = self._load_feedback()
        self.improvement_metrics = {}

    # ---------- Internal Helper Methods ----------

    def _load_feedback(self):
        """Load feedback data from file if it exists."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load feedback data: {e}")
        return []

    def _save_feedback(self):
        """Persist feedback data to a file."""
        try:
            with open(self.feedback_file, "w") as f:
                json.dump(self.feedback_loop, f, default=str, indent=4)
        except Exception as e:
            print(f"❌ Failed to save feedback data: {e}")

    # ---------- Prefect Tasks ----------

    @task(log_prints=True)
    def collect_feedback(self, prediction: Dict, actual_outcome: int) -> Dict:
        """
        Collect feedback from prediction and actual outcome.

        Args:
            prediction (Dict): Prediction dictionary, expected to contain prediction info.
            actual_outcome (int): Actual outcome label (0 or 1).

        Returns:
            Dict: Feedback record collected.
        """

        # Extract predicted risk (failure probability)
        predicted_risk = None

        try:
            predicted_risk = prediction.get("prediction", {}).get("failure_probability")
        except Exception:
            predicted_risk = None

        # Fallbacks
        if predicted_risk is None:
            if "probability" in prediction:
                predicted_risk = prediction["probability"]
            elif "explanation" in prediction and "risk_score" in prediction["explanation"]:
                predicted_risk = prediction["explanation"]["risk_score"]

        # Default fallback
        if predicted_risk is None:
            predicted_risk = 0.0

        # Convert numpy types to native Python types
        if isinstance(predicted_risk, (np.generic,)):
            predicted_risk = float(predicted_risk)
        if isinstance(actual_outcome, (np.generic,)):
            actual_outcome = int(actual_outcome)

        prediction_correct = (predicted_risk > 0.5) == bool(actual_outcome)

        feedback = {
            "timestamp": datetime.now().isoformat(),
            "deployment_id": prediction.get("deployment_id", None),
            "predicted_risk": predicted_risk,
            "actual_failure": actual_outcome,
            "prediction_correct": prediction_correct,
        }

        self.feedback_loop.append(feedback)
        self._save_feedback()  # persist to disk

        print(f"✅ Feedback collected: {feedback}")
        print(f"📈 Total feedback entries stored: {len(self.feedback_loop)}")
        return feedback

    @task(log_prints=True)
    def analyze_performance(self) -> Dict:
        """
        Analyze feedback to measure performance and decide if retraining is needed.

        Returns:
            Dict: Performance summary and retraining recommendation.
        """
        total = len(self.feedback_loop)
        if total < 30:
            print(f"⚠️ Insufficient data for analysis: {total} entries collected.")
            return {"status": "insufficient_data", "total_predictions": total}

        correct_predictions = sum(1 for f in self.feedback_loop if f["prediction_correct"])
        accuracy = correct_predictions / total

        result = {
            "accuracy": round(accuracy, 3),
            "total_predictions": total,
            "improvement_needed": accuracy < 0.85,
            "retrain_recommended": accuracy < 0.85,
        }

        print(f"📊 Performance analysis: {result}")

        # Optional retraining trigger
        if result["retrain_recommended"] and self.retraining_callback:
            print("⚙️ Triggering retraining callback...")
            self.retraining_callback()

        return result
