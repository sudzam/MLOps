import pandas as pd
import os
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow
from config import REPORT_DIR, DRIFT_THRESHOLD

class DriftDetector:
    def __init__(self):
        self.drift_history = []

    # Detect drift in text data using Evidently
    def detect_text_drift(self, reference_data, current_data, text_column="text"):
        reference_df = pd.DataFrame(reference_data)
        current_df = pd.DataFrame(current_data)

        # Validate that the required text column exists
        if text_column not in reference_df.columns or text_column not in current_df.columns:
            raise ValueError(f"Both dataframes must contain the '{text_column}' column.")

        # Use only DataDriftPreset for current Evidently (2025+)
        report = Report(metrics=[
            DataDriftPreset(),
        ])

        # Run report
        report.run(
            reference_data=reference_df,
            current_data=current_df
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report.save_html(report_path)

        # Extract drift metrics robustly
        report_dict = report.as_dict()
        dataset_drift = None
        drift_score = None
        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "DataDriftPreset":
                dataset_drift = metric["result"].get("dataset_drift")
                drift_score = metric["result"].get("drift_share")
                break
        if dataset_drift is None or drift_score is None:
            raise ValueError("Drift metrics not found in report output.")

        # Log metrics to MLflow
        with mlflow.start_run():
            mlflow.log_metric("drift_score", drift_score)
            mlflow.log_metric("dataset_drift", int(dataset_drift))
            mlflow.log_artifact(report_path)

        # Update drift history
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drift_detected': dataset_drift,
            'drift_score': drift_score
        })

        return dataset_drift, drift_score, report_path

    # Determine if model should be retrained based on drift history
    def should_retrain(self):
        if len(self.drift_history) < 3:
            return False
        recent_drifts = [d['drift_detected'] for d in self.drift_history[-3:]]
        return all(recent_drifts)  # Retrain if all of the last three detections show drift

def check_and_handle_drift(reference_data_path, current_data_path, text_column="text"):
    reference_df = pd.read_csv(reference_data_path)
    current_df = pd.read_csv(current_data_path)

    detector = DriftDetector()
    drift_detected, drift_score, report_path = detector.detect_text_drift(
        reference_df, current_df, text_column
    )

    print(f"Drift detected: {drift_detected}")
    print(f"Drift score: {drift_score:.4f}")
    print(f"Report saved: {report_path}")

    should_retrain = detector.should_retrain()
    return drift_detected, drift_score, should_retrain