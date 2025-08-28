from metaflow import FlowSpec, step, Parameter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import load_and_prepare_data, preprocess_data, simulate_drift_data
from src.drift_detection import check_and_handle_drift

print ("before importing train_and_log_model")
from src.model_training import train_and_log_model

from config import *
import mlflow

class SentimentMLOpsFlow(FlowSpec):
    model_version = Parameter('model_version', default='1.0')
    simulate_drift = Parameter('simulate_drift', default=True)
    
    @step
    def start(self):
        self._debug = True
        """Initialize MLflow experiment"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        # print mlflow tracking uri and experiment name
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow Experiment: {EXPERIMENT_NAME}")
        
        print("MLOps Pipeline Started")
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """Load and prepare training data"""
        print("Loading and preparing data...")
        
        # Load data
        self.raw_data = load_and_prepare_data()
        self.processed_data, self.processed_path = preprocess_data(self.raw_data)
        
        # Simulate drift if requested
        if self.simulate_drift:
            self.drift_data, self.drift_path = simulate_drift_data(self.raw_data)
        
        print(f"Loaded {len(self.raw_data)} samples")
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """Train the GPT-2 model"""
        print("Training model...")
        
        self.model_path = train_and_log_model(
            self.processed_path, 
            self.model_version
        )
        print(f"Model trained and saved to: {self.model_path}")
        self.next(self.end)
    """
    @step
    def check_drift(self):
        try:
            print(f"[DEBUG] simulate_drift={self.simulate_drift}, has_drift_path={hasattr(self, 'drift_path')}")
            if self.simulate_drift and hasattr(self, 'drift_path'):
                print("Checking for data drift...")
                try:
                    self.drift_detected, self.drift_score, self.should_retrain = check_and_handle_drift(
                        os.path.join(DATA_DIR, "raw_data.csv"),
                        self.drift_path
                    )
                except Exception as e:
                    # If drift detection fails, default to not retraining
                    import traceback
                    print("[WARN] Drift detection failed:", e)
                    traceback.print_exc()
                    self.drift_detected = False
                    self.should_retrain = False

                if self.should_retrain:
                    print("Drift detected! Retraining recommended.")
                    self.next(self.retrain_model)
                    return
                else:
                    print("No significant drift detected.")
                    self.next(self.end)
                    return
            else:
                print("No drift simulation requested or drift_path missing.")
                self.next(self.end)
                return

        except Exception as e:
            # Last-resort error catcher → send flow to end instead of dying
            import traceback
            print("[ERROR in check_drift]:", e)
            traceback.print_exc()
            self.drift_detected = False
            self.next(self.end)
    

    @step
    def retrain_model(self):
        # Retrain model if drift is detected
        print("Retraining model due to drift...")
        
        # Increment version for retrained model
        new_version = f"{float(self.model_version) + 0.1:.1f}"
        
        self.retrained_model_path = train_and_log_model(
            self.drift_path,  # Use drift data for retraining
            new_version
        )
        
        print(f"Model retrained with version {new_version}")
        self.next(self.end)
    """

    @step
    def end(self):
        # Pipeline completed
        print("MLOps Pipeline completed successfully!")
        
        # Log final results
        with mlflow.start_run():
            mlflow.log_param("pipeline_status", "completed")
            if hasattr(self, 'drift_detected'):
                mlflow.log_metric("final_drift_detected", int(self.drift_detected))

if __name__ == '__main__':
    print("Starting Sentiment MLOps Flow")
    SentimentMLOpsFlow()