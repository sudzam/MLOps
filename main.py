import os
import mlflow
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

def setup_mlflow():
    # Setup MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def run_initial_pipeline():
    # Run the initial training pipeline
    print("Running initial training pipeline...")
    os.system("cd flows && python training_flow.py run --model_version 1.0")

def run_drift_simulation():
    # Run pipeline with drift simulation
    print("Running pipeline with drift simulation...")
    os.system("cd flows && python training_flow.py run --model_version 1.1 --simulate_drift True")

def start_mlflow_ui():
    # Start MLflow UI
    print("Starting MLflow UI...")
    print("Access at: http://localhost:5000")
    os.system("mlflow ui")

if __name__ == "__main__":
    # Setup
    setup_mlflow()
    print("=== Sentiment Analysis MLOps Pipeline ===")
    print("1. Run Initial Training")
    print("2. Run with Drift Simulation")
    print("3. Start MLflow UI")
    
    choice = input("Choose option (1-3): ")
    
    if choice == "1":
        run_initial_pipeline()
    elif choice == "2":
        run_drift_simulation()
    elif choice == "3":
        start_mlflow_ui()
    else:
        print("Invalid choice")