import os
# what does this do?
# get the absolute path of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# form the respective paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

# Model configuration
MODEL_NAME = "gpt2"  # 124M parameters - runs on CPU
MAX_LENGTH = 128
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "sentiment_analysis_gpt2"

# Drift detection thresholds
DRIFT_THRESHOLD = 0.1
RETRAIN_THRESHOLD = 3  # Retrain after 3 consecutive drift detections