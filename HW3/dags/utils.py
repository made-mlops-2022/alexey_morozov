from datetime import timedelta
from docker.types import Mount

DEFAULT_ARGS = {
    "owner": "admin",
    "email": ["admin@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    'email_on_failure': True
}

MOUNT = [Mount(
    source="/mnt/c/1 semester/MLOPS/alexey_morozov/HW3/data",
    target="/data",
    type="bind"
    )]

RAW_DATA_PATH = "/data/raw/{{ ds }}"
PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
MODELS_PATH = "/data/models/{{ ds }}"
PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
METRICS_PATH = "data/metrics/{{ ds }}"
PREPROCESSOR_NAME = "preprocessor.pickle"
MODEL_NAME = "rf.pickle"
