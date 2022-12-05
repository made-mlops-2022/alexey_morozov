from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from utils import MOUNT, DEFAULT_ARGS, RAW_DATA_PATH, MODELS_PATH, MODEL_NAME, PREDICTIONS_PATH, PREPROCESSOR_NAME

with DAG(
    dag_id="predict",
    default_args=DEFAULT_ARGS,
    description="DAG for generating data",
    schedule_interval="@daily",
    start_date=days_ago(7)
) as dag:
    predict = DockerOperator(
        image="airflow-predict",
        command=f'--data-dir {RAW_DATA_PATH} --models-dir {MODELS_PATH} --output-dir {PREDICTIONS_PATH} --preprocessor-name {PREPROCESSOR_NAME} --model-name {MODEL_NAME}',
        task_id="docker-airflow-predict",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        mounts=MOUNT
    )

    predict
