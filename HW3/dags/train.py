from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from utils import RAW_DATA_PATH, MOUNT, DEFAULT_ARGS, PREPROCESSOR_NAME, PROCESSED_DATA_PATH, MODELS_PATH, MODEL_NAME, METRICS_PATH

with DAG(
    dag_id="preprocess_split_train_validate",
    default_args=DEFAULT_ARGS,
    description="DAG for generating data",
    schedule_interval="@daily",
    start_date=days_ago(7)
) as dag:
    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f'--raw-data-path {RAW_DATA_PATH} --processed-data-path {PROCESSED_DATA_PATH}',
        task_id="docker-airflow-preprocess",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        mounts=MOUNT
    )

    split = DockerOperator(
        image="airflow-split",
        command=f'--processed-data-path {PROCESSED_DATA_PATH} --output-dir {PROCESSED_DATA_PATH} --models-dir {MODELS_PATH} --preprocessor-name {PREPROCESSOR_NAME}',
        task_id="docker-airflow-split",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        mounts=MOUNT
    )

    train = DockerOperator(
        image="airflow-train",
        command=f'--processed-data-path {PROCESSED_DATA_PATH} --models-dir {MODELS_PATH} --preprocessor-name {PREPROCESSOR_NAME} --model-name {MODEL_NAME}',
        task_id="docker-airflow-train",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        mounts=MOUNT
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f'--data-dir {PROCESSED_DATA_PATH} --models-dir {MODELS_PATH} --output-dir {METRICS_PATH} --preprocessor-name {PREPROCESSOR_NAME} --model-name {MODEL_NAME}',
        task_id="docker-airflow-validate",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        mounts=MOUNT
    )

    preprocess >> split >> train >> validate
