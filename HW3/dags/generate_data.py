from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from utils import RAW_DATA_PATH, MOUNT, DEFAULT_ARGS

with DAG(
    dag_id="generate_data",
    default_args=DEFAULT_ARGS,
    description="DAG for generating data",
    schedule_interval="@daily",
    start_date=days_ago(7)
) as dag:
    get_dataset = DockerOperator(
        image="airflow-download",
        command=f"--output_path {RAW_DATA_PATH}",
        task_id="docker-airflow-download",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        mounts=MOUNT
    )

    get_dataset
