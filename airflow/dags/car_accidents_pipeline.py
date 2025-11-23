from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(dag_id='car_accidents_etl', start_date=datetime(2025,1,1), schedule_interval='@daily', default_args=default_args, catchup=False) as dag:
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python /opt/airflow/dags/../..//src/data_preprocessing.py --input /opt/airflow/dags/../..//data/raw/accidents.csv --output /opt/airflow/dags/../..//data/processed/processed.csv'
    )

    train = BashOperator(
        task_id='train',
        bash_command='python /opt/airflow/dags/../..//src/model_training_mlflow.py --input /opt/airflow/dags/../..//data/processed/processed.csv --k 5 --output /opt/airflow/dags/../..//models/best_model.joblib'
    )

    preprocess >> train
