from prefect import flow, task
import subprocess
import os

@task(log_prints=True, retries=1)
def run_preprocessing(input_path: str, output_path: str):
    cmd = ['python', 'src/data_preprocessing.py', '--input', input_path, '--output', output_path]
    print('Running preprocessing:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError('Preprocessing failed')

@task(log_prints=True, retries=1)
def run_training(input_path: str, output_model: str, k: int = 5):
    cmd = ['python', 'src/model_training_mlflow.py', '--input', input_path, '--k', str(k), '--output', output_model]
    print('Running training:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError('Training failed')

@flow(name='car-accidents-etl-train')
def etl_and_train(raw_csv: str = 'data/raw/accidents.csv',
                  processed_csv: str = 'data/processed/processed.csv',
                  model_path: str = 'models/best_model.joblib',
                  k: int = 5):
    # ensure directories exist
    os.makedirs(os.path.dirname(processed_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    run_preprocessing(raw_csv, processed_csv)
    run_training(processed_csv, model_path, k)

if __name__ == '__main__':
    etl_and_train()
