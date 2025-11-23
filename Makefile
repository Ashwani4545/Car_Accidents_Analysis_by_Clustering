.PHONY: install deps run-preprocess train serve mlflow

install:
	python -m venv venv && . venv/bin/activate && pip install -r requirements.txt

run-preprocess:
	python src/data_preprocessing.py --input data/raw/accidents.csv --output data/processed/processed.csv

train:
	python src/model_training_mlflow.py --input data/processed/processed.csv --k 5 --output models/best_model.joblib

serve:
	uvicorn src.serve_fastapi:app --host 0.0.0.0 --port 8080

mlflow:
	docker-compose up -d mlflow
