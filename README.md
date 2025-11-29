# Car Accidents Analysis Using Clustering 

## Project Overview

This project analyzes vehicle accident records and groups them into **clusters** based on time, location, and severity patterns. The goal is to:
```
- Identify hidden behavioral and environmental patterns
- Support traffic safety planning
- Detect high-risk accident zones
- Enable future real-time risk scoring

Clustering algorithms used:

- **K-Means**
- **Gaussian Mixture Models (GMM)**

The end-to-end workflow includes:

- ETL preprocessing
- Feature engineering
- Scalable model training
- Experiment tracking
- Model serving via API
- Docker & Kubernetes deployment
- Full pipeline orchestration
```
---

## Key Features
```
✔ Modern end-to-end MLOps workflow
✔ Modular Python source code (`src/`)
✔ Data versioning using **DVC**
✔ Training experiments tracked using **MLflow**

✔ Workflow orchestration using **Apache Airflow**

✔ Real-time model inference using **FastAPI + Uvicorn**

✔ Containerization using **Docker**

✔ Production deployment via **Kubernetes manifests**

✔ Monitoring using **Prometheus metrics exporter**

✔ Optional distributed processing using **PySpark / Dask** stubs
```
---

# Architecture Diagram

```
Raw Data → DVC Storage → Preprocessing → MLflow Tracking → Model Registry
         ↓ Airflow DAG → Training → Best Model → FastAPI Serving → Monitoring (Prometheus)
                        ↓ Docker/K8s Deployment → Autoscaling
```

---

# Technology Stack (Advanced)

### 🧠 Machine Learning

- Scikit-learn (Clustering Models)
- PySpark / Dask (Optional distributed pipeline)

### 📦 Data Engineering

- **DVC** for dataset versioning
- **Airflow** for ETL orchestration
- **PyArrow** for optimized file handling

### 📊 Experiment Tracking & Model Registry

- **MLflow** (UI, metrics, parameters, artifact storage)

### 🚀 Model Serving & Deployment

- **FastAPI** for live model inference
- **Docker** for containerization
- **Kubernetes** deployment manifests

### 🛠 DevOps & Automation

- **GitHub Actions** (CI/CD pipeline)
- **Docker Compose** for local orchestrated environments

### 📡 Monitoring & Observability

- **Prometheus client** for metrics
- Grafana (recommended setup)

---

## Project Structure

```
Car_Accidents_AdvancedStack_Project/
│
├── src/
│   ├── data_preprocessing.py        # ETL processing
│   ├── model_training_mlflow.py     # Training + MLflow logging
│   ├── serve_fastapi.py             # API for model prediction
│   ├── metrics_exporter.py          # Prometheus exporter
│   └── utils.py                     # Helper functions
│
├── mlflow/                          # MLflow configuration
├── airflow/
│   └── dags/car_accidents_pipeline.py
│
├── deploy/
│   ├── Dockerfile
│   ├── k8s/deployment.yaml
│   └── k8s/service.yaml (optional)
│
├── dvc.yaml                         # DVC pipeline definition
├── docker-compose.yml               # MLflow + API local stack
├── requirements.txt
├── Makefile
└── README.md
```

---

# How to Run the Project

### 🔧 Install Dependencies

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 📂 DVC Setup

```
dvc init
dvc repro          # runs `dvc.yaml` pipeline
```

### 🔬 MLflow Tracking

Start MLflow UI:

```
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 --port 5000
```

Visit:

```
http://localhost:5000
```

### 🚦 Run the Pipeline Manually

Preprocess Data

```
python src/data_preprocessing.py --input data/raw/accidents.csv --output data/processed/processed.csv
```

### Train Model

```
python src/model_training_mlflow.py --input data/processed/processed.csv --k 5 --output models/best_model.joblib
```

### 🌐 Run FastAPI Model Server

```
uvicorn src.serve_fastapi:app --host 0.0.0.0 --port 8080
```

### API Endpoints

```
GET  /health
POST /predict
```

**Example Request:**

```json
{
  "hour": 14,
  "dayofweek": 2,
  "month": 8,
  "severity": 3
}
```

### 🐳 Docker Deployment

Build the image:

```
docker build -t accident-api:latest ./deploy
```

Run the container:

```
docker run -p 8080:8080 accident-api:latest
```

### ☸️ Kubernetes Deployment

Apply manifests:

```
kubectl apply -f deploy/k8s/deployment.yaml
```

Check pods:

```
kubectl get pods
```

### 📈 Monitoring with Prometheus

Start exporter:

```
python src/metrics_exporter.py
```

Visit metrics at:

```
http://localhost:8000
```

---

# CI/CD Pipeline (GitHub Actions)

Every push to `main` triggers:

- Dependency install
- Linting
- Future: automated DVC + MLflow jobs

YAML in:

```
.github/workflows/ci.yml
```

---

# Recommended Improvements

- Add DB integration (Snowflake, PostgreSQL, BigQuery)
- Create a Streamlit dashboard for interactive cluster visualization
- Add HDBSCAN + UMAP for advanced clustering
- Enable GPU-powered clustering
- Build full production observability via Grafana dashboards

---

# License

MIT License — free for personal and commercial use.
