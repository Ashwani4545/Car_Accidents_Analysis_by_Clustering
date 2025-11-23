## Running MLflow locally (quickstart)

Start MLflow server:
```
docker-compose up -d mlflow
```

Then run training with:
```
mlflow run src -e train --experiment-name car-accidents-clustering
```

MLflow UI available at http://localhost:5000
