"""Train clustering models with MLflow tracking and optional Dask/PySpark hooks.
Saves the best model as an MLflow artifact and to a local `models/` folder.
"""
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import os

def load_data(path):
    return pd.read_csv(path)

def train_kmeans(X, k=5):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    return km, score

def train_gmm(X, k=5):
    gm = GaussianMixture(n_components=k, random_state=42)
    labels = gm.fit_predict(X)
    score = silhouette_score(X, labels)
    return gm, score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experiment', default='car-accidents-clustering')
    args = parser.parse_args()

    df = load_data(args.input)
    feature_cols = [c for c in df.columns if c in ['hour','dayofweek','month','severity']]
    X = df[feature_cols].values

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_param('k', args.k)
        km, score_km = train_kmeans(X, k=args.k)
        mlflow.log_metric('silhouette_km', float(score_km))
        mlflow.sklearn.log_model(km, artifact_path='kmeans_model')

        gm, score_gm = train_gmm(X, k=args.k)
        mlflow.log_metric('silhouette_gmm', float(score_gm))
        mlflow.sklearn.log_model(gm, artifact_path='gmm_model')

        # choose best
        if score_km >= score_gm:
            best_model = km
            best_type = 'kmeans'
            best_score = score_km
        else:
            best_model = gm
            best_type = 'gmm'
            best_score = score_gm

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        joblib.dump({'model': best_model, 'type': best_type, 'features': feature_cols}, args.output)
        mlflow.log_artifact(args.output, artifact_path='best_model')
        mlflow.log_metric('best_silhouette', float(best_score))
        print('Training complete. Best model saved to', args.output)
