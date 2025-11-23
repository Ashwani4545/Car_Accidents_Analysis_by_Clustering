"""Data preprocessing for Car Accidents project.
Includes hooks for DVC (data versioning) and optional Spark/Dask processing.
Tracked with MLflow-friendly outputs.
"""
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from joblib import dump

def load_csv(path):
    return pd.read_csv(path)

def basic_cleaning(df):
    df = df.drop_duplicates()
    # example latitude/longitude columns
    if {'latitude','longitude'}.issubset(df.columns):
        df = df.dropna(subset=['latitude','longitude'])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

def feature_engineering(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
    if 'severity' in df.columns:
        df['severity'] = pd.to_numeric(df['severity'], errors='coerce').fillna(0)
    return df

def scale_and_save(df, features, out_csv, scaler_path):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df_scaled.to_csv(out_csv, index=False)
    dump(scaler, scaler_path)
    return out_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scaler', required=False, default='models/scaler.joblib')
    args = parser.parse_args()

    df = load_csv(args.input)
    df = basic_cleaning(df)
    df = feature_engineering(df)
    candidate_features = [c for c in ['hour','dayofweek','month','severity'] if c in df.columns]
    if not candidate_features:
        raise SystemExit('No features detected for scaling. Provide a dataset with timestamp/severity columns.')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scale_and_save(df, candidate_features, args.output, args.scaler)
    print('Preprocessing complete. Output saved to', args.output)
