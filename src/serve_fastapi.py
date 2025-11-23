"""FastAPI app to serve clustering model predictions.
Run: uvicorn src.serve_fastapi:app --host 0.0.0.0 --port 8080
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.joblib')

model_bundle = None
try:
    model_bundle = joblib.load(MODEL_PATH)
except Exception as e:
    print('Warning: failed to load model at', MODEL_PATH, e)

app = FastAPI(title='Car Accidents Clustering - Model Serve')

class InputFeatures(BaseModel):
    hour: int
    dayofweek: int
    month: int
    severity: float

@app.get('/health')
def health():
    return {'status':'ok', 'model_loaded': model_bundle is not None}

@app.post('/predict')
def predict(inp: InputFeatures):
    if model_bundle is None:
        return {'error':'model not loaded'}
    model = model_bundle['model']
    features = model_bundle['features']
    x = [[getattr(inp, f) for f in features]]
    labels = model.predict(x)
    return {'cluster': int(labels[0])}
