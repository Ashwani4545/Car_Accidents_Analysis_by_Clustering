FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV MODEL_PATH=/app/models/best_model.joblib
EXPOSE 8080
CMD ["uvicorn", "src.serve_fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
