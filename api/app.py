from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "FastAPI ML API running"}

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}