# src/app.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------
# Load saved models
# -------------------------
rf_model = joblib.load("models/best_randomforest.pkl")
xgb_model = joblib.load("models/best_xgboost.pkl")

# -------------------------
# Define input schema
# -------------------------
class Transaction(BaseModel):
    CUSTOMER_ID: int
    TERMINAL_ID: int
    TX_AMOUNT: float
    TX_TIME_SECONDS: int
    TX_TIME_DAYS: int

# -------------------------
# Create FastAPI app
# -------------------------
app = FastAPI(title="Fraud Detection API")

@app.get("/")
def home():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert request to DataFrame
    df = pd.DataFrame([transaction.dict()])

    # Run predictions
    pred_rf = rf_model.predict(df)[0]
    pred_xgb = xgb_model.predict(df)[0]
    proba_rf = rf_model.predict_proba(df)[0, 1]
    proba_xgb = xgb_model.predict_proba(df)[0, 1]

    return {
        "RandomForest": {"prediction": int(pred_rf), "fraud_probability": float(proba_rf)},
        "XGBoost": {"prediction": int(pred_xgb), "fraud_probability": float(proba_xgb)}
    }
