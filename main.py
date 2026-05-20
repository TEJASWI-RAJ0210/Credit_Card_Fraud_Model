"""
Credit Card Fraud Detection API
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
import shap

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost model with SHAP explanations for real-time fraud detection.",
    version="1.0.0",
)

# ── CORS (allow your React dev server) ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & SHAP explainer once at startup ───────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model, explainer
    with open("xgb_credit_fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)

MODEL_FEATURES = ["V14", "V17", "V12", "V10", "V4", "V11", "Amount_scaled"]

LABEL_MAP = {
    "V14":           "Authentication Pattern",
    "V17":           "Spending Velocity",
    "V12":           "Merchant Category Deviation",
    "V10":           "Geographic Anomaly Score",
    "V4":            "Card Usage Frequency",
    "V11":           "Time-of-Day Behaviour",
    "Amount_scaled": "Transaction Amount (€)",
}

# ── Request / Response schemas ────────────────────────────────────────────────
class TransactionInput(BaseModel):
    V14:           float = Field(0.0,  ge=-20.0, le=10.0,    description="Authentication Pattern")
    V17:           float = Field(0.0,  ge=-20.0, le=10.0,    description="Spending Velocity")
    V12:           float = Field(0.0,  ge=-20.0, le=10.0,    description="Merchant Category Deviation")
    V10:           float = Field(0.0,  ge=-20.0, le=10.0,    description="Geographic Anomaly Score")
    V4:            float = Field(0.0,  ge=-10.0, le=20.0,    description="Card Usage Frequency")
    V11:           float = Field(0.0,  ge=-10.0, le=20.0,    description="Time-of-Day Behaviour")
    Amount_scaled: float = Field(50.0, ge=0.0,   le=25000.0, description="Transaction Amount (€)")

class ShapContribution(BaseModel):
    feature:      str
    label:        str
    input_value:  float
    shap_value:   float
    effect:       str   # "fraud" | "legit"

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    risk_level:        str           # "low" | "moderate" | "high" | "critical"
    shap_base_value:   float
    contributions:     list[ShapContribution]

# ── Helper ────────────────────────────────────────────────────────────────────
def risk_label(prob: float) -> str:
    if prob < 0.2:   return "low"
    if prob < 0.5:   return "moderate"
    if prob < 0.8:   return "high"
    return "critical"

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": "XGBoost Fraud Detector v1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionInput):
    try:
        input_dict = tx.model_dump()
        input_df   = pd.DataFrame([input_dict])[MODEL_FEATURES]

        # Fraud probability
        prob     = float(model.predict_proba(input_df)[0][1])
        is_fraud = prob >= 0.5

        # SHAP values
        shap_explanation = explainer(input_df)
        shap_vals        = shap_explanation.values[0].tolist()
        base_value       = float(shap_explanation.base_values[0])

        contributions = [
            ShapContribution(
                feature=feat,
                label=LABEL_MAP[feat],
                input_value=float(input_dict[feat]),
                shap_value=round(sv, 6),
                effect="fraud" if sv > 0 else "legit",
            )
            for feat, sv in zip(MODEL_FEATURES, shap_vals)
        ]
        # Sort by absolute SHAP impact descending
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return PredictionResponse(
            fraud_probability=round(prob, 6),
            is_fraud=is_fraud,
            risk_level=risk_label(prob),
            shap_base_value=round(base_value, 6),
            contributions=contributions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))