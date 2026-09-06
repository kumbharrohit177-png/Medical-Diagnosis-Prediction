"""
CardioSense AI - Central Model Integration Module
Lead: Rohit (ML & Integration Lead)

This module provides the central interface for:
- Loading saved model artifacts and scaler
- Performing inference on patient parameters
- Formatting diagnostic results and recommendations
- Providing benchmark metrics for UI integration (Streamlit / Umar's App)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

_artifacts = {
    "model": None,
    "scaler": None,
    "metrics": None
}


def load_artifacts():
    """Loads and caches the model, scaler, and benchmark metrics."""
    if _artifacts["model"] is None and os.path.exists(MODEL_PATH):
        _artifacts["model"] = joblib.load(MODEL_PATH)
        
    if _artifacts["scaler"] is None and os.path.exists(SCALER_PATH):
        _artifacts["scaler"] = joblib.load(SCALER_PATH)
        
    if _artifacts["metrics"] is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            _artifacts["metrics"] = json.load(f)
            
    return _artifacts


def get_available_models():
    """Returns a list of available trained model files in models/."""
    available = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pkl") and f != "scaler.pkl":
                available.append(f)
    return available


def get_metrics():
    """Returns the benchmark metrics dictionary."""
    artifacts = load_artifacts()
    return artifacts["metrics"] or {}


def predict_patient(patient_dict, model_name=None):
    """
    Central inference function.
    
    Args:
        patient_dict: dict containing the 13 clinical biomarkers:
                      age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal
        model_name: optional model filename to override default best model
                    (e.g. 'logistic_regression.pkl', 'random_forest.pkl')
                    
    Returns:
        dict containing:
            - prediction (0 or 1)
            - has_heart_disease (bool)
            - probability (float: 0.0 to 1.0)
            - risk_level ('Low' or 'High')
            - model_used (str)
            - recommendations (list of str)
    """
    artifacts = load_artifacts()
    scaler = artifacts["scaler"]
    
    if model_name:
        custom_model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(custom_model_path):
            active_model = joblib.load(custom_model_path)
            model_display_name = model_name.replace(".pkl", "").replace("_", " ").title()
        else:
            active_model = artifacts["model"]
            model_display_name = artifacts["metrics"].get("best_model_name", "Best Model") if artifacts["metrics"] else "Best Model"
    else:
        active_model = artifacts["model"]
        model_display_name = artifacts["metrics"].get("best_model_name", "Random Forest Classifier") if artifacts["metrics"] else "Random Forest Classifier"

    if active_model is None or scaler is None:
        raise RuntimeError("Model or Scaler not found. Run `python train_model.py` first.")

    input_data = {}
    for feature in FEATURE_ORDER:
        if feature not in patient_dict:
            raise ValueError(f"Missing required clinical biomarker: '{feature}'")
        input_data[feature] = [float(patient_dict[feature])]
        
    df_features = pd.DataFrame(input_data)
    scaled_array = scaler.transform(df_features)
    scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_ORDER)
    
    prediction = int(active_model.predict(scaled_df)[0])
    probabilities = active_model.predict_proba(scaled_df)[0]
    prob_disease = float(probabilities[1])
    
    is_positive = (prediction == 1)
    risk_level = "High" if is_positive else "Low"
    
    recommendations = []
    if is_positive:
        recommendations.append("Immediate consultation recommended with a cardiologist for diagnostic evaluation.")
        recommendations.append("Schedule functional cardiac assessments (Stress ECG / Echocardiogram).")
        if float(patient_dict.get("chol", 0)) > 200:
            recommendations.append("Serum cholesterol elevated (>200 mg/dl); discuss lipid-lowering lifestyle or medication.")
        if float(patient_dict.get("trestbps", 0)) > 130:
            recommendations.append("Resting blood pressure elevated; monitor daily and reduce dietary sodium.")
    else:
        recommendations.append("Parameters are within healthy baseline cardiac ranges.")
        recommendations.append("Maintain regular cardiovascular exercise and annual biometric checkups.")
        
    return {
        "status": "success",
        "prediction": prediction,
        "has_heart_disease": is_positive,
        "probability": prob_disease,
        "probability_percent": round(prob_disease * 100, 2),
        "risk_level": risk_level,
        "model_used": model_display_name,
        "recommendations": recommendations
    }
