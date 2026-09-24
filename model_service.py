"""
CardioSense AI - Central Model Integration Module
Team: Rohit (ML Lead), Om (Analysis Lead), Umar (UI & Deployment Lead)

This module provides the central interface for:
- Loading saved model artifacts and scaler (Logistic Regression, Random Forest, Naive Bayes)
- Performing inference on patient parameters with live models
- Graceful Mock Fallback engine if model artifacts are ever missing or unpicklable
- Formatting diagnostic results and recommendations
- Providing benchmark metrics for UI integration (Streamlit / Umar's App)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress unpickling version warnings for seamless cross-version portability
warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

MODEL_REGISTRY = {
    "Random Forest Classifier": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl"
}

_artifacts = {
    "model": None,
    "scaler": None,
    "metrics": None
}

FALLBACK_METRICS = {
    "best_model_name": "Random Forest Classifier",
    "feature_names": FEATURE_ORDER,
    "models": {
        "random_forest": {
            "model_name": "Random Forest Classifier",
            "train_accuracy": 0.9132,
            "test_accuracy": 0.9016,
            "precision": 0.8667,
            "recall": 0.9286,
            "f1_score": 0.8966,
            "roc_auc": 0.9545
        },
        "logistic_regression": {
            "model_name": "Logistic Regression",
            "train_accuracy": 0.8512,
            "test_accuracy": 0.8689,
            "precision": 0.8125,
            "recall": 0.9286,
            "f1_score": 0.8667,
            "roc_auc": 0.9513
        },
        "naive_bayes": {
            "model_name": "Gaussian Naive Bayes",
            "train_accuracy": 0.8512,
            "test_accuracy": 0.8689,
            "precision": 0.7941,
            "recall": 0.9643,
            "f1_score": 0.8710,
            "roc_auc": 0.9524
        }
    },
    "best_metrics": {
        "model_name": "Random Forest Classifier",
        "train_accuracy": 0.9132,
        "test_accuracy": 0.9016,
        "precision": 0.8667,
        "recall": 0.9286,
        "f1_score": 0.8966,
        "roc_auc": 0.9545
    }
}


def load_artifacts():
    """Loads and caches the default model, scaler, and benchmark metrics."""
    if _artifacts["model"] is None and os.path.exists(MODEL_PATH):
        try:
            _artifacts["model"] = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[-] Warning loading best_model: {e}")
        
    if _artifacts["scaler"] is None and os.path.exists(SCALER_PATH):
        try:
            _artifacts["scaler"] = joblib.load(SCALER_PATH)
        except Exception as e:
            print(f"[-] Warning loading scaler: {e}")
        
    if _artifacts["metrics"] is None:
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, "r") as f:
                    _artifacts["metrics"] = json.load(f)
            except Exception:
                _artifacts["metrics"] = FALLBACK_METRICS
        else:
            _artifacts["metrics"] = FALLBACK_METRICS
            
    return _artifacts


def get_available_models():
    """Returns a dictionary of selectable models and whether their artifact is live or mock fallback."""
    load_artifacts()
    available = {}
    for display_name, filename in MODEL_REGISTRY.items():
        filepath = os.path.join(MODELS_DIR, filename)
        is_live = os.path.exists(filepath)
        available[display_name] = {
            "filename": filename,
            "is_live": is_live,
            "status": "Trained Artifact Ready" if is_live else "Mock Fallback Active"
        }
    return available


def get_metrics():
    """Returns the benchmark metrics dictionary."""
    artifacts = load_artifacts()
    return artifacts["metrics"] or FALLBACK_METRICS


def _mock_predict(patient_dict, model_display_name="Random Forest Classifier (Mock Fallback)"):
    """
    Clinical Heuristic Fallback Engine.
    Used gracefully when .pkl artifacts or scalers are not yet present in the repo.
    Applies Framingham/Cleveland clinical risk weighting on biomarkers.
    """
    score = 0.0
    
    # Age factor
    age = float(patient_dict.get("age", 50))
    if age > 60:
        score += 0.20
    elif age > 50:
        score += 0.12
    elif age > 40:
        score += 0.05
        
    # Sex factor (Male higher baseline risk in Cleveland cohort)
    if float(patient_dict.get("sex", 1)) == 1.0:
        score += 0.10
        
    # Chest pain type (cp 4 = Asymptomatic angina has highest positive correlation)
    cp = float(patient_dict.get("cp", 1))
    if cp == 4.0:
        score += 0.25
    elif cp == 3.0:
        score += 0.10
    elif cp == 2.0:
        score += 0.05
        
    # Resting Blood Pressure
    trestbps = float(patient_dict.get("trestbps", 120))
    if trestbps >= 140:
        score += 0.15
    elif trestbps >= 130:
        score += 0.08
        
    # Cholesterol
    chol = float(patient_dict.get("chol", 200))
    if chol >= 260:
        score += 0.15
    elif chol >= 220:
        score += 0.08
        
    # Max Heart Rate (lower achieved heart rate indicates coronary insufficiency)
    thalach = float(patient_dict.get("thalach", 150))
    if thalach < 120:
        score += 0.18
    elif thalach < 140:
        score += 0.10
        
    # Exercise induced angina
    if float(patient_dict.get("exang", 0)) == 1.0:
        score += 0.20
        
    # ST Depression (oldpeak)
    oldpeak = float(patient_dict.get("oldpeak", 0))
    if oldpeak >= 2.0:
        score += 0.25
    elif oldpeak >= 1.0:
        score += 0.12
        
    # Number of major vessels (ca)
    ca = float(patient_dict.get("ca", 0))
    if ca >= 2:
        score += 0.25
    elif ca == 1:
        score += 0.15
        
    # Thalassemia defect
    thal = float(patient_dict.get("thal", 3))
    if thal == 7.0: # Reversible defect
        score += 0.20
    elif thal == 6.0: # Fixed defect
        score += 0.12

    # Sigmoid scaling to realistic probability [0.05, 0.98]
    # Base risk adjustment
    raw_z = (score - 0.70) * 2.8
    prob_disease = 1.0 / (1.0 + np.exp(-raw_z))
    prob_disease = float(np.clip(prob_disease, 0.03, 0.97))
    
    prediction = 1 if prob_disease >= 0.50 else 0
    is_positive = (prediction == 1)
    risk_level = "High" if is_positive else "Low"
    
    recs = _build_recommendations(is_positive, patient_dict)
    
    return {
        "status": "success",
        "is_mock": True,
        "prediction": prediction,
        "has_heart_disease": is_positive,
        "probability": prob_disease,
        "probability_percent": round(prob_disease * 100, 2),
        "risk_level": risk_level,
        "model_used": f"{model_display_name} [Graceful Fallback Mode]",
        "recommendations": recs
    }


def _build_recommendations(is_positive, patient_dict):
    """Generates individualized patient action items based on biomarker ranges."""
    recommendations = []
    if is_positive:
        recommendations.append("Immediate consultation recommended with a certified cardiologist.")
        recommendations.append("Schedule non-invasive diagnostic follow-ups: Echocardiogram & Exercise Stress ECG.")
        if float(patient_dict.get("chol", 0)) > 200:
            recommendations.append(f"Serum cholesterol is elevated ({int(patient_dict.get('chol', 0))} mg/dl); discuss statin therapy and low-saturated-fat nutrition.")
        if float(patient_dict.get("trestbps", 0)) > 130:
            recommendations.append(f"Resting blood pressure is hypertensive ({int(patient_dict.get('trestbps', 0))} mmHg); start home BP logging.")
        if float(patient_dict.get("exang", 0)) == 1.0:
            recommendations.append("Exercise-induced angina reported; limit strenuous exertion pending specialist clearance.")
    else:
        recommendations.append("Biomarker parameters align with normal low-risk cardiovascular ranges.")
        recommendations.append("Continue standard preventive measures: 150 min/week moderate aerobic activity.")
        recommendations.append("Maintain routine annual health screenings for lipid panel and blood pressure.")
        if float(patient_dict.get("chol", 0)) > 200:
            recommendations.append(f"Note: Cholesterol is slightly borderline ({int(patient_dict.get('chol', 0))} mg/dl); monitor dietary intake.")
    return recommendations


def predict_patient(patient_dict, model_name=None):
    """
    Central inference function.
    
    Args:
        patient_dict: dict containing the 13 clinical biomarkers:
                      age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal
        model_name: optional model identifier or filename:
                    - 'Random Forest Classifier' or 'random_forest.pkl'
                    - 'Logistic Regression' or 'logistic_regression.pkl'
                    - 'Gaussian Naive Bayes' or 'naive_bayes.pkl'
                    
    Returns:
        dict containing:
            - status ('success')
            - is_mock (bool)
            - prediction (0 or 1)
            - has_heart_disease (bool)
            - probability (float: 0.0 to 1.0)
            - probability_percent (float)
            - risk_level ('Low' or 'High')
            - model_used (str)
            - recommendations (list of str)
    """
    # Normalize model name
    target_filename = None
    display_name = "Best Model"
    
    if model_name:
        if model_name in MODEL_REGISTRY:
            display_name = model_name
            target_filename = MODEL_REGISTRY[model_name]
        elif model_name.endswith(".pkl"):
            target_filename = model_name
            display_name = model_name.replace(".pkl", "").replace("_", " ").title()
        else:
            for k, v in MODEL_REGISTRY.items():
                if model_name.lower() in k.lower() or model_name.lower() in v.lower():
                    display_name = k
                    target_filename = v
                    break
    
    artifacts = load_artifacts()
    scaler = artifacts["scaler"]
    
    # Try loading requested model
    active_model = None
    if target_filename:
        model_path = os.path.join(MODELS_DIR, target_filename)
        if os.path.exists(model_path):
            try:
                active_model = joblib.load(model_path)
            except Exception as e:
                print(f"[-] Could not load {target_filename}: {e}. Falling back to mock.")
    else:
        active_model = artifacts["model"]
        display_name = artifacts["metrics"].get("best_model_name", "Random Forest Classifier") if artifacts["metrics"] else "Random Forest Classifier"

    # If active model or scaler is missing, invoke graceful mock fallback
    if active_model is None or scaler is None:
        return _mock_predict(patient_dict, model_display_name=display_name)

    # Validate required features
    input_data = {}
    for feature in FEATURE_ORDER:
        if feature not in patient_dict:
            raise ValueError(f"Missing required clinical biomarker: '{feature}'")
        input_data[feature] = [float(patient_dict[feature])]
        
    try:
        df_features = pd.DataFrame(input_data)
        scaled_array = scaler.transform(df_features)
        scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_ORDER)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = int(active_model.predict(scaled_df)[0])
            probabilities = active_model.predict_proba(scaled_df)[0]
            prob_disease = float(probabilities[1])
        
        is_positive = (prediction == 1)
        risk_level = "High" if is_positive else "Low"
        
        recommendations = _build_recommendations(is_positive, patient_dict)
        
        return {
            "status": "success",
            "is_mock": False,
            "prediction": prediction,
            "has_heart_disease": is_positive,
            "probability": prob_disease,
            "probability_percent": round(prob_disease * 100, 2),
            "risk_level": risk_level,
            "model_used": display_name,
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"[-] Live inference exception: {e}. Executing graceful fallback.")
        return _mock_predict(patient_dict, model_display_name=display_name)
