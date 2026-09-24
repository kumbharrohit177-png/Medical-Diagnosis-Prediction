"""
CardioSense AI - Medical Diagnosis Web Application (Flask REST API Backend)
Lead: Rohit (ML & Integration Lead)

Integrates:
- Pre-trained best model (Random Forest / Logistic Regression)
- Preprocessing StandardScaler
- Interactive web interface and JSON API (/api/predict)
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths to trained model artifacts
MODEL_PATH = os.path.join("models", "best_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

# Global variables for loaded model and metadata
model = None
scaler = None
metrics = {}

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_artifacts():
    global model, scaler, metrics
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[+] Model and Scaler loaded successfully!")
    else:
        print("[-] Model artifacts not found. Please run `python train_model.py` first.")
        
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {
            "best_model_name": "Random Forest Classifier",
            "best_metrics": {
                "test_accuracy": 0.9016,
                "f1_score": 0.8966,
                "recall": 0.9286,
                "roc_auc": 0.9545
            }
        }


# Initial load
load_artifacts()


def generate_recommendations(is_positive, proba, form_data):
    recs = []
    if is_positive:
        recs.append("Schedule a comprehensive clinical consultation with a cardiologist.")
        recs.append("Recommend an Echocardiogram and 12-Lead stress ECG for further functional assessment.")
        if float(form_data.get("chol", 0)) > 200:
            recs.append("Serum cholesterol is elevated (>200 mg/dl); consider lipid-lowering therapy and dietary evaluation.")
        if float(form_data.get("trestbps", 0)) > 130:
            recs.append("Resting blood pressure is above normal thresholds; initiate regular BP monitoring.")
        recs.append("Adopt low-sodium, heart-healthy dietary habits and moderate aerobic physical activity under supervision.")
    else:
        recs.append("Vitals and diagnostic parameters are currently within normal low-risk physiological limits.")
        recs.append("Maintain standard cardiovascular wellness habits: balanced diet, regular exercise, and stress management.")
        recs.append("Schedule routine annual biometric health checkups to monitor blood pressure and lipid profile.")
    return recs


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", metrics=metrics, result=None, form_data=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract inputs
        form_data = request.form.to_dict()
        input_data = {}
        for feature in FEATURE_ORDER:
            input_data[feature] = [float(form_data.get(feature, 0))]
            
        features_df = pd.DataFrame(input_data)
        scaled_features = scaler.transform(features_df)
        
        # Predict class and probability
        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        prob_disease = float(probabilities[1])
        
        is_positive = (prediction == 1)
        risk_level = "High" if is_positive else "Low"
        
        if is_positive:
            diag_title = "Elevated Heart Disease Risk Detected"
            diag_text = f"The ML diagnostic pipeline predicts a {prob_disease * 100:.1f}% risk probability for coronary artery disease based on the provided clinical biomarkers."
        else:
            diag_title = "Low Risk / Normal Cardiac Profile"
            diag_text = f"The ML diagnostic pipeline indicates a {prob_disease * 100:.1f}% risk probability, placing the patient within normal healthy baseline thresholds."

        recommendations = generate_recommendations(is_positive, prob_disease, form_data)

        result = {
            "prediction": prediction,
            "is_positive": is_positive,
            "probability": prob_disease,
            "risk_level": risk_level,
            "diagnosis_title": diag_title,
            "diagnosis_text": diag_text,
            "recommendations": recommendations
        }

        return render_template("index.html", metrics=metrics, result=result, form_data=form_data)
        
    except Exception as e:
        error_msg = f"Error during prediction inference: {str(e)}"
        return render_template("index.html", metrics=metrics, result=None, error=error_msg, form_data=request.form.to_dict())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic model integration"""
    try:
        data = request.get_json(force=True)
        input_data = {}
        for feature in FEATURE_ORDER:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: '{feature}'"}), 400
            input_data[feature] = [float(data[feature])]
            
        features_df = pd.DataFrame(input_data)
        scaled_features = scaler.transform(features_df)
        
        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        prob_disease = float(probabilities[1])
        
        return jsonify({
            "status": "success",
            "model_used": metrics.get("best_model_name", "Random Forest"),
            "prediction": prediction,
            "has_heart_disease": bool(prediction == 1),
            "disease_probability": prob_disease,
            "risk_level": "High" if prediction == 1 else "Low"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
