"""
CardioSense AI - Medical Diagnosis Web Application
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
import model_service

app = Flask(__name__)

# Pre-load artifacts via central service
model_service.load_artifacts()


@app.route("/", methods=["GET"])
def index():
    artifacts = model_service.load_artifacts()
    available_models = model_service.get_available_models()
    return render_template(
        "index.html",
        metrics=artifacts["metrics"] or {},
        available_models=available_models,
        result=None,
        form_data=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()
        selected_model = form_data.get("model_name", None)
        if selected_model == "default" or not selected_model:
            selected_model = None
            
        # Use central model_service for inference
        inference_res = model_service.predict_patient(form_data, model_name=selected_model)
        
        is_positive = inference_res["has_heart_disease"]
        prob_disease = inference_res["probability"]
        model_used = inference_res["model_used"]
        risk_level = inference_res["risk_level"]
        
        if is_positive:
            diag_title = "Elevated Heart Disease Risk Detected"
            diag_text = f"The {model_used} model predicts a {prob_disease * 100:.1f}% risk probability for coronary artery disease based on the provided clinical biomarkers."
        else:
            diag_title = "Low Risk / Normal Cardiac Profile"
            diag_text = f"The {model_used} model indicates a {prob_disease * 100:.1f}% risk probability, placing the patient within normal healthy baseline thresholds."

        result = {
            "prediction": inference_res["prediction"],
            "is_positive": is_positive,
            "probability": prob_disease,
            "risk_level": risk_level,
            "model_used": model_used,
            "diagnosis_title": diag_title,
            "diagnosis_text": diag_text,
            "recommendations": inference_res["recommendations"]
        }

        artifacts = model_service.load_artifacts()
        available_models = model_service.get_available_models()
        return render_template(
            "index.html",
            metrics=artifacts["metrics"] or {},
            available_models=available_models,
            result=result,
            form_data=form_data
        )
        
    except Exception as e:
        error_msg = f"Error during prediction inference: {str(e)}"
        artifacts = model_service.load_artifacts()
        available_models = model_service.get_available_models()
        return render_template(
            "index.html",
            metrics=artifacts["metrics"] or {},
            available_models=available_models,
            result=None,
            error=error_msg,
            form_data=request.form.to_dict()
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic model integration"""
    try:
        data = request.get_json(force=True)
        selected_model = data.get("model_name", None)
        res = model_service.predict_patient(data, model_name=selected_model)
        return jsonify({
            "status": "success",
            "model_used": res["model_used"],
            "prediction": res["prediction"],
            "has_heart_disease": res["has_heart_disease"],
            "disease_probability": res["probability"],
            "risk_level": res["risk_level"],
            "recommendations": res["recommendations"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
