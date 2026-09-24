"""
CardioSense AI - Flask REST API Server
Provides programmatic RESTful endpoints for hospital systems, mobile apps, and third-party clients.
"""

from flask import Flask, request, jsonify
import model_service

app = Flask(__name__)

# Preload model artifacts
model_service.load_artifacts()


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint to verify API status."""
    return jsonify({
        "status": "online",
        "service": "CardioSense AI Diagnostic API",
        "version": "1.0.0",
        "available_models": list(model_service.get_available_models().keys())
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for heart disease risk prediction.
    Accepts JSON payload with 13 clinical biomarkers.
    Returns JSON response with prediction, probability, risk level, and recommendations.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided."}), 400

        model_name = data.get("model_name", "Random Forest Classifier")
        patient_data = data.get("patient", data)  # Supports direct dictionary or nested {"patient": {...}}

        result = model_service.predict_patient(patient_data, model_name=model_name)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns benchmark performance metrics for all trained models."""
    return jsonify(model_service.get_metrics()), 200


if __name__ == "__main__":
    print("🚀 Starting CardioSense AI REST API Server on http://127.0.0.1:5000")
    print("📌 POST Endpoint: http://127.0.0.1:5000/api/predict")
    app.run(host="127.0.0.1", port=5000, debug=True)
