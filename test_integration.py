"""
Medical Diagnosis Prediction - Integration Test Suite
Lead: Rohit (ML & Integration Lead)

Verifies:
1. Model artifacts loading from models/
2. Feature scaling pipeline
3. Central inference via model_service.py
4. Individual model queries (Logistic Regression vs Random Forest)
"""

import unittest
import numpy as np
import model_service


class TestCentralModelIntegration(unittest.TestCase):

    def setUp(self):
        self.artifacts = model_service.load_artifacts()

    def test_artifacts_loaded(self):
        """Verify model, scaler, and metrics are loaded."""
        self.assertIsNotNone(self.artifacts["model"], "Best model must be loaded")
        self.assertIsNotNone(self.artifacts["scaler"], "Scaler must be loaded")
        self.assertIsNotNone(self.artifacts["metrics"], "Metrics must be loaded")

    def test_healthy_patient_prediction(self):
        """Verify inference for a low-risk healthy profile."""
        healthy_patient = {
            "age": 45, "sex": 0, "cp": 2, "trestbps": 115, "chol": 190,
            "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0,
            "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
        }
        res = model_service.predict_patient(healthy_patient)
        self.assertEqual(res["prediction"], 0)
        self.assertEqual(res["risk_level"], "Low")
        self.assertLess(res["probability"], 0.50)
        print(f"\n[PASS] Healthy Profile -> Risk: {res['risk_level']}, Prob: {res['probability_percent']}%")

    def test_high_risk_patient_prediction(self):
        """Verify inference for an elevated cardiac risk profile."""
        risk_patient = {
            "age": 65, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
            "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1,
            "oldpeak": 2.8, "slope": 2, "ca": 3, "thal": 7
        }
        res = model_service.predict_patient(risk_patient)
        self.assertEqual(res["prediction"], 1)
        self.assertEqual(res["risk_level"], "High")
        self.assertGreater(res["probability"], 0.50)
        print(f"[PASS] High Risk Profile -> Risk: {res['risk_level']}, Prob: {res['probability_percent']}%")

    def test_individual_models(self):
        """Verify Umar's UI can query individual models (LR, RF, and NB)."""
        patient = {
            "age": 60, "sex": 1, "cp": 4, "trestbps": 150, "chol": 270,
            "fbs": 0, "restecg": 2, "thalach": 120, "exang": 1,
            "oldpeak": 2.2, "slope": 2, "ca": 2, "thal": 7
        }
        res_lr = model_service.predict_patient(patient, model_name="logistic_regression.pkl")
        res_rf = model_service.predict_patient(patient, model_name="random_forest.pkl")
        res_nb = model_service.predict_patient(patient, model_name="naive_bayes.pkl")
        
        self.assertIn("Logistic Regression", res_lr["model_used"])
        self.assertIn("Random Forest", res_rf["model_used"])
        self.assertIn("Naive Bayes", res_nb["model_used"])
        self.assertFalse(res_lr["is_mock"])
        self.assertFalse(res_rf["is_mock"])
        self.assertFalse(res_nb["is_mock"])
        print(f"[PASS] Model Queries: LR={res_lr['probability_percent']}%, RF={res_rf['probability_percent']}%, NB={res_nb['probability_percent']}%")

    def test_graceful_mock_fallback(self):
        """Verify graceful fallback mock when querying a non-existent model."""
        patient = {
            "age": 65, "sex": 1, "cp": 4, "trestbps": 160, "chol": 290,
            "fbs": 1, "restecg": 2, "thalach": 110, "exang": 1,
            "oldpeak": 2.5, "slope": 2, "ca": 3, "thal": 7
        }
        res_mock = model_service.predict_patient(patient, model_name="non_existent_model.pkl")
        self.assertTrue(res_mock["is_mock"])
        self.assertEqual(res_mock["status"], "success")
        self.assertGreater(res_mock["probability_percent"], 50.0)
        print(f"[PASS] Mock Fallback -> Prob: {res_mock['probability_percent']}%, Risk: {res_mock['risk_level']}")


if __name__ == "__main__":
    unittest.main()
