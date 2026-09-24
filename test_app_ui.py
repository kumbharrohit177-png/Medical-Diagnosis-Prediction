"""
CardioSense AI - Streamlit UI Automated Test Suite
Author: Umar (Software/UI & Deployment Lead)

Verifies the 4-page Streamlit application using streamlit.testing.v1.AppTest:
1. Page 1 (Home): Title, metrics cards, Start Assessment button
2. Page 2 (Patient Details): 2-column form, presets, model selector, diagnostic submission
3. Page 3 (Diagnostic Results): Alert banner, risk probability, clinical recommendations
4. Page 4 (Model Comparison): Comparison table, Plotly chart, visual assets
"""

import unittest
from streamlit.testing.v1 import AppTest


class TestStreamlitAppWorkflow(unittest.TestCase):

    def test_page_1_home(self):
        """Verify Page 1 (Home) renders title and hero cards."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        self.assertFalse(at.exception)
        
        # Verify title and brand content
        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("CardioSense AI", markdown_text)
        self.assertIn("UCI Cleveland Clinical Records", markdown_text)
        # Verify clicking Start Assessment button transitions to Page 2
        start_btn = [b for b in at.button if "Start Patient Assessment" in b.label][0]
        start_btn.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.current_page, "Page 2: Patient Details")
        print("[PASS] Page 1 (Home) loaded successfully and navigated to Page 2 via CTA button.")

    def test_page_2_patient_form_and_prediction(self):
        """Verify Page 2 intake form and submission to Page 3."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        
        # Navigate to Page 2
        at.sidebar.radio[0].set_value("Page 2: Patient Details").run()
        self.assertFalse(at.exception)
        
        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("Patient Clinical Intake", markdown_text)
        self.assertIn("Predictive Algorithm Selector", markdown_text)
        print("[PASS] Page 2 (Patient Details) loaded successfully.")

        # Submit form to run diagnostic prediction
        submit_btn = [b for b in at.button if "Run Diagnostic Prediction" in b.label][0]
        submit_btn.click().run()
        self.assertFalse(at.exception)
        
        # Should now be on Page 3: Diagnostic Results
        self.assertEqual(at.session_state.current_page, "Page 3: Diagnostic Results")
        res_markdown = " ".join([m.value for m in at.markdown])
        self.assertIn("Diagnostic Results", res_markdown)
        self.assertIsNotNone(at.session_state.prediction_result)
        print(f"[PASS] Page 3 (Results) verified with prediction: {at.session_state.prediction_result['risk_level']} Risk ({at.session_state.prediction_result['probability_percent']}%)")

    def test_page_4_model_comparison(self):
        """Verify Page 4 (Model Comparison) table, charts, and metrics."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        
        # Navigate to Page 4
        at.sidebar.radio[0].set_value("Page 4: Model Comparison").run()
        self.assertFalse(at.exception)
        
        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("Comparative Algorithm Performance", markdown_text)
        self.assertIn("Precision vs. Recall", markdown_text)
        self.assertGreater(len(at.dataframe), 0)
        print("[PASS] Page 4 (Model Comparison) verified with benchmark tables.")


if __name__ == "__main__":
    unittest.main()
