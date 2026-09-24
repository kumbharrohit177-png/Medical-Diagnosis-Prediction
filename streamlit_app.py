"""
CardioSense AI - Multi-Page Streamlit Web Application
Lead: Umar (Software / UI + Deployment Lead)
ML Integration: Rohit (ML Lead) & Om (EDA Lead)

Application Structure:
- Page 1: Home (Project introduction, CTA [ Start Prediction ])
- Page 2: Patient Details (13 clinical biomarker input widgets, [ Predict ] button)
- Page 3: Result (Selected Model, Risk prediction, Probability %, Recommendations)
- Page 4: Model Comparison (LR, RF, Naive Bayes comparison, Accuracy bar chart, Confusion Matrix, Best-performing model)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import model_service

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioSense AI | Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin: 1rem 0 2rem 0;
    }
    .risk-badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 2px solid #EF4444;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .risk-badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22C55E;
        border: 2px solid #22C55E;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Page 1 — Home"

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 230,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 3
    }

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "random_forest.pkl"

def set_page(page_name):
    st.session_state.current_page = page_name


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
st.sidebar.markdown("## **CardioSense AI**")
st.sidebar.caption("Heart Disease Risk Prediction System")

pages = [
    "Page 1 — Home",
    "Page 2 — Patient Details",
    "Page 3 — Result",
    "Page 4 — Model Comparison"
]

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    pages,
    index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
)

# Sync sidebar click with session state
if selected_nav != st.session_state.current_page:
    st.session_state.current_page = selected_nav
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Project Team")
st.sidebar.markdown("""
- **Rohit** — *ML & Integration Lead*
- **Om** — *EDA & Naive Bayes Lead*
- **Umar** — *UI & Deployment Lead*
""")
st.sidebar.caption("Dataset: UCI Cleveland (303 records)")


# =============================================================================
# PAGE 1 — HOME
# =============================================================================
if st.session_state.current_page == "Page 1 — Home":
    st.markdown('<div class="main-title">🫀 Medical Diagnosis Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predict Heart Disease Using Machine Learning</div>', unsafe_allow_html=True)

    # Hero Card
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #00D2FF; margin-bottom: 0.5rem;">Clinical Diagnostic Intelligence System</h2>
        <p style="font-size: 1.15rem; color: #CBD5E1; max-width: 750px; margin: 0 auto 1.5rem auto;">
            An end-to-end cardiovascular risk assessment platform powered by machine learning algorithms 
            trained on 13 key physiological biomarkers from the UCI Cleveland Heart Disease Dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Big CTA Button
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
            set_page("Page 2 — Patient Details")
            st.rerun()

    st.markdown("---")

    # Key Highlights
    st.markdown("### 🌟 System Highlights")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-box"><h4>🎯 Test Accuracy</h4><h2 style="color:#00D2FF;">90.16%</h2><small>Random Forest Ensemble</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-box"><h4>🔬 Sensitivity (Recall)</h4><h2 style="color:#22C55E;">92.86%</h2><small>Minimizes False Negatives</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-box"><h4>📈 ROC-AUC Score</h4><h2 style="color:#A855F7;">0.9545</h2><small>High Discriminative Power</small></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-box"><h4>📋 Biomarkers</h4><h2 style="color:#F59E0B;">13 Features</h2><small>UCI Clinical Dataset</small></div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — PATIENT DETAILS
# =============================================================================
elif st.session_state.current_page == "Page 2 — Patient Details":
    st.markdown('<div class="main-title">📋 Page 2 — Patient Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter the patient clinical biomarkers below to perform diagnosis</div>', unsafe_allow_html=True)

    # Preset quick-load buttons
    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    with col_p1:
        if st.button("🟢 Load Normal Profile"):
            st.session_state.patient_data = {
                "age": 45, "sex": 0, "cp": 2, "trestbps": 115, "chol": 190,
                "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0,
                "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
            }
            st.rerun()
    with col_p2:
        if st.button("🔴 Load High-Risk Profile"):
            st.session_state.patient_data = {
                "age": 65, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
                "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1,
                "oldpeak": 2.8, "slope": 2, "ca": 3, "thal": 7
            }
            st.rerun()

    st.markdown("---")

    # Patient Details Form
    with st.form("patient_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 General Biomarkers")
            age = st.number_input("Age (in years)", min_value=18, max_value=100, value=int(st.session_state.patient_data["age"]))
            sex = st.selectbox("Sex", options=[(1, "1 = Male"), (0, "0 = Female")], format_func=lambda x: x[1], index=0 if st.session_state.patient_data["sex"]==1 else 1)[0]
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[
                    (1, "1: Typical Angina"),
                    (2, "2: Atypical Angina"),
                    (3, "3: Non-Anginal Pain"),
                    (4, "4: Asymptomatic (High Risk)")
                ],
                format_func=lambda x: x[1],
                index=int(st.session_state.patient_data["cp"]) - 1
            )[0]
            trestbps = st.slider("Resting Blood Pressure (trestbps in mm Hg)", 80, 220, int(st.session_state.patient_data["trestbps"]))
            chol = st.slider("Serum Cholesterol (chol in mg/dl)", 100, 600, int(st.session_state.patient_data["chol"]))

        with col2:
            st.markdown("##### 🩸 Blood & Cardiac Rhythm")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[(0, "0 = Normal (<=120)"), (1, "1 = Elevated (>120)")], format_func=lambda x: x[1], index=int(st.session_state.patient_data["fbs"]))[0]
            restecg = st.selectbox(
                "Resting ECG (restecg)",
                options=[
                    (0, "0: Normal"),
                    (1, "1: ST-T Wave Abnormality"),
                    (2, "2: Left Ventricular Hypertrophy")
                ],
                format_func=lambda x: x[1],
                index=int(st.session_state.patient_data["restecg"])
            )[0]
            thalach = st.slider("Maximum Heart Rate Achieved (thalach in bpm)", 60, 220, int(st.session_state.patient_data["thalach"]))
            exang = st.selectbox("Exercise Induced Angina (exang)", options=[(0, "0 = No"), (1, "1 = Yes")], format_func=lambda x: x[1], index=int(st.session_state.patient_data["exang"]))[0]

        with col3:
            st.markdown("##### 🔬 Stress Test & Fluoroscopy")
            oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.5, float(st.session_state.patient_data["oldpeak"]), step=0.1)
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment (slope)",
                options=[(1, "1: Upsloping"), (2, "2: Flat"), (3, "3: Downsloping")],
                format_func=lambda x: x[1],
                index=int(st.session_state.patient_data["slope"]) - 1
            )[0]
            ca = st.selectbox("Number of Major Vessels via Fluoroscopy (ca)", options=[0, 1, 2, 3], index=int(st.session_state.patient_data["ca"]))
            thal = st.selectbox(
                "Thallium Scintigraphy (thal)",
                options=[(3, "3: Normal Blood Flow"), (6, "6: Fixed Defect"), (7, "7: Reversible Defect")],
                format_func=lambda x: x[1],
                index=0 if st.session_state.patient_data["thal"]==3 else (1 if st.session_state.patient_data["thal"]==6 else 2)
            )[0]

        st.markdown("---")
        
        # Model Selection (Dynamically discovers Naive Bayes if Om adds it)
        available_model_options = ["random_forest.pkl", "logistic_regression.pkl"]
        nb_path = os.path.join(os.path.dirname(__file__), "models", "naive_bayes.pkl")
        if os.path.exists(nb_path) and "naive_bayes.pkl" not in available_model_options:
            available_model_options.append("naive_bayes.pkl")

        def format_model_label(x):
            if "random_forest" in x:
                return "🥇 Random Forest Classifier (Champion - 90.16% Acc)"
            elif "logistic" in x:
                return "🥈 Logistic Regression (Baseline - 86.89% Acc)"
            elif "naive_bayes" in x:
                return "🥉 Gaussian Naive Bayes (Om's Model - 85.25% Acc)"
            return x

        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            model_choice = st.selectbox(
                "Select Machine Learning Model for Prediction:",
                options=available_model_options,
                format_func=format_model_label
            )
        with col_m2:
            st.write("")
            st.write("")
            predict_btn = st.form_submit_button("🔍 [ Predict ]", type="primary", use_container_width=True)

    if predict_btn:
        patient_dict = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        st.session_state.patient_data = patient_dict
        st.session_state.selected_model_name = model_choice
        
        # Perform Inference
        result = model_service.predict_patient(patient_dict, model_name=model_choice)
        st.session_state.prediction_result = result
        
        # Transition directly to Page 3 — Result
        set_page("Page 3 — Result")
        st.rerun()


# =============================================================================
# PAGE 3 — RESULT
# =============================================================================
elif st.session_state.current_page == "Page 3 — Result":
    st.markdown('<div class="main-title">🩺 Page 3 — Prediction Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Cardiovascular Diagnostic Intelligence Report</div>', unsafe_allow_html=True)

    if st.session_state.prediction_result is None:
        st.warning("No prediction performed yet. Please enter patient details on Page 2 first.")
        if st.button("⬅️ Go to Patient Details"):
            set_page("Page 2 — Patient Details")
            st.rerun()
    else:
        res = st.session_state.prediction_result
        
        col_res_left, col_res_right = st.columns([1, 1])
        
        with col_res_left:
            st.markdown("### 📊 Diagnostic Evaluation")
            st.markdown(f"**Selected Model:** `{res['model_used']}`")
            
            if res["has_heart_disease"]:
                st.markdown('<div class="risk-badge-high">⚠️ Heart Disease Detected</div>', unsafe_allow_html=True)
                st.markdown(f"### **Probability:** <span style='color:#EF4444; font-size:2rem;'>{res['probability_percent']}%</span>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-badge-low">✅ No Heart Disease Detected (Healthy)</div>', unsafe_allow_html=True)
                st.markdown(f"### **Probability:** <span style='color:#22C55E; font-size:2rem;'>{res['probability_percent']}%</span>", unsafe_allow_html=True)
                
            st.progress(res["probability"])
            
        with col_res_right:
            st.markdown("### 💡 Clinical Action Plan & Recommendations")
            for rec in res["recommendations"]:
                st.info(f"• {rec}")
                
        st.markdown("---")
        
        # Navigation buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Test Another Patient", use_container_width=True):
                set_page("Page 2 — Patient Details")
                st.rerun()
        with col_btn2:
            if st.button("📊 View Model Comparison Hub ➡️", type="primary", use_container_width=True):
                set_page("Page 4 — Model Comparison")
                st.rerun()


# =============================================================================
# PAGE 4 — MODEL COMPARISON
# =============================================================================
elif st.session_state.current_page == "Page 4 — Model Comparison":
    st.markdown('<div class="main-title">🏆 Page 4 — Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Benchmarking Logistic Regression, Random Forest, and Gaussian Naive Bayes</div>', unsafe_allow_html=True)

    # 1. Model Accuracy Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <h4>🥈 Logistic Regression</h4>
            <h2 style="color:#00D2FF;">86.89%</h2>
            <p>Sensitivity: <b>92.86%</b> | ROC-AUC: <b>0.9513</b></p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-box" style="border: 2px solid #22C55E;">
            <h4>🥇 Random Forest (Champion)</h4>
            <h2 style="color:#22C55E;">90.16%</h2>
            <p>Sensitivity: <b>92.86%</b> | ROC-AUC: <b>0.9545</b></p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-box">
            <h4>🥉 Gaussian Naive Bayes</h4>
            <h2 style="color:#A855F7;">85.25%</h2>
            <p>Sensitivity: <b>89.29%</b> | ROC-AUC: <b>0.9123</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Accuracy Comparison Bar Chart
    st.markdown("### 📊 Accuracy & Performance Comparison Chart")
    model_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest (Champion)", "Gaussian Naive Bayes"],
        "Test Accuracy (%)": [86.89, 90.16, 85.25],
        "Sensitivity / Recall (%)": [92.86, 92.86, 89.29],
        "Precision (%)": [81.25, 86.67, 80.65],
        "ROC-AUC": [0.9513, 0.9545, 0.9123]
    })
    
    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        st.bar_chart(model_df.set_index("Model")[["Test Accuracy (%)", "Sensitivity / Recall (%)"]])
    with col_table:
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3. Confusion Matrices & Discriminative Visuals
    st.markdown("### 🔲 Confusion Matrices & Feature Explainability")
    col_m1, col_m2 = st.columns(2)
    
    cm_path = os.path.join(os.path.dirname(__file__), "assets", "confusion_matrix.png")
    feat_path = os.path.join(os.path.dirname(__file__), "assets", "feature_importance.png")
    
    with col_m1:
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True, caption="Confusion Matrices Comparison")
        else:
            st.info("Run `python train_model.py` to generate confusion matrix plots.")
            
    with col_m2:
        if os.path.exists(feat_path):
            st.image(feat_path, use_container_width=True, caption="Key Biomarker Feature Importance (Gini)")
        else:
            st.info("Run `python train_model.py` to generate feature importance plots.")

    st.success("🏆 **Best-Performing Model:** **Random Forest Classifier** achieved the highest Test Accuracy (90.16%) and maximum Clinical Sensitivity (92.86%), making it the optimal champion model for clinical deployment.")
