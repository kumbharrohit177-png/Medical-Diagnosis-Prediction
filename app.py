"""
CardioSense AI - Medical Diagnosis Prediction System
Author: Umar (Software/UI & Deployment Lead)
Teammates: Rohit (ML Lead), Om (Data Analysis Lead)
Dataset: UCI Cleveland Heart Disease Dataset (303 patient records, 13 biomarkers)
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import model_service

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioSense AI | Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Medical Dark & Clean CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    /* Gradient Brand Accent Header */
    .brand-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Metric / Feature Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 0.35rem;
    }

    /* Status Badges */
    .badge-live {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-fallback {
        display: inline-flex;
        align-items: center;
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Alert Banners */
    .alert-high-risk {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.35) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #fef2f2;
    }

    .alert-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(4, 120, 87, 0.35) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #f0fdf4;
    }

    /* Team Pills */
    .team-badge {
        background: rgba(51, 65, 85, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
    }

    /* Stepper / Nav indicators */
    .nav-active {
        font-weight: 700;
        color: #38bdf8 !important;
    }

    /* Streamlit button enhancements */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s ease-in-out;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Page 1: Home"

# Preset patient profiles for rapid testing
PRESETS = {
    "Healthy": {
        "age": 45, "sex": 0, "cp": 2, "trestbps": 115, "chol": 190,
        "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    },
    "Elevated Risk": {
        "age": 65, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
        "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1,
        "oldpeak": 2.8, "slope": 2, "ca": 3, "thal": 7
    }
}

if "patient_input" not in st.session_state:
    st.session_state.patient_input = PRESETS["Healthy"].copy()

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "Random Forest Classifier"

# Helper navigation function
def navigate_to(page_name):
    st.session_state.current_page = page_name
    st.rerun()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & REPO HEALTH MONITOR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫀 CardioSense AI")
    st.caption("Clinical Decision Support Platform")
    
    pages = [
        "Page 1: Home",
        "Page 2: Patient Details",
        "Page 3: Diagnostic Results",
        "Page 4: Model Comparison"
    ]
    
    selected_page = st.radio(
        "Workflow Navigation",
        options=pages,
        index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
        label_visibility="collapsed"
    )
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
        
    st.markdown("---")
    
    # Model Artifact Status Monitor
    st.markdown("#### ⚙️ Artifact Registry")
    avail_models = model_service.get_available_models()
    for name, info in avail_models.items():
        if info["is_live"]:
            st.markdown(f"🟢 **{name}**  \n`<small style='color:#10b981;'>Live Artifact: {info['filename']}</small>`", unsafe_allow_html=True)
        else:
            st.markdown(f"🟡 **{name}**  \n`<small style='color:#f59e0b;'>Mock Fallback Active</small>`", unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("#### 👥 Team Roles")
    st.markdown(
        """
        <div class="team-badge">
            <strong>Rohit</strong><br>
            <small style="color:#94a3b8;">ML Pipeline & Backend Lead</small>
        </div>
        <div class="team-badge">
            <strong>Om</strong><br>
            <small style="color:#94a3b8;">EDA & Naive Bayes Lead</small>
        </div>
        <div class="team-badge">
            <strong>Umar</strong><br>
            <small style="color:#38bdf8;">UI, Integration & Deployment Lead</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.caption("Cleveland Heart Disease Study | 303 Patients")

# -----------------------------------------------------------------------------
# PAGE 1: HOME & PROJECT OVERVIEW
# -----------------------------------------------------------------------------
if st.session_state.current_page == "Page 1: Home":
    st.markdown('<div class="brand-title">CardioSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Automated Machine Learning Diagnostic System for Early Coronary Artery Disease Detection</div>', unsafe_allow_html=True)

    # Hero stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#38bdf8;">303</div>
                <div class="metric-label">Patient Cohort</div>
                <small style="color:#64748b;">UCI Cleveland Clinical Records</small>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#818cf8;">13</div>
                <div class="metric-label">Clinical Biomarkers</div>
                <small style="color:#64748b;">Hemodynamic, ECG, Fluoroscopy</small>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#c084fc;">3</div>
                <div class="metric-label">ML Classifiers</div>
                <small style="color:#64748b;">RF, Logistic Reg, Naive Bayes</small>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#10b981;">90.16%</div>
                <div class="metric-label">Top Benchmark Accuracy</div>
                <small style="color:#64748b;">Random Forest Ensemble</small>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Narrative & Clinical Motivation
    col_left, col_right = st.columns([1.6, 1.0])
    
    with col_left:
        st.markdown("### 🩺 Clinical Overview & Problem Statement")
        st.write(
            """
            Cardiovascular diseases (CVDs) remain the leading cause of mortality globally, 
            accounting for an estimated 17.9 million deaths annually. Traditional manual risk 
            assessments rely heavily on subjective symptom interpretation and single-variable cutoffs, 
            often missing early-stage ischemia and coronary blockages.

            **CardioSense AI** bridges this clinical gap by integrating multi-modal patient biomarkers—ranging from 
            exercise-induced ST depression and fluoroscopic vessel counts to resting hemodynamics—into trained 
            probabilistic classification models.
            """
        )

        st.markdown("#### 🔬 Core Diagnostic Architecture")
        st.markdown(
            """
            1. **Intake & Preprocessing:** 13 verified clinical biomarkers are transformed and standardized via a calibrated `StandardScaler`.
            2. **Multi-Model Inference:** Predictions are rendered via Random Forest, Logistic Regression, or Gaussian Naive Bayes with calibrated posterior probabilities.
            3. **Decision Support & Action Items:** Patients receive an immediate risk classification (Low vs High) coupled with evidence-based dietary, lifestyle, and cardiology referral directives.
            """
        )

    with col_right:
        st.markdown("### 🚀 Fast Assessment")
        st.info(
            "Ready to test patient parameters? Start an interactive clinical intake session or explore model benchmarks."
        )
        if st.button("🩺 Start Patient Assessment", use_container_width=True, type="primary"):
            navigate_to("Page 2: Patient Details")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("📊 Explore Model Comparisons", use_container_width=True):
            navigate_to("Page 4: Model Comparison")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔒 Reliability & Fallback Architecture")
        st.markdown(
            """
            <div style="background:rgba(30,41,59,0.5); padding:1rem; border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                <span class="badge-live">Live ML Integration</span>
                <p style="font-size:0.85rem; color:#94a3b8; margin-top:0.5rem; margin-bottom:0;">
                    Connected to serialized teammate pipelines in <code>models/</code>. Equipped with an intelligent fallback heuristic engine ensuring zero crashes even during asynchronous model updates.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# PAGE 2: PATIENT DETAILS & CLINICAL INTAKE FORM
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "Page 2: Patient Details":
    st.markdown('<div class="brand-title">Patient Clinical Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Enter the 13 clinical biomarkers for algorithmic cardiac risk assessment</div>', unsafe_allow_html=True)

    # Preset quick-load bar
    p_col1, p_col2, p_col3 = st.columns([1.5, 1.5, 3])
    with p_col1:
        if st.button("🌿 Load Healthy Baseline Profile", use_container_width=True):
            st.session_state.patient_input = PRESETS["Healthy"].copy()
            st.success("Loaded Healthy Patient Preset!")
            st.rerun()
    with p_col2:
        if st.button("⚠️ Load High-Risk Cardiac Profile", use_container_width=True):
            st.session_state.patient_input = PRESETS["Elevated Risk"].copy()
            st.warning("Loaded High-Risk Patient Preset!")
            st.rerun()

    st.markdown("---")

    current = st.session_state.patient_input

    # Form with clean 2-column layout
    with st.form("clinical_intake_form"):
        col1, col2 = st.columns(2)

        # ----------------- COLUMN 1: Demographics & Resting Metrics -----------------
        with col1:
            st.markdown("#### 📋 Demographics & Baseline Hemodynamics")
            
            age = st.slider(
                "1. Age (Years)",
                min_value=20, max_value=90,
                value=int(current.get("age", 54)),
                help="Patient age at examination"
            )

            sex_opts = ["Male", "Female"]
            default_sex_idx = 0 if current.get("sex", 1) == 1 else 1
            sex_choice = st.selectbox(
                "2. Biological Sex",
                options=sex_opts,
                index=default_sex_idx,
                help="1 = Male, 0 = Female"
            )
            sex = 1.0 if sex_choice == "Male" else 0.0

            cp_mapping = {
                "1: Typical Angina": 1.0,
                "2: Atypical Angina": 2.0,
                "3: Non-Anginal Pain": 3.0,
                "4: Asymptomatic Angina": 4.0
            }
            default_cp_val = current.get("cp", 1.0)
            cp_idx = [1.0, 2.0, 3.0, 4.0].index(default_cp_val) if default_cp_val in [1.0, 2.0, 3.0, 4.0] else 0
            cp_choice = st.selectbox(
                "3. Chest Pain Type (cp)",
                options=list(cp_mapping.keys()),
                index=cp_idx,
                help="Asymptomatic angina (4) is strongly associated with ischemic heart disease"
            )
            cp = cp_mapping[cp_choice]

            trestbps = st.number_input(
                "4. Resting Blood Pressure (mm Hg)",
                min_value=80, max_value=220,
                value=int(current.get("trestbps", 130)),
                step=1,
                help="Normal systolic BP is typically <120 mmHg; >130 mmHg indicates hypertension"
            )

            chol = st.number_input(
                "5. Serum Cholesterol (mg/dl)",
                min_value=100, max_value=600,
                value=int(current.get("chol", 240)),
                step=1,
                help="Desirable cholesterol level is <200 mg/dl; elevated levels increase coronary plaque"
            )

            fbs_choice = st.radio(
                "6. Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=["No (Normal ≤ 120 mg/dl)", "Yes (Elevated > 120 mg/dl)"],
                index=1 if current.get("fbs", 0) == 1 else 0,
                horizontal=True,
                help="Indicator for diabetes or prediabetic metabolic condition"
            )
            fbs = 1.0 if "Yes" in fbs_choice else 0.0

            restecg_mapping = {
                "0: Normal": 0.0,
                "1: ST-T Wave Abnormality": 1.0,
                "2: Left Ventricular Hypertrophy (LVH)": 2.0
            }
            default_restecg = current.get("restecg", 0.0)
            recg_idx = [0.0, 1.0, 2.0].index(default_restecg) if default_restecg in [0.0, 1.0, 2.0] else 0
            restecg_choice = st.selectbox(
                "7. Resting ECG (restecg)",
                options=list(restecg_mapping.keys()),
                index=recg_idx,
                help="Resting electrocardiogram status"
            )
            restecg = restecg_mapping[restecg_choice]

        # ----------------- COLUMN 2: Stress Testing & Imaging -----------------
        with col2:
            st.markdown("#### 🏃 Exercise Stress & Imaging Biomarkers")

            thalach = st.slider(
                "8. Maximum Heart Rate Achieved (bpm)",
                min_value=60, max_value=220,
                value=int(current.get("thalach", 150)),
                help="Maximum heart rate attained during graded treadmill stress testing"
            )

            exang_choice = st.radio(
                "9. Exercise Induced Angina (exang)",
                options=["No", "Yes"],
                index=1 if current.get("exang", 0) == 1 else 0,
                horizontal=True,
                help="Ischemic chest pain provoked during physical exercise"
            )
            exang = 1.0 if exang_choice == "Yes" else 0.0

            oldpeak = st.slider(
                "10. ST Depression Induced by Exercise (oldpeak)",
                min_value=0.0, max_value=6.5,
                value=float(current.get("oldpeak", 1.0)),
                step=0.1,
                help="ST depression in mm relative to baseline rest; values >1.0 mm signal myocardial ischemia"
            )

            slope_mapping = {
                "1: Upsloping (Healthy response)": 1.0,
                "2: Flat (Moderate ischemia risk)": 2.0,
                "3: Downsloping (High ischemia risk)": 3.0
            }
            default_slope = current.get("slope", 1.0)
            slope_idx = [1.0, 2.0, 3.0].index(default_slope) if default_slope in [1.0, 2.0, 3.0] else 0
            slope_choice = st.selectbox(
                "11. Slope of Peak Exercise ST Segment (slope)",
                options=list(slope_mapping.keys()),
                index=slope_idx,
                help="Electrocardiographic trajectory during peak treadmill stress"
            )
            slope = slope_mapping[slope_choice]

            ca = st.slider(
                "12. Number of Major Vessels Colored by Fluoroscopy (0 - 3)",
                min_value=0, max_value=3,
                value=int(current.get("ca", 0)),
                help="Number of major coronary blood vessels visualized with contrast agent"
            )

            thal_mapping = {
                "3: Normal Blood Flow": 3.0,
                "6: Fixed Perfusion Defect": 6.0,
                "7: Reversible Perfusion Defect": 7.0
            }
            default_thal = current.get("thal", 3.0)
            thal_idx = [3.0, 6.0, 7.0].index(default_thal) if default_thal in [3.0, 6.0, 7.0] else 0
            thal_choice = st.selectbox(
                "13. Thalassemia Scintigraphy (thal)",
                options=list(thal_mapping.keys()),
                index=thal_idx,
                help="Thallium stress test myocardial perfusion imaging"
            )
            thal = thal_mapping[thal_choice]

        st.markdown("---")

        # ----------------- MODEL SELECTOR ROW -----------------
        st.markdown("#### 🤖 Predictive Algorithm Selector")
        m_col1, m_col2 = st.columns([2, 1.5])
        with m_col1:
            model_options = [
                "Random Forest Classifier",
                "Logistic Regression",
                "Gaussian Naive Bayes"
            ]
            default_m_idx = model_options.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_options else 0
            chosen_model = st.selectbox(
                "Choose Machine Learning Model for Evaluation:",
                options=model_options,
                index=default_m_idx
            )
            st.session_state.selected_model_name = chosen_model

        with m_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            model_reg = model_service.get_available_models().get(chosen_model, {})
            if model_reg.get("is_live", False):
                st.markdown('<span class="badge-live">● Live Serialized Artifact Ready</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-fallback">⚡ Graceful Heuristic Fallback Active</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🩺 Run Diagnostic Prediction", use_container_width=True, type="primary")

    if submitted:
        patient_dict = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
            "thal": thal
        }
        st.session_state.patient_input = patient_dict

        with st.spinner("Processing clinical biomarkers and calculating probability..."):
            res = model_service.predict_patient(patient_dict, model_name=chosen_model)
            st.session_state.prediction_result = res

        navigate_to("Page 3: Diagnostic Results")

# -----------------------------------------------------------------------------
# PAGE 3: DIAGNOSTIC RESULTS
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "Page 3: Diagnostic Results":
    st.markdown('<div class="brand-title">Diagnostic Results & Risk Stratification</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Evidence-based clinical risk analysis generated from patient biomarkers</div>', unsafe_allow_html=True)

    res = st.session_state.prediction_result
    patient = st.session_state.patient_input

    if res is None:
        st.warning("No diagnostic results found. Please enter patient biomarkers on Page 2 first.")
        if st.button("Go to Patient Details Form"):
            navigate_to("Page 2: Patient Details")
    else:
        is_positive = res["has_heart_disease"]
        prob_pct = res["probability_percent"]
        model_used = res["model_used"]

        # Alert Banner
        if is_positive:
            st.markdown(
                f"""
                <div class="alert-high-risk">
                    <h3 style="margin:0 0 0.3rem 0; color:#fecaca; display:flex; align-items:center; gap:0.5rem;">
                        ⚠️ Heart Disease Risk Detected (Positive Diagnosis)
                    </h3>
                    <p style="margin:0; font-size:1.05rem; opacity:0.95;">
                        The diagnostic model identifies significant physiological indicators consistent with coronary artery disease. 
                        Calculated probability: <strong>{prob_pct:.1f}%</strong> via <strong>{model_used}</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="alert-normal">
                    <h3 style="margin:0 0 0.3rem 0; color:#a7f3d0; display:flex; align-items:center; gap:0.5rem;">
                        ✅ Low Cardiac Risk (Normal Diagnostic Profile)
                    </h3>
                    <p style="margin:0; font-size:1.05rem; opacity:0.95;">
                        Patient biomarkers remain within normal physiological baselines. 
                        Calculated disease probability: <strong>{prob_pct:.1f}%</strong> via <strong>{model_used}</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Gauge & Summary Columns
        col_gauge, col_info = st.columns([1.2, 1.0])

        with col_gauge:
            st.markdown("#### 🎯 Disease Probability Gauge")
            
            # Interactive Plotly Gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Probability (%)", 'font': {'size': 20, 'color': '#f8fafc'}},
                number={'suffix': "%", 'font': {'size': 38, 'color': '#ffffff'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#ef4444" if is_positive else "#10b981", 'thickness': 0.3},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'borderwidth': 1,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.25)"},
                        {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': "#f87171", 'width': 4},
                        'thickness': 0.8,
                        'value': 50.0
                    }
                }
            ))
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#f8fafc", 'family': "Outfit"},
                height=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_info:
            st.markdown("#### 📊 Diagnostic Summary Card")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                        <span style="color:#94a3b8; font-weight:500;">Risk Category</span>
                        <strong style="color:{'#ef4444' if is_positive else '#10b981'}; font-size:1.15rem;">
                            {'High Risk' if is_positive else 'Low Risk'}
                        </strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                        <span style="color:#94a3b8; font-weight:500;">Binary Prediction</span>
                        <span style="color:#f8fafc; font-weight:600;">Class {res['prediction']} ({'Disease' if is_positive else 'Healthy'})</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                        <span style="color:#94a3b8; font-weight:500;">Classifier Engine</span>
                        <span style="color:#38bdf8; font-weight:600;">{model_used}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <span style="color:#94a3b8; font-weight:500;">Inference Mode</span>
                        <span style="color:{'#f59e0b' if res.get('is_mock') else '#10b981'}; font-weight:600;">
                            {'⚡ Fallback Mock Heuristic' if res.get('is_mock') else '🟢 Live Scaled Pipeline'}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 💡 Clinical Recommendations")
            for rec in res.get("recommendations", []):
                st.markdown(f"- 🩺 {rec}")

        st.markdown("---")

        # Patient Input Summary Matrix
        st.markdown("### 📋 Input Biomarker Summary Matrix")
        
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Age / Sex", f"{int(patient.get('age', 0))} yrs / {'M' if patient.get('sex', 0) == 1 else 'F'}")
            st.metric("Resting Blood Pressure", f"{int(patient.get('trestbps', 0))} mm Hg", delta="Elevated" if patient.get("trestbps", 0) > 130 else "Normal", delta_color="inverse")
        with b2:
            st.metric("Serum Cholesterol", f"{int(patient.get('chol', 0))} mg/dl", delta="Elevated" if patient.get("chol", 0) > 200 else "Desirable", delta_color="inverse")
            st.metric("Fasting Blood Sugar", "> 120 mg/dl" if patient.get("fbs", 0) == 1 else "≤ 120 mg/dl")
        with b3:
            st.metric("Max Heart Rate", f"{int(patient.get('thalach', 0))} bpm")
            st.metric("ST Depression (oldpeak)", f"{patient.get('oldpeak', 0.0):.1f} mm", delta="Ischemia Risk" if patient.get("oldpeak", 0) >= 1.0 else "Normal", delta_color="inverse")
        with b4:
            st.metric("Fluoroscopy Vessels (ca)", f"{int(patient.get('ca', 0))} colored")
            st.metric("Exercise Angina", "Yes (Induced)" if patient.get("exang", 0) == 1 else "No")

        st.markdown("<br>", unsafe_allow_html=True)

        # Action Buttons
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("🔄 Test Another Model on This Patient", use_container_width=True):
                navigate_to("Page 2: Patient Details")
        with btn_c2:
            if st.button("📝 Perform New Patient Assessment", use_container_width=True):
                st.session_state.patient_input = PRESETS["Healthy"].copy()
                st.session_state.prediction_result = None
                navigate_to("Page 2: Patient Details")
        with btn_c3:
            if st.button("📊 View Model Comparison Benchmarks", use_container_width=True, type="primary"):
                navigate_to("Page 4: Model Comparison")

# -----------------------------------------------------------------------------
# PAGE 4: MODEL COMPARISON & ANALYTICS
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "Page 4: Model Comparison":
    st.markdown('<div class="brand-title">Comparative Algorithm Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Quantitative evaluation of Logistic Regression, Random Forest, and Gaussian Naive Bayes</div>', unsafe_allow_html=True)

    metrics_dict = model_service.get_metrics()
    models_data = metrics_dict.get("models", {})

    # Top KPI comparison cards
    k1, k2, k3 = st.columns(3)
    
    rf_m = models_data.get("random_forest", {})
    lr_m = models_data.get("logistic_regression", {})
    nb_m = models_data.get("naive_bayes", {})

    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight:700; color:#38bdf8; font-size:1.1rem; margin-bottom:0.4rem;">🌲 Random Forest</div>
                <div class="metric-number">{rf_m.get('test_accuracy', 0.9016)*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <div style="margin-top:0.6rem; font-size:0.85rem; color:#94a3b8;">
                    F1-Score: <strong>{rf_m.get('f1_score', 0.8966)*100:.1f}%</strong> | Recall: <strong>{rf_m.get('recall', 0.9286)*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight:700; color:#818cf8; font-size:1.1rem; margin-bottom:0.4rem;">📈 Logistic Regression</div>
                <div class="metric-number">{lr_m.get('test_accuracy', 0.8689)*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <div style="margin-top:0.6rem; font-size:0.85rem; color:#94a3b8;">
                    F1-Score: <strong>{lr_m.get('f1_score', 0.8667)*100:.1f}%</strong> | Recall: <strong>{lr_m.get('recall', 0.9286)*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight:700; color:#c084fc; font-size:1.1rem; margin-bottom:0.4rem;">⚡ Gaussian Naive Bayes</div>
                <div class="metric-number">{nb_m.get('test_accuracy', 0.8689)*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <div style="margin-top:0.6rem; font-size:0.85rem; color:#94a3b8;">
                    F1-Score: <strong>{nb_m.get('f1_score', 0.8710)*100:.1f}%</strong> | Recall: <strong style="color:#10b981;">{nb_m.get('recall', 0.9643)*100:.1f}% (Top)</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison Table
    st.markdown("### 📊 Benchmark Metrics Comparison Table")
    
    rows = []
    display_names = {
        "random_forest": "Random Forest Classifier (Rohit)",
        "logistic_regression": "Logistic Regression (Rohit)",
        "naive_bayes": "Gaussian Naive Bayes (Om)"
    }
    
    for key, name in display_names.items():
        if key in models_data:
            m = models_data[key]
            rows.append({
                "Algorithm": name,
                "Train Accuracy": f"{m.get('train_accuracy', 0)*100:.2f}%",
                "Test Accuracy": f"{m.get('test_accuracy', 0)*100:.2f}%",
                "Precision": f"{m.get('precision', 0)*100:.2f}%",
                "Recall (Sensitivity)": f"{m.get('recall', 0)*100:.2f}%",
                "F1-Score": f"{m.get('f1_score', 0)*100:.2f}%",
                "ROC-AUC": f"{m.get('roc_auc', 0):.4f}"
            })

    df_comp = pd.DataFrame(rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Plotly Grouped Bar Chart
    st.markdown("### 📈 Interactive Performance Visualization")
    
    plot_data = []
    for key, name in display_names.items():
        if key in models_data:
            m = models_data[key]
            plot_data.extend([
                {"Model": key.replace("_", " ").title(), "Metric": "Test Accuracy", "Score (%)": round(m.get('test_accuracy', 0)*100, 2)},
                {"Model": key.replace("_", " ").title(), "Metric": "Precision", "Score (%)": round(m.get('precision', 0)*100, 2)},
                {"Model": key.replace("_", " ").title(), "Metric": "Recall", "Score (%)": round(m.get('recall', 0)*100, 2)},
                {"Model": key.replace("_", " ").title(), "Metric": "F1-Score", "Score (%)": round(m.get('f1_score', 0)*100, 2)}
            ])
            
    df_plot = pd.DataFrame(plot_data)
    
    fig = px.bar(
        df_plot,
        x="Metric",
        y="Score (%)",
        color="Model",
        barmode="group",
        text="Score (%)",
        color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc"]
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Outfit"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(range=[60, 100], gridcolor="rgba(148, 163, 184, 0.15)"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_traces(textposition='outside', textfont_size=11)
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Visual Artifacts (Confusion Matrix, ROC Curve, Feature Importance)
    st.markdown("### 🖼️ Diagnostic Evaluation Artifacts")
    tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrices", "📈 ROC-AUC Curves", "🌟 Clinical Feature Importance"])

    with tab1:
        cm_path = os.path.join("assets", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices: Logistic Regression vs Random Forest (Test Set)", use_container_width=True)
        else:
            st.info("Confusion matrix asset not found in assets/ directory.")

    with tab2:
        roc_path = os.path.join("assets", "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Receiver Operating Characteristic (ROC-AUC) Curves", use_container_width=True)
        else:
            st.info("ROC Curve asset not found in assets/ directory.")

    with tab3:
        fi_path = os.path.join("assets", "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Biomarker Gini Importance Scores (Random Forest Classifier)", use_container_width=True)
        else:
            st.info("Feature importance asset not found in assets/ directory.")

    st.markdown("---")

    # Clinical Discussion of Tradeoffs
    st.markdown("### 🏥 Clinical Tradeoff Insights: Precision vs. Recall")
    st.info(
        """
        **Why Recall (Sensitivity) is Critical in Cardiology:**
        - **False Negative (FN) Cost:** Missing an actual heart disease patient leads to untreated coronary artery disease, potentially causing sudden myocardial infarction or death.
        - **False Positive (FP) Cost:** Misclassifying a healthy patient as positive triggers harmless non-invasive follow-up tests (e.g. Echocardiogram, Stress ECG).
        - **Model Takeaway:** Gaussian Naive Bayes achieves an exceptional **96.43% Recall**, making it an ideal rapid screening filter, while Random Forest provides the highest overall accuracy (**90.16%**) and balanced F1-score (**89.66%**).
        """
    )
