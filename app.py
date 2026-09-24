"""
CardioSense AI - Clinical Cardiovascular Diagnostic Intelligence Platform
Lead Team: Rohit (ML & Integration), Om (EDA & Naive Bayes), Umar (UI & Deployment)
Dataset: UCI Cleveland Heart Disease Dataset (303 patient records, 13 biomarkers)
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    go = None
    px = None
    HAS_PLOTLY = False

import model_service

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioSense AI | Cardiovascular Decision Support",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
    }

    /* Modern Background & Container Tweaks */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Gradient Brand Title */
    .brand-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 14px 30px -5px rgba(56, 189, 248, 0.15);
    }

    .metric-number {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-top: 0.4rem;
    }

    /* Clinical Alert Banners */
    .alert-high-risk {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 6px solid #EF4444;
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #FEF2F2;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.15);
    }

    .alert-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(4, 120, 87, 0.3) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-left: 6px solid #10B981;
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.5rem;
        color: #F0FDF4;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
    }

    /* Section Panels */
    .content-panel {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Team Badges */
    .team-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    /* Button Enhancements */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: all 0.2s ease-in-out;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT & CLINICAL PRESETS
# -----------------------------------------------------------------------------
PRESETS = {
    "Healthy": {
        "age": 45, "sex": 0, "cp": 2, "trestbps": 118, "chol": 185,
        "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    },
    "Elevated Risk": {
        "age": 64, "sex": 1, "cp": 4, "trestbps": 158, "chol": 286,
        "fbs": 1, "restecg": 2, "thalach": 112, "exang": 1,
        "oldpeak": 2.8, "slope": 2, "ca": 3, "thal": 7
    }
}

if "current_page" not in st.session_state:
    st.session_state.current_page = "Page 1: Home"

if "patient_input" not in st.session_state:
    st.session_state.patient_input = PRESETS["Healthy"].copy()

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "Random Forest Classifier"

def navigate_to(page_name):
    st.session_state.current_page = page_name
    st.rerun()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM ARTIFACT MONITOR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
            <span style="font-size:1.8rem;">🫀</span>
            <div>
                <h3 style="margin:0; font-weight:800; font-size:1.35rem; color:#F8FAFC;">CardioSense AI</h3>
            </div>
        </div>
        <p style="color:#94A3B8; font-size:0.8rem; margin-top:0.1rem; margin-bottom:1.2rem;">
            Clinical Decision Support Platform
        </p>
        """,
        unsafe_allow_html=True
    )
    
    pages = [
        "Page 1: Home",
        "Page 2: Patient Details",
        "Page 3: Diagnostic Results",
        "Page 4: Model Comparison"
    ]
    
    nav_icons = {
        "Page 1: Home": "🏠 1. Overview & Protocol",
        "Page 2: Patient Details": "🩺 2. Patient Clinical Intake",
        "Page 3: Diagnostic Results": "📊 3. Diagnostic Report",
        "Page 4: Model Comparison": "🏆 4. Model Benchmark Hub"
    }

    selected_page = st.radio(
        "Workflow Navigation",
        options=pages,
        format_func=lambda p: nav_icons.get(p, p),
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
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.65); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 9px; padding: 0.5rem 0.75rem; margin-bottom: 0.45rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #F8FAFC;">🟢 {name}</span>
                        <span style="background: rgba(16, 185, 129, 0.2); color: #10B981; font-size: 0.68rem; padding: 2px 6px; border-radius: 4px; font-weight: 700;">LIVE</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #94A3B8; margin-top: 0.25rem;">
                        📁 <code>{info['filename']}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.65); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 9px; padding: 0.5rem 0.75rem; margin-bottom: 0.45rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #F8FAFC;">🟡 {name}</span>
                        <span style="background: rgba(245, 158, 11, 0.2); color: #F59E0B; font-size: 0.68rem; padding: 2px 6px; border-radius: 4px; font-weight: 700;">MOCK</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #94A3B8; margin-top: 0.25rem;">
                        ⚠️ Fallback active
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown("---")
    st.markdown("#### 👥 Development Team")
    st.markdown(
        """
        <div class="team-card">
            <span style="font-size:1.1rem;">💻</span>
            <div>
                <strong style="color:#F8FAFC; font-size:0.88rem;">Rohit</strong><br>
                <span style="color:#94A3B8; font-size:0.75rem;">ML Pipeline & Backend Lead</span>
            </div>
        </div>
        <div class="team-card">
            <span style="font-size:1.1rem;">📊</span>
            <div>
                <strong style="color:#F8FAFC; font-size:0.88rem;">Om</strong><br>
                <span style="color:#94A3B8; font-size:0.75rem;">EDA & Naive Bayes Lead</span>
            </div>
        </div>
        <div class="team-card">
            <span style="font-size:1.1rem;">🎨</span>
            <div>
                <strong style="color:#38BDF8; font-size:0.88rem;">Umar</strong><br>
                <span style="color:#38BDF8; font-size:0.75rem;">UI & Deployment Lead</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.caption("UCI Cleveland Cohort • 303 Records")


# =============================================================================
# PAGE 1: HOME & CLINICAL PROTOCOL OVERVIEW
# =============================================================================
if st.session_state.current_page == "Page 1: Home":
    st.markdown('<div class="brand-title">CardioSense AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Automated Machine Learning Diagnostic System for Early Coronary Artery Disease Detection</div>',
        unsafe_allow_html=True
    )

    # 4 Top Performance KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#38BDF8;">303</div>
                <div class="metric-label">Patient Cohort</div>
                <small style="color:#64748B;">UCI Cleveland Clinical Records</small>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#818CF8;">13</div>
                <div class="metric-label">Clinical Biomarkers</div>
                <small style="color:#64748B;">Hemodynamic, ECG, Fluoroscopy</small>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#C084FC;">96.43%</div>
                <div class="metric-label">Peak Screening Recall</div>
                <small style="color:#64748B;">Gaussian Naive Bayes (Om)</small>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number" style="color:#10B981;">90.16%</div>
                <div class="metric-label">Champion Accuracy</div>
                <small style="color:#64748B;">Random Forest (Rohit)</small>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Clinical Motivation & Architecture
    col_left, col_right = st.columns([1.5, 1.0])
    
    with col_left:
        st.markdown("### 🩺 Clinical Overview & Problem Statement")
        st.write(
            """
            Cardiovascular diseases (CVDs) represent the leading cause of global mortality.
            Standard diagnostic pathways frequently struggle to synthesize non-linear interactions across
            resting hemodynamics, exercise-induced ST segment shifts, and fluoroscopic major vessel counts.
            
            **CardioSense AI** provides deterministic, calibrated diagnostic risk assessments by applying
            rigorously validated machine learning ensembles trained on the gold-standard UCI Cleveland dataset.
            """
        )

        st.markdown("#### 🔬 3-Stage Diagnostic Engine")
        st.markdown(
            """
            1. **Intake & Standardization:** 13 physiological biomarkers are normalized with a calibrated `StandardScaler`.
            2. **Multi-Algorithm Inference:** Patient records are processed through Random Forest, Logistic Regression, or Gaussian Naive Bayes classifiers.
            3. **Clinical Stratification:** Provides an immediate risk classification, disease probability percentage, and tailored action plans.
            """
        )

    with col_right:
        st.markdown("### 🚀 Fast Assessment")
        st.info("Ready to evaluate a patient? Launch the clinical intake form or review benchmark comparisons.")
        
        if st.button("🩺 Start Patient Assessment", type="primary"):
            navigate_to("Page 2: Patient Details")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("📊 Explore Model Comparisons"):
            navigate_to("Page 4: Model Comparison")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:rgba(30,41,59,0.5); padding:1rem; border-radius:10px; border:1px solid rgba(148,163,184,0.15);">
                <span style="background:rgba(16,185,129,0.2); color:#10B981; font-weight:700; font-size:0.75rem; padding:2px 8px; border-radius:4px;">
                    ● Live ML Integration
                </span>
                <p style="font-size:0.83rem; color:#94A3B8; margin-top:0.5rem; margin-bottom:0; line-height:1.4;">
                    Directly linked to serialized <code>models/</code>. Equipped with zero-downtime heuristic fallbacks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =============================================================================
# PAGE 2: PATIENT DETAILS & CLINICAL INTAKE FORM
# =============================================================================
elif st.session_state.current_page == "Page 2: Patient Details":
    st.markdown('<div class="brand-title">Patient Clinical Intake</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Enter the 13 clinical biomarkers for algorithmic cardiac risk assessment</div>',
        unsafe_allow_html=True
    )

    # Preset quick-load bar
    p_col1, p_col2, p_col3 = st.columns([1.5, 1.5, 3])
    with p_col1:
        if st.button("🌿 Load Healthy Baseline Profile"):
            st.session_state.patient_input = PRESETS["Healthy"].copy()
            st.success("Loaded Healthy Patient Preset!")
            st.rerun()
    with p_col2:
        if st.button("⚠️ Load High-Risk Cardiac Profile"):
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
                help="Patient age in years"
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
                help="Asymptomatic angina (4) is strongly associated with ischemic coronary disease"
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
                help="Desirable cholesterol level is <200 mg/dl"
            )

            fbs_choice = st.radio(
                "6. Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=["No (Normal ≤ 120 mg/dl)", "Yes (Elevated > 120 mg/dl)"],
                index=1 if current.get("fbs", 0) == 1 else 0,
                horizontal=True,
                help="Marker for diabetes or metabolic syndrome"
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
                help="Resting electrocardiogram waveform interpretation"
            )
            restecg = restecg_mapping[restecg_choice]

        # ----------------- COLUMN 2: Stress Testing & Imaging -----------------
        with col2:
            st.markdown("#### 🏃 Exercise Stress & Imaging Biomarkers")

            thalach = st.slider(
                "8. Maximum Heart Rate Achieved (bpm)",
                min_value=60, max_value=220,
                value=int(current.get("thalach", 150)),
                help="Maximum heart rate attained during graded treadmill stress test"
            )

            exang_choice = st.radio(
                "9. Exercise Induced Angina (exang)",
                options=["No", "Yes"],
                index=1 if current.get("exang", 0) == 1 else 0,
                horizontal=True,
                help="Ischemic chest pain provoked during physical exertion"
            )
            exang = 1.0 if exang_choice == "Yes" else 0.0

            oldpeak = st.slider(
                "10. ST Depression Induced by Exercise (oldpeak)",
                min_value=0.0, max_value=6.5,
                value=float(current.get("oldpeak", 1.0)),
                step=0.1,
                help="ST depression in mm relative to rest baseline; values ≥1.0 mm signal myocardial ischemia"
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
                "12. Major Vessels Colored by Fluoroscopy (0 - 3)",
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
                "Choose Machine Learning Model for Diagnostic Inference:",
                options=model_options,
                index=default_m_idx
            )
            st.session_state.selected_model_name = chosen_model

        with m_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            model_reg = model_service.get_available_models().get(chosen_model, {})
            if model_reg.get("is_live", False):
                st.markdown(
                    '<span style="background:rgba(16,185,129,0.2); color:#10B981; font-weight:700; font-size:0.8rem; padding:4px 10px; border-radius:6px;">● Live Serialized Artifact Ready</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span style="background:rgba(245,158,11,0.2); color:#F59E0B; font-weight:700; font-size:0.8rem; padding:4px 10px; border-radius:6px;">⚡ Graceful Fallback Engine Active</span>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🩺 Run Diagnostic Prediction", type="primary")

    if submitted:
        patient_dict = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
            "thal": thal
        }
        st.session_state.patient_input = patient_dict

        with st.spinner("Standardizing biomarkers and calculating calibrated risk..."):
            res = model_service.predict_patient(patient_dict, model_name=chosen_model)
            st.session_state.prediction_result = res

        navigate_to("Page 3: Diagnostic Results")


# =============================================================================
# PAGE 3: DIAGNOSTIC RESULTS & CLINICAL REPORT
# =============================================================================
elif st.session_state.current_page == "Page 3: Diagnostic Results":
    st.markdown('<div class="brand-title">Diagnostic Results & Risk Stratification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Evidence-based clinical risk analysis generated from patient biomarkers</div>',
        unsafe_allow_html=True
    )

    res = st.session_state.prediction_result
    patient = st.session_state.patient_input

    if res is None:
        st.warning("No diagnostic results found. Please enter patient biomarkers on Page 2 first.")
        if st.button("⬅️ Go to Patient Intake Form"):
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
                    <h3 style="margin:0 0 0.3rem 0; color:#FECACA; font-weight:800; display:flex; align-items:center; gap:0.5rem;">
                        ⚠️ Heart Disease Risk Detected (Positive Diagnosis)
                    </h3>
                    <p style="margin:0; font-size:1.05rem; opacity:0.95;">
                        Biomarkers indicate clinical indicators consistent with ischemic heart disease. 
                        Calculated probability: <strong style="color:#EF4444; font-size:1.2rem;">{prob_pct:.1f}%</strong> via <strong>{model_used}</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="alert-normal">
                    <h3 style="margin:0 0 0.3rem 0; color:#A7F3D0; font-weight:800; display:flex; align-items:center; gap:0.5rem;">
                        ✅ Low Cardiac Risk (Normal Diagnostic Profile)
                    </h3>
                    <p style="margin:0; font-size:1.05rem; opacity:0.95;">
                        Patient biomarkers remain within healthy baselines. 
                        Calculated disease probability: <strong style="color:#10B981; font-size:1.2rem;">{prob_pct:.1f}%</strong> via <strong>{model_used}</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Gauge & Summary Columns
        col_gauge, col_info = st.columns([1.2, 1.0])

        with col_gauge:
            st.markdown("#### 🎯 Disease Probability Gauge")
            
            if HAS_PLOTLY and go is not None:
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Calculated Risk Probability", 'font': {'size': 18, 'color': '#F8FAFC'}},
                    number={'suffix': "%", 'font': {'size': 38, 'color': '#FFFFFF'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                        'bar': {'color': "#EF4444" if is_positive else "#10B981", 'thickness': 0.35},
                        'bgcolor': "rgba(30, 41, 59, 0.5)",
                        'borderwidth': 1,
                        'bordercolor': "#475569",
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.25)"},
                            {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                        ],
                        'threshold': {
                            'line': {'color': "#F87171", 'width': 4},
                            'thickness': 0.8,
                            'value': 50.0
                        }
                    }
                ))
                gauge_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#F8FAFC", 'family': "Plus Jakarta Sans"},
                    height=320,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(gauge_fig)
            else:
                st.progress(float(res.get("probability", 0.0)))
                st.metric("Risk Probability", f"{prob_pct:.2f}%")

        with col_info:
            st.markdown("#### 📊 Diagnostic Summary Card")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <span style="color:#94A3B8; font-weight:500;">Risk Category</span>
                        <strong style="color:{'#EF4444' if is_positive else '#10B981'}; font-size:1.15rem;">
                            {'High Risk (Presence)' if is_positive else 'Low Risk (Healthy)'}
                        </strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <span style="color:#94A3B8; font-weight:500;">Binary Prediction</span>
                        <span style="color:#F8FAFC; font-weight:600;">Class {res['prediction']} ({'Disease' if is_positive else 'No Disease'})</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                        <span style="color:#94A3B8; font-weight:500;">Inference Model</span>
                        <span style="color:#38BDF8; font-weight:600;">{model_used}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                        <span style="color:#94A3B8; font-weight:500;">Execution Pipeline</span>
                        <span style="color:{'#F59E0B' if res.get('is_mock') else '#10B981'}; font-weight:600;">
                            {'⚡ Fallback Mock Engine' if res.get('is_mock') else '🟢 Live Scaled Pipeline'}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 💡 Clinical Action Directives")
            for rec in res.get("recommendations", []):
                st.info(f"• {rec}")

        st.markdown("---")

        # Patient Input Summary Matrix
        st.markdown("### 📋 Input Biomarker Summary Matrix")
        
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Age / Sex", f"{int(patient.get('age', 0))} yrs / {'Male' if patient.get('sex', 0) == 1 else 'Female'}")
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
            if st.button("🔄 Test Another Model on This Patient"):
                navigate_to("Page 2: Patient Details")
        with btn_c2:
            if st.button("📝 Perform New Patient Assessment"):
                st.session_state.patient_input = PRESETS["Healthy"].copy()
                st.session_state.prediction_result = None
                navigate_to("Page 2: Patient Details")
        with btn_c3:
            if st.button("📊 View Model Comparison Benchmarks", type="primary"):
                navigate_to("Page 4: Model Comparison")


# =============================================================================
# PAGE 4: MODEL COMPARISON & BENCHMARK HUB
# =============================================================================
elif st.session_state.current_page == "Page 4: Model Comparison":
    st.markdown('<div class="brand-title">Comparative Algorithm Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Quantitative evaluation of Logistic Regression, Random Forest, and Gaussian Naive Bayes</div>',
        unsafe_allow_html=True
    )

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
            <div class="metric-card" style="border: 2px solid rgba(56, 189, 248, 0.4);">
                <div style="font-weight:700; color:#38BDF8; font-size:1.1rem; margin-bottom:0.4rem;">🌲 Random Forest (Champion)</div>
                <div class="metric-number" style="color:#38BDF8;">{rf_m.get('test_accuracy', 0.9016)*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <div style="margin-top:0.6rem; font-size:0.85rem; color:#94A3B8;">
                    F1-Score: <strong style="color:#F8FAFC;">{rf_m.get('f1_score', 0.8966)*100:.1f}%</strong> | Recall: <strong style="color:#F8FAFC;">{rf_m.get('recall', 0.9286)*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight:700; color:#818CF8; font-size:1.1rem; margin-bottom:0.4rem;">📈 Logistic Regression (Baseline)</div>
                <div class="metric-number" style="color:#818CF8;">{lr_m.get('test_accuracy', 0.8689)*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <div style="margin-top:0.6rem; font-size:0.85rem; color:#94A3B8;">
                    F1-Score: <strong style="color:#F8FAFC;">{lr_m.get('f1_score', 0.8667)*100:.1f}%</strong> | Recall: <strong style="color:#F8FAFC;">{lr_m.get('recall', 0.9286)*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-weight:700; color:#C084FC; font-size:1.1rem; margin-bottom:0.4rem;">⚡ Gaussian Naive Bayes (Om)</div>
                <div class="metric-number" style="color:#C084FC;">{nb_m.get('test_accuracy', 0.8689)*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <div style="margin-top:0.6rem; font-size:0.85rem; color:#94A3B8;">
                    F1-Score: <strong style="color:#F8FAFC;">{nb_m.get('f1_score', 0.8710)*100:.1f}%</strong> | Recall: <strong style="color:#10B981;">{nb_m.get('recall', 0.9643)*100:.1f}% (Top Sensitivity)</strong>
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
    st.dataframe(df_comp, hide_index=True)

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
    
    if HAS_PLOTLY and px is not None:
        fig = px.bar(
            df_plot,
            x="Metric",
            y="Score (%)",
            color="Model",
            barmode="group",
            text="Score (%)",
            color_discrete_sequence=["#38BDF8", "#818CF8", "#C084FC"]
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC", family="Plus Jakarta Sans"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(range=[60, 100], gridcolor="rgba(148, 163, 184, 0.15)"),
            xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_traces(textposition='outside', textfont_size=11)
        st.plotly_chart(fig)
    else:
        st.dataframe(df_plot)

    st.markdown("---")

    # Visual Artifacts (Confusion Matrix, ROC Curve, Feature Importance)
    st.markdown("### 🖼️ Diagnostic Evaluation Artifacts")
    tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrices", "📈 ROC-AUC Curves", "🌟 Clinical Feature Importance"])

    with tab1:
        cm_path = os.path.join("assets", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices: Logistic Regression vs Random Forest vs Naive Bayes")
        else:
            st.info("Confusion matrix asset not found in assets/ directory.")

    with tab2:
        roc_path = os.path.join("assets", "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Receiver Operating Characteristic (ROC-AUC) Curves")
        else:
            st.info("ROC Curve asset not found in assets/ directory.")

    with tab3:
        fi_path = os.path.join("assets", "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Biomarker Gini Importance Scores (Random Forest Classifier)")
        else:
            st.info("Feature importance asset not found in assets/ directory.")

    st.markdown("---")

    # Clinical Discussion of Tradeoffs
    st.markdown("### 🏥 Clinical Tradeoff Insights: Precision vs. Recall")
    st.info(
        """
        **Why Recall (Sensitivity) is Paramount in Cardiology:**
        - **False Negative (FN) Cost:** Missing an actual heart disease patient leads to untreated coronary artery disease, potentially causing sudden myocardial infarction or death.
        - **False Positive (FP) Cost:** Misclassifying a healthy patient as positive triggers harmless non-invasive follow-up tests (e.g. Echocardiogram, Stress ECG).
        - **Model Takeaway:** Gaussian Naive Bayes achieves an exceptional **96.43% Recall**, making it an ideal rapid screening filter, while Random Forest provides the highest overall accuracy (**90.16%**) and balanced F1-score (**89.66%**).
        """
    )
