"""
Medical Diagnosis Prediction - ML Training & Evaluation Pipeline
Lead: Rohit (ML & Integration Lead)
Dataset: UCI Heart Disease Dataset (Cleveland)

Roles & Team Work Division:
- Rohit (You): Dataset Preprocessing Pipeline, Logistic Regression, Random Forest, Evaluation & Model Selection, Central Integration API & Repository Management.
- Om: Exploratory Data Analysis (EDA graphs) & Gaussian Naive Bayes Model.
- Umar: Streamlit Multi-Page Web Application UI & Deployment.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
import joblib


COLUMN_METADATA = {
    "age": "Age in years",
    "sex": "Sex (1 = Male, 0 = Female)",
    "cp": "Chest Pain Type (1: Typical Angina, 2: Atypical Angina, 3: Non-anginal, 4: Asymptomatic)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)",
    "restecg": "Resting ECG (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy)",
    "thalach": "Maximum Heart Rate Achieved (bpm)",
    "exang": "Exercise Induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression Induced by Exercise Relative to Rest",
    "slope": "Slope of Peak Exercise ST Segment (1: Upsloping, 2: Flat, 3: Downsloping)",
    "ca": "Number of Major Vessels Colored by Fluoroscopy (0-3)",
    "thal": "Thalassemia (3: Normal, 6: Fixed Defect, 7: Reversible Defect)",
    "target": "Diagnosis (0 = No Heart Disease, 1 = Presence of Heart Disease)"
}


def save_figure_safely(fig, filepath, dpi=300):
    """
    Safely saves a matplotlib figure, handling Windows file locks, permission attributes,
    and open preview tabs.
    """
    import stat
    abs_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    
    if os.path.exists(abs_path):
        try:
            os.chmod(abs_path, stat.S_IWRITE)
        except Exception:
            pass
        try:
            os.remove(abs_path)
        except Exception:
            pass
            
    try:
        fig.savefig(abs_path, dpi=dpi)
        print(f"  -> Saved figure to: {filepath}")
    except OSError:
        base, ext = os.path.splitext(abs_path)
        alt_path = f"{base}_updated{ext}"
        try:
            if os.path.exists(alt_path):
                try:
                    os.chmod(alt_path, stat.S_IWRITE)
                    os.remove(alt_path)
                except Exception:
                    pass
            fig.savefig(alt_path, dpi=dpi)
            print(f"  [!] Note: '{os.path.basename(filepath)}' is currently open/locked in another window (e.g. VS Code preview).")
            print(f"      Saved updated figure to: {os.path.basename(alt_path)}")
        except Exception as e:
            print(f"  [!] Warning: Could not save {filepath}: {e}")


def load_and_preprocess_data(filepath="data/heart_disease.csv"):
    """
    Part 1: Dataset Pipeline (Rohit)
    - Loads dataset from data/heart_disease.csv (or downloads if missing)
    - Handles missing values via median imputation
    - Removes duplicates
    - Standardizes target into binary classification (0 = Healthy, 1 = Disease)
    - Performs 80/20 stratified split & standardizes features with StandardScaler
    """
    print("=" * 70)
    print("PART 1: DATASET INSPECTION & PREPROCESSING (UCI Heart Disease)")
    print("=" * 70)
    
    # 1. Load Dataset
    if not os.path.exists(filepath):
        print(f"[-] File not found locally. Downloading from UCI repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = list(COLUMN_METADATA.keys())
        df = pd.read_csv(url, names=cols, na_values="?")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath, na_values="?")
        
    print(f"[+] Dataset successfully loaded: {filepath}")
    print(f"[+] Dataset dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n")
    
    # 2. Understand Columns & Types
    print("[+] Clinical Features & Column Descriptions:")
    for col in df.columns:
        desc = COLUMN_METADATA.get(col, "Clinical feature")
        print(f"  - {col:10s} | Type: {str(df[col].dtype):8s} | Unique: {df[col].nunique():3d} | {desc}")
        
    print("\n[+] Summary Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])
    
    # 3. Check for Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n[+] Duplicate rows found: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"[+] Dropped duplicates. New shape: {df.shape}")
        
    # 4. Check & Handle Missing Values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n[+] Total missing (NaN) values: {total_nulls}")
    if total_nulls > 0:
        print("[+] Missing values by column:")
        for col, cnt in null_counts[null_counts > 0].items():
            impute_val = df[col].median()
            df[col] = df[col].fillna(impute_val)
            print(f"  - {col}: {cnt} missing values -> Imputed with median ({impute_val})")
            
    # 5. Target Standardization (Binary classification: 0 = Healthy, 1 = Disease)
    target_col = "target" if "target" in df.columns else df.columns[-1]
    df[target_col] = (df[target_col] > 0).astype(int)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\n[+] Target Class Distribution:")
    target_dist = y.value_counts(normalize=True).rename({0: "0 (No Disease)", 1: "1 (Heart Disease)"})
    for k, v in target_dist.items():
        print(f"  - {k}: {v*100:.2f}% ({y.value_counts().loc[int(str(k)[0])]} samples)")
        
    # 6. Train-Test Split (80/20 Stratified Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[+] Stratified Train-Test Split:")
    print(f"  - Training Set: {X_train.shape[0]} samples (80%)")
    print(f"  - Testing Set:  {X_test.shape[0]} samples (20%)")
    
    # 7. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return {
        "df": df,
        "feature_names": list(X.columns),
        "target_name": target_col,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }


def train_and_evaluate(data_dict):
    """
    Part 2 & Part 3: Algorithms (Logistic Regression & Random Forest) + Evaluation (Rohit)
    """
    print("\n" + "=" * 70)
    print("PART 2: ALGORITHMS IMPLEMENTATION (Logistic Regression & Random Forest)")
    print("=" * 70)
    
    X_train_scaled = data_dict["X_train_scaled"]
    X_test_scaled = data_dict["X_test_scaled"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    feature_names = data_dict["feature_names"]
    
    os.makedirs("assets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # ------------------------------------------------------------------
    # Algorithm 1: Logistic Regression (Rohit)
    # ------------------------------------------------------------------
    print("\n[+] 1. Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    lr_train_pred = log_reg.predict(X_train_scaled)
    lr_test_pred = log_reg.predict(X_test_scaled)
    lr_test_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    lr_metrics = {
        "model_name": "Logistic Regression",
        "train_accuracy": float(accuracy_score(y_train, lr_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, lr_test_pred)),
        "precision": float(precision_score(y_test, lr_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, lr_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, lr_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, lr_test_proba))
    }
    
    # ------------------------------------------------------------------
    # Algorithm 2: Random Forest Classifier (Rohit)
    # ------------------------------------------------------------------
    print("[+] 2. Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    rf_clf.fit(X_train_scaled, y_train)
    
    rf_train_pred = rf_clf.predict(X_train_scaled)
    rf_test_pred = rf_clf.predict(X_test_scaled)
    rf_test_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]
    
    rf_metrics = {
        "model_name": "Random Forest Classifier",
        "train_accuracy": float(accuracy_score(y_train, rf_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, rf_test_pred)),
        "precision": float(precision_score(y_test, rf_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, rf_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, rf_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, rf_test_proba))
    }
    
    # ------------------------------------------------------------------
    # Algorithm 3: Gaussian Naive Bayes (Om)
    # ------------------------------------------------------------------
    print("[+] 3. Training Gaussian Naive Bayes (Om)...")
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)
    
    nb_train_pred = gnb.predict(X_train_scaled)
    nb_test_pred = gnb.predict(X_test_scaled)
    nb_test_proba = gnb.predict_proba(X_test_scaled)[:, 1]
    
    nb_metrics = {
        "model_name": "Gaussian Naive Bayes",
        "train_accuracy": float(accuracy_score(y_train, nb_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, nb_test_pred)),
        "precision": float(precision_score(y_test, nb_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, nb_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, nb_test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, nb_test_proba))
    }

    print("\n" + "=" * 70)
    print("PART 3: EVALUATION & COMPARISON (3-MODEL MULTI-ALGORITHM BENCHMARK)")
    print("=" * 70)
    
    # 1. Performance Comparison Table
    print("\n[+] Multi-Model Performance Comparison Table:")
    print("-" * 105)
    print(f"{'Metric':<22} | {'Logistic Regression':<22} | {'Gaussian Naive Bayes (Om)':<26} | {'Random Forest':<22}")
    print("-" * 105)
    print(f"{'Train Accuracy':<22} | {lr_metrics['train_accuracy']*100:>20.2f}% | {nb_metrics['train_accuracy']*100:>24.2f}% | {rf_metrics['train_accuracy']*100:>20.2f}%")
    print(f"{'Test Accuracy':<22} | {lr_metrics['test_accuracy']*100:>20.2f}% | {nb_metrics['test_accuracy']*100:>24.2f}% | {rf_metrics['test_accuracy']*100:>20.2f}%")
    print(f"{'Precision (Sensitivity)':<22} | {lr_metrics['precision']*100:>20.2f}% | {nb_metrics['precision']*100:>24.2f}% | {rf_metrics['precision']*100:>20.2f}%")
    print(f"{'Recall':<22} | {lr_metrics['recall']*100:>20.2f}% | {nb_metrics['recall']*100:>24.2f}% | {rf_metrics['recall']*100:>20.2f}%")
    print(f"{'F1-Score':<22} | {lr_metrics['f1_score']*100:>20.2f}% | {nb_metrics['f1_score']*100:>24.2f}% | {rf_metrics['f1_score']*100:>20.2f}%")
    print(f"{'ROC-AUC Score':<22} | {lr_metrics['roc_auc']:>21.4f} | {nb_metrics['roc_auc']:>25.4f} | {rf_metrics['roc_auc']:>21.4f}")
    print("-" * 105)
    
    # 2. Classification Reports
    print("\n[+] Classification Report: Logistic Regression")
    print(classification_report(y_test, lr_test_pred, target_names=["No Disease", "Heart Disease"]))

    print("[+] Classification Report: Gaussian Naive Bayes (Om)")
    print(classification_report(y_test, nb_test_pred, target_names=["No Disease", "Heart Disease"]))
    
    print("[+] Classification Report: Random Forest Classifier")
    print(classification_report(y_test, rf_test_pred, target_names=["No Disease", "Heart Disease"]))
    
    # 3. Confusion Matrix Visualizations (3 Models)
    print("[+] Generating & Saving 3-Model Confusion Matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    cm_lr = confusion_matrix(y_test, lr_test_pred)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=["No Disease", "Heart Disease"])
    disp_lr.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title(f"Logistic Regression\nAccuracy: {lr_metrics['test_accuracy']*100:.1f}% | F1: {lr_metrics['f1_score']*100:.1f}%", fontsize=11, fontweight="bold")
    axes[0].grid(False)

    cm_nb = confusion_matrix(y_test, nb_test_pred)
    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=["No Disease", "Heart Disease"])
    disp_nb.plot(ax=axes[1], cmap="Purples", values_format="d")
    axes[1].set_title(f"Gaussian Naive Bayes (Om)\nAccuracy: {nb_metrics['test_accuracy']*100:.1f}% | Recall: {nb_metrics['recall']*100:.1f}%", fontsize=11, fontweight="bold")
    axes[1].grid(False)
    
    cm_rf = confusion_matrix(y_test, rf_test_pred)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["No Disease", "Heart Disease"])
    disp_rf.plot(ax=axes[2], cmap="Greens", values_format="d")
    axes[2].set_title(f"Random Forest Classifier\nAccuracy: {rf_metrics['test_accuracy']*100:.1f}% | F1: {rf_metrics['f1_score']*100:.1f}%", fontsize=11, fontweight="bold")
    axes[2].grid(False)
    
    plt.tight_layout()
    cm_path = os.path.join("assets", "confusion_matrix.png")
    save_figure_safely(fig, cm_path, dpi=300)
    plt.close(fig)
    
    # 4. ROC Curves (3 Models)
    print("[+] Generating & Saving 3-Model ROC-AUC Curves...")
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_proba)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_test_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_proba)
    
    fig_roc = plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.3f})", color="#2b6cb0", lw=2)
    plt.plot(fpr_nb, tpr_nb, label=f"Gaussian Naive Bayes (AUC = {nb_metrics['roc_auc']:.3f})", color="#805ad5", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_metrics['roc_auc']:.3f})", color="#2f855a", lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves Comparison for Heart Disease Diagnosis", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join("assets", "roc_curve.png")
    save_figure_safely(fig_roc, roc_path, dpi=300)
    plt.close(fig_roc)
    
    # 5. Feature Importance Plot
    print("[+] Generating & Saving Feature Importance Plot...")
    feat_importances = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig_fi = plt.figure(figsize=(9, 5))
    feat_importances.plot(kind="barh", color="#3182ce")
    plt.title("Clinical Biomarker Importance (Random Forest)", fontweight="bold")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    fi_path = os.path.join("assets", "feature_importance.png")
    save_figure_safely(fig_fi, fi_path, dpi=300)
    plt.close(fig_fi)

    # 6. Generate & Save Om's 4 EDA Distribution Graphs to assets/
    print("[+] Exporting Om's 4 EDA Distribution Graphs to assets/...")
    df_raw = data_dict["df"]
    
    # Graph 1: Age Distribution
    fig_g1 = plt.figure(figsize=(8, 5))
    plt.hist(df_raw['age'], bins=20, color='#2b6cb0', edgecolor='white', alpha=0.75)
    mean_age = df_raw['age'].mean()
    plt.axvline(mean_age, color='#c53030', linestyle='--', linewidth=2, label=f'Mean Age ({mean_age:.1f} yrs)')
    plt.title('Patient Age Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Patient Count')
    plt.legend()
    plt.tight_layout()
    save_figure_safely(fig_g1, os.path.join("assets", "eda_age_distribution.png"), dpi=300)
    plt.close(fig_g1)

    # Graph 2: Heart Disease vs Age
    fig_g2 = plt.figure(figsize=(8, 5))
    healthy_ages = df_raw[df_raw['target'] == 0]['age']
    disease_ages = df_raw[df_raw['target'] > 0]['age']
    plt.boxplot([healthy_ages, disease_ages])
    plt.xticks([1, 2], ['No Disease (0)', 'Heart Disease (1)'])
    plt.title('Heart Disease vs Patient Age', fontsize=13, fontweight='bold')
    plt.xlabel('Diagnosis Target')
    plt.ylabel('Age (years)')
    plt.tight_layout()
    save_figure_safely(fig_g2, os.path.join("assets", "eda_disease_vs_age.png"), dpi=300)
    plt.close(fig_g2)

    # Graph 3: Serum Cholesterol Distribution
    fig_g3 = plt.figure(figsize=(8, 5))
    plt.hist(df_raw['chol'], bins=25, color='#319795', edgecolor='white', alpha=0.75)
    plt.axvline(200, color='#e53e3e', linestyle='--', linewidth=2, label='Threshold (200 mg/dl)')
    plt.title('Serum Cholesterol Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Cholesterol (mg/dl)')
    plt.ylabel('Patient Count')
    plt.legend()
    plt.tight_layout()
    save_figure_safely(fig_g3, os.path.join("assets", "eda_cholesterol_distribution.png"), dpi=300)
    plt.close(fig_g3)

    # Graph 4: Target Class Distribution
    fig_g4 = plt.figure(figsize=(6, 4))
    plt.bar(['No Disease (0)', 'Heart Disease (1)'], [len(healthy_ages), len(disease_ages)], color=['#4299e1', '#f56565'], edgecolor='white')
    plt.title('Target Class Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Diagnosis')
    plt.ylabel('Patient Count')
    plt.tight_layout()
    save_figure_safely(fig_g4, os.path.join("assets", "eda_target_distribution.png"), dpi=300)
    plt.close(fig_g4)
    print("  -> Saved all 4 EDA graphs to assets/ successfully.")

    # 7. Select Best Model
    all_models = [
        ("Random Forest Classifier", rf_clf, rf_metrics),
        ("Logistic Regression", log_reg, lr_metrics),
        ("Gaussian Naive Bayes", gnb, nb_metrics)
    ]
    # Sort by F1-Score
    all_models.sort(key=lambda x: x[2]["f1_score"], reverse=True)
    best_name, best_model, best_metrics = all_models[0]
        
    print("\n" + "=" * 70)
    print(f"[+] BEST MODEL SELECTED: {best_name}")
    print(f"    - Test Accuracy : {best_metrics['test_accuracy']*100:.2f}%")
    print(f"    - F1-Score      : {best_metrics['f1_score']*100:.2f}%")
    print(f"    - Recall        : {best_metrics['recall']*100:.2f}%")
    print(f"    - ROC-AUC Score : {best_metrics['roc_auc']:.4f}")
    print("=" * 70)
    
    # 8. Model Serialization (Part 4 Integration Artifacts)
    joblib.dump(log_reg, os.path.join("models", "logistic_regression.pkl"))
    joblib.dump(rf_clf, os.path.join("models", "random_forest.pkl"))
    joblib.dump(gnb, os.path.join("models", "naive_bayes.pkl"))
    joblib.dump(best_model, os.path.join("models", "best_model.pkl"))
    joblib.dump(data_dict["scaler"], os.path.join("models", "scaler.pkl"))
    
    metrics_export = {
        "best_model_name": best_name,
        "feature_names": feature_names,
        "models": {
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics,
            "gaussian_naive_bayes": nb_metrics
        },
        "best_metrics": best_metrics
    }
    
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_export, f, indent=4)
        
    print(f"\n[+] Saved models to models/ directory (logistic_regression.pkl, random_forest.pkl, naive_bayes.pkl, best_model.pkl, scaler.pkl)")
    print(f"[+] Saved evaluation metrics to: {metrics_path}")
    
    return metrics_export


if __name__ == "__main__":
    data_dict = load_and_preprocess_data("data/heart_disease.csv")
    metrics_summary = train_and_evaluate(data_dict)
    print("\n[SUCCESS] ML Pipeline Execution Completed Successfully!")
