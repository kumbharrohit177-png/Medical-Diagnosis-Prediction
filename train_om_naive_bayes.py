"""
Medical Diagnosis Prediction - Gaussian Naive Bayes Pipeline Runner
Lead: Om (EDA & Naive Bayes Lead)

This script trains Gaussian Naive Bayes using the standardized features,
evaluates performance metrics, displays the classification report,
and exports the model artifact to models/naive_bayes.pkl.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


def save_figure_safely(fig, filepath, dpi=300):
    """
    Safely saves a matplotlib figure, handling Windows file locks and permission attributes.
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
            print(f"  [!] Note: '{os.path.basename(filepath)}' is currently open/locked in an image previewer.")
            print(f"      Saved updated figure to: {os.path.basename(alt_path)}")
        except Exception as e:
            print(f"  [!] Warning: Could not save {filepath}: {e}")


def main():
    print("=" * 65)
    print("OM'S ML PIPELINE: GAUSSIAN NAIVE BAYES TRAINING & EXPORT")
    print("=" * 65)

    # 1. Locate dataset
    data_path = "data/heart_disease.csv" if os.path.exists("data/heart_disease.csv") else "../data/heart_disease.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset at {data_path}")

    df = pd.read_csv(data_path)
    print(f"[+] Loaded dataset: {data_path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # 2. Impute missing values with column medians (consistent with central pipeline)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        for col in null_counts[null_counts > 0].index:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"    - Imputed missing values in '{col}' with median: {med}")

    # 3. Standardize target to binary (0 = Healthy, 1 = Heart Disease)
    target_col = "target" if "target" in df.columns else df.columns[-1]
    df[target_col] = (df[target_col] > 0).astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Split: {len(X_train)} Train / {len(X_test)} Test samples")

    # 5. Load pre-fitted StandardScaler
    scaler_path = "models/scaler.pkl" if os.path.exists("models/scaler.pkl") else "../models/scaler.pkl"
    scaler = joblib.load(scaler_path)
    print(f"[+] Loaded StandardScaler from: {scaler_path}")

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # 6. Train Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train)
    print("[+] Gaussian Naive Bayes fitted successfully!")

    # 7. Evaluate on test set
    y_train_pred = gnb.predict(X_train_scaled)
    y_test_pred = gnb.predict(X_test_scaled)
    y_test_proba = gnb.predict_proba(X_test_scaled)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    print("\n" + "=" * 65)
    print("GAUSSIAN NAIVE BAYES EVALUATION RESULTS")
    print("=" * 65)
    print(f"Train Accuracy:        {train_acc * 100:.2f}%")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Precision:             {prec * 100:.2f}%")
    print(f"Recall (Sensitivity):  {rec * 100:.2f}%")
    print(f"F1-Score:              {f1 * 100:.2f}%")
    print(f"ROC-AUC Score:         {roc_auc:.4f}")
    print("=" * 65)
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["No Disease", "Heart Disease"]))

    # 8. Export model artifact
    models_dir = "models" if os.path.exists("models") else "../models"
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "naive_bayes.pkl")
    joblib.dump(gnb, out_path)
    print(f"\n[+] Saved model artifact to: {out_path}")

    # 9. Update metrics.json
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        
        metrics_data.setdefault("models", {})["gaussian_naive_bayes"] = {
            "model_name": "Gaussian Naive Bayes",
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"[+] Updated central metrics in: {metrics_path}")

    # 10. Export Om's 4 EDA Distribution Graphs to assets/
    assets_dir = "assets" if os.path.exists("assets") else "../assets"
    os.makedirs(assets_dir, exist_ok=True)
    print("\n[+] Generating and saving Om's 4 EDA Distribution Graphs to assets/...")

    # Graph 1: Age Distribution
    fig_g1 = plt.figure(figsize=(8, 5))
    plt.hist(df['age'], bins=20, color='#2b6cb0', edgecolor='white', alpha=0.75)
    mean_age = df['age'].mean()
    plt.axvline(mean_age, color='#c53030', linestyle='--', linewidth=2, label=f'Mean Age ({mean_age:.1f} yrs)')
    plt.title('Patient Age Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Patient Count')
    plt.legend()
    plt.tight_layout()
    save_figure_safely(fig_g1, os.path.join(assets_dir, "eda_age_distribution.png"), dpi=300)
    plt.close(fig_g1)

    # Graph 2: Heart Disease vs Age
    fig_g2 = plt.figure(figsize=(8, 5))
    healthy_ages = df[df['target'] == 0]['age']
    disease_ages = df[df['target'] > 0]['age']
    plt.boxplot([healthy_ages, disease_ages])
    plt.xticks([1, 2], ['No Disease (0)', 'Heart Disease (1)'])
    plt.title('Heart Disease vs Patient Age', fontsize=13, fontweight='bold')
    plt.xlabel('Diagnosis Target')
    plt.ylabel('Age (years)')
    plt.tight_layout()
    save_figure_safely(fig_g2, os.path.join(assets_dir, "eda_disease_vs_age.png"), dpi=300)
    plt.close(fig_g2)

    # Graph 3: Serum Cholesterol Distribution
    fig_g3 = plt.figure(figsize=(8, 5))
    plt.hist(df['chol'], bins=25, color='#319795', edgecolor='white', alpha=0.75)
    plt.axvline(200, color='#e53e3e', linestyle='--', linewidth=2, label='Threshold (200 mg/dl)')
    plt.title('Serum Cholesterol Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Cholesterol (mg/dl)')
    plt.ylabel('Patient Count')
    plt.legend()
    plt.tight_layout()
    save_figure_safely(fig_g3, os.path.join(assets_dir, "eda_cholesterol_distribution.png"), dpi=300)
    plt.close(fig_g3)

    # Graph 4: Target Class Distribution
    fig_g4 = plt.figure(figsize=(6, 4))
    plt.bar(['No Disease (0)', 'Heart Disease (1)'], [len(healthy_ages), len(disease_ages)], color=['#4299e1', '#f56565'], edgecolor='white')
    plt.title('Target Class Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Diagnosis')
    plt.ylabel('Patient Count')
    plt.tight_layout()
    save_figure_safely(fig_g4, os.path.join(assets_dir, "eda_target_distribution.png"), dpi=300)
    plt.close(fig_g4)

    print("  -> Saved all 4 EDA graphs into assets/ folder successfully!")

    print("\n[SUCCESS] Om's Naive Bayes model pipeline completed successfully!")

if __name__ == "__main__":
    main()
