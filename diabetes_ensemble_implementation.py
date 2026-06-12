#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
Dual-Stage Explainable Ensemble Learning Model for Diabetes Diagnosis
=============================================================================
Implementation based on:
  Elgendy et al. (2025) - Expert Systems With Applications 274, 126899
  DOI: https://doi.org/10.1016/j.eswa.2025.126899

Datasets:
  - PID  : Pima Indians Diabetes Database (Kaggle / UCI)
  - MIMIC-IV: Simulated (real dataset is restricted; requires PhysioNet access)

Requirements:
  pip install scikit-learn imbalanced-learn xgboost catboost shap
              pandas numpy matplotlib seaborn reportlab

Author note: MIMIC-IV is simulated here for reproducibility. Download the
  real dataset from https://physionet.org/content/mimiciv/3.1/
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 – Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
    BaggingClassifier, RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap

print("All libraries imported successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 – Load PID Dataset
# ─────────────────────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
df_pid = pd.read_csv('pima_diabetes.csv')   # adjust path as needed
X_pid_raw = df_pid.drop('Outcome', axis=1)
y_pid_raw = df_pid['Outcome']
pid_features = X_pid_raw.columns.tolist()

# Replace biologically impossible zeros with NaN (as in the paper)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X_pid_raw[zero_cols] = X_pid_raw[zero_cols].replace(0, np.nan)

print(f"PID Dataset: {X_pid_raw.shape}")
print("Missing values per feature:")
print(X_pid_raw.isnull().sum())
print(f"\nClass distribution:\n{y_pid_raw.value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 – Simulate MIMIC-IV Dataset
# (Replace this cell with real MIMIC-IV loading if you have PhysioNet access)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N_TOTAL  = 37246
N_DIABETIC = 7472
N_NONDIAB  = N_TOTAL - N_DIABETIC

def simulate_mimic_patients(diabetic: bool, n: int) -> pd.DataFrame:
    """Generate synthetic patients with MIMIC-IV-like feature distributions."""
    age_m  = 65 if diabetic else 55
    sys_m  = 135 if diabetic else 120
    dia_m  = 80  if diabetic else 72
    bmi_m  = 32  if diabetic else 27
    return pd.DataFrame({
        'Gender':        np.random.randint(0, 2, n),
        'Age':           np.clip(np.random.normal(age_m, 15, n), 18, 101).astype(int),
        'Race':          np.random.randint(0, 5, n),
        'Systolic':      np.clip(np.random.normal(sys_m, 20, n), 31, 240),
        'Diastolic':     np.clip(np.random.normal(dia_m, 12, n), 0, 147),
        'BMI':           np.clip(np.random.normal(bmi_m, 6, n), 9.9, 48.4),
        'Hypertension':  np.random.choice([0,1], n, p=[0.35,0.65] if diabetic else [0.75,0.25]),
        'Kidney_Failure':np.random.choice([0,1], n, p=[0.55,0.45] if diabetic else [0.90,0.10]),
        'Pregnant':      np.random.choice([0,1], n, p=[0.85,0.15]),
        'Smoker':        np.random.choice([0,1], n, p=[0.65,0.35]),
        'Diagnosis':     np.ones(n, int) if diabetic else np.zeros(n, int),
    })

df_mimic = pd.concat([
    simulate_mimic_patients(True,  N_DIABETIC),
    simulate_mimic_patients(False, N_NONDIAB),
], ignore_index=True)

# Inject 5% missing values into continuous features
for col in ['Systolic', 'Diastolic', 'BMI']:
    idx = np.random.choice(df_mimic.index, int(0.05 * N_TOTAL), replace=False)
    df_mimic.loc[idx, col] = np.nan

X_mimic_raw = df_mimic.drop('Diagnosis', axis=1)
y_mimic_raw = df_mimic['Diagnosis']
mimic_features = X_mimic_raw.columns.tolist()

print(f"\nMIMIC-IV Simulated Dataset: {X_mimic_raw.shape}")
print(f"Missing values: {X_mimic_raw.isnull().sum().sum()}")
print(f"Class distribution:\n{y_mimic_raw.value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 – Preprocessing Pipeline
# Paper: LOF outlier removal → Autoencoder reconstruction → KNN imputation → SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(X: pd.DataFrame, y: pd.Series,
                        lof_neighbors: int = 30,
                        lof_contamination: float = 0.05,
                        knn_impute_k: int = 5,
                        smote_k: int = 3,
                        dataset_name: str = '') -> tuple:
    """
    Full preprocessing pipeline matching the paper:
      1. KNN impute temporarily for LOF (LOF requires complete data)
      2. Apply LOF outlier detection (n_neighbors=30 per paper)
      3. KNN imputation on cleaned data (autoencoder-equivalent)
      4. SMOTE to balance classes
    Returns: X_balanced, y_balanced
    """
    print(f"\n{'='*55}\nPreprocessing: {dataset_name}")
    print(f"  Input: {X.shape} | Missing: {X.isnull().sum().sum()}")

    # Step 1: Temporary imputation for LOF
    temp_imputer = KNNImputer(n_neighbors=knn_impute_k)
    X_temp = pd.DataFrame(temp_imputer.fit_transform(X), columns=X.columns)

    # Step 2: LOF outlier detection & removal
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=lof_contamination)
    lof_preds = lof.fit_predict(X_temp)
    inlier_mask = lof_preds == 1
    X_clean = X_temp[inlier_mask].reset_index(drop=True)
    y_clean = y[inlier_mask].reset_index(drop=True)
    removed = (~inlier_mask).sum()
    print(f"  LOF removed {removed} outliers → {X_clean.shape}")

    # Step 3: KNN imputation (mimics autoencoder-based reconstruction)
    imputer = KNNImputer(n_neighbors=knn_impute_k)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)

    # Step 4: SMOTE to address class imbalance
    sm = SMOTE(random_state=42, k_neighbors=smote_k)
    X_balanced, y_balanced = sm.fit_resample(X_imputed, y_clean)
    print(f"  After SMOTE: {X_balanced.shape} | Classes: {pd.Series(y_balanced).value_counts().to_dict()}")

    return np.array(X_balanced), np.array(y_balanced)


X_pid, y_pid = preprocess_pipeline(
    X_pid_raw.copy(), y_pid_raw.copy(), dataset_name='PID'
)
X_mimic, y_mimic = preprocess_pipeline(
    X_mimic_raw.copy(), y_mimic_raw.copy(), dataset_name='MIMIC-IV (Simulated)'
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 – Define Base and Meta Models
# Hyperparameters from Table 3 of the paper
# ─────────────────────────────────────────────────────────────────────────────

def get_base_models() -> dict:
    return {
        'AdaBoost':   AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=42),
        'GBoost':     GradientBoostingClassifier(n_estimators=200, learning_rate=0.5,
                                                  max_depth=3, min_samples_leaf=50, random_state=42),
        'HistGBoost': HistGradientBoostingClassifier(max_iter=300, learning_rate=0.5,
                                                      max_depth=3, min_samples_leaf=30, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'CatBoost':   CatBoostClassifier(iterations=300, learning_rate=0.5, depth=5,
                                          random_state=42, verbose=0),
        'XGBoost':    XGBClassifier(n_estimators=300, learning_rate=0.5, max_depth=5,
                                     random_state=42, eval_metric='logloss', verbosity=0),
        'KNN':        KNeighborsClassifier(n_neighbors=3, metric='cityblock'),
    }

def get_meta_models() -> dict:
    return {
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'RF':      RandomForestClassifier(n_estimators=100, max_depth=5,
                                           min_samples_split=10, random_state=42),
        'MLP':     MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                                  solver='adam', learning_rate_init=0.001,
                                  max_iter=500, random_state=42),
        'LR':      LogisticRegression(C=100, penalty='l2', solver='lbfgs',
                                       max_iter=1000, random_state=42),
    }

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 – Stacking Ensemble Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Accuracy':    round(accuracy_score(y_true, y_pred) * 100, 2),
        'F1-Score':    round(f1_score(y_true, y_pred) * 100, 2),
        'AUC':         round(roc_auc_score(y_true, y_prob) * 100, 2),
        'Precision':   round(precision_score(y_true, y_pred) * 100, 2),
        'Sensitivity': round(recall_score(y_true, y_pred) * 100, 2),
        'Specificity': round((tn / (tn + fp)) * 100, 2),
    }

def run_stacking_experiment(X: np.ndarray, y: np.ndarray, dataset_name: str) -> dict:
    """
    Full stacking ensemble pipeline:
      Level 0: 7 base models
      Level 1: 4 meta models trained on base model predictions
    Returns complete results dictionary.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n{'='*60}\n{dataset_name}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    base_models = get_base_models()
    meta_models_dict = get_meta_models()
    results = {}

    # ── Train base models ──
    meta_feats_train = np.zeros((len(X_train), len(base_models)))
    meta_feats_test  = np.zeros((len(X_test),  len(base_models)))

    print("\n[BASE MODELS]")
    for i, (name, model) in enumerate(base_models.items()):
        model.fit(X_train, y_train)
        pred_tr = model.predict(X_train)
        pred_te = model.predict(X_test)
        prob_te = model.predict_proba(X_test)[:, 1]
        meta_feats_train[:, i] = pred_tr
        meta_feats_test[:, i]  = pred_te
        metrics = compute_metrics(y_test, pred_te, prob_te)
        results[name] = metrics
        print(f"  {name:<14} Acc={metrics['Accuracy']}%  F1={metrics['F1-Score']}%  AUC={metrics['AUC']}%")

    # ── Simple Average Ensemble (SAE) ──
    sae_pred = (meta_feats_test.mean(axis=1) >= 0.5).astype(int)
    sae_prob = meta_feats_test.mean(axis=1)
    results['SAE'] = compute_metrics(y_test, sae_pred, sae_prob)
    print(f"  {'SAE':<14} Acc={results['SAE']['Accuracy']}%")

    # ── Train meta models (stacking) ──
    print("\n[STACKING ENSEMBLE]")
    for mname, meta in meta_models_dict.items():
        meta.fit(meta_feats_train, y_train)
        pred_te = meta.predict(meta_feats_test)
        prob_te = meta.predict_proba(meta_feats_test)[:, 1]
        metrics = compute_metrics(y_test, pred_te, prob_te)
        results[f'Stacking {mname}'] = metrics
        print(f"  Stacking {mname:<8} Acc={metrics['Accuracy']}%  F1={metrics['F1-Score']}%  AUC={metrics['AUC']}%")

    return results

results_pid   = run_stacking_experiment(X_pid,   y_pid,   'PID Dataset')
results_mimic = run_stacking_experiment(X_mimic, y_mimic, 'MIMIC-IV Simulated')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 – Results Table
# ─────────────────────────────────────────────────────────────────────────────
def results_to_df(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results).T[['Accuracy','F1-Score','AUC','Precision','Sensitivity','Specificity']]

print("\n\n=== PID RESULTS ===")
print(results_to_df(results_pid).to_string())
print("\n\n=== MIMIC-IV (Simulated) RESULTS ===")
print(results_to_df(results_mimic).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 – Feature Importance (Stage 1 Explainability)
# ─────────────────────────────────────────────────────────────────────────────
X_train_pid, X_test_pid, y_train_pid, y_test_pid = train_test_split(
    X_pid, y_pid, test_size=0.2, random_state=42, stratify=y_pid
)
base_models_fitted = get_base_models()
for m in base_models_fitted.values():
    m.fit(X_train_pid, y_train_pid)

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()
for idx, (name, model) in enumerate(base_models_fitted.items()):
    ax = axes[idx]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        pi = permutation_importance(model, X_test_pid, y_test_pid, n_repeats=5, random_state=42)
        importances = np.clip(pi.importances_mean, 0, None)
    si = np.argsort(importances)
    ax.barh(np.array(pid_features)[si], importances[si], color='steelblue', edgecolor='black', alpha=0.85)
    ax.set_title(name, fontweight='bold', fontsize=10)
    ax.set_xlabel('Importance', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
axes[-1].set_visible(False)
axes[-2].set_visible(False)
plt.suptitle('Feature Importance – All 7 Base Models (PID Dataset)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance figure saved.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 – SHAP Analysis (Stage 2 Explainability)
# SHAP evaluates each base model's contribution to the MLP meta model
# ─────────────────────────────────────────────────────────────────────────────
model_names_list = list(base_models_fitted.keys())

# Build meta feature matrices
meta_tr = np.column_stack([m.predict(X_train_pid) for m in base_models_fitted.values()])
meta_te = np.column_stack([m.predict(X_test_pid)  for m in base_models_fitted.values()])

mlp_meta = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu',
    solver='adam', learning_rate_init=0.001, max_iter=500, random_state=42)
mlp_meta.fit(meta_tr, y_train_pid)

# KernelExplainer for class 1 probability
background = shap.sample(pd.DataFrame(meta_te, columns=model_names_list), 30)
explainer   = shap.KernelExplainer(lambda x: mlp_meta.predict_proba(x)[:, 1], background)
shap_vals   = explainer.shap_values(pd.DataFrame(meta_te[:80], columns=model_names_list),
                                    nsamples=50, silent=True)

mean_shap = np.abs(shap_vals).mean(axis=0)
si = np.argsort(mean_shap)
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#E53935' if i == si[-1] else '#1565C0' for i in range(len(model_names_list))]
bars = ax.barh(np.array(model_names_list)[si], mean_shap[si],
               color=[colors[i] for i in si], edgecolor='black', alpha=0.85)
for bar, val in zip(bars, mean_shap[si]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax.set_title('Base Model Contributions to MLP Meta Model (SHAP)\nRed = Most Influential', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('shap_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP analysis figure saved.")
print("SHAP values:", dict(zip(model_names_list, mean_shap.round(4))))

print("\n=== Implementation Complete ===")
