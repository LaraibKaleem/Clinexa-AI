"""
Clinexa AI — Phase 1: ML Risk Prediction Model
Trains RandomForest on synthetic triage data + SHAP explainability
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(".")
DATA_PATH = BASE / "data" / "synthetic_triage.csv"
MODEL_DIR = BASE / "phase1_ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VITAL_FEATURES = [
    "age", "heart_rate", "systolic_bp", "diastolic_bp",
    "temperature", "spo2", "respiratory_rate", "pain_scale"
]

SYMPTOM_FEATURES = [
    "chest_pain", "shortness_of_breath", "high_fever", "severe_headache",
    "loss_of_consciousness", "abdominal_pain", "nausea_vomiting",
    "dizziness", "fatigue", "cough", "back_pain", "rash",
    "confusion", "palpitations", "leg_swelling"
]

CATEGORICAL_FEATURES = ["gender", "comorbidity", "medication"]
ALL_FEATURES = VITAL_FEATURES + SYMPTOM_FEATURES + CATEGORICAL_FEATURES


def prepare_data(df):
    df = df.copy()
    le_dict = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    X = df[ALL_FEATURES]
    y = df["risk_label"]
    label_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    y_encoded = y.map(label_order)
    return X, y_encoded, le_dict, label_order


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_order):
    y_pred = model.predict(X_test)
    label_names = {v: k for k, v in label_order.items()}
    target_names = [label_names[i] for i in sorted(label_names)]
    print("\n" + "="*50)
    print("CLINEXA AI — MODEL EVALUATION")
    print("="*50)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="accuracy")
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    importances = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)
    print(f"\nTop 10 Feature Importances:")
    print(importances.head(10).to_string())
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "cv_mean": round(cv_scores.mean(), 4),
    }


def explain_single_patient(explainer, model, X_sample, label_order, feature_names):
    shap_values = explainer.shap_values(X_sample)
    pred_class_idx = int(model.predict(X_sample)[0])
    pred_proba = model.predict_proba(X_sample)[0]
    if isinstance(shap_values, list):
        sv = np.array(shap_values[pred_class_idx]).flatten()
    else:
        sv = np.array(shap_values).flatten()
    label_names = {v: k for k, v in label_order.items()}
    risk_label = label_names[pred_class_idx]
    sv_floats = sv.tolist()
    feature_contributions = sorted(
        zip(feature_names, sv_floats),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_factors = []
    for feat, val in feature_contributions[:5]:
        actual_val = float(X_sample[feat].values[0])
        direction = "increases" if val > 0 else "decreases"
        top_factors.append({
            "feature": feat,
            "value": actual_val,
            "shap_impact": round(float(val), 4),
            "direction": direction,
            "explanation": f"{feat} = {actual_val} {direction} risk"
        })
    confidence = {
        label_names[i]: round(float(p), 4)
        for i, p in enumerate(pred_proba)
    }
    return {
        "predicted_risk": risk_label,
        "confidence": confidence,
        "top_factors": top_factors,
    }


def predict_patient(model, explainer, le_dict, label_order, patient_data: dict):
    df_input = pd.DataFrame([patient_data])
    for col in CATEGORICAL_FEATURES:
        if col in df_input.columns and col in le_dict:
            le = le_dict[col]
            val = str(df_input[col].values[0])
            if val in le.classes_:
                df_input[col] = le.transform([val])
            else:
                df_input[col] = le.transform([le.classes_[0]])
    X = df_input[ALL_FEATURES]
    return explain_single_patient(explainer, model, X, label_order, ALL_FEATURES)


def main():
    print("="*50)
    print("CLINEXA AI — PHASE 1: ML TRAINING PIPELINE")
    print("="*50)
    print(f"\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records.")
    X, y, le_dict, label_order = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print("\nTraining RandomForest model...")
    model = train_model(X_train, y_train)
    print("Training complete.")
    metrics = evaluate_model(model, X_test, y_test, label_order)
    print("\nBuilding SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    print("\nTesting sample patient explanation...")
    sample = {
        "age": 68, "gender": "M", "heart_rate": 125,
        "systolic_bp": 185, "diastolic_bp": 110,
        "temperature": 38.9, "spo2": 91,
        "respiratory_rate": 26, "pain_scale": 8,
        "comorbidity": "heart_disease", "medication": "aspirin",
        "chest_pain": 1, "shortness_of_breath": 1, "high_fever": 0,
        "severe_headache": 0, "loss_of_consciousness": 0,
        "abdominal_pain": 0, "nausea_vomiting": 0, "dizziness": 1,
        "fatigue": 1, "cough": 0, "back_pain": 0, "rash": 0,
        "confusion": 0, "palpitations": 1, "leg_swelling": 1,
    }
    result = predict_patient(model, explainer, le_dict, label_order, sample)
    print(json.dumps(result, indent=2))
    print("\nSaving model artifacts...")
    joblib.dump(model,     MODEL_DIR / "risk_model.joblib")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.joblib")
    joblib.dump(le_dict,   MODEL_DIR / "label_encoders.joblib")
    joblib.dump(label_order, MODEL_DIR / "label_order.joblib")
    joblib.dump(ALL_FEATURES, MODEL_DIR / "feature_list.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"All artifacts saved to {MODEL_DIR}/")
    print("\nPhase 1 complete!")


if __name__ == "__main__":
    main()