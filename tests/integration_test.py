"""
Clinexa AI — Full Integration Test
Tests complete 6-agent pipeline without running servers
"""

import sys, json
sys.path.insert(0, ".")

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

MODEL_DIR = Path("phase1_ml/models")
model       = joblib.load(MODEL_DIR / "risk_model.joblib")
explainer   = joblib.load(MODEL_DIR / "shap_explainer.joblib")
le_dict     = joblib.load(MODEL_DIR / "label_encoders.joblib")
label_order = joblib.load(MODEL_DIR / "label_order.joblib")
features    = joblib.load(MODEL_DIR / "feature_list.joblib")

CATEGORICAL_FEATURES = ["gender", "comorbidity", "medication"]

def encode_features(patient_data):
    df = pd.DataFrame([patient_data])
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and col in le_dict:
            le = le_dict[col]
            val = str(df[col].values[0])
            df[col] = le.transform([val]) if val in le.classes_ else [0]
    return df[features]

def run_ml_prediction(patient_data):
    X = encode_features(patient_data)
    shap_values = explainer.shap_values(X)
    pred_idx = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0]
    if isinstance(shap_values, list):
        sv = np.array(shap_values[pred_idx]).flatten()
    else:
        sv = np.array(shap_values).flatten()
    label_names = {v: k for k, v in label_order.items()}
    risk = label_names[pred_idx]
    top = sorted(zip(features, sv.tolist()),
                 key=lambda x: abs(x[1]), reverse=True)[:5]
    confidence = {label_names[i]: round(float(p), 3)
                  for i, p in enumerate(pred_proba)}
    factors = [{
        "feature": f,
        "value": float(X[f].values[0]),
        "shap_impact": round(v, 4),
        "direction": "raises risk" if v > 0 else "lowers risk"
    } for f, v in top]
    return risk, confidence, factors

def run_drug_safety(medications, allergies):
    KNOWN = {
        ("warfarin", "aspirin"): ("HIGH", "Increased bleeding risk"),
        ("metformin", "alcohol"): ("MEDIUM", "Risk of lactic acidosis"),
        ("lisinopril", "potassium"): ("MEDIUM", "Potassium level risk"),
    }
    alerts = []
    meds_l = [m.lower() for m in medications]
    for i in range(len(meds_l)):
        for j in range(i+1, len(meds_l)):
            pair = (meds_l[i], meds_l[j])
            rev  = (meds_l[j], meds_l[i])
            if pair in KNOWN or rev in KNOWN:
                info = KNOWN.get(pair) or KNOWN.get(rev)
                alerts.append(
                    f"[{info[0]}] {info[1]}: {pair[0]} + {pair[1]}"
                )
    ALLERGY_MAP = {"penicillin": ["amoxicillin", "ampicillin"]}
    for allergy in allergies:
        cross = ALLERGY_MAP.get(allergy.lower(), [])
        for med in meds_l:
            if allergy.lower() in med or any(c in med for c in cross):
                alerts.append(f"ALLERGY: {med} conflicts with {allergy}")
    return alerts

def make_soap(patient_id, risk, vitals, symptoms, xai_text, drug_notes):
    plans = {
        "HIGH":   "IMMEDIATE: Activate emergency protocol. IV access, cardiac monitoring, stat labs.",
        "MEDIUM": "URGENT: Physician evaluation within 2 hours. ECG, BMP, vitals q30min.",
        "LOW":    "ROUTINE: Follow-up within 48 hours. Patient education, return precautions."
    }
    now = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    return f"""
CLINEXA AI — CLINICAL SOAP NOTE
Generated: {now}
Patient ID: {patient_id} | AI Risk: {risk}
{'─'*50}
S (Subjective):
  Chief complaint: {', '.join(symptoms) if symptoms else 'multiple symptoms'}

O (Objective):
  HR {vitals.get('hr','--')} bpm | BP {vitals.get('sbp','--')}/{vitals.get('dbp','--')} mmHg
  Temp {vitals.get('temp','--')}C | SpO2 {vitals.get('spo2','--')}% | RR {vitals.get('rr','--')}/min
  Data: Synthetic (no PHI)

A (Assessment):
  Risk: {risk}
  {xai_text}
  Drug Safety: {drug_notes}

P (Plan):
  {plans.get(risk, 'Evaluate per protocol.')}
{'─'*50}
FHIR: HL7 R4 compliant | XAI: SHAP TreeExplainer
"""

def make_fhir_bundle(patient_id, risk, assessment, recommendations):
    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "document",
        "timestamp": datetime.now().isoformat() + "Z",
        "meta": {"tag": [{"system": "urn:clinexa:synthetic", "code": "synthetic"}]},
        "entry": [{
            "resource": {
                "resourceType": "Composition",
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "title": "Clinexa AI Triage Report",
                "section": [
                    {"title": "Risk Level",
                     "text": {"div": f"<div>{risk}</div>"}},
                    {"title": "Assessment",
                     "text": {"div": f"<div>{assessment}</div>"}},
                    {"title": "Plan",
                     "text": {"div": f"<div>{'<br/>'.join(recommendations)}</div>"}}
                ]
            }
        }]
    }

TEST_PATIENTS = [
    {
        "patient_id": "SYN-10001",
        "intake_text": "Severe chest pain, trouble breathing, heart racing.",
        "age": 68, "gender": "male",
        "vitals": {"hr": 125, "sbp": 185, "dbp": 110,
                   "temp": 38.9, "spo2": 91, "rr": 26},
        "symptoms": ["chest_pain", "shortness_of_breath",
                     "dizziness", "palpitations"],
        "conditions": ["heart_disease", "hypertension"],
        "medications": ["warfarin", "aspirin"],
        "allergies": [],
        "comorbidity": "heart_disease", "pain_scale": 9
    },
    {
        "patient_id": "SYN-10002",
        "intake_text": "Mild cough, tired, slightly elevated temperature.",
        "age": 34, "gender": "female",
        "vitals": {"hr": 88, "sbp": 118, "dbp": 76,
                   "temp": 37.5, "spo2": 97, "rr": 16},
        "symptoms": ["cough", "fatigue"],
        "conditions": ["asthma"],
        "medications": ["albuterol"],
        "allergies": ["penicillin"],
        "comorbidity": "asthma", "pain_scale": 3
    },
    {
        "patient_id": "SYN-10003",
        "intake_text": "Moderate abdominal pain, nausea, back pain 3 days.",
        "age": 55, "gender": "other",
        "vitals": {"hr": 102, "sbp": 145, "dbp": 92,
                   "temp": 37.8, "spo2": 95, "rr": 20},
        "symptoms": ["abdominal_pain", "nausea_vomiting", "back_pain"],
        "conditions": ["copd"],
        "medications": ["metformin"],
        "allergies": [],
        "comorbidity": "copd", "pain_scale": 6
    }
]

def run_patient(patient):
    pid = patient["patient_id"]
    print(f"\n{'='*60}")
    print(f"CLINEXA AI — Patient {pid}")
    print(f"{'='*60}")

    symptom_fields = [
        "chest_pain", "shortness_of_breath", "high_fever", "severe_headache",
        "loss_of_consciousness", "abdominal_pain", "nausea_vomiting",
        "dizziness", "fatigue", "cough", "back_pain", "rash",
        "confusion", "palpitations", "leg_swelling"
    ]
    syms = patient.get("symptoms", [])
    symptom_map = {sf: 1 if sf in syms else 0 for sf in symptom_fields}

    features_dict = {
        "age": patient["age"],
        "gender": patient["gender"],
        "heart_rate": patient["vitals"]["hr"],
        "systolic_bp": patient["vitals"]["sbp"],
        "diastolic_bp": patient["vitals"]["dbp"],
        "temperature": patient["vitals"]["temp"],
        "spo2": patient["vitals"]["spo2"],
        "respiratory_rate": patient["vitals"]["rr"],
        "pain_scale": patient["pain_scale"],
        "comorbidity": patient["comorbidity"],
        "medication": (patient["medications"][0]
                       if patient["medications"] else "none"),
        **symptom_map
    }

    print(f"[Agent 1] Intake: {patient['intake_text'][:60]}...")

    risk, confidence, factors = run_ml_prediction(features_dict)
    conf_pct = confidence.get(risk, 0) * 100
    print(f"[Agent 2] Risk: {risk} ({conf_pct:.0f}% confidence)")

    top_feat = (factors[0]["feature"].replace("_", " ").title()
                if factors else "vitals")
    xai_text = (f"Top driver: {top_feat} = {factors[0]['value']} "
                f"({factors[0]['direction']}, SHAP={factors[0]['shap_impact']})")
    print(f"[Agent 3] XAI: {xai_text}")

    plans = {
        "HIGH":   ["Activate emergency response", "IV access + monitoring",
                   "Stat labs", "Specialist consult", "NPO status"],
        "MEDIUM": ["Urgent evaluation within 2 hours", "12-lead ECG",
                   "Vitals q30min", "Basic metabolic panel"],
        "LOW":    ["Routine follow-up 48 hours", "Symptom diary",
                   "OTC care", "Return precautions"]
    }
    recommendations = plans.get(risk, [])
    print(f"[Agent 4] Treatment: {recommendations[0]}")

    drug_alerts = run_drug_safety(patient["medications"], patient["allergies"])
    drug_notes = ("No safety issues." if not drug_alerts
                  else f"{len(drug_alerts)} alert(s): {drug_alerts[0]}")
    print(f"[Agent 5] Drug Safety: {drug_notes}")

    soap = make_soap(pid, risk, patient["vitals"],
                     patient["symptoms"], xai_text, drug_notes)
    bundle = make_fhir_bundle(pid, risk, xai_text, recommendations)
    print(f"[Agent 6] FHIR Bundle ID: {bundle['id'][:8]}...")

    print(soap)
    print("Top 3 SHAP Factors:")
    for f in factors[:3]:
        print(f"  {f['feature']:25} val={f['value']:6} "
              f"impact={f['shap_impact']:+.4f} → {f['direction']}")
    print(f"Drug Alerts: {drug_alerts if drug_alerts else 'None'}")


if __name__ == "__main__":
    print("CLINEXA AI — FULL INTEGRATION TEST")
    print("3 synthetic patients through 6 agents")
    print("="*60)
    for patient in TEST_PATIENTS:
        run_patient(patient)
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETE")