"""
Clinexa AI - Synthetic Patient Data Generator
Generates HIPAA-safe synthetic triage data (NO real PHI)
"""

import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

SYMPTOMS = {
    "chest_pain":        {"weight": 0.9, "category": "cardiac"},
    "shortness_of_breath": {"weight": 0.85, "category": "respiratory"},
    "high_fever":        {"weight": 0.7,  "category": "infection"},
    "severe_headache":   {"weight": 0.75, "category": "neuro"},
    "loss_of_consciousness": {"weight": 0.95, "category": "neuro"},
    "abdominal_pain":    {"weight": 0.6,  "category": "gi"},
    "nausea_vomiting":   {"weight": 0.4,  "category": "gi"},
    "dizziness":         {"weight": 0.5,  "category": "neuro"},
    "fatigue":           {"weight": 0.3,  "category": "general"},
    "cough":             {"weight": 0.35, "category": "respiratory"},
    "back_pain":         {"weight": 0.4,  "category": "musculo"},
    "rash":              {"weight": 0.3,  "category": "derma"},
    "confusion":         {"weight": 0.8,  "category": "neuro"},
    "palpitations":      {"weight": 0.65, "category": "cardiac"},
    "leg_swelling":      {"weight": 0.55, "category": "vascular"},
}

COMORBIDITIES = [
    "diabetes", "hypertension", "heart_disease",
    "asthma", "copd", "kidney_disease", "cancer", "none"
]

MEDICATIONS = [
    "metformin", "lisinopril", "atorvastatin",
    "aspirin", "albuterol", "none", "warfarin", "insulin"
]

def compute_risk_score(row):
    score = 0.0
    if row["heart_rate"] > 120 or row["heart_rate"] < 50:
        score += 0.25
    if row["systolic_bp"] > 180 or row["systolic_bp"] < 90:
        score += 0.25
    if row["temperature"] > 39.5 or row["temperature"] < 35.5:
        score += 0.15
    if row["spo2"] < 92:
        score += 0.3
    if row["respiratory_rate"] > 25 or row["respiratory_rate"] < 10:
        score += 0.2
    for sym, meta in SYMPTOMS.items():
        if row.get(sym, 0) == 1:
            score += meta["weight"] * 0.15
    if row["age"] > 70:
        score += 0.15
    elif row["age"] > 60:
        score += 0.08
    if row["comorbidity"] in ["heart_disease", "cancer", "kidney_disease"]:
        score += 0.2
    elif row["comorbidity"] in ["diabetes", "copd"]:
        score += 0.1
    score += (row["pain_scale"] / 10.0) * 0.2
    return min(score, 1.0)

def assign_risk_label(score):
    if score >= 0.65:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"

def generate_patient_id():
    return f"SYN-{random.randint(10000, 99999)}"

def generate_synthetic_dataset(n=1000):
    records = []
    for _ in range(n):
        age = int(np.random.normal(50, 18))
        age = max(18, min(95, age))
        heart_rate = int(np.random.normal(80, 20))
        systolic_bp = int(np.random.normal(125, 25))
        diastolic_bp = int(systolic_bp * 0.65 + np.random.normal(0, 5))
        temperature = round(np.random.normal(37.0, 0.8), 1)
        spo2 = int(np.random.normal(97, 3))
        spo2 = max(80, min(100, spo2))
        respiratory_rate = int(np.random.normal(16, 4))
        pain_scale = random.randint(0, 10)
        symptom_values = {}
        for sym in SYMPTOMS:
            prob = 0.2 if pain_scale < 5 else 0.45
            symptom_values[sym] = int(random.random() < prob)
        comorbidity = random.choice(COMORBIDITIES)
        medication = random.choice(MEDICATIONS)
        gender = random.choice(["M", "F", "Other"])
        row = {
            "patient_id": generate_patient_id(),
            "age": age,
            "gender": gender,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "temperature": temperature,
            "spo2": spo2,
            "respiratory_rate": respiratory_rate,
            "pain_scale": pain_scale,
            "comorbidity": comorbidity,
            "medication": medication,
            **symptom_values,
        }
        score = compute_risk_score(row)
        row["risk_score"] = round(score, 4)
        row["risk_label"] = assign_risk_label(score)
        records.append(row)
    df = pd.DataFrame(records)
    print(f"Dataset generated: {len(df)} records")
    print(f"Risk distribution:\n{df['risk_label'].value_counts()}")
    return df

if __name__ == "__main__":
    df = generate_synthetic_dataset(1000)
    df.to_csv("data/synthetic_triage.csv", index=False)
    print("\nSaved to data/synthetic_triage.csv")
    print(df.head(3).to_string())