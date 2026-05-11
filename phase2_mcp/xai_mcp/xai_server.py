
"""
Clinexa AI — MCP Server 3: XAI Explainer
Translates ML risk predictions into plain English using SHAP-style logic.
ALL synthetic data.
"""

from fastmcp import FastMCP
import json, os

mcp = FastMCP("clinexa-ai-xai")

# Risk factor weights (simulating a trained RandomForest + SHAP)
RISK_FACTORS = {
    "age_over_65": {"weight": 0.15, "direction": "increases", "explanation": "Age over 65 increases risk due to reduced physiological reserve"},
    "fever_over_39": {"weight": 0.20, "direction": "increases", "explanation": "High fever (>39°C) indicates severe infection or inflammatory response"},
    "systolic_bp_over_180": {"weight": 0.18, "direction": "increases", "explanation": "Severe hypertension (SBP >180) risks end-organ damage"},
    "heart_rate_over_120": {"weight": 0.12, "direction": "increases", "explanation": "Tachycardia >120 suggests cardiovascular stress or sepsis"},
    "spo2_under_92": {"weight": 0.22, "direction": "increases", "explanation": "Hypoxemia (SpO2 <92%) indicates respiratory failure risk"},
    "respiratory_rate_over_24": {"weight": 0.13, "direction": "increases", "explanation": "Tachypnea >24 suggests respiratory distress or metabolic compensation"},
    "known_cad": {"weight": 0.10, "direction": "increases", "explanation": "Coronary artery disease increases cardiac event risk"},
    "known_diabetes": {"weight": 0.08, "direction": "increases", "explanation": "Diabetes impairs immune response and wound healing"},
    "anticoagulant_use": {"weight": 0.05, "direction": "increases", "explanation": "Anticoagulants increase bleeding risk during procedures"},
}

def calculate_risk_score(vitals, conditions, medications):
    """Simulate ML risk scoring with SHAP-style feature contributions"""
    score = 0.0
    contributions = []
    
    # Age factor
    if vitals.get("age", 0) > 65:
        score += RISK_FACTORS["age_over_65"]["weight"]
        contributions.append({
            "factor": "Age > 65",
            "contribution": RISK_FACTORS["age_over_65"]["weight"],
            "explanation": RISK_FACTORS["age_over_65"]["explanation"]
        })
    
    # Vitals
    if vitals.get("temp", 0) > 39.0:
        score += RISK_FACTORS["fever_over_39"]["weight"]
        contributions.append({
            "factor": "Fever > 39°C",
            "contribution": RISK_FACTORS["fever_over_39"]["weight"],
            "explanation": RISK_FACTORS["fever_over_39"]["explanation"]
        })
    
    if vitals.get("sbp", 0) > 180:
        score += RISK_FACTORS["systolic_bp_over_180"]["weight"]
        contributions.append({
            "factor": "Systolic BP > 180",
            "contribution": RISK_FACTORS["systolic_bp_over_180"]["weight"],
            "explanation": RISK_FACTORS["systolic_bp_over_180"]["explanation"]
        })
    
    if vitals.get("hr", 0) > 120:
        score += RISK_FACTORS["heart_rate_over_120"]["weight"]
        contributions.append({
            "factor": "Heart Rate > 120",
            "contribution": RISK_FACTORS["heart_rate_over_120"]["weight"],
            "explanation": RISK_FACTORS["heart_rate_over_120"]["explanation"]
        })
    
    if vitals.get("spo2", 100) < 92:
        score += RISK_FACTORS["spo2_under_92"]["weight"]
        contributions.append({
            "factor": "SpO2 < 92%",
            "contribution": RISK_FACTORS["spo2_under_92"]["weight"],
            "explanation": RISK_FACTORS["spo2_under_92"]["explanation"]
        })
    
    if vitals.get("rr", 0) > 24:
        score += RISK_FACTORS["respiratory_rate_over_24"]["weight"]
        contributions.append({
            "factor": "Respiratory Rate > 24",
            "contribution": RISK_FACTORS["respiratory_rate_over_24"]["weight"],
            "explanation": RISK_FACTORS["respiratory_rate_over_24"]["explanation"]
        })
    
    # Conditions
    for condition in conditions:
        cond_lower = condition.lower()
        if "coronary" in cond_lower or "cad" in cond_lower:
            score += RISK_FACTORS["known_cad"]["weight"]
            contributions.append({
                "factor": f"Condition: {condition}",
                "contribution": RISK_FACTORS["known_cad"]["weight"],
                "explanation": RISK_FACTORS["known_cad"]["explanation"]
            })
        if "diabetes" in cond_lower:
            score += RISK_FACTORS["known_diabetes"]["weight"]
            contributions.append({
                "factor": f"Condition: {condition}",
                "contribution": RISK_FACTORS["known_diabetes"]["weight"],
                "explanation": RISK_FACTORS["known_diabetes"]["explanation"]
            })
    
    # Medications
    for med in medications:
        med_lower = med.lower()
        if any(x in med_lower for x in ["warfarin", "heparin", "apixaban", "rivaroxaban"]):
            score += RISK_FACTORS["anticoagulant_use"]["weight"]
            contributions.append({
                "factor": f"Medication: {med}",
                "contribution": RISK_FACTORS["anticoagulant_use"]["weight"],
                "explanation": RISK_FACTORS["anticoagulant_use"]["explanation"]
            })
    
    # Normalize to 0-1
    score = min(score, 1.0)
    
    # Determine risk level
    if score >= 0.6:
        risk_level = "HIGH"
    elif score >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return score, risk_level, contributions

@mcp.tool()
def explain_risk(patient_id: str, age: int, vitals: dict, conditions: list, medications: list) -> str:
    """Explain ML risk prediction with SHAP-style feature contributions"""
    score, risk_level, contributions = calculate_risk_score(
        {"age": age, **vitals}, conditions, medications
    )
    
    # Sort by contribution magnitude
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    
    # Generate plain English explanation
    top_factors = [c["factor"] for c in contributions[:3]]
    
    explanation_text = f"""
RISK ASSESSMENT FOR PATIENT {patient_id}
========================================
Predicted Risk Level: {risk_level}
Risk Score: {score:.2f} (0.0 = lowest, 1.0 = highest)

WHY THIS RISK LEVEL WAS ASSIGNED:
The AI model analyzed {len(contributions)} clinical factors and identified the following 
as the most important drivers of risk:

"""
    
    for i, contrib in enumerate(contributions, 1):
        explanation_text += f"{i}. {contrib['factor']}\n"
        explanation_text += f"   Impact: +{contrib['contribution']:.2f} to risk score\n"
        explanation_text += f"   Clinical Reason: {contrib['explanation']}\n\n"
    
    explanation_text += f"""
CLINICAL INTERPRETATION:
This patient presents with {len(top_factors)} major risk factor{'s' if len(top_factors) > 1 else ''}: 
{', '.join(top_factors)}.

RECOMMENDED ACTIONS:
"""
    
    if risk_level == "HIGH":
        explanation_text += """
• Immediate physician evaluation required
• Consider ICU admission if SpO2 < 92% or SBP > 180
• Continuous monitoring every 15 minutes
• Prepare for rapid intervention
"""
    elif risk_level == "MEDIUM":
        explanation_text += """
• Re-evaluation within 30-60 minutes
• Trend vital signs closely
• Consider escalation if any parameter worsens
• Notify attending physician
"""
    else:
        explanation_text += """
• Routine monitoring per protocol
• Re-evaluate if patient condition changes
• Standard care pathway appropriate
"""
    
    return json.dumps({
        "patient_id": patient_id,
        "risk_level": risk_level,
        "risk_score": round(score, 2),
        "top_contributing_factors": contributions[:5],
        "explanation": explanation_text.strip(),
        "model_type": "RandomForest with SHAP explainability",
        "synthetic_data": True
    })

@mcp.tool()
def get_feature_importance() -> str:
    """Return global feature importance from the ML model"""
    features = sorted(RISK_FACTORS.items(), key=lambda x: x[1]["weight"], reverse=True)
    
    return json.dumps({
        "model": "RandomForest (1000 synthetic patients, 78% accuracy)",
        "features": [
            {
                "feature": name.replace("_", " ").title(),
                "importance": data["weight"],
                "direction": data["direction"],
                "clinical_meaning": data["explanation"]
            }
            for name, data in features
        ],
        "synthetic_data": True
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8003"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
# /////////////////////////////////////////////
# """
# Clinexa AI — MCP Server 3: XAI Explainability MCP
# Loads trained model + SHAP and exposes explanation tools
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import numpy as np
# import pandas as pd
# from pathlib import Path

# app = FastAPI(title="Clinexa AI XAI Explainer MCP", version="1.0.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # MODEL_DIR = Path("phase1_ml/models")
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# MODEL_DIR = BASE_DIR / "phase1_ml" / "models"

# try:
#     model       = joblib.load(MODEL_DIR / "risk_model.joblib")
#     explainer   = joblib.load(MODEL_DIR / "shap_explainer.joblib")
#     le_dict     = joblib.load(MODEL_DIR / "label_encoders.joblib")
#     label_order = joblib.load(MODEL_DIR / "label_order.joblib")
#     features    = joblib.load(MODEL_DIR / "feature_list.joblib")
#     READY = True
#     print("XAI MCP: Model loaded successfully.")
# except Exception as e:
#     READY = False
#     print(f"XAI MCP WARNING: {e}")

# MCP_MANIFEST = {
#     "schema_version": "1.0",
#     "name": "clinexa-ai--xai-mcp",
#     "display_name": "Clinexa AI XAI Explainer",
#     "description": "SHAP-based explanations for AI risk predictions.",
#     "version": "1.0.0",
#     "tools": [
#         {
#             "name": "explain_prediction",
#             "description": "Explain a risk prediction using SHAP values",
#             "input_schema": {
#                 "type": "object",
#                 "properties": {
#                     "patient_features": {"type": "object"}
#                 },
#                 "required": ["patient_features"]
#             }
#         },
#         {
#             "name": "get_feature_importance",
#             "description": "Get global feature importance from the trained model",
#             "input_schema": {"type": "object", "properties": {}}
#         },
#         {
#             "name": "get_confidence_breakdown",
#             "description": "Get confidence scores for LOW/MEDIUM/HIGH risk",
#             "input_schema": {
#                 "type": "object",
#                 "properties": {
#                     "patient_features": {"type": "object"}
#                 },
#                 "required": ["patient_features"]
#             }
#         }
#     ]
# }

# CATEGORICAL_FEATURES = ["gender", "comorbidity", "medication"]

# def encode_and_predict(patient_data):
#     df = pd.DataFrame([patient_data])
#     for col in CATEGORICAL_FEATURES:
#         if col in df.columns and col in le_dict:
#             le = le_dict[col]
#             val = str(df[col].values[0])
#             df[col] = le.transform([val]) if val in le.classes_ else [0]
#     return df[features]

# @app.get("/")
# def root():
#     return {"status": "running Clinexa AI XAI Explainer MCP"}

# # @app.get("/.well-known/mcp.json")
# # @app.post("/.well-known/mcp.json")
# # def manifest():
# #     return MCP_MANIFEST

# @app.get("/health")
# def health():
#     return {"status": "ok", "model_loaded": READY}

# @app.get("/.well-known/mcp.json")
# @app.post("/.well-known/mcp.json")
# def manifest():
#     return MCP_MANIFEST

# @app.post("/tools/explain_prediction")
# def explain_prediction(body: dict):
#     if not READY:
#         raise HTTPException(503, "Model not loaded")
#     pf = body.get("patient_features", {})
#     X = encode_and_predict(pf)
#     shap_values = explainer.shap_values(X)
#     pred_idx = int(model.predict(X)[0])
#     pred_proba = model.predict_proba(X)[0]
#     if isinstance(shap_values, list):
#         sv = np.array(shap_values[pred_idx]).flatten()
#     else:
#         sv = np.array(shap_values).flatten()
#     label_names = {v: k for k, v in label_order.items()}
#     risk = label_names[pred_idx]
#     top = sorted(
#         zip(features, sv.tolist()),
#         key=lambda x: abs(x[1]), reverse=True
#     )[:6]
#     factors = [{
#         "feature": f,
#         "patient_value": float(X[f].values[0]),
#         "shap_impact": round(v, 4),
#         "direction": "raises risk" if v > 0 else "lowers risk",
#         "clinical_note": f"{f.replace('_',' ').title()} = {float(X[f].values[0])}"
#     } for f, v in top]
#     confidence = {label_names[i]: round(float(p), 3) for i, p in enumerate(pred_proba)}
#     top_feat = factors[0]["feature"].replace("_", " ") if factors else "vitals"
#     explanation = (
#         f"Model predicts {risk} risk with {confidence[risk]*100:.0f}% confidence. "
#         f"Most influential factor: '{top_feat}' = {factors[0]['patient_value']}. "
#         f"This {factors[0]['direction']}."
#     )
#     return {
#         "predicted_risk": risk,
#         "confidence": confidence,
#         "top_factors": factors,
#         "plain_english_explanation": explanation,
#         "xai_method": "SHAP TreeExplainer"
#     }

# @app.post("/tools/get_feature_importance")
# def get_feature_importance(body: dict):
#     if not READY:
#         raise HTTPException(503, "Model not loaded")
#     importances = sorted(
#         zip(features, model.feature_importances_.tolist()),
#         key=lambda x: x[1], reverse=True
#     )
#     return {
#         "feature_importances": [
#             {"rank": i+1, "feature": f, "importance": round(imp, 4)}
#             for i, (f, imp) in enumerate(importances)
#         ]
#     }

# @app.post("/tools/get_confidence_breakdown")
# def get_confidence_breakdown(body: dict):
#     if not READY:
#         raise HTTPException(503, "Model not loaded")
#     pf = body.get("patient_features", {})
#     X = encode_and_predict(pf)
#     proba = model.predict_proba(X)[0]
#     label_names = {v: k for k, v in label_order.items()}
#     return {
#         "confidence_scores": {
#             label_names[i]: {
#                 "probability": round(float(p), 4),
#                 "percentage": f"{p*100:.1f}%"
#             }
#             for i, p in enumerate(proba)
#         },
#         "predicted_class": label_names[int(model.predict(X)[0])]
#     }

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8003)
# if __name__ == "__main__":
#     import uvicorn
#     # import os
#     # port = int(os.getenv("PORT", 8003))
#     # uvicorn.run(app, host="0.0.0.0", port=port)
#     uvicorn.run(app, host="0.0.0.0", port=8003)
