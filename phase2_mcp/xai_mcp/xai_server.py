"""
Clinexa AI — MCP Server 3: XAI Explainability MCP
Loads trained model + SHAP and exposes explanation tools
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(title="Clinexa AI XAI Explainer MCP", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# MODEL_DIR = Path("phase1_ml/models")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "phase1_ml" / "models"

try:
    model       = joblib.load(MODEL_DIR / "risk_model.joblib")
    explainer   = joblib.load(MODEL_DIR / "shap_explainer.joblib")
    le_dict     = joblib.load(MODEL_DIR / "label_encoders.joblib")
    label_order = joblib.load(MODEL_DIR / "label_order.joblib")
    features    = joblib.load(MODEL_DIR / "feature_list.joblib")
    READY = True
    print("XAI MCP: Model loaded successfully.")
except Exception as e:
    READY = False
    print(f"XAI MCP WARNING: {e}")

MCP_MANIFEST = {
    "schema_version": "1.0",
    "name": "clinexa-ai--xai-mcp",
    "display_name": "Clinexa AI XAI Explainer",
    "description": "SHAP-based explanations for AI risk predictions.",
    "version": "1.0.0",
    "tools": [
        {
            "name": "explain_prediction",
            "description": "Explain a risk prediction using SHAP values",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_features": {"type": "object"}
                },
                "required": ["patient_features"]
            }
        },
        {
            "name": "get_feature_importance",
            "description": "Get global feature importance from the trained model",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_confidence_breakdown",
            "description": "Get confidence scores for LOW/MEDIUM/HIGH risk",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_features": {"type": "object"}
                },
                "required": ["patient_features"]
            }
        }
    ]
}

CATEGORICAL_FEATURES = ["gender", "comorbidity", "medication"]

def encode_and_predict(patient_data):
    df = pd.DataFrame([patient_data])
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and col in le_dict:
            le = le_dict[col]
            val = str(df[col].values[0])
            df[col] = le.transform([val]) if val in le.classes_ else [0]
    return df[features]

@app.get("/")
def root():
    return {"status": "running Clinexa AI XAI Explainer MCP"}

@app.get("/.well-known/mcp.json")
def manifest():
    return MCP_MANIFEST

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": READY}

@app.post("/tools/explain_prediction")
def explain_prediction(body: dict):
    if not READY:
        raise HTTPException(503, "Model not loaded")
    pf = body.get("patient_features", {})
    X = encode_and_predict(pf)
    shap_values = explainer.shap_values(X)
    pred_idx = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0]
    if isinstance(shap_values, list):
        sv = np.array(shap_values[pred_idx]).flatten()
    else:
        sv = np.array(shap_values).flatten()
    label_names = {v: k for k, v in label_order.items()}
    risk = label_names[pred_idx]
    top = sorted(
        zip(features, sv.tolist()),
        key=lambda x: abs(x[1]), reverse=True
    )[:6]
    factors = [{
        "feature": f,
        "patient_value": float(X[f].values[0]),
        "shap_impact": round(v, 4),
        "direction": "raises risk" if v > 0 else "lowers risk",
        "clinical_note": f"{f.replace('_',' ').title()} = {float(X[f].values[0])}"
    } for f, v in top]
    confidence = {label_names[i]: round(float(p), 3) for i, p in enumerate(pred_proba)}
    top_feat = factors[0]["feature"].replace("_", " ") if factors else "vitals"
    explanation = (
        f"Model predicts {risk} risk with {confidence[risk]*100:.0f}% confidence. "
        f"Most influential factor: '{top_feat}' = {factors[0]['patient_value']}. "
        f"This {factors[0]['direction']}."
    )
    return {
        "predicted_risk": risk,
        "confidence": confidence,
        "top_factors": factors,
        "plain_english_explanation": explanation,
        "xai_method": "SHAP TreeExplainer"
    }

@app.post("/tools/get_feature_importance")
def get_feature_importance(body: dict):
    if not READY:
        raise HTTPException(503, "Model not loaded")
    importances = sorted(
        zip(features, model.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True
    )
    return {
        "feature_importances": [
            {"rank": i+1, "feature": f, "importance": round(imp, 4)}
            for i, (f, imp) in enumerate(importances)
        ]
    }

@app.post("/tools/get_confidence_breakdown")
def get_confidence_breakdown(body: dict):
    if not READY:
        raise HTTPException(503, "Model not loaded")
    pf = body.get("patient_features", {})
    X = encode_and_predict(pf)
    proba = model.predict_proba(X)[0]
    label_names = {v: k for k, v in label_order.items()}
    return {
        "confidence_scores": {
            label_names[i]: {
                "probability": round(float(p), 4),
                "percentage": f"{p*100:.1f}%"
            }
            for i, p in enumerate(proba)
        },
        "predicted_class": label_names[int(model.predict(X)[0])]
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8003)
if __name__ == "__main__":
    import uvicorn
    # import os
    # port = int(os.getenv("PORT", 8003))
    # uvicorn.run(app, host="0.0.0.0", port=port)
    uvicorn.run(app, host="0.0.0.0", port=8003)
