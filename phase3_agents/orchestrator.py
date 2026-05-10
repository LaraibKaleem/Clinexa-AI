"""
Clinexa AI — Phase 3: Full Agent Orchestration Pipeline
6 agents working in sequence:
Intake → Risk → XAI → Treatment → Drug Safety → FHIR Output
"""

import json, httpx, asyncio
from datetime import datetime
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Clinexa AI Orchestrator", version="1.0.0")

# MCP_URLS = {
#     "fhir":  "http://localhost:8001",
#     "drug":  "http://localhost:8002",
#     "xai":   "http://localhost:8003",
#     "grok":  "http://localhost:8004",
# }
MCP_URLS = {
    "fhir":  "fhir-production-mcp.up.railway.app",
    "drug":  "drug-production-mcp.up.railway.app",
    "xai":   "xai-production-mcp.up.railway.app",
    "grok":  "grok-production-mcp.up.railway.app",
}

class TriageRequest(BaseModel):
    patient_id: str
    intake_text: str
    age: int
    gender: str = "unknown"
    vitals: dict = {}
    symptoms: List[str] = []
    conditions: List[str] = []
    medications: List[str] = []
    allergies: List[str] = []
    comorbidity: str = "none"
    pain_scale: int = 5

async def call_mcp(server, tool, data):
    url = f"{MCP_URLS[server]}/tools/{tool}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=data)

            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}", "fallback": True}

            return r.json()

    except Exception as e:
        return {
            "error": str(e),
            "fallback": True,
            "server": server,
            "tool": tool
        }
# async def call_mcp(server, tool, data):
#     url = f"{MCP_URLS[server]}/tools/{tool}"
#     try:
#         async with httpx.AsyncClient(timeout=15.0) as client:
#             r = await client.post(url, json=data)
#             return r.json()
#     except Exception as e:
#         return {"error": str(e), "fallback": True}

async def intake_agent(req):
    print(f"\n[Agent 1] Intake Agent — {req.patient_id}")
    result = await call_mcp("grok", "parse_patient_intake",
                            {"intake_text": req.intake_text})
    return {
        "agent": "IntakeAgent",
        "patient_id": req.patient_id,
        "chief_complaint": result.get("chief_complaint", req.intake_text[:100]),
        "symptoms_detected": result.get("symptoms_detected", req.symptoms),
        "duration": result.get("duration", "unknown"),
        "severity_text": result.get("severity", "unknown"),
    }

async def risk_agent(req, intake):
    print(f"[Agent 2] Risk Prediction Agent")
    vitals = req.vitals or {}
    symptom_fields = [
        "chest_pain", "shortness_of_breath", "high_fever", "severe_headache",
        "loss_of_consciousness", "abdominal_pain", "nausea_vomiting",
        "dizziness", "fatigue", "cough", "back_pain", "rash",
        "confusion", "palpitations", "leg_swelling"
    ]
    symptom_map = {sf: 1 if sf in req.symptoms else 0 for sf in symptom_fields}
    patient_features = {
        "age": req.age,
        "gender": req.gender,
        "heart_rate": vitals.get("hr", 80),
        "systolic_bp": vitals.get("sbp", 120),
        "diastolic_bp": vitals.get("dbp", 80),
        "temperature": vitals.get("temp", 37.0),
        "spo2": vitals.get("spo2", 97),
        "respiratory_rate": vitals.get("rr", 16),
        "pain_scale": req.pain_scale,
        "comorbidity": req.comorbidity,
        "medication": req.medications[0] if req.medications else "none",
        **symptom_map
    }
    result = await call_mcp("xai", "explain_prediction",
                            {"patient_features": patient_features})
    return {
        "agent": "RiskPredictionAgent",
        "predicted_risk": result.get("predicted_risk", "MEDIUM"),
        "confidence": result.get("confidence", {}),
        "top_factors": result.get("top_factors", []),
        "xai_method": result.get("xai_method", "SHAP"),
    }

async def xai_agent(risk_result):
    print(f"[Agent 3] XAI Explanation Agent")
    risk = risk_result.get("predicted_risk", "MEDIUM")
    factors = risk_result.get("top_factors", [])
    confidence = risk_result.get("confidence", {})
    if factors:
        top = factors[0]
        feat_name = top["feature"].replace("_", " ").title()
        conf_pct = confidence.get(risk, 0) * 100
        explanation = (
            f"AI predicted {risk} risk ({conf_pct:.0f}% confidence). "
            f"Top driver: {feat_name} = {top['patient_value']}. "
            f"This {top['direction']}."
        )
        clinical_notes = [
            f"Factor {i+1}: {f['feature'].replace('_',' ').title()} "
            f"({f['direction']}, impact={f['shap_impact']})"
            for i, f in enumerate(factors[:3])
        ]
    else:
        explanation = f"Patient assessed as {risk} risk."
        clinical_notes = []
    return {
        "agent": "XAIExplanationAgent",
        "risk_level": risk,
        "plain_english": explanation,
        "clinical_factor_notes": clinical_notes,
    }

async def treatment_agent(req, risk_result, xai_result):
    print(f"[Agent 4] Treatment Recommendation Agent")
    risk = risk_result.get("predicted_risk", "MEDIUM")
    top_factors = risk_result.get("top_factors", [])
    top_factor_name = (top_factors[0]["feature"].replace("_", " ")
                       if top_factors else "vitals")
    result = await call_mcp("grok", "generate_treatment_plan", {
        "risk_level": risk,
        "primary_symptoms": req.symptoms or ["unspecified"],
        "conditions": req.conditions,
        "age": req.age,
        "xai_top_factor": top_factor_name
    })
    return {
        "agent": "TreatmentRecommendationAgent",
        "risk_level": risk,
        "recommendations": result.get("recommendations", []),
        "urgency": result.get("urgency", risk),
        "follow_up": result.get("follow_up", "Per clinical judgment"),
        "red_flags": result.get("red_flags", []),
    }

async def drug_safety_agent(req, treatment):
    print(f"[Agent 5] Drug Safety Agent")
    meds = req.medications
    allergies = req.allergies
    interaction_result = {"interactions": [], "safe": True}
    if len(meds) >= 2:
        interaction_result = await call_mcp("drug", "check_drug_interactions",
                                            {"drugs": meds})
    allergy_result = {"risks": [], "safe": True}
    if meds and allergies:
        allergy_result = await call_mcp("drug", "check_allergy_risk", {
            "medications": meds,
            "allergies": allergies
        })
    drug_safe = (interaction_result.get("safe", True) and
                 allergy_result.get("safe", True))
    alerts = []
    for i in interaction_result.get("interactions", []):
        alerts.append(
            f"INTERACTION [{i.get('severity','?')}]: {i.get('description','')}"
        )
    for r in allergy_result.get("risks", []):
        alerts.append(f"ALLERGY RISK: {r.get('recommendation','')}")
    return {
        "agent": "DrugSafetyAgent",
        "medications_checked": meds,
        "overall_safe": drug_safe,
        "alerts": alerts,
        "alert_count": len(alerts),
        "summary": ("All clear." if drug_safe
                    else f"{len(alerts)} safety alert(s) detected.")
    }

async def fhir_formatter_agent(req, risk_result, xai_result,
                                treatment, drug_safety):
    print(f"[Agent 6] FHIR Formatter Agent")
    risk = risk_result.get("predicted_risk", "MEDIUM")
    assessment_text = xai_result.get("plain_english", "AI assessment complete.")
    soap_result = await call_mcp("grok", "generate_soap_note", {
        "patient_id": req.patient_id,
        "risk_level": risk,
        "symptoms": req.symptoms,
        "vitals": req.vitals,
        "xai_explanation": assessment_text,
        "drug_safety_notes": drug_safety.get("summary", "")
    })
    bundle = await call_mcp("fhir", "create_triage_bundle", {
        "patient_id": req.patient_id,
        "risk_level": risk,
        "assessment_text": assessment_text,
        "recommendations": treatment.get("recommendations", [])[:5]
    })
    return {
        "agent": "FHIRFormatterAgent",
        "fhir_bundle": bundle,
        "soap_note": soap_result.get("soap_note", "SOAP generation failed"),
        "fhir_compliant": True,
        "standard": "HL7 FHIR R4",
        "phi_present": False,
        "data_type": "synthetic"
    }

@app.post("/triage")
async def run_triage(req: TriageRequest):
    try:
        start = datetime.utcnow()
        intake    = await intake_agent(req)
        risk      = await risk_agent(req, intake)
        xai       = await xai_agent(risk)
        treatment = await treatment_agent(req, risk, xai)
        drug      = await drug_safety_agent(req, treatment)
        fhir      = await fhir_formatter_agent(req, risk, xai, treatment, drug)
        # return {"status": "success"}
        return {
        "status": "debug_ok",
        "patient_id": req.patient_id,
        "message": "orchestrator is alive"
    }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
# @app.post("/triage")
# async def run_triage(req: TriageRequest):
#     start = datetime.utcnow()
#     print(f"\n{'='*60}")
#     print(f"CLINEXA AI — PIPELINE STARTED — {req.patient_id}")
#     print(f"{'='*60}")
#     intake    = await intake_agent(req)
#     risk      = await risk_agent(req, intake)
#     xai       = await xai_agent(risk)
#     treatment = await treatment_agent(req, risk, xai)
#     drug      = await drug_safety_agent(req, treatment)
#     fhir      = await fhir_formatter_agent(req, risk, xai, treatment, drug)
#     elapsed = (datetime.utcnow() - start).total_seconds()
#     return {
#         "clinexa_version": "1.0.0",
#         "timestamp": start.isoformat() + "Z",
#         "processing_time_seconds": round(elapsed, 2),
#         "patient_id": req.patient_id,
#         "final_risk": risk.get("predicted_risk"),
#         "confidence": risk.get("confidence", {}),
#         "safety_cleared": drug.get("overall_safe"),
#         "agents": {
#             "1_intake":      intake,
#             "2_risk":        risk,
#             "3_xai":         xai,
#             "4_treatment":   treatment,
#             "5_drug_safety": drug,
#             "6_fhir":        fhir,
#         },
#         "summary": {
#             "risk_level": risk.get("predicted_risk"),
#             "chief_complaint": intake.get("chief_complaint"),
#             "plain_english": xai.get("plain_english"),
#             "top_recommendations": treatment.get("recommendations", [])[:3],
#             "drug_alerts": drug.get("alerts", []),
#             "fhir_bundle_ready": True
#         }
#     }

@app.get("/health")
def health():
    return {"status": "ok", "agents": 6, "mcp_servers": 4}

@app.get("/")
def root():
    return {
        "name": "Clinexa AI",
        "description": "Multi-agent clinical triage system",
        "agents": [
            "IntakeAgent", "RiskPredictionAgent", "XAIAgent",
            "TreatmentAgent", "DrugSafetyAgent", "FHIRFormatterAgent"
        ]
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import uvicorn
    import os
    # port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=8000)