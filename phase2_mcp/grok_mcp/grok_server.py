"""
Clinexa AI — MCP Server 4: Grok LLM MCP
Clinical text generation using xAI Grok
"""

from fastapi import FastAPI
import httpx, json, os

app = FastAPI(title="Clinexa Grok LLM MCP", version="1.0.0")

GROK_API_KEY = os.getenv("GROK_API_KEY", "YOUR_GROK_API_KEY_HERE")
GROK_BASE_URL = "https://api.x.ai/v1"
GROK_MODEL = "grok-3-mini"

MCP_MANIFEST = {
    "schema_version": "1.0",
    "name": "clinexa-grok-llm-mcp",
    "display_name": "Clinexa Grok LLM Engine",
    "description": "Clinical text generation using xAI Grok.",
    "version": "1.0.0",
    "tools": [
        {
            "name": "parse_patient_intake",
            "description": "Parse free-text symptoms into structured clinical data",
            "input_schema": {
                "type": "object",
                "properties": {
                    "intake_text": {"type": "string"}
                },
                "required": ["intake_text"]
            }
        },
        {
            "name": "generate_soap_note",
            "description": "Generate SOAP clinical note from assessment data",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "risk_level": {"type": "string"},
                    "symptoms": {"type": "array", "items": {"type": "string"}},
                    "vitals": {"type": "object"},
                    "xai_explanation": {"type": "string"},
                    "drug_safety_notes": {"type": "string"}
                },
                "required": ["patient_id", "risk_level", "symptoms", "vitals"]
            }
        },
        {
            "name": "generate_treatment_plan",
            "description": "Generate treatment recommendations based on risk level",
            "input_schema": {
                "type": "object",
                "properties": {
                    "risk_level": {"type": "string"},
                    "primary_symptoms": {"type": "array", "items": {"type": "string"}},
                    "conditions": {"type": "array", "items": {"type": "string"}},
                    "age": {"type": "integer"},
                    "xai_top_factor": {"type": "string"}
                },
                "required": ["risk_level", "primary_symptoms"]
            }
        },
        {
            "name": "summarize_clinical_notes",
            "description": "Summarize clinical notes into brief structured summary",
            "input_schema": {
                "type": "object",
                "properties": {
                    "notes_text": {"type": "string"},
                    "max_words": {"type": "integer"}
                },
                "required": ["notes_text"]
            }
        }
    ]
}

async def call_grok(system_prompt, user_message):
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 800,
        "temperature": 0.3
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{GROK_BASE_URL}/chat/completions",
                              headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

def fallback(tool, data):
    if tool == "intake":
        return json.dumps({
            "chief_complaint": data.get("intake_text", "")[:100],
            "symptoms_detected": ["symptom reported"],
            "duration": "acute",
            "severity": "moderate",
            "structured": True
        })
    elif tool == "soap":
        risk = data.get("risk_level", "MEDIUM")
        vitals = data.get("vitals", {})
        symptoms = data.get("symptoms", [])
        plans = {
            "HIGH": "IMMEDIATE: Activate emergency protocol. IV access, cardiac monitoring, stat labs.",
            "MEDIUM": "URGENT: Physician evaluation within 2 hours. ECG, vitals q30min.",
            "LOW": "ROUTINE: Follow-up within 48 hours. Patient education."
        }
        return f"""SOAP NOTE — Clinexa AI
S: Patient presents with {', '.join(symptoms)}.
O: HR {vitals.get('hr','--')} | BP {vitals.get('sbp','--')}/{vitals.get('dbp','--')} | SpO2 {vitals.get('spo2','--')}% | Temp {vitals.get('temp','--')}C
A: AI Risk: {risk}. {data.get('xai_explanation','SHAP analysis complete.')}
P: {plans.get(risk,'Evaluate per protocol.')}"""
    elif tool == "treatment":
        risk = data.get("risk_level", "MEDIUM")
        plans = {
            "HIGH": ["Activate emergency response", "IV access + monitoring",
                     "Stat labs", "Specialist consult", "NPO status"],
            "MEDIUM": ["Urgent evaluation within 2 hours", "12-lead ECG",
                       "Basic metabolic panel", "Vitals q30min"],
            "LOW": ["Routine follow-up 48 hours", "Symptom diary",
                    "OTC care", "Return precautions"]
        }
        return json.dumps({
            "recommendations": plans.get(risk, []),
            "urgency": risk,
            "follow_up": "Per clinical judgment"
        })
    return "Clinical analysis complete."


@app.get("/.well-known/mcp.json")
def manifest():
    return MCP_MANIFEST

@app.get("/health")
def health():
    api_ready = GROK_API_KEY != "YOUR_GROK_API_KEY_HERE"
    return {"status": "ok", "grok_configured": api_ready}

@app.post("/tools/parse_patient_intake")
async def parse_patient_intake(body: dict):
    text = body.get("intake_text", "")
    system = """You are a clinical intake parser. Extract structured info from patient symptoms.
Return ONLY valid JSON with keys: chief_complaint, symptoms_detected (list),
duration, severity, structured (true)."""
    try:
        result = await call_grok(system, f"Parse this: {text}")
        return json.loads(result)
    except Exception:
        return json.loads(fallback("intake", {"intake_text": text}))

@app.post("/tools/generate_soap_note")
async def generate_soap_note(body: dict):
    system = """You are a clinical documentation AI. Generate professional SOAP notes.
Format: S (Subjective), O (Objective), A (Assessment), P (Plan). Max 250 words."""
    user_msg = f"""Generate SOAP note:
Patient: {body.get('patient_id')}
Risk: {body.get('risk_level')}
Symptoms: {', '.join(body.get('symptoms', []))}
Vitals: {json.dumps(body.get('vitals', {}))}
AI Explanation: {body.get('xai_explanation', 'N/A')}
Drug Notes: {body.get('drug_safety_notes', 'None')}"""
    try:
        return {"soap_note": await call_grok(system, user_msg)}
    except Exception:
        return {"soap_note": fallback("soap", body)}

@app.post("/tools/generate_treatment_plan")
async def generate_treatment_plan(body: dict):
    system = """You are a clinical decision support AI. Generate evidence-based treatment plans.
Return JSON with keys: recommendations (list), urgency, follow_up, red_flags (list)."""
    user_msg = f"""Treatment plan for:
Risk: {body.get('risk_level')}
Symptoms: {', '.join(body.get('primary_symptoms', []))}
Conditions: {', '.join(body.get('conditions', ['none']))}
Age: {body.get('age', 'unknown')}
Top AI Factor: {body.get('xai_top_factor', 'N/A')}"""
    try:
        result = await call_grok(system, user_msg)
        return json.loads(result)
    except Exception:
        return json.loads(fallback("treatment", body))

@app.post("/tools/summarize_clinical_notes")
async def summarize_clinical_notes(body: dict):
    notes = body.get("notes_text", "")
    max_words = body.get("max_words", 150)
    system = f"Summarize clinical notes in max {max_words} words. Keep key findings and action items."
    try:
        summary = await call_grok(system, f"Summarize: {notes}")
        return {"summary": summary}
    except Exception:
        return {"summary": notes[:500]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8004)
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)