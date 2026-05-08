"""
Clinexa AI — MCP Server 1: FHIR R4 Server
ALL data is synthetic — zero real PHI
"""

from fastapi import FastAPI
from typing import List
import json, uuid, random
from datetime import datetime

app = FastAPI(title="Clinexa FHIR MCP Server", version="1.0.0")

MCP_MANIFEST = {
    "schema_version": "1.0",
    "name": "clinexa-fhir-mcp",
    "display_name": "Clinexa FHIR Patient Data",
    "description": "Exposes FHIR R4 synthetic patient data.",
    "version": "1.0.0",
    "tools": [
        {
            "name": "get_patient",
            "description": "Retrieve synthetic FHIR Patient resource by patient ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"}
                },
                "required": ["patient_id"]
            }
        },
        {
            "name": "get_observations",
            "description": "Get vital signs for a patient",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"}
                },
                "required": ["patient_id"]
            }
        },
        {
            "name": "get_conditions",
            "description": "Get active conditions for a patient",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"}
                },
                "required": ["patient_id"]
            }
        },
        {
            "name": "get_medications",
            "description": "Get current medications for a patient",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"}
                },
                "required": ["patient_id"]
            }
        },
        {
            "name": "create_triage_bundle",
            "description": "Create a FHIR Bundle containing triage assessment",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "risk_level": {"type": "string"},
                    "assessment_text": {"type": "string"},
                    "recommendations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["patient_id", "risk_level", "assessment_text"]
            }
        }
    ]
}

SYNTHETIC_PATIENTS = {
    "SYN-10001": {
        "id": "SYN-10001", "name": "Alex Johnson", "age": 68,
        "gender": "male", "dob": "1957-03-14",
        "conditions": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Lisinopril 10mg", "Metformin 500mg"],
        "vitals": {"hr": 125, "sbp": 185, "dbp": 110, "temp": 38.9, "spo2": 91, "rr": 26}
    },
    "SYN-10002": {
        "id": "SYN-10002", "name": "Morgan Lee", "age": 34,
        "gender": "female", "dob": "1991-07-22",
        "conditions": ["Asthma"],
        "medications": ["Albuterol inhaler"],
        "vitals": {"hr": 88, "sbp": 118, "dbp": 76, "temp": 37.1, "spo2": 97, "rr": 16}
    },
    "SYN-10003": {
        "id": "SYN-10003", "name": "Taylor Smith", "age": 55,
        "gender": "other", "dob": "1970-11-05",
        "conditions": ["Coronary Artery Disease", "COPD"],
        "medications": ["Aspirin 81mg", "Atorvastatin 40mg"],
        "vitals": {"hr": 102, "sbp": 155, "dbp": 95, "temp": 37.8, "spo2": 93, "rr": 22}
    }
}

def make_fhir_patient(p):
    return {
        "resourceType": "Patient",
        "id": p["id"],
        "identifier": [{"system": "urn:clinexa:synthetic", "value": p["id"]}],
        "name": [{"use": "official", "text": p["name"]}],
        "gender": p["gender"],
        "birthDate": p["dob"],
        "extension": [{
            "url": "http://promptopinion.ai/sharp/patient-context",
            "valueString": json.dumps({"patientId": p["id"], "phi": False})
        }]
    }

def make_fhir_observations(patient_id, vitals):
    obs_map = {
        "hr":  ("8867-4", "Heart rate", "beats/minute"),
        "sbp": ("8480-6", "Systolic blood pressure", "mmHg"),
        "dbp": ("8462-4", "Diastolic blood pressure", "mmHg"),
        "temp": ("8310-5", "Body temperature", "Cel"),
        "spo2": ("59408-5", "Oxygen saturation", "%"),
        "rr":  ("9279-1", "Respiratory rate", "/min"),
    }
    observations = []
    for key, (code, display, unit) in obs_map.items():
        observations.append({
            "resourceType": "Observation",
            "id": str(uuid.uuid4())[:8],
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "valueQuantity": {"value": vitals[key], "unit": unit}
        })
    return observations

def make_fhir_conditions(patient_id, conditions):
    return [{
        "resourceType": "Condition",
        "id": str(uuid.uuid4())[:8],
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "code": {"coding": [{"display": cond}]},
        "subject": {"reference": f"Patient/{patient_id}"}
    } for cond in conditions]

def make_fhir_medications(patient_id, meds):
    return [{
        "resourceType": "MedicationRequest",
        "id": str(uuid.uuid4())[:8],
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {"text": med},
        "subject": {"reference": f"Patient/{patient_id}"}
    } for med in meds]

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/.well-known/mcp.json")
def get_manifest():
    return MCP_MANIFEST

@app.get("/health")
def health():
    return {"status": "ok", "server": "clinexa-fhir-mcp"}

@app.post("/tools/get_patient")
def get_patient(body: dict):
    pid = body.get("patient_id", "")
    p = SYNTHETIC_PATIENTS.get(pid, {
        "id": pid, "name": f"Synthetic Patient {pid}", "age": 45,
        "gender": "unknown", "dob": "1980-01-01",
        "conditions": ["Unknown"], "medications": ["None"],
        "vitals": {"hr": 80, "sbp": 120, "dbp": 80, "temp": 37.0, "spo2": 97, "rr": 16}
    })
    return make_fhir_patient(p)

@app.post("/tools/get_observations")
def get_observations(body: dict):
    pid = body.get("patient_id", "")
    p = SYNTHETIC_PATIENTS.get(pid, list(SYNTHETIC_PATIENTS.values())[0])
    return {"resourceType": "Bundle", "type": "searchset",
            "entry": [{"resource": o} for o in make_fhir_observations(pid, p["vitals"])]}

@app.post("/tools/get_conditions")
def get_conditions(body: dict):
    pid = body.get("patient_id", "")
    p = SYNTHETIC_PATIENTS.get(pid, list(SYNTHETIC_PATIENTS.values())[0])
    return {"resourceType": "Bundle", "type": "searchset",
            "entry": [{"resource": c} for c in make_fhir_conditions(pid, p["conditions"])]}

@app.post("/tools/get_medications")
def get_medications(body: dict):
    pid = body.get("patient_id", "")
    p = SYNTHETIC_PATIENTS.get(pid, list(SYNTHETIC_PATIENTS.values())[0])
    return {"resourceType": "Bundle", "type": "searchset",
            "entry": [{"resource": m} for m in make_fhir_medications(pid, p["medications"])]}

@app.post("/tools/create_triage_bundle")
def create_triage_bundle(body: dict):
    pid = body.get("patient_id", "SYN-00000")
    risk = body.get("risk_level", "MEDIUM")
    assessment = body.get("assessment_text", "")
    recommendations = body.get("recommendations", [])
    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "document",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "meta": {"tag": [{"system": "urn:clinexa:synthetic", "code": "synthetic-data"}]},
        "entry": [{
            "resource": {
                "resourceType": "Composition",
                "status": "final",
                "subject": {"reference": f"Patient/{pid}"},
                "title": "Clinexa AI Triage Assessment",
                "section": [
                    {"title": "Risk Level", "text": {"div": f"<div>{risk}</div>"}},
                    {"title": "Assessment", "text": {"div": f"<div>{assessment}</div>"}},
                    {"title": "Recommendations", "text": {"div": f"<div>{'<br/>'.join(recommendations)}</div>"}}
                ]
            }
        }]
    }

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8001)
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)