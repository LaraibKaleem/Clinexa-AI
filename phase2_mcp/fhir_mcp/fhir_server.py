"""
Clinexa AI — FHIR R4 Server (SIMPLE SSE VERSION)
"""

from mcp.server.fastmcp import FastMCP
import json, uuid
from datetime import datetime
import uvicorn

mcp = FastMCP("clinexa-ai-fhir")

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
        "birthDate": p["dob"]
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

@mcp.tool()
def get_patient(patient_id: str) -> str:
    """Retrieve FHIR Patient"""
    p = SYNTHETIC_PATIENTS.get(patient_id, list(SYNTHETIC_PATIENTS.values())[0])
    return json.dumps(make_fhir_patient(p))

@mcp.tool()
def get_observations(patient_id: str) -> str:
    """Get vital signs"""
    p = SYNTHETIC_PATIENTS.get(patient_id, list(SYNTHETIC_PATIENTS.values())[0])
    bundle = {
        "resourceType": "Bundle", "type": "searchset",
        "entry": [{"resource": o} for o in make_fhir_observations(patient_id, p["vitals"])]
    }
    return json.dumps(bundle)

@mcp.tool()
def get_conditions(patient_id: str) -> str:
    """Get conditions"""
    p = SYNTHETIC_PATIENTS.get(patient_id, list(SYNTHETIC_PATIENTS.values())[0])
    bundle = {
        "resourceType": "Bundle", "type": "searchset",
        "entry": [{"resourceType": "Condition", "id": str(uuid.uuid4())[:8], "code": {"display": c}} for c in p["conditions"]]
    }
    return json.dumps(bundle)

@mcp.tool()
def get_medications(patient_id: str) -> str:
    """Get medications"""
    p = SYNTHETIC_PATIENTS.get(patient_id, list(SYNTHETIC_PATIENTS.values())[0])
    bundle = {
        "resourceType": "Bundle", "type": "searchset",
        "entry": [{"resourceType": "MedicationRequest", "id": str(uuid.uuid4())[:8], "medicationCodeableConcept": {"text": m}} for m in p["medications"]]
    }
    return json.dumps(bundle)

@mcp.tool()
def create_triage_bundle(patient_id: str, risk_level: str, assessment_text: str, recommendations: list = None) -> str:
    """Create FHIR Bundle"""
    if recommendations is None:
        recommendations = []
    return json.dumps({
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "document",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "entry": [{"resource": {"resourceType": "Composition", "status": "final", "subject": {"reference": f"Patient/{patient_id}"}, "title": "Triage"}}]
    })

if __name__ == "__main__":
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    async def health_check(request):
        return JSONResponse({"status": "ok", "server": "clinexa-fhir", "tools": 5})
    
    # Create main app with health check
    app = Starlette(routes=[
        Route("/", endpoint=health_check),
    ])
    
    # Get MCP SSE app and merge routes
    mcp_app = mcp.sse_app()
    app.router.routes.extend(mcp_app.router.routes)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

# if __name__ == "__main__":
#     uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=8001)
    
    
# """
# Clinexa AI — MCP Server 1: FHIR R4 Server (WORKING — Official SDK)
# ALL data is synthetic — zero real PHI
# """

# from mcp.server.fastmcp import FastMCP
# from starlette.applications import Starlette
# from starlette.routing import Route
# from mcp.server.sse import SseServerTransport
# from starlette.responses import JSONResponse
# import uvicorn
# import json, uuid, os
# from datetime import datetime

# # ─── Initialize FastMCP ──────────────────────────────────────────────────────
# mcp = FastMCP("clinexa-ai-fhir")

# # ─── Synthetic Patient Data ───────────────────────────────────────────────────
# SYNTHETIC_PATIENTS = {
#     "SYN-10001": {
#         "id": "SYN-10001", "name": "Alex Johnson", "age": 68,
#         "gender": "male", "dob": "1957-03-14",
#         "conditions": ["Hypertension", "Type 2 Diabetes"],
#         "medications": ["Lisinopril 10mg", "Metformin 500mg"],
#         "vitals": {"hr": 125, "sbp": 185, "dbp": 110, "temp": 38.9, "spo2": 91, "rr": 26}
#     },
#     "SYN-10002": {
#         "id": "SYN-10002", "name": "Morgan Lee", "age": 34,
#         "gender": "female", "dob": "1991-07-22",
#         "conditions": ["Asthma"],
#         "medications": ["Albuterol inhaler"],
#         "vitals": {"hr": 88, "sbp": 118, "dbp": 76, "temp": 37.1, "spo2": 97, "rr": 16}
#     },
#     "SYN-10003": {
#         "id": "SYN-10003", "name": "Taylor Smith", "age": 55,
#         "gender": "other", "dob": "1970-11-05",
#         "conditions": ["Coronary Artery Disease", "COPD"],
#         "medications": ["Aspirin 81mg", "Atorvastatin 40mg"],
#         "vitals": {"hr": 102, "sbp": 155, "dbp": 95, "temp": 37.8, "spo2": 93, "rr": 22}
#     }
# }

# # ─── FHIR Helpers ─────────────────────────────────────────────────────────────
# def make_fhir_patient(p):
#     return {
#         "resourceType": "Patient",
#         "id": p["id"],
#         "identifier": [{"system": "urn:clinexa:synthetic", "value": p["id"]}],
#         "name": [{"use": "official", "text": p["name"]}],
#         "gender": p["gender"],
#         "birthDate": p["dob"],
#         "extension": [{
#             "url": "http://promptopinion.ai/sharp/patient-context",
#             "valueString": json.dumps({"patientId": p["id"], "phi": False})
#         }]
#     }

# def make_fhir_observations(patient_id, vitals):
#     obs_map = {
#         "hr":  ("8867-4", "Heart rate", "beats/minute"),
#         "sbp": ("8480-6", "Systolic blood pressure", "mmHg"),
#         "dbp": ("8462-4", "Diastolic blood pressure", "mmHg"),
#         "temp": ("8310-5", "Body temperature", "Cel"),
#         "spo2": ("59408-5", "Oxygen saturation", "%"),
#         "rr":  ("9279-1", "Respiratory rate", "/min"),
#     }
#     observations = []
#     for key, (code, display, unit) in obs_map.items():
#         observations.append({
#             "resourceType": "Observation",
#             "id": str(uuid.uuid4())[:8],
#             "status": "final",
#             "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
#             "subject": {"reference": f"Patient/{patient_id}"},
#             "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
#             "valueQuantity": {"value": vitals[key], "unit": unit}
#         })
#     return observations

# def make_fhir_conditions(patient_id, conditions):
#     return [{
#         "resourceType": "Condition",
#         "id": str(uuid.uuid4())[:8],
#         "clinicalStatus": {"coding": [{"code": "active"}]},
#         "code": {"coding": [{"display": cond}]},
#         "subject": {"reference": f"Patient/{patient_id}"}
#     } for cond in conditions]

# def make_fhir_medications(patient_id, meds):
#     return [{
#         "resourceType": "MedicationRequest",
#         "id": str(uuid.uuid4())[:8],
#         "status": "active",
#         "intent": "order",
#         "medicationCodeableConcept": {"text": med},
#         "subject": {"reference": f"Patient/{patient_id}"}
#     } for med in meds]

# # ─── MCP Tools ────────────────────────────────────────────────────────────────

# @mcp.tool()
# def get_patient(patient_id: str) -> str:
#     """Retrieve synthetic FHIR Patient resource by patient ID"""
#     p = SYNTHETIC_PATIENTS.get(patient_id, {
#         "id": patient_id, "name": f"Synthetic Patient {patient_id}", "age": 45,
#         "gender": "unknown", "dob": "1980-01-01",
#         "conditions": ["Unknown"], "medications": ["None"],
#         "vitals": {"hr": 80, "sbp": 120, "dbp": 80, "temp": 37.0, "spo2": 97, "rr": 16}
#     })
#     return json.dumps(make_fhir_patient(p))

# @mcp.tool()
# def get_observations(patient_id: str) -> str:
#     """Get vital signs for a patient"""
#     p = SYNTHETIC_PATIENTS.get(patient_id, list(SYNTHETIC_PATIENTS.values())[0])
#     bundle = {
#         "resourceType": "Bundle", "type": "searchset",
#         "entry": [{"resource": o} for o in make_fhir_observations(patient_id, p["vitals"])]
#     }
#     return json.dumps(bundle)

# @mcp.tool()
# def get_conditions(patient_id: str) -> str:
#     """Get active conditions for a patient"""
#     p = SYNTHETIC_PATIENTS.get(patient_id, list(SYNTHETIC_PATIENTS.values())[0])
#     bundle = {
#         "resourceType": "Bundle", "type": "searchset",
#         "entry": [{"resource": c} for c in make_fhir_conditions(patient_id, p["conditions"])]
#     }
#     return json.dumps(bundle)

# @mcp.tool()
# def get_medications(patient_id: str) -> str:
#     """Get medications for a patient"""
#     p = SYNTHETIC_PATIENTS.get(patient_id, list(SYNTHETIC_PATIENTS.values())[0])
#     bundle = {
#         "resourceType": "Bundle", "type": "searchset",
#         "entry": [{"resource": m} for m in make_fhir_medications(patient_id, p["medications"])]
#     }
#     return json.dumps(bundle)

# @mcp.tool()
# def create_triage_bundle(patient_id: str, risk_level: str, assessment_text: str, recommendations: list = None) -> str:
#     """Create a FHIR Bundle containing triage assessment"""
#     if recommendations is None:
#         recommendations = []
#     bundle = {
#         "resourceType": "Bundle",
#         "id": str(uuid.uuid4()),
#         "type": "document",
#         "timestamp": datetime.utcnow().isoformat() + "Z",
#         "meta": {"tag": [{"system": "urn:clinexa:synthetic", "code": "synthetic-data"}]},
#         "entry": [{
#             "resource": {
#                 "resourceType": "Composition",
#                 "status": "final",
#                 "subject": {"reference": f"Patient/{patient_id}"},
#                 "title": "Clinexa AI Triage Assessment",
#                 "section": [
#                     {"title": "Risk Level", "text": {"div": f"<div>{risk_level}</div>"}},
#                     {"title": "Assessment", "text": {"div": f"<div>{assessment_text}</div>"}},
#                     {"title": "Recommendations", "text": {"div": f"<div>{'<br/>'.join(recommendations)}</div>"}}
#                 ]
#             }
#         }]
#     }
#     return json.dumps(bundle)

# # ─── SSE Transport Setup ──────────────────────────────────────────────────────
# sse = SseServerTransport("/messages/")

# async def handle_sse(request):
#     async with sse.connect_sse(
#         request.scope,
#         request.receive,
#         request.send
#     ) as streams:
#         await mcp.run(
#             streams[0],
#             streams[1],
#             mcp.create_initialization_options()
#         )

# async def handle_messages(request):
#     await sse.handle_post_message(
#         request.scope,
#         request.receive,
#         request.send
#     )

# async def health_check(request):
#     return JSONResponse({"status": "ok", "server": "clinexa-fhir", "tools": 5})

# app = Starlette(routes=[
#     Route("/", endpoint=health_check),
#     Route("/sse", endpoint=handle_sse),
#     Route("/messages", endpoint=handle_messages, methods=["POST"]),
# ])

# # ─── Run Server ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     print(f"Routes registered: {[r.path for r in app.routes]}", flush=True)
#     uvicorn.run(app, host="0.0.0.0", port=8001)