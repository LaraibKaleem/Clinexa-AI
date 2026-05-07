"""
Clinexa AI — Prompt Opinion Marketplace Registration Guide
Follow this file step by step on Day 6
"""

# ─────────────────────────────────────────────
# STEP 1 — Go to app.promptopinion.ai
# Click: Marketplace → Add New Tool
# Register each MCP server below one by one
# ─────────────────────────────────────────────

MCP_SERVERS = [
    {
        "marketplace_name": "Clinexa FHIR Patient Data",
        "description": "Retrieves synthetic FHIR R4 patient records including demographics, vitals, conditions, and medications. Zero PHI — synthetic data only.",
        "railway_port": 8001,
        "file": "phase2_mcp/fhir_mcp/fhir_server.py",
        "manifest_endpoint": "/.well-known/mcp.json",
        "tools": [
            "get_patient",
            "get_observations",
            "get_conditions",
            "get_medications",
            "create_triage_bundle"
        ],
        "uses_phi": False,
        "data_type": "synthetic"
    },
    {
        "marketplace_name": "Clinexa Drug Safety Checker",
        "description": "Checks drug-drug interactions and allergy conflicts using OpenFDA data. Flags dangerous combinations before treatment.",
        "railway_port": 8002,
        "file": "phase2_mcp/drug_mcp/drug_server.py",
        "manifest_endpoint": "/.well-known/mcp.json",
        "tools": [
            "check_drug_interactions",
            "get_drug_warnings",
            "check_allergy_risk"
        ],
        "uses_phi": False,
        "data_type": "synthetic"
    },
    {
        "marketplace_name": "Clinexa XAI Explainer",
        "description": "Provides SHAP-based explanations for AI risk predictions. Tells clinicians exactly WHY a risk level was assigned.",
        "railway_port": 8003,
        "file": "phase2_mcp/xai_mcp/xai_server.py",
        "manifest_endpoint": "/.well-known/mcp.json",
        "tools": [
            "explain_prediction",
            "get_feature_importance",
            "get_confidence_breakdown"
        ],
        "uses_phi": False,
        "data_type": "synthetic"
    },
    {
        "marketplace_name": "Clinexa Grok LLM Engine",
        "description": "Powers clinical text generation using xAI Grok. Handles intake parsing, SOAP notes, treatment plans, and note summarization.",
        "railway_port": 8004,
        "file": "phase2_mcp/grok_mcp/grok_server.py",
        "manifest_endpoint": "/.well-known/mcp.json",
        "tools": [
            "parse_patient_intake",
            "generate_soap_note",
            "generate_treatment_plan",
            "summarize_clinical_notes"
        ],
        "env_vars": {
            "GROK_API_KEY": "your-xai-key-here"
        },
        "uses_phi": False,
        "data_type": "synthetic"
    }
]

# ─────────────────────────────────────────────
# STEP 2 — Register these 6 agents on Prompt Opinion
# Click: Agents → Create New Agent
# ─────────────────────────────────────────────

A2A_AGENTS = [
    {
        "agent_name": "Clinexa Intake Agent",
        "description": "Parses free-text patient symptoms into structured clinical data using Grok LLM.",
        "mcp_tools": ["Clinexa Grok LLM Engine → parse_patient_intake"],
        "input": "Free-text patient symptom description",
        "output": "Structured JSON with chief complaint, symptoms, severity"
    },
    {
        "agent_name": "Clinexa Risk Prediction Agent",
        "description": "ML-powered risk classifier predicting LOW/MEDIUM/HIGH using RandomForest on 26 clinical features.",
        "mcp_tools": ["Clinexa XAI Explainer → explain_prediction"],
        "input": "Patient vitals and symptom features",
        "output": "Risk level with SHAP confidence scores"
    },
    {
        "agent_name": "Clinexa XAI Explanation Agent",
        "description": "Translates ML predictions into plain English clinical explanations using SHAP values.",
        "mcp_tools": [
            "Clinexa XAI Explainer → explain_prediction",
            "Clinexa XAI Explainer → get_feature_importance"
        ],
        "input": "Risk prediction and SHAP values",
        "output": "Plain English explanation with top 5 clinical factors"
    },
    {
        "agent_name": "Clinexa Treatment Agent",
        "description": "Generates evidence-based treatment recommendations based on risk level and symptoms.",
        "mcp_tools": ["Clinexa Grok LLM Engine → generate_treatment_plan"],
        "input": "Risk level, symptoms, conditions, age",
        "output": "Treatment plan with recommendations, urgency, red flags"
    },
    {
        "agent_name": "Clinexa Drug Safety Agent",
        "description": "Checks medications for dangerous interactions and allergy conflicts.",
        "mcp_tools": [
            "Clinexa Drug Safety Checker → check_drug_interactions",
            "Clinexa Drug Safety Checker → check_allergy_risk"
        ],
        "input": "Medication list and allergy list",
        "output": "Safety alerts with severity levels"
    },
    {
        "agent_name": "Clinexa FHIR Formatter Agent",
        "description": "Produces HL7 FHIR R4 compliant bundles and SOAP notes from full triage assessment.",
        "mcp_tools": [
            "Clinexa FHIR Patient Data → create_triage_bundle",
            "Clinexa Grok LLM Engine → generate_soap_note"
        ],
        "input": "Full triage assessment output",
        "output": "FHIR R4 Bundle and SOAP note — synthetic data, no PHI"
    }
]

# ─────────────────────────────────────────────
# STEP 3 — Test on Prompt Opinion
# After registering all servers and agents:
# Go to Marketplace → find Clinexa AI
# Click Invoke → send this test payload
# ─────────────────────────────────────────────

TEST_PAYLOAD = {
    "patient_id": "SYN-10001",
    "intake_text": "I have severe chest pain and trouble breathing. Heart is racing.",
    "age": 68,
    "gender": "male",
    "vitals": {
        "hr": 125,
        "sbp": 185,
        "dbp": 110,
        "temp": 38.9,
        "spo2": 91,
        "rr": 26
    },
    "symptoms": ["chest_pain", "shortness_of_breath", "palpitations"],
    "conditions": ["heart_disease", "hypertension"],
    "medications": ["warfarin", "aspirin"],
    "allergies": [],
    "comorbidity": "heart_disease",
    "pain_scale": 9
}