
# Clinexa AI — Multi-Agent Clinical Triage System

## Elevator Pitch
6 AI agents that work like a hospital team —
from patient intake to FHIR output — in seconds.

---

## What It Does

A patient describes their symptoms.
Clinexa AI sends them through 6 agents:

1. Intake Agent — reads symptoms, structures them
2. Risk Agent — predicts LOW / MEDIUM / HIGH risk
3. XAI Agent — explains WHY using SHAP
4. Treatment Agent — recommends care plan
5. Drug Safety Agent — checks medication conflicts
6. FHIR Agent — produces HL7 R4 clinical report

---

## Tech Stack

- Python 3.11
- FastAPI — MCP servers
- Scikit-learn — RandomForest ML model
- SHAP — Explainable AI
- xAI Grok — LLM for clinical text
- HL7 FHIR R4 — clinical output standard
- MCP + A2A — agent communication
- Prompt Opinion — deployment platform

---

## Project Structure
clinexai/
├── requirements.txt
├── README.md
├── .env
├── data/
│   ├── generate_data.py
│   └── synthetic_triage.csv
├── phase1_ml/
│   ├── train_model.py
│   └── models/
├── phase2_mcp/
│   ├── fhir_mcp/fhir_server.py
│   ├── drug_mcp/drug_server.py
│   ├── xai_mcp/xai_server.py
│   └── grok_mcp/grok_server.py
├── phase3_agents/
│   └── orchestrator.py
├── phase4_deploy/
│   └── prompt_opinion_config.py
├── phase5_demo/
│   └── demo_script.txt
└── tests/
└── integration_test.py
---

## How To Run

### Step 1 — Install packages
```bash
pip install -r requirements.txt
```

### Step 2 — Generate synthetic data
```bash
python data/generate_data.py
```

### Step 3 — Train ML model
```bash
python phase1_ml/train_model.py
```

### Step 4 — Run integration test
```bash
python tests/integration_test.py
```

### Step 5 — Start all MCP servers
Open 4 terminals in VS Code:
```bash
# Terminal 1
python phase2_mcp/fhir_mcp/fhir_server.py

# Terminal 2
python phase2_mcp/drug_mcp/drug_server.py

# Terminal 3
python phase2_mcp/xai_mcp/xai_server.py

# Terminal 4
python phase2_mcp/grok_mcp/grok_server.py
```

### Step 6 — Start orchestrator
```bash
python phase3_agents/orchestrator.py
```

### Step 7 — Test full pipeline
```bash
curl -X POST http://localhost:8000/triage \
-H "Content-Type: application/json" \
-d "{
  \"patient_id\": \"SYN-10001\",
  \"intake_text\": \"Severe chest pain and trouble breathing\",
  \"age\": 68,
  \"gender\": \"male\",
  \"vitals\": {\"hr\": 125, \"sbp\": 185, \"dbp\": 110,
               \"temp\": 38.9, \"spo2\": 91, \"rr\": 26},
  \"symptoms\": [\"chest_pain\", \"shortness_of_breath\"],
  \"medications\": [\"warfarin\", \"aspirin\"],
  \"comorbidity\": \"heart_disease\",
  \"pain_scale\": 9
}"
```

---

## Judging Criteria

### AI Factor
LLM + ML + SHAP combined.
Not rule-based — genuinely intelligent.

### Potential Impact
Faster triage. Catches drug interactions.
Auto-generates clinical documentation.
Explains every decision.

### Feasibility
FHIR R4 compliant output.
Synthetic data only — zero PHI.
Built on open standards — MCP, A2A, FHIR.
Runs on Prompt Opinion platform.

---

## Safety
- All data is synthetic
- Zero real patient information
- FHIR bundles tagged as synthetic
- No PHI stored or transmitted

---

## Hackathon
Agents Assemble — Healthcare AI Endgame
Submission Deadline: May 12, 2026
Platform: Prompt Opinion Marketplace