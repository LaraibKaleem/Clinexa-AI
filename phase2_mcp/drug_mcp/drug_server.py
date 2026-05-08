"""
Clinexa AI — MCP Server 2: Drug Safety MCP
Checks drug interactions and allergy risks
"""

from fastapi import FastAPI
import httpx

app = FastAPI(title="Clinexa Drug Safety MCP", version="1.0.0")

MCP_MANIFEST = {
    "schema_version": "1.0",
    "name": "clinexa-drug-safety-mcp",
    "display_name": "Clinexa Drug Safety Checker",
    "description": "Checks drug interactions, contraindications, and allergy risks.",
    "version": "1.0.0",
    "tools": [
        {
            "name": "check_drug_interactions",
            "description": "Check for known interactions between two or more drugs",
            "input_schema": {
                "type": "object",
                "properties": {
                    "drugs": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["drugs"]
            }
        },
        {
            "name": "get_drug_warnings",
            "description": "Get black box warnings for a drug",
            "input_schema": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"}
                },
                "required": ["drug_name"]
            }
        },
        {
            "name": "check_allergy_risk",
            "description": "Check if patient allergies conflict with medications",
            "input_schema": {
                "type": "object",
                "properties": {
                    "medications": {"type": "array", "items": {"type": "string"}},
                    "allergies": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["medications", "allergies"]
            }
        }
    ]
}

KNOWN_INTERACTIONS = {
    ("warfarin", "aspirin"): {
        "severity": "HIGH",
        "description": "Increased bleeding risk.",
        "recommendation": "Monitor INR closely."
    },
    ("metformin", "alcohol"): {
        "severity": "MEDIUM",
        "description": "Increased risk of lactic acidosis.",
        "recommendation": "Advise patient to avoid alcohol."
    },
    ("lisinopril", "potassium"): {
        "severity": "MEDIUM",
        "description": "ACE inhibitors can increase potassium levels.",
        "recommendation": "Monitor serum potassium levels."
    },
    ("atorvastatin", "clarithromycin"): {
        "severity": "HIGH",
        "description": "Increased statin levels, risk of myopathy.",
        "recommendation": "Temporarily discontinue statin."
    },
    ("insulin", "beta_blockers"): {
        "severity": "MEDIUM",
        "description": "Beta-blockers can mask hypoglycemia symptoms.",
        "recommendation": "Educate patient on signs of hypoglycemia."
    },
}

ALLERGY_CROSS_REACTIONS = {
    "penicillin": ["amoxicillin", "ampicillin", "piperacillin"],
    "sulfa": ["sulfamethoxazole", "trimethoprim"],
    "nsaid": ["ibuprofen", "naproxen", "aspirin", "celecoxib"],
}

def check_interactions_local(drugs):
    interactions = []
    drugs_lower = [d.lower().strip() for d in drugs]
    for i in range(len(drugs_lower)):
        for j in range(i + 1, len(drugs_lower)):
            pair = (drugs_lower[i], drugs_lower[j])
            pair_rev = (drugs_lower[j], drugs_lower[i])
            if pair in KNOWN_INTERACTIONS:
                interactions.append({"drugs": list(pair), **KNOWN_INTERACTIONS[pair]})
            elif pair_rev in KNOWN_INTERACTIONS:
                interactions.append({"drugs": list(pair_rev), **KNOWN_INTERACTIONS[pair_rev]})
    return interactions

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/.well-known/mcp.json")
def manifest():
    return MCP_MANIFEST

@app.get("/health")
def health():
    return {"status": "ok", "server": "clinexa-drug-safety-mcp"}

@app.post("/tools/check_drug_interactions")
async def check_drug_interactions(body: dict):
    drugs = body.get("drugs", [])
    if len(drugs) < 2:
        return {"interactions": [], "message": "Provide at least 2 drugs."}
    interactions = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for drug in drugs[:3]:
                url = f"https://api.fda.gov/drug/label.json?search=drug_interactions:{drug}&limit=1"
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    if results and "drug_interactions" in results[0]:
                        text = results[0]["drug_interactions"][0][:300]
                        interactions.append({
                            "drugs": [drug],
                            "severity": "REVIEW",
                            "description": text,
                            "source": "OpenFDA"
                        })
    except Exception:
        pass
    local = check_interactions_local(drugs)
    interactions.extend(local)
    return {
        "checked_drugs": drugs,
        "interactions_found": len(interactions),
        "interactions": interactions,
        "safe": len(interactions) == 0,
        "summary": f"Found {len(interactions)} interaction(s)."
    }

@app.post("/tools/get_drug_warnings")
async def get_drug_warnings(body: dict):
    drug = body.get("drug_name", "")
    warnings_list = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug}&limit=1"
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                if results:
                    res = results[0]
                    if "boxed_warning" in res:
                        warnings_list.append({"type": "BLACK_BOX", "text": res["boxed_warning"][0][:400]})
                    if "warnings" in res:
                        warnings_list.append({"type": "WARNING", "text": res["warnings"][0][:400]})
    except Exception:
        warnings_list.append({"type": "INFO", "text": f"No FDA data for {drug}."})
    return {"drug": drug, "warnings_count": len(warnings_list), "warnings": warnings_list}

@app.post("/tools/check_allergy_risk")
def check_allergy_risk(body: dict):
    medications = [m.lower() for m in body.get("medications", [])]
    allergies = [a.lower() for a in body.get("allergies", [])]
    risks = []
    for allergy in allergies:
        cross_react = ALLERGY_CROSS_REACTIONS.get(allergy, [])
        for med in medications:
            if allergy in med or any(cr in med for cr in cross_react):
                risks.append({
                    "allergy": allergy,
                    "conflicting_medication": med,
                    "severity": "HIGH",
                    "recommendation": f"Patient allergic to {allergy}. {med} may cause reaction."
                })
    return {
        "allergy_risks_found": len(risks),
        "risks": risks,
        "safe": len(risks) == 0,
        "summary": "No allergy conflicts found." if not risks else f"{len(risks)} conflict(s) detected!"
    }

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8002)
    
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)