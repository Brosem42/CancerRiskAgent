# NG12 Cancer Risk Agent

Clinical decision-support agent that performs evidence-grounded suspected cancer referral assessment using NICE NG12.

## Features

- Tool-calling agent workflow:
  - get_patient(patient_id) → retrieve_guideline_evidence(query) → structured decision JSON
- Evidence-grounded output with citations (source/page/excerpt)
- FastAPI:
  - POST /assess
  - POST /chat (memory + grounded retrieval)
- Streamlit UI for quick demo
- Docker-ready packaging

## Setup (local)

Create env + install deps

```bash
pip install -r requirements.txt
