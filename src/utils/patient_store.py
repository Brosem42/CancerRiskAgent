# SOLUTION PLANNING: load patients.json then invoke get_patient()
from __future__ import annotations


#imports
import json
from pathlib import Path
from typing import Any, Dict

#data path to fetch patient data
data_path = Path(__file__).resolve().parents[2] /"data"/"patients.json"

#load the data via path
with open(data_path, "r") as f:
    PATIENTS: Dict[str, Dict[str, Any]] = json.load(f)
class PatientNotFound(Exception):
    pass

#tool defintion, vertexAI requires tools to be defined in global scope vs. the @tool method in python with langchain
def get_patient(patient_id: str) -> Dict[str: Any]:
    patient = PATIENTS.get(patient_id)
    if not patient:
        raise PatientNotFound(f"Patient '{patient_id}' not found in system database. Try entry again.")
    return {"patient_id": patient_id, **patient}
    