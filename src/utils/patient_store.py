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
#align with the schema of my patient data for reasoning agent
def get_patient(patient_id: str) -> Dict[str: Any]:
    patient = PATIENTS.get(patient_id)
    if not patient:
        raise PatientNotFound(f"Patient '{patient_id}' not found in our system database. Try entry again or contact your admin.")
    
    # implementing my controlled generation for tooling schema + normalize key values 
    age = patient.get("age")
    if age is not None:
        try:
            age = int(age)
        except Exception:
            pass

    symptoms = patient.get("symptoms") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    
    duration = patient.get("symptom_duration_days")
    if duration is not None:
        try:
            duration = int(duration)
        except Exception:
            pass
    
    return {
        "patient_id": patient.get("patient_id", patient_id),
        "name": patient.get("name"),
        "age": age,
        "gender": patient.get("gender"),
        "smoking_history": patient.get("smoking_history"),
        "symptoms": symptoms,
        "symptom_duration_days": duration
        }
