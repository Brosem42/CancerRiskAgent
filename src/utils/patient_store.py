# SOLUTION PLANNING: load patients.json then invoke get_patient()
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "patients.json"


class PatientNotFound(Exception):
    pass


def _load_patients() -> Dict[str, Dict[str, Any]]:
    """
    Supports patients.json being either:
    - a list of patient objects [{...}, {...}]
    - OR a dict keyed by patient_id {"PT-110": {...}}
    Returns a dict keyed by patient_id either way.
    """
    with open(DATA_PATH, "r") as f:
        raw: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]] = json.load(f)

    if isinstance(raw, dict):
        # already keyed
        return raw

    if isinstance(raw, list):
        indexed: Dict[str, Dict[str, Any]] = {}
        for p in raw:
            pid = p.get("patient_id")
            if not pid:
                continue
            indexed[pid] = p
        return indexed

    raise RuntimeError("patients.json must be a list[dict] or dict keyed by patient_id")


_PATIENT_INDEX = _load_patients()


def get_patient(patient_id: str) -> Dict[str, Any]:
    patient = _PATIENT_INDEX.get(patient_id)
    if not patient:
        raise PatientNotFound(f"Patient '{patient_id}' not found.")

    # Normalize commonly-used fields
    age = patient.get("age")
    try:
        age = int(age) if age is not None else None
    except Exception:
        pass

    symptoms = patient.get("symptoms") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms]

    duration = patient.get("symptom_duration_days")
    try:
        duration = int(duration) if duration is not None else None
    except Exception:
        pass

    return {
        "patient_id": patient.get("patient_id", patient_id),
        "name": patient.get("name"),
        "age": age,
        "gender": patient.get("gender"),
        "smoking_history": patient.get("smoking_history"),
        "symptoms": symptoms,
        "symptom_duration_days": duration,
    }