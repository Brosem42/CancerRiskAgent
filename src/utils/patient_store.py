# SOLUTION PLANNING: load patients.json then invoke get_patient()

#chaging approach to more deterministic for my query
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "patients.json"


class PatientNotFound(Exception):
    pass


def _load_patients() -> Dict[str, Dict[str, Any]]:
    with open(DATA_PATH, "r") as f:
        raw: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]] = json.load(f)

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, list):
        indexed: Dict[str, Dict[str, Any]] = {}
        for p in raw:
            pid = p.get("patient_id")
            if pid:
                indexed[pid] = p
        return indexed

    raise RuntimeError("patients.json must be a list[dict] or dict keyed by patient_id")


_PATIENT_INDEX = _load_patients()


def _build_retrieval_query(patient: Dict[str, Any]) -> str:
    symptoms = patient.get("symptoms") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms]

    age = patient.get("age")
    smoking = (patient.get("smoking_history") or "").strip()
    dur = patient.get("symptom_duration_days")

    parts: List[str] = []
    if age is not None:
        parts.append(f"age {age}")
        parts.append(f"{age} years")
        parts.append(f"{age} year old")
    if smoking:
        parts.append(smoking)
    if dur is not None:
        parts.append(f"duration {dur} days")

    # exact symptom strings (best for retrieval)
    parts.extend([s for s in symptoms if isinstance(s, str) and s.strip()])

    return " ".join(parts).strip()


def get_patient(patient_id: str) -> Dict[str, Any]:
    patient = _PATIENT_INDEX.get(patient_id)
    if not patient:
        raise PatientNotFound(f"Patient '{patient_id}' not found.")

    # normalize types
    age = patient.get("age")
    try:
        age = int(age) if age is not None else None
    except Exception:
        pass

    dur = patient.get("symptom_duration_days")
    try:
        dur = int(dur) if dur is not None else None
    except Exception:
        pass

    symptoms = patient.get("symptoms") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms]

    normalized = {
        "patient_id": patient.get("patient_id", patient_id),
        "name": patient.get("name"),
        "age": age,
        "gender": patient.get("gender"),
        "smoking_history": patient.get("smoking_history"),
        "symptoms": symptoms,
        "symptom_duration_days": dur,
    }

    normalized["retrieval_query"] = _build_retrieval_query(normalized)
    return normalized