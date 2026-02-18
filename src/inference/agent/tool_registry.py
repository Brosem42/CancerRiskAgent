from __future__ import annotations

from typing import Any, Callable, Dict, List

from vertexai.preview.generative_models import FunctionDeclaration, Tool

from src.utils.patient_store import get_patient
from src.RAG.retriever_prod import retrieve_docs

# query expansion
def expand_query(query: str) -> str:
    q = (query or "").lower()

    expansions: List[str] = []
    # pelling + phrasing
    if "haematuria" in q or "hematuria" in q or "blood in urine" in q:
        expansions += [
            "hematuria", "haematuria", "visible haematuria", "visible hematuria",
            "blood in urine", "urological cancer", "bladder cancer", "renal cancer"
        ]

    if "dysphagia" in q or "swallow" in q:
        expansions += [
            "dysphagia", "difficulty swallowing", "swallowing difficulty",
            "oesophageal cancer", "esophageal cancer"
        ]

    if expansions:
        return query + " " + " ".join(sorted(set(expansions)))

    return query
# Tool 1: Normalize symptoms
SYMPTOM_MAP = {
    "blood in urine": "visible haematuria",
    "visible hematuria": "visible haematuria",
    "hematuria": "haematuria",
    "difficulty swallowing": "dysphagia",
    "swallowing difficulty": "dysphagia",
    "shortness of breath": "shortness of breath",
    "persistent cough": "persistent cough",
    "weight loss": "weight loss"
}
def normalize_symptoms(symptoms: List[str]) -> List[str]:
    """
    Normalize symptom phrases into guideline-friendly terms.
    Returns a list of strings.
    """
    output: List[str] = []
    for s in symptoms or []:
        if not isinstance(s, str):
            continue
        key = s.strip().lower()
        output.append(SYMPTOM_MAP.get(key, s.strip()))
    # preserve order, remove empties
    return [x for x in output if x]

#Tool 2 Build retrival query
def build_retrieval_query(patient: Dict[str, Any]) -> str:
    """
    Build a retrieval query from patient fields:
    - symptoms (exact strings)
    - age
    - smoking_history
    - duration
    """
    symptoms = patient.get("symptoms") or []
    if isinstance(symptoms, str):
        symptoms = [symptoms]

    # normalize symptoms (best effort)
    normalized = normalize_symptoms([s for s in symptoms if isinstance(s, str)])

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

    parts.extend([s for s in normalized if isinstance(s, str) and s.strip()])

    return " ".join(parts).strip()

#Tool 3 retrieve guidleins evidence --> wraps retriever
def retrieve_guideline_evidence(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    query2 = expand_query(query)
    docs = retrieve_docs(query2, top_k=top_k)

    output: List[Dict[str, Any]] = []
    for d in docs:
        md = dict(d.metadata or {})

        text = (d.page_content or "").strip()
        if len(text) > 800:
            text = text[:800].rstrip() + "…"

        page = md.get("page", None)
        try:
            page = int(page) if page is not None else None
        except Exception:
            pass

        output.append(
            {
                "excerpt": text,
                "source": md.get("source", "Unknown source"),
                "page": page,
                "referral": md.get("referral", None),
            }
        )

    return output

# Tool 4: Evaluate referral criteria
def evaluate_referral_criteria(evidence: List[Dict[str, Any]]) -> str:
    """
    Determine referral decision based on retrieved evidence metadata.
    Very simple rule-based priority:
    - if any evidence suggests urgent/very urgent/suspected cancer pathway -> Urgent Referral
    - else if any evidence exists -> Non-urgent Referral
    - else -> Not Met / Insufficient Evidence
    """
    if not evidence:
        return "Not Met / Insufficient Evidence"

    def norm(x: Optional[str]) -> str: #type: ignore
        return (x or "").strip().lower()

    urgent_markers = [
        "very urgent",
        "urgent",
        "suspected cancer pathway",
        "immediate",
        "2-week",
        "two week",
    ]

    for e in evidence:
        r = norm(e.get("referral"))
        if any(m in r for m in urgent_markers):
            return "Urgent Referral"

    return "Non-urgent Referral"

#Tool 5 Extract citations
def extract_citations(evidence: List[Dict[str, Any]], max_citations: int = 4) -> List[Dict[str, Any]]:
    """
    Convert evidence list -> citations list: [{source, page, excerpt}, ...]
    """
    citations: List[Dict[str, Any]] = []
    for e in evidence or []:
        excerpt = (e.get("excerpt") or "").strip()
        if not excerpt:
            continue
        citations.append(
            {
                "source": e.get("source", "Unknown source"),
                "page": e.get("page", None),
                "excerpt": excerpt[:300] + ("…" if len(excerpt) > 300 else ""),
            }
        )
        if len(citations) >= max_citations:
            break
    return citations

# Tool 6 recommended imaging
def recommend_imaging(decision: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Provide post-referral imaging suggestion.
    Conservative: only recommend when referral is urgent.
    Uses evidence snippets to infer modality keywords if present.
    """
    d = (decision or "").lower()
    if "urgent" not in d:
        return "No imaging recommended."

    # simple keyword scan (keeps it grounded in retrieved text)
    joined = " ".join([(e.get("excerpt") or "") for e in (evidence or [])]).lower()

    if "ct" in joined and "contrast" in joined:
        return "Recommend contrast-enhanced CT scan as indicated in the retrieved guideline evidence."
    if "ct" in joined:
        return "Recommend CT imaging as indicated in the retrieved guideline evidence."
    if "x-ray" in joined or "xray" in joined:
        return "Recommend an initial X-ray as indicated in the retrieved guideline evidence."

    return "Recommend imaging as indicated by the retrieved guideline evidence for the suspected cancer pathway."


# =========================================================
# Vertex Function Declarations (schemas)
# =========================================================

GET_PATIENT_DECL = FunctionDeclaration(
    name="get_patient",
    description="Fetch structured patient data by patient_id.",
    parameters={
        "type": "object",
        "properties": {"patient_id": {"type": "string"}},
        "required": ["patient_id"],
        "additionalProperties": False,
    },
)

NORMALIZE_SYMPTOMS_DECL = FunctionDeclaration(
    name="normalize_symptoms",
    description="Normalize symptom phrases into guideline-friendly terms.",
    parameters={
        "type": "object",
        "properties": {
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["symptoms"],
        "additionalProperties": False,
    },
)

BUILD_QUERY_DECL = FunctionDeclaration(
    name="build_retrieval_query",
    description="Build a retrieval query string from structured patient fields.",
    parameters={
        "type": "object",
        "properties": {
            "patient": {"type": "object"}
        },
        "required": ["patient"],
        "additionalProperties": True,  # patient can contain arbitrary fields
    },
)

RETRIEVE_EVIDENCE_DECL = FunctionDeclaration(
    name="retrieve_guideline_evidence",
    description="Retrieve relevant guideline evidence from NG12 for a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

EVALUATE_REFERRAL_DECL = FunctionDeclaration(
    name="evaluate_referral_criteria",
    description="Decide referral status based on retrieved guideline evidence metadata.",
    parameters={
        "type": "object",
        "properties": {
            "evidence": {"type": "array", "items": {"type": "object"}}
        },
        "required": ["evidence"],
        "additionalProperties": False,
    },
)

EXTRACT_CITATIONS_DECL = FunctionDeclaration(
    name="extract_citations",
    description="Extract citation objects (source/page/excerpt) from evidence list.",
    parameters={
        "type": "object",
        "properties": {
            "evidence": {"type": "array", "items": {"type": "object"}},
            "max_citations": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
        },
        "required": ["evidence"],
        "additionalProperties": False,
    },
)

RECOMMEND_IMAGING_DECL = FunctionDeclaration(
    name="recommend_imaging",
    description="Recommend post-referral imaging when appropriate based on decision and evidence.",
    parameters={
        "type": "object",
        "properties": {
            "decision": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["decision", "evidence"],
        "additionalProperties": False,
    },
)

VERTEX_TOOLS = [
    Tool(
        function_declarations=[
            GET_PATIENT_DECL,
            NORMALIZE_SYMPTOMS_DECL,
            BUILD_QUERY_DECL,
            RETRIEVE_EVIDENCE_DECL,
            EVALUATE_REFERRAL_DECL,
            EXTRACT_CITATIONS_DECL,
            RECOMMEND_IMAGING_DECL,
        ]
    )
]

TOOL_EXECUTORS: Dict[str, Callable[..., Any]] = {
    "get_patient": get_patient,
    "normalize_symptoms": normalize_symptoms,
    "build_retrieval_query": build_retrieval_query,
    "retrieve_guideline_evidence": retrieve_guideline_evidence,
    "evaluate_referral_criteria": evaluate_referral_criteria,
    "extract_citations": extract_citations,
    "recommend_imaging": recommend_imaging
}

