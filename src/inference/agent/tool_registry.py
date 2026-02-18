from __future__ import annotations

from typing import Any, Callable, Dict, List

from vertexai.preview.generative_models import FunctionDeclaration, Tool

from src.utils.patient_store import get_patient
from src.RAG.retriever_prod import retrieve_docs


def _expand_query(query: str) -> str:
    q = (query or "").lower()

    expansions: List[str] = []
    # US/UK spelling + lay phrasing
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


def retrieve_guideline_evidence(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    query2 = _expand_query(query)
    docs = retrieve_docs(query2, top_k=top_k)

    # ✅ debug (leave on while validating; remove later if you want)
    print(f"[retrieve_guideline_evidence] query='{query2}' -> {len(docs)} docs")

    output: List[Dict[str, Any]] = []
    for d in docs:
        md = dict(d.metadata or {})

        text = (d.page_content or "").strip()
        if len(text) > 700:
            text = text[:700].rstrip() + "…"

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

RETRIEVE_EVIDENCE_DECL = FunctionDeclaration(
    name="retrieve_guideline_evidence",
    description="Retrieve relevant guideline evidence from NG12 for a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

VERTEX_TOOLS = [Tool(function_declarations=[GET_PATIENT_DECL, RETRIEVE_EVIDENCE_DECL])]

TOOL_EXECUTORS: Dict[str, Callable[..., Any]] = {
    "get_patient": get_patient,
    "retrieve_guideline_evidence": retrieve_guideline_evidence,
}