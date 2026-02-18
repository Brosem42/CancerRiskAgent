# src/inference/agent/tool_registry.py
from __future__ import annotations

from typing import Any, Callable, Dict, List

from vertexai.preview.generative_models import FunctionDeclaration, Tool

from src.utils.patient_store import get_patient
from src.RAG.retriever_prod import retrieve_docs


def retrieve_guideline_evidence(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    docs = retrieve_docs(query, top_k=top_k)

    output: List[Dict[str, Any]] = []
    for d in docs:
        md = dict(d.metadata or {})
        output.append(
            {
                "text": d.page_content,
                "source": md.get("source", "Unknown source"),
                "page": md.get("page", None),
                "metadata": md,
                "referral": md.get("referral", None),
            }
        )
    return output


GET_PATIENT_DECL = FunctionDeclaration(
    name="get_patient",
    description="Fetch structured patient data by patient_id.",
    parameters={
        "type": "object",
        "properties": {
            "patient_id": {"type": "string"}
        },
        "required": ["patient_id"],
        "additionalProperties": False,
    },
)

RETRIEVE_EVIDENCE_DECL = FunctionDeclaration(
    name="retrieve_guideline_evidence",
    description="Retrieve relevant guideline evidence.",
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

VERTEX_TOOLS = [
    Tool(function_declarations=[GET_PATIENT_DECL, RETRIEVE_EVIDENCE_DECL])
]

TOOL_EXECUTORS: Dict[str, Callable[..., Any]] = {
    "get_patient": get_patient,
    "retrieve_guideline_evidence": retrieve_guideline_evidence,
}