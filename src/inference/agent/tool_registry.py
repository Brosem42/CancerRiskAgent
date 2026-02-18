# 

#imports
from __future__ import annotations
from typing import Any, Callable, Dict, List
from vertexai.preview.generative_models import FunctionDeclaration, Tool

#import modules from my src
from src.utils.patient_store import get_patient
from src.RAG.retriever_prod import retrieve_docs

#tool to wrap around my retriever, only returns the JSON structure
def retrieve_guideline_evidence(query: str, top_k: int=8) -> List[Dict[str, Any]]:
    """
    This is a tool that is wrapping around the retriever. You must return JSON-serializable structure.
    Here are the relevant parameters for the tool:
    
    Docstring for retrieve_guideline_evidence
    
    :param query: Description
    :type query: str
    :param top_k: Description
    :type top_k: int
    :return: Description
    :rtype: List[Dict[str, Any]]

    """
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
                "referral": md.get("referral", None)
            }
        )
    return output


# vertex AI tool decl
# defining how to fetch data based on schema defined above --based on patients.json schema
GET_PATIENT_DECL = FunctionDeclaration(
    name="get_patient",
    description="Fetch and retrieve structured patient data from patients.json by patient_id.",
    parameters={
        "type": "object",
        "properties": {
            "patient_id": {"type": "string", "description": "Unique patient identifier"}
        },
        "required": ["patient_id"],
        "additionalProperties": False
    }
)

RETRIEVE_EVIDENCE_DECL = FunctionDeclaration(
    name="retrieve_guideline_evidence",
    description="Retrieve relevant guideline text chunks from NG12 Chroma DB for a given query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to retrieve",
                "minimum": 1,
                "maximum": 20,
                "default": 8
            }, 
        },
        "required": ["query"],
        "additionalProperties": False
    },
)


VERTEX_TOOLS = [
    Tool(function_declarations=[GET_PATIENT_DECL, RETRIEVE_EVIDENCE_DECL])
]

#implement execution map
TOOL_EXECUTORS: Dict[str, Callable[..., Any]] = {
    "get_patient": get_patient,
    "retrieve_guideline_evidence": retrieve_guideline_evidence
}