from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference.assessor import assess_patient
from src.RAG.retriever_prod import retrieve_docs
from src.inference.agent.vertex_client import get_model

app = FastAPI(title="CancerRiskAgent", version="1.0")

# super simple in-memory chat store (good enough for take-home)
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}


class AssessRequest(BaseModel):
    patient_id: str = Field(..., examples=["PT-110"])
    top_k: int = 8


class ChatRequest(BaseModel):
    session_id: str = Field(..., examples=["demo"])
    message: str
    top_k: int = 6


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/assess")
def assess(req: AssessRequest) -> Dict[str, Any]:
    return assess_patient(req.patient_id, top_k=req.top_k)


@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    # store memory
    history = CHAT_MEMORY.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})

    # retrieve evidence
    docs = retrieve_docs(req.message, top_k=req.top_k)
    evidence = []
    for d in docs:
        md = dict(d.metadata or {})
        txt = (d.page_content or "").strip()
        if len(txt) > 800:
            txt = txt[:800].rstrip() + "…"
        evidence.append(
            {
                "excerpt": txt,
                "source": md.get("source", "Unknown source"),
                "page": md.get("page", None),
                "referral": md.get("referral", None),
            }
        )

    # grounded response with citations
    model = get_model()
    prompt = f"""
You are a precise and accurate clinical AI assistant.
Answer the user based ONLY on the evidence list below.
If evidence is empty, say you cannot find support in the indexed guideline corpus.

User message: {req.message}

Evidence (list of objects with excerpt/source/page/referral):
{evidence}

Return a helpful answer and include citations inline using:
[source: <source>, page: <page>]
""".strip()

    resp = model.generate_content([{"role": "user", "parts": [{"text": prompt}]}])
    text = ""
    candidates = getattr(resp, "candidates", None) or []
    if candidates and getattr(candidates[0], "content", None):
        parts = candidates[0].content.parts or []
        text = "".join([getattr(p, "text", "") or "" for p in parts]).strip()

    history.append({"role": "assistant", "content": text})

    return {
        "session_id": req.session_id,
        "answer": text,
        "evidence_count": len(evidence),
    }