# vertex tooling
from __future__ import annotations
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel

#vertex client tool
def init_vertex() -> None:
    """
    Init Vertex AI client. Uses ENV vars if set.
    """
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")

    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT (OR GCLOUD_PROJECT) is not set.")
    
    vertexai.init(project=project, location=location)

#gemni model tool
def get_model(model_name: str | None=None) -> GenerativeModel:
    """
    Retruns my model instance for my Gemini 2.5-pro flash, GenerativeModel in VertexAI
    """
    init_vertex()
    name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    return GenerativeModel(name)
