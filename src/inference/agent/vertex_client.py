# vertex tooling
from __future__ import annotations
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel

#vertex client tool gemni model tool
def get_model(model_name: str | None=None) -> GenerativeModel:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")
    model_name = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gemini-2.5-pro")

    if not project:
        raise RuntimeError("Missing env var GOOGLE_CLOUD_PROJECT")

    vertexai.init(project=project, location=location)
    return GenerativeModel(model_name)
