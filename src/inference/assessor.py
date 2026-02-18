# assessor of orchestration for decision-making workflow

from __future__ import annotations
import json
from typing import Any, Dict

from src.inference.agent.vertex_client import get_model
from src.inference.agent.tool_loop import run_tool_calling

# query expansion + rewriting while keeping main context for base prompts 
assessor_system_prompt = """
    Agent module type: You have inherited a persona profile module as your agentic architecture.
    Your Role: You are a precise and accurate clinical AI assistant that provides expert knowledge on
    clinical decision-making for those who encounter direct patient care.
    Do not make up information or provide personal opinions in your responses without verifying answers with evidence.

    Main Task: Your main task is to determine whether the presented patient requires an urgent referral or not,
    this is based on the patient_id the user submits for lookup wihin the patients.json file. 
    Use the information provided from the text corpus below to perform clinical decision-making on patient cases outside of the structured data provided in your patients.json.

    Utilize your citation tools to site claims and provide evidence for every response you output to the user. 
    The source is: The National Institute for Health and Care Excellence (NICE) Guideline for Suspected cancer: recognition and referral NICE guideline.
    Once, main task is complete and accurate, you must provide a recommendation that decides post-referral instructions corresponding to the most relevant medical imaging practices.
    If patient does not meet urgent referral criteria, do not recommend any medical imaging practices.

    Your evidence must include this citation format:
    - Citations is a list of {source, page, excerpt} and post-referral recommendations if any.",
    - Every clinical response must be in rationale and supported with at least one citation excerpt.

""".strip()


# how to correctly assess the status of the patient 
def assess_patient(patient_id: str, top_k: int=8) -> Dict[str, Any]:
    """
    Returns the structured output of the assessment dictionary.
    """
    model = get_model()
    user_prompt = f"""
Assess this patient-000 based on NCIE NG12 The National Institute for Health and Care Excellence (NICE) Guideline for Suspected cancer: recognition and referral NICE guideline.

Step 1: Call get_patient with patient_id="{patient_id}".
Step 2: From the returned patient fields (age, smoking_history, symptoms, symptom_duration_days),
compose a retrieval query that includes:
- the main symptom phrases (exact strings in symptoms)
- age (e.g., "age 60" or "60 years")
- smoking_history when relevant (e.g., "never smoked", "ex-smoker")
- duration when relevant (e.g., "2 days", "28 days")

Step 3: Call retrieve_guideline_evidence with that query and top_k={top_k}.
Step 4: Decide: If status requires an Urgent Referral or Non-urgent vs Not Met/Insufficient Evidence.
Step 5: Once, patient referral status decided, determine post-referral instructions that correspond to the most relevant medical imaging exam to be ordered, factor in cost-effectiveness.
Step 6: Return JSON ONLY with keys:
patient_id, decision, rationale, citations. 
Citations rules:
- citations is a list of objects: {{source, page, excerpt}}
- Every clinical claim in rationale must be supported by at least one citation excerpt
- If evidence is insufficient, say so and choose "Not Met / Insufficient Evidence".

Return JSON only, no markdown.
""".strip()
    
    raw = run_tool_calling(
        model=model,
        system_instructions=assessor_system_prompt,
        user_prompt=user_prompt,
        allowed_tools=["get_patient", "retrieve_guideline_evidence"],
        max_steps=6
    )
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Agent did not return valid JSON. Raw Raw output:\n{raw}") from e