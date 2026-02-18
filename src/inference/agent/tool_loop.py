# src/inference/agent/tool_loop.py
# src/inference/agent/tool_loop.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from vertexai.preview.generative_models import Content, Part, ToolConfig

from src.inference.agent.tool_registry import TOOL_EXECUTORS, VERTEX_TOOLS


def find_function_call(response) -> Optional[Tuple[str, Dict[str, Any]]]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc:
            name = getattr(fc, "name", None)
            args = getattr(fc, "args", None) or {}
            return name, dict(args)

    return None


def extract_text(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    return "".join([getattr(p, "text", "") or "" for p in parts]).strip()


def run_tool_calling(
    model,
    system_instructions: str,
    user_prompt: str,
    allowed_tools: Optional[List[str]] = None,
    max_steps: int = 8,
) -> str:
    """
    Vertex function-calling loop.

    Important constraints in your environment:
    - role="system" Content is NOT supported
    - system_instruction kwarg is NOT supported

    So we prepend system instructions into the first user message.
    """

    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
            allowed_function_names=allowed_tools or list(TOOL_EXECUTORS.keys()),
        )
    )

    combined_prompt = f"""INSTRUCTIONS (follow strictly):
{system_instructions}

USER REQUEST:
{user_prompt}
""".strip()

    contents: List[Content] = [
        Content(role="user", parts=[Part.from_text(combined_prompt)]),
    ]

    for _ in range(max_steps):
        response = model.generate_content(
            contents=contents,
            tools=VERTEX_TOOLS,
            tool_config=tool_config,
        )

        func_call = find_function_call(response)
        if not func_call:
            return extract_text(response)

        func_name, func_args = func_call

        if func_name not in TOOL_EXECUTORS:
            raise RuntimeError(f"Unknown tool requested: {func_name}")

        # 1) Append the model's function_call message to history
        #    (this is required so the model "remembers" it called the tool)
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None) is not None:
            contents.append(candidates[0].content)

        # 2) Execute tool
        result = TOOL_EXECUTORS[func_name](**func_args)

        # 3) Append function response as *user* content
        #    (Vertex examples use role="user" for function_response parts)
        contents.append(
            Content(
                role="user",
                parts=[
                    Part.from_function_response(
                        name=func_name,
                        response={"result": result},
                    )
                ],
            )
        )

    raise RuntimeError(f"Tool loop exceeded max_steps={max_steps}")