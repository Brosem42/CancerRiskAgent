
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
    max_steps: int = 10,
) -> str:
    """
    Vertex function-calling loop compatible with your SDK constraints:
    - No system role Content
    - No system_instruction kwarg
    - If restricting tools, must use Mode.ANY

    This version also prevents infinite tool loops by forcing a final
    non-tool answer on the last step.
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

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        # On the last step, DISABLE tools and force a final answer
        if last_step:
            contents.append(
                Content(
                    role="user",
                    parts=[
                        Part.from_text(
                            "Finalize now using the tool results already provided above. "
                            "Do NOT call any tools. Return the final answer."
                        )
                    ],
                )
            )
            response = model.generate_content(contents=contents)
            return extract_text(response)

        # Normal tool-enabled step
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

        # Append the model's tool call content (important)
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None) is not None:
            contents.append(candidates[0].content)

        # Execute tool
        result = TOOL_EXECUTORS[func_name](**func_args)

        # Append tool result back (use role="user" for function_response)
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