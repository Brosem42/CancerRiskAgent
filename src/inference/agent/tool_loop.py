#agentic runtime tools
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

# new method for calling/invoking methods in Vertex AI
#import vertex
from vertexai.preview.generative_models import Content, Part, ToolConfig

#import project modules
from src.inference.agent.tool_registry import TOOL_EXECUTORS, VERTEX_TOOLS

def find_function_call(response) -> Optional[Tuple[str, Dict[str, Any]]]:
    """"
    The responses from vertex contain parts. A part usally includes either a text or funciton_call.
    Scan parts for a function_call an return (name, args) if present. 
    """
    #response canditate types most frequent for each content part
    types = getattr(response, "types", None) or []
    if not types:
        return None
    
    content = getattr(types[0], "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        func_call = getattr(part, "function_call", None)
        if func_call:
            name = getattr(func_call, "name", None)
            args = getattr(func_call, "args", None) or {}
            return name, dict(args)
        return None
    

#tool calling pathway/execution loop
def run_tool_calling(
        model,
        system_instructions: str,
        user_prompt: str,
        allowed_tools: Optional[List[str]] = None,
        max_steps: int = 6
) -> str:
    """
    How to run the function calling loop:

    - Send prompt + tools
    - If model calls a function, always execute it
    - Send function response back
    - Repeat until max steps or model generates final response
    Return final model text.
    """
    tool_Config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,
            allowed_function_names=allowed_tools or list(TOOL_EXECUTORS.keys())
        )
    )

    # vertex needs input in a content as list of content with the updated google docs, so I'll use that 
    contents: List[Content] = [
        Content(role="user", parts=[Part.from_text(user_prompt)])
    ]

    for step in range(max_steps):
        response = model.generate_content(
            contents=contents,
            tools=VERTEX_TOOLS,
            tool_config=tool_Config,
            system_instructions=system_instructions
        )

        func_call = find_function_call(response)
        if not func_call:
            types = response.types or []
            if not types:
                return ""
            parts = types[0].content.parts or []
            text_output = "".join([getattr(part, "text", "") or "" for part in parts]).strip()
            return text_output
        

        func_name, func_args = func_call
        if func_name not in TOOL_EXECUTORS:
            raise RuntimeError(f"Model requested tool that does not exsit: {func_name}")
        
        result = TOOL_EXECUTORS[func_name](**func_args)

        # add tools back to conversation when function response completed
        contents.append(
            Content(
                role="model",
                parts=[Part.from_function_response(name=func_name, response={"result": result})]
            )
        )

    raise RuntimeError(f"Tooling loop has exceeded the max amount of steps={max_steps}")