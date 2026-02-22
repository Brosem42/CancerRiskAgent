from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

#module import

from scripts.model import llm
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

#request
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    reply: str
@router.post("app/routers/chat.py", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        #prepare messages including history for content
        formatted_messages = []
        for m in request.history:
            if m["role"] == "user":
                formatted_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                formatted_messages.append(AIMessage(content=m["content"]))

        #add current message
        formatted_messages.append(HumanMessage(content=request.message))
        response = await llm.invoke(formatted_messages)
        return ChatResponse(reply=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))