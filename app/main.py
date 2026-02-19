# init fast API
from fastapi import FastAPI, Request
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
import uvicorn
from scripts.model import llm
from fastapi import APIRouter


app = FastAPI()

# init with fastapi
@app.post("/chat")

async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return {"response": "No messages provided"}
    
    #human message
    messages = [HumanMessage(content=user_message)]
    repsonse = llm.invoke(messages)
    return {"response": str.content}


