# init fast API
from fastapi import FastAPI, Request, APIRouter, WebSocket, WebSocketDisconnect
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
import uvicorn
from scripts.model import llm
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import json
from streamlit import logger
from app.routers import chat

app = FastAPI()

app.include_router(chat.router)
# init with fastapi
@app.post("/chat")

async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return {"response": "No messages provided"}
    
    #human message
    messages = [HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    return {"response": response.content}

# adding websockets
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    #process messages
    try:
        while True:
            user_data = await websocket.receive_text()
            data = json.loads(user_data)
            user_message = data.get("message", "")

            async for chunk in llm.astream([HumanMessage(content=user_message)]):
                if chunk.content:
                    await websocket.send_json({"token": chunk.content})

    except WebSocketDisconnect:
        print("Client disconnected")

