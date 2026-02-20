# init fast API
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, APIRouter, WebSocket, WebSocketDisconnect
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
import uvicorn
from scripts.model import llm
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import json
import os
import shutil
from streamlit import logger
from app.routers import chat

from scripts.doc_retrieval import DocumentBaseRetriever
app = FastAPI()
retriever_instance = DocumentBaseRetriever()

# temp directory
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        retriever_instance.add_uploaded_docs([file_path])
        
        return {"message": f"Successfully uploaded + indexed current data {file.filename}"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Upload Failed, try again or contact admin: {str(e)}")
# add routers
app.include_router(chat.router)
# init with fastapi
@app.post("/chat")

async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return {"reply": "No messages provided"}
    
    #human message
    messages = [HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    return {"reply": response.content}

# adding websockets
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    #process messages
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
            print("Client disconnected")

@app.get("/")
async def home():
            return {"message": "API is online, go to test endpoints."}
