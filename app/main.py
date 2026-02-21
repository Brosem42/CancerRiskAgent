# init fast API
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, APIRouter, WebSocket, WebSocketDisconnect

from langchain_core.messages import HumanMessage
from scripts.model import llm
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import json
import os
import shutil

from routers import chat

#adding CORS
from fastapi.middleware.cors import CORSMiddleware

from routers import chat as chat_router
from scripts.doc_retrieval import DocumentBaseRetriever
from scripts.model import llm

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"]
)

retriever_instance = DocumentBaseRetriever()
chat_router.router.RETRIEVER = retriever_instance
chat_router.LLM = llm


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

@app.get("/")
async def home():
     return {"message": "API is online."}

# add routers
app.include_router(chat.router)
