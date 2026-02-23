# imports
import asyncio
import nest_asyncio
# pkgs for imports
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from ragPipeline.rag import config, retriever
from langchain_core.documents import Document

from scripts.doc_retrieval import DocumentBaseRetriever
from scripts.model import llm
from scripts.document_loader import DocumentLoader
from langchain_core.messages import HumanMessage, AIMessage
from app.routers.chat import ChatRequest, ChatResponse

nest_asyncio.apply()
import gradio as gr

def process_question(ChatRequest):
    result = DocumentBaseRetriever.invoke(ChatRequest)
    relevance = result['answer']['relevance']

    final_answer = result['answer']['final_answer']
    sources = [DocumentLoader.metadata]






