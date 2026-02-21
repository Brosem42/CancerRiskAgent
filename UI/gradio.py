# imports
import asyncio
import nest_asyncio
# pkgs for imports
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from ragPipeline.rag import graph, config, retriever
from scripts.doc_retrieval import DocumentBaseRetriever
from scripts.embeddings import EMBEDDINGS
from scripts.model import llm
from scripts.document_loader import load_document 

from scripts.document_loader import DocumentLoader

nest_asyncio.apply()
import gradio as gr




