import os
from pathlib import Path
from typing import List, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from scripts.document_loader import load_document 
from scripts.embeddings import EMBEDDINGS

# setup our vector store for retriver
VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)

# load and split documents into chunks that will be better for the retriever
def split_documents(docs: List[Document]) -> list[Document]:
    """
    Split each docuemnt.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=256
    )
    return text_splitter(docs)



