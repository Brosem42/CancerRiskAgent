import os
from pathlib import Path
from typing import List, Any
import tempfile
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

# def retriever from base--> creating BaseRetriever object to call
class DocumentBaseRetriever(BaseRetriever):
    """
    Retriever that contains the top k documents for the user query.
    """
    documents: List[Document] = []
    k: int = 5

#add context to model as ctx
    def model_post_init(self, ctx: Any) -> None:
        self.store_documents(self.documents)

    @staticmethod
    def store_documents(docs: List[Document]) -> None:
        """
        Adding my docs to vector store.
        """
        splits = split_documents(docs)
        VECTOR_STORE.add_documents(splits)
    
    def add_uploaded_docs(self, uploaded_files):
        """
        Uploading files to vector store.
        """
        docs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                temp_filepath = os.path.join(temp_dir, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())
                try:
                    docs.extend(load_document(temp_filepath))
                except Exception as e:
                    print(f"Failed to load {file.name}: {e}")
                    continue

             


