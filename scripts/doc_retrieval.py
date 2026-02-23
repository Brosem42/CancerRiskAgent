import os
from pathlib import Path
from typing import List, Any
import tempfile
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

#module imports
from scripts.document_loader import DocumentLoader
from scripts.embeddings import EMBEDDINGS

# setup our vector store for retriver
VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)

# load and split documents into chunks that will be better for the retriever
class TextChunking(RecursiveCharacterTextSplitter):
    def split_documents(docs: List[Document]) -> list[Document]:
        """
        Split each docuemnt.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=256
        )
        return text_splitter.split_documents(docs)
    
    def store_documents(docs: List[Document]) -> None:
            """
            Adding my docs to vector store.
            """
            splits = RecursiveCharacterTextSplitter.split_documents(docs)
            VECTOR_STORE.add_documents(splits)

    def add_uploaded_docs(self, file_paths: List[str]):
        """
        Add your uploaded file paths and files to vector store.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            for path in file_paths:
                try:
                    new_docs = DocumentLoader(path)

                    if new_docs:
                        self.documents.extend(new_docs)
                        self.store_documents(new_docs)
                except Exception as e:
                    print(f"Failed to load document at {path}: {e}")

    def model_post_init(self, ctx: Any) -> None:
         self.store_documents(self.documents)     

# def retriever from base--> creating BaseRetriever object to call
class DocumentBaseRetriever(BaseRetriever):
    """
    Retriever that contains the top k documents for the user query.
    """
    documents: List[Document] = []
    k: int = 5

    def _get_relevant_documents(
              self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
              """
              Sync integration for retriever.
              """
              if len(self.documents) == 0:
                return []
              return VECTOR_STORE.max_marginal_relevance_search(query, k=self.k, fetch_k=20, lambda_mult=0.5)


    
    

             


