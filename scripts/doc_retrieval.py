import os
from pathlib import Path
from typing import List, Any
import tempfile
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

#module imports
from scripts.embeddings import EMBEDDINGS
from document_loader import VECTOR_STORE, store_documents
# def retriever from base--> creating BaseRetriever object to call
dense_retriever = VECTOR_STORE.as_retriever(
     search_kwargs={"k": 10}
)

    def _get_relevant_documents(
              self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
              """
              Sync integration for retriever.
              """
              if len(self.documents) == 0:
                return []
              return VECTOR_STORE.max_marginal_relevance_search(query, k=self.k, fetch_k=20, lambda_mult=0.5)


    
    

             


