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
from scripts.document_loader import VECTOR_STORE, split_documents

def store_documents(docs: List[Document]) -> List[Document]:
  """
  Adding my docs to vector store.
  """
  chunks = split_documents(docs)
  for i, d in enumerate(chunks):
       d.metadata = {**(d.metadata or {}), "chunk_id": str(i)}
  VECTOR_STORE.add_documents(chunks)
  return chunks


docs = [Document(page_content=text, metadata={"id": str(i)}) for i, text in enumerate()]
#dense retriever
dense_retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 10})
#sparse retriever
sparse_retriever = BM25Retriever.from_documents(documents=documents, k=10)




# def retriever from base--> creating BaseRetriever object to call
dense_retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 10})




#sparse--lexical vector for keyword search
sparse_retriever = BM25Retriever.from_documents(documents=Docdocsuments, k=10)

    def _get_relevant_documents(
              self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
              """
              Sync integration for retriever.
              """
              if len(self.documents) == 0:
                return []
              return VECTOR_STORE.max_marginal_relevance_search(query, k=self.k, fetch_k=20, lambda_mult=0.5)


    
    

             


