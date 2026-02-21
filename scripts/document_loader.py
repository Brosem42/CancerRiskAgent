# loading pdfs
import logging 
import os
import pathlib
import tempfile
from typing import Any, List

from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader, PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#migrating to gradio
logger_log = logging.getLogger(__name__) 
logging.basicConfig(level=logging.INFO)

#document reader
class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], **unstructured_kwargs: Any):
        super().__init__(file_path, **unstructured_kwargs, mode="elements", strategy="fast")

class DocumentLoaderException(Exception):
    pass

#document loader that aligns with each file ext type-->keeping this open for future solutions later on
class DocumentLoader(object):
    """
    Loads in a document with a supported extension.
    """
    supported_extension = {
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".epub": EpubReader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }
#load document fucntion
def load_document(temp_filepath: str) -> List[Document]:
    """
    Load a file and return a list of documents.
    """
    ext = pathlib.Path(temp_filepath).suffix.lower()
    loader = DocumentLoader.supported_extension.get(ext)
    if not loader:
        raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file"
        )
    loaded = loader(temp_filepath)
    docs = loaded.load()

    logger_log.info("Loading docs..", len(docs), temp_filepath)
    return docs


# split our data into chunks
def split_documents(docs: List[Document]) -> List[Document]:
    """
    Splitting each pdf document.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=256
    )
    return text_splitter.split_documents(docs)
