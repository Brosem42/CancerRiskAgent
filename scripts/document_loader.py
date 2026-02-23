import logging
import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_WS = re.compile(r"[ \t]+")
_LINEBREAK = re.compile(r"\n+")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:!?])")
_SPACE_BEFORE_DOT = re.compile(r"\s+\.")
_SPACE_AFTER_DOT = re.compile(r"\.\s+(?=[A-Za-z])")
_SPACE_AFTER_OPEN = re.compile(r"([(/])\s+")

_SINGLE_LETTER_PREFIX = re.compile(r"(?i)\b([a-z])\s+([a-z]{2,})\b")
_SINGLE_LETTER_SUFFIX = re.compile(r"(?i)\b([a-z]{2,})\s+([a-z])\b")
_TWO_LETTER_SUFFIX = re.compile(r"(?i)\b([a-z]{4,})\s+([a-z]{1,2})\b")

_COMMON_WORDS = {
    "and", "the", "for", "with", "from", "this", "that", "have", "has", "was", "were",
    "will", "can", "not", "you", "your", "are", "but", "into", "over", "under", "upon",
    "than", "then", "also", "such", "only", "more", "most", "less", "each", "page",
    "nice", "guideline", "published", "last", "updated"
}

_MIDWORD_CANDIDATE = re.compile(r"\b([A-Za-z]{3,})\s+([A-Za-z]{3,})\b")

def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _LINEBREAK.sub("\n", text)
    text = _WS.sub(" ", text)

    for _ in range(6):
        new_text = text
        new_text = _SINGLE_LETTER_PREFIX.sub(r"\1\2", new_text)
        new_text = _SINGLE_LETTER_SUFFIX.sub(r"\1\2", new_text)
        new_text = _TWO_LETTER_SUFFIX.sub(r"\1\2", new_text)

        def _join(m):
            a, b = m.group(1), m.group(2)
            if a.lower() in _COMMON_WORDS:
                return m.group(0)
            if a[0].isupper() and b[0].islower():
                return m.group(0)
            if b.lower() in _COMMON_WORDS:
                return m.group(0)
            if len(a) + len(b) < 8:
                return m.group(0)
            return a + b

        new_text2 = _MIDWORD_CANDIDATE.sub(_join, new_text)
        if new_text2 == text:
            break
        text = new_text2

    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_BEFORE_DOT.sub(".", text)
    text = _SPACE_AFTER_DOT.sub(". ", text)
    text = _WS.sub(" ", text)
    return text.strip()

def load_pdfs(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    try:
        loader = UnstructuredPDFLoader(
            file_path=file_path,
            mode="elements",
            strategy="hi_res",
            infer_table_structure=True
        )
        docs = loader.load()
        logger.info("Loaded %d element docs (unstructured) from %s", len(docs), file_path)
        return docs

    except ImportError as e:
        logger.warning("Unstructured not available (%s). Falling back to PyPDFLoader.", e)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info("Loaded %d page docs (pypdf) from %s", len(docs), file_path)

        for d in docs:
            d.metadata.setdefault("source", file_path)
            d.metadata["page"] = d.metadata.get("page", d.metadata.get("page_number", 0))
            cleaned = clean_pdf_text(d.page_content)
            if cleaned:
                d.page_content = cleaned

        return docs

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=128
    )
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if c.page_content and c.page_content.strip()]

def load_and_chunk_pdf(file_path: str) -> List[Document]:
    docs = load_pdfs(file_path)
    chunks = split_documents(docs)
    logger.info("Created %d chunks from %s", len(chunks), file_path)
    return chunks

if __name__ == "__main__":
    file_path = "/Users/briannamitchell/Downloads/CancerRiskAgent/data/NG12_pdf.pdf"

    docs = load_pdfs(file_path)
    print("CLEANED PAGE 0 PREVIEW:\n", docs[0].page_content[:250])

    chunks = split_documents(docs)
    print("\nMetadata:\n", chunks[0].metadata)
    print("\nTotal chunks:", len(chunks))