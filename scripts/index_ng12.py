from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def main() -> None:
    load_dotenv()

    project_root = Path(__file__).resolve().parents[1]
    pdf_dir = project_root / "docs" / "ng12"
    if not pdf_dir.exists():
        raise RuntimeError(f"Missing {pdf_dir}. Create it and put NG12 PDFs there.")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {pdf_dir}")

    chroma_path = Path(os.getenv("CHROMA_PATH", str(project_root / "src" / "RAG" / "chroma_db"))).resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    print("Indexing PDFs:", [p.name for p in pdfs])
    print("CHROMA_PATH:", chroma_path)

    docs = []
    for p in pdfs:
        loader = PyPDFLoader(str(p))
        pages = loader.load()
        for d in pages:
            md = dict(d.metadata or {})
            md["source"] = md.get("source") or p.name
            d.metadata = md
        docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print("Pages:", len(docs), "Chunks:", len(chunks))

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    client = chromadb.PersistentClient(path=str(chroma_path))

    # rebuild collection cleanly
    try:
        client.delete_collection("ng12")
        print("Deleted existing collection ng12")
    except Exception:
        pass

    vs = Chroma(collection_name="ng12", embedding_function=embeddings, client=client)

    batch = 128
    for i in range(0, len(chunks), batch):
        b = chunks[i:i+batch]
        vs.add_texts(
            texts=[d.page_content for d in b],
            metadatas=[dict(d.metadata or {}) for d in b],
        )
        print(f"Upserted {min(i+batch, len(chunks))}/{len(chunks)}")

    print("DONE. count =", vs._collection.count())


if __name__ == "__main__":
    main()