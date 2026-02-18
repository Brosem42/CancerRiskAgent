# how to structure citations for responses
import re
_PAGE_RE = re.compile(r"Page\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)

def infer_page_from_text(text: str):
    match = _PAGE_RE.search(text)
    if not match:
        return None
    return int(match.group(1))

#source formatted strings from documents
def format_sources_with_citations(documents):
    formatted_sources = []
    for i, doc in enumerate(documents, 1):
        source_info = doc.metadata.get('source', 'Unknown source')

        page = doc.metadata.get("page", None)

        if page is None:
            page = infer_page_from_text(doc.page_content)

        page_str = ""
        if page is not None:
            try:
                page_str = f", page {int(page)}"
            except Exception:
                page_str = f", page {page}"

        header = f"[{i} {source_info}{page_str}]"
        formatted_sources.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted_sources)
    