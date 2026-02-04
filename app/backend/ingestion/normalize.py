import json
import uuid
import re


VALID_TYPES = {"text", "table", "figure"}


def infer_section(text: str) -> str:
    text_upper = text.upper()
    if "METHOD" in text_upper:
        return "Methods"
    if "RESULT" in text_upper:
        return "Results"
    if "DISCUSS" in text_upper:
        return "Discussion"
    if "INTRO" in text_upper:
        return "Introduction"
    return "Unknown"


def validate_chunk(chunk: dict):
    assert chunk["type"] in VALID_TYPES
    assert isinstance(chunk["id"], str)
    assert isinstance(chunk["chunk_index"], int)
    assert chunk["content"] is not None


def normalize_text_chunks(pages, doc_id, granularity="paragraph"):
    chunks = []
    idx = 0

    for page in pages:
        text = page["text"]
        if not text.strip():
            continue

        if granularity == "sentence":
            units = re.split(r'(?<=[.!?])\s+', text)
        elif granularity == "sectional":
            units = re.split(r'\n(?=[A-Z][A-Za-z\s]{3,}:)', text)
        else:
            units = re.split(r'\n\s*\n', text)

        for unit in units:
            if len(unit.strip()) < 40:
                continue

            chunk = {
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "chunk_index": idx,
                "type": "text",
                "section": infer_section(unit),
                "content": unit.strip(),
                "page": page["page_num"],
                "metadata": {}
            }

            validate_chunk(chunk)
            chunks.append(chunk)
            idx += 1

    return chunks


def normalize_tables(tables_jsonl, doc_id, start_idx):
    chunks = []
    idx = start_idx

    for line in tables_jsonl:
        obj = json.loads(line)

        chunk = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "chunk_index": idx,
            "type": "table",
            "section": "Results",
            "content": obj["content"],
            "page": obj.get("source", {}).get("page"),
            "metadata": {
                "caption": obj["content"].get("caption")
            }
        }

        validate_chunk(chunk)
        chunks.append(chunk)
        idx += 1

    return chunks


def normalize_figures(figures_jsonl, doc_id, start_idx):
    chunks = []
    idx = start_idx

    for line in figures_jsonl:
        obj = json.loads(line)

        chunk = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "chunk_index": idx,
            "type": "figure",
            "section": "Results",
            "content": obj["content"],
            "page": obj.get("source", {}).get("page"),
            "metadata": {
                "caption": obj["content"].get("caption")
            }
        }

        validate_chunk(chunk)
        chunks.append(chunk)
        idx += 1

    return chunks
