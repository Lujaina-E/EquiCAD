import base64
import json
import fitz
import os
from openai import OpenAI
import re

_client = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        _client = OpenAI(api_key=api_key)
    return _client


TABLE_EXTRACTION_SYSTEM = """
You are a biomedical document parser.

Extract ALL TABLES from the provided page.

Output a JSON ARRAY.
Each item MUST follow this schema exactly:

{
  "caption": string | null,
  "table": {
    "columns": [ { "key": string, "label": string } ],
    "rows": [
      {
        "label": string,
        "cells": {
          "<column_key>": {
            "display": string,
            "count": number | null,
            "percent": number | null
          }
        }
      }
    ]
  }
}

Rules:
- Output JSON only
- One object per table
- Do not include prose
- If no tables exist, output []
"""


def extract_caption(text):
    """Extract caption from OCR text"""
    if not text:
        return "Caption not detected"
    lines = text.splitlines()
    for line in lines[:5]:
        if re.search(r"(table|figure)\s*\d+", line, re.I):
            return line.strip()
    return "Caption not detected"


def extract_columns(text):
    """Extract column headers from table text"""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    # First line is typically headers
    return [c.strip() for c in re.split(r"\s{2,}|\t", lines[0])]


def extract_rows(text):
    """Extract rows from table text"""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return []

    columns = extract_columns(text)
    if not columns:
        return []
    
    rows = []
    for line in lines[1:]:
        values = re.split(r"\s{2,}|\t", line)
        row = {col: values[i] if i < len(values) else "" for i, col in enumerate(columns)}
        rows.append(row)

    return rows


def extract_notes(text):
    """Extract notes or footnotes from table text"""
    for line in text.splitlines():
        if "note" in line.lower() or "*" in line:
            return line.strip()
    return ""


def extract_tables_from_page_fitz(pdf_bytes):
    """Heuristic table extraction using PyMuPDF"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_tables = []
    
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0:  # text block
                lines = b["lines"]
                if any(re.search(r"\t|\s{2,}", line["spans"][0]["text"]) for line in lines if line.get("spans")):
                    # Potential table detected
                    table_text = "\n".join(span["text"] for line in lines for span in line.get("spans", []))
                    all_tables.append({
                        "page": page_num,
                        "text": table_text
                    })
    return all_tables


def extract_tables_from_page(page_text, page_num, file_id):
    """
    Calls OpenAI to extract tables from a single page.
    Returns JSONL lines (one per table).
    """
    response = get_openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": TABLE_EXTRACTION_SYSTEM
            },
            {
                "role": "user",
                "content": page_text
            }
        ],
        temperature=0
    )

    llm_output = response.choices[0].message.content.strip()

    if llm_output.startswith("```"):
        llm_output = re.sub(r"^```json|```$", "", llm_output, flags=re.MULTILINE).strip()

    try:
        json_tables = json.loads(llm_output)
    except json.JSONDecodeError as e:
        print(f"LLM returned invalid JSON: {e}\nOutput was:\n{llm_output}")
        return []

    jsonl_lines = []
    for table in json_tables:
        try:
            validate_table_schema(table)
            jsonl_line = json.dumps({
                "type": "table",
                "content": table,
                "source": {
                    "page": page_num,
                    "file_id": file_id
                }
            })
            jsonl_lines.append(jsonl_line)
        except ValueError as e:
            print(f"Invalid table schema on page {page_num}: {e}")
            continue

    return jsonl_lines


def validate_table_schema(table: dict):
    """Validate table schema"""
    if not isinstance(table, dict):
        raise ValueError("Table must be an object")

    if "caption" not in table or "table" not in table:
        raise ValueError("Missing required keys: caption, table")

    table_obj = table["table"]
    if not isinstance(table_obj, dict):
        raise ValueError("table must be an object")

    if "columns" not in table_obj or "rows" not in table_obj:
        raise ValueError("table.columns or table.rows missing")

    if not isinstance(table_obj["columns"], list):
        raise ValueError("table.columns must be a list")

    if not isinstance(table_obj["rows"], list):
        raise ValueError("table.rows must be a list")

      
def perform_ocr_on_image(image_data):
    """
    OCR temporarily disabled to avoid runtime failures.
    Returns empty string for now.
    """
    return ""

      
def extract_tables_from_images(pdf_bytes, file_id):
    """
    Scan all images in the PDF for tables using OCR.
    Returns JSONL lines for detected tables.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    ocr_table_jsonl = []

    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Send image to OCR model for table text extraction
                ocr_text = perform_ocr_on_image(image_bytes)
                
                if not ocr_text:
                    continue

                # Convert OCR text into table JSON
                table_obj = {
                    "caption": extract_caption(ocr_text),
                    "table": {
                        "columns": extract_columns(ocr_text),
                        "rows": extract_rows(ocr_text)
                    },
                    "notes": extract_notes(ocr_text)
                }

                jsonl_line = json.dumps({
                    "type": "table",
                    "content": table_obj,
                    "source": {"page": page_num, "file_id": file_id}
                })
                ocr_table_jsonl.append(jsonl_line)
            except Exception as e:
                print(f"OCR failed for image {img_index} on page {page_num}: {e}")
                continue

    return ocr_table_jsonl


def extract_tables_with_fallback(pdf_bytes, pages, file_id):
    """
    Attempt multiple methods to extract tables: LLM, heuristics, OCR from images.
    Returns list of JSONL lines.
    """
    all_table_jsonl = []

    # --- Method 1: LLM-based extraction ---
    for page in pages:
        try:
            page_tables = extract_tables_from_page(page["text"], page["page_num"], file_id)
            if page_tables:
                all_table_jsonl.extend(page_tables)
        except Exception as e:
            print(f"LLM table extraction failed on page {page['page_num']}: {e}")

    # --- Method 2: Heuristic text extraction ---
    try:
        heuristic_tables = extract_tables_from_page_fitz(pdf_bytes)
        for t in heuristic_tables:
            jsonl_line = json.dumps({
                "type": "table",
                "content": {
                    "caption": extract_caption(t["text"]),
                    "table": {
                        "columns": extract_columns(t["text"]),
                        "rows": extract_rows(t["text"])
                    }
                },
                "source": {
                    "page": t["page"],
                    "file_id": file_id
                }
            })
            all_table_jsonl.append(jsonl_line)
    except Exception as e:
        print(f"Heuristic table extraction failed: {e}")

    # --- Method 3: OCR from images ---
    try:
        ocr_tables = extract_tables_from_images(pdf_bytes, file_id)
        if ocr_tables:
            all_table_jsonl.extend(ocr_tables)
    except Exception as e:
        print(f"OCR table extraction failed: {e}")

    # Return empty list if no tables found (don't raise error)
    return all_table_jsonl
