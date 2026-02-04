import json
import os
from openai import OpenAI

FIGURE_EXTRACTION_SYSTEM = """
You are a biomedical document parser.

Extract ALL FIGURES from the provided page.

Output a JSON ARRAY.
Each item MUST follow this schema exactly:

{
  "caption": string | null,
  "figure": {
    "figure_type": string | null,
    "overall_description": string | null,
    "axes": object | null,
    "legend": object | null,
    "panels": array | null
  }
}

Rules:
- Output JSON only
- One object per figure
- If no figures exist, output []
- Do NOT include prose outside the JSON
"""

_client = None

def get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        _client = OpenAI(api_key=api_key)
    return _client



def extract_figures_from_page(
    page_text,
    page_num,
    file_id,
    model="gpt-4.1"
):
    """
    Calls OpenAI to extract figures from a single page.
    Returns JSONL lines (one per figure).
    """

    response = get_openai_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": FIGURE_EXTRACTION_SYSTEM
            },
            {
                "role": "user",
                "content": page_text
            }
        ],
        temperature=0
    )

    llm_output = response.choices[0].message.content.strip()

    try:
        json_figures = json.loads(llm_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output from figure extraction: {e}\nOutput: {llm_output}")

    jsonl_lines = []
    for figure in json_figures:
        validate_figure_schema(figure)
        jsonl_line = json.dumps({
            "type": "figure",
            "content": figure,
            "source": {
                "page": page_num,
                "file_id": file_id
            }
        })
        jsonl_lines.append(jsonl_line)

    return jsonl_lines


def validate_figure_schema(figure: dict):
    """
    Ensure the figure output matches the expected schema.
    """
    required_keys = {"caption", "figure"}
    if not required_keys.issubset(figure.keys()):
        raise ValueError(f"Invalid figure schema: missing {required_keys - figure.keys()}")

    if not isinstance(figure["figure"], dict):
        raise ValueError("Invalid figure schema: 'figure' must be a dict")
