#handle installing of appropriate imports
# %pip install torch transformers accelerate peft trl
# %pip install tqdm==4.64
 

import json
import tiktoken # for token counting
import numpy as np
from collections import defaultdict

import csv
import json
import sys

input_path = "../data/Training.csv"
output_path = "../data/Training.jsonl"


SYSTEM_MESSAGE = """You are a biomedical assistant that detects sex-based discrimination against women in Coronary Artery Disease (CAD) research.

Analyze the input and respond in EXACTLY this format (no extra text):

Label: [Bias OR No Bias]
Category: [one category from the list below]

If Label is "Bias", Category must be ONE of:
- Sampling Bias: exclusion or underrepresentation of women in study samples
- Diagnostic Uncertainty / Bias: differential diagnostic criteria or misdiagnosis affecting women
- Symptom Misinterpretation: dismissing or misattributing women's CAD symptoms

If Label is "No Bias", Category must be ONE of:
- Biological / Physiological Differences: legitimate sex-based biological variations in CAD presentation
- Factual / Neutral Observed Outcome: objective reporting of sex-disaggregated results without bias

CRITICAL RULES:
1. Use EXACT category names as written above
2. Choose the MOST applicable category - you MUST select one
3. Output ONLY the two lines specified - no explanations, no reasoning, no additional text
4. If input is ambiguous, choose the most likely category based on available evidence
"""


def escape_newlines_inside_strings(s: str) -> str:
    """
    Replace literal newline characters only inside JSON string values
    (i.e., between quotes), leaving structural newlines intact.
    """
    out = []
    in_string = False
    escape = False

    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue

        if ch == "\\":
            out.append(ch)
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue

        if in_string and ch == "\n":
            out.append("\\n")  # escape newline inside string
        else:
            out.append(ch)

    return "".join(out)


with open(input_path, newline="", encoding="utf-8-sig") as csvfile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    reader = csv.DictReader(csvfile)

    if "Data Point" not in reader.fieldnames:
        raise ValueError(f"'Data Point' column not found. Columns: {reader.fieldnames}")

    for row_index, row in enumerate(reader, start=2):
        raw_json = row["Data Point"]

        # Print entire raw cell (as you requested)
        print(f"\n----- ROW {row_index} RAW CELL START -----\n{raw_json}\n----- ROW {row_index} RAW CELL END -----\n")

        if not raw_json or raw_json.strip() == "":
            print(f"WARNING: Empty 'Data Point' cell at row {row_index}. Skipping.", file=sys.stderr)
            continue

        # Fix JSON by escaping newlines inside strings
        fixed_json = escape_newlines_inside_strings(raw_json.strip())

        # Try parsing JSON
        try:
            data = json.loads(fixed_json)
        except json.JSONDecodeError as e:
            print(f"ERROR: Still invalid JSON at row {row_index}.")
            print(f"Reason: {e}")
            print(f"Fixed JSON preview:\n{fixed_json[:500]}...\n", file=sys.stderr)
            continue

        # Extract fields
        text = data.get("Text")
        caption = data.get("Caption")
        table_or_figure = data.get("table") or data.get("figure") or data.get("Table")

        label = data.get("Assigned Label")
        category = data.get("Category")

        if label is None or category is None:
            print(f"WARNING: Missing 'Assigned Label' or 'Category' at row {row_index}. Skipping.", file=sys.stderr)
            continue

        # Build user content
        user_content_parts = []
        if text:
            user_content_parts.append(f"Text: {text}")
        if caption:
            user_content_parts.append(f"Caption: {caption}")
        if table_or_figure:
            user_content_parts.append(f"Table/Figure: {json.dumps(table_or_figure, ensure_ascii=False)}")

        if not user_content_parts:
            print(f"WARNING: No usable user content at row {row_index}. Skipping.", file=sys.stderr)
            continue

        user_content = "\n".join(user_content_parts)

        # Build JSONL record
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"Assigned Label: {label}\nCategory: {category}"}
            ]
        }

        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Conversion complete! Output saved to {output_path}")


# Load the dataset to verify no rows have been lost

data_path = "../data/Training.jsonl"


with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]:
    print(message)
    
    
import csv
import json
import sys

input_path = "../data/Testing.csv"
output_path = "../data/Testing.jsonl"


SYSTEM_MESSAGE = """You are a biomedical assistant that detects sex-based discrimination against women in Coronary Artery Disease (CAD) research.

Analyze the input and respond in EXACTLY this format (no extra text):

Label: [Bias OR No Bias]
Category: [one category from the list below]

If Label is "Bias", Category must be ONE of:
- Sampling Bias: exclusion or underrepresentation of women in study samples
- Diagnostic Uncertainty / Bias: differential diagnostic criteria or misdiagnosis affecting women
- Symptom Misinterpretation: dismissing or misattributing women's CAD symptoms

If Label is "No Bias", Category must be ONE of:
- Biological / Physiological Differences: legitimate sex-based biological variations in CAD presentation
- Factual / Neutral Observed Outcome: objective reporting of sex-disaggregated results without bias

CRITICAL RULES:
1. Use EXACT category names as written above
2. Choose the MOST applicable category - you MUST select one
3. Output ONLY the two lines specified - no explanations, no reasoning, no additional text
4. If input is ambiguous, choose the most likely category based on available evidence
"""


def escape_newlines_inside_strings(s: str) -> str:
    """
    Replace literal newline characters only inside JSON string values
    (i.e., between quotes), leaving structural newlines intact.
    """
    out = []
    in_string = False
    escape = False

    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue

        if ch == "\\":
            out.append(ch)
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue

        if in_string and ch == "\n":
            out.append("\\n")  # escape newline inside string
        else:
            out.append(ch)

    return "".join(out)


with open(input_path, newline="", encoding="utf-8-sig") as csvfile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    reader = csv.DictReader(csvfile)

    if "Data Point" not in reader.fieldnames:
        raise ValueError(f"'Data Point' column not found. Columns: {reader.fieldnames}")

    for row_index, row in enumerate(reader, start=2):
        raw_json = row["Data Point"]

        # Print entire raw cell (as you requested)
        print(f"\n----- ROW {row_index} RAW CELL START -----\n{raw_json}\n----- ROW {row_index} RAW CELL END -----\n")

        if not raw_json or raw_json.strip() == "":
            print(f"WARNING: Empty 'Data Point' cell at row {row_index}. Skipping.", file=sys.stderr)
            continue

        # Fix JSON by escaping newlines inside strings
        fixed_json = escape_newlines_inside_strings(raw_json.strip())

        # Try parsing JSON
        try:
            data = json.loads(fixed_json)
        except json.JSONDecodeError as e:
            print(f"ERROR: Still invalid JSON at row {row_index}.")
            print(f"Reason: {e}")
            print(f"Fixed JSON preview:\n{fixed_json[:500]}...\n", file=sys.stderr)
            continue

        # Extract fields
        text = data.get("Text")
        caption = data.get("Caption")
        table_or_figure = data.get("table") or data.get("figure") or data.get("Table")

        label = data.get("Assigned Label")
        category = data.get("Category")

        if label is None or category is None:
            print(f"WARNING: Missing 'Assigned Label' or 'Category' at row {row_index}. Skipping.", file=sys.stderr)
            continue

        # Build user content
        user_content_parts = []
        if text:
            user_content_parts.append(f"Text: {text}")
        if caption:
            user_content_parts.append(f"Caption: {caption}")
        if table_or_figure:
            user_content_parts.append(f"Table/Figure: {json.dumps(table_or_figure, ensure_ascii=False)}")

        if not user_content_parts:
            print(f"WARNING: No usable user content at row {row_index}. Skipping.", file=sys.stderr)
            continue

        user_content = "\n".join(user_content_parts)

        # Build JSONL record
        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"Assigned Label: {label}\nCategory: {category}"}
            ]
        }

        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Conversion complete! Output saved to {output_path}")


# Load the dataset

data_path = "../data/Testing.jsonl"

with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]:
    print(message)