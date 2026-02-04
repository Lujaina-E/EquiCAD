import json
import re
from openai import OpenAI
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)

from dotenv import load_dotenv
import os


load_dotenv()
 
# -------------------------------
# configuration
# -------------------------------
api_key = os.getenv("api_key")
MODEL_ID = os.getenv("MODEL_ID")
 
SYSTEM_MESSAGE = """You are a biomedical assistant that detects sex-based discrimination against women in Coronary Artery Disease (CAD) research.
 
Analyze the input and respond in EXACTLY this format (no extra text):
 
Label: [Bias OR No Bias]
Category: [one category from the list below]
 
If Label is "Bias", Category must be ONE of:
- Sampling Bias
- Diagnostic Uncertainty / Bias
- Symptom Misinterpretation
 
If Label is "No Bias", Category must be ONE of:
- Biological / Physiological Differences
- Factual / Neutral Observed Outcome
 
CRITICAL RULES:
1. Use EXACT category names as written above
2. Choose the MOST applicable category - you MUST select one
3. Output ONLY the two lines specified - no explanations, no reasoning, no additional text
4. If input is ambiguous, choose the most likely category based on available evidence
"""
 
client = OpenAI(api_key=api_key)
 
def wrap_input_for_model(user_content: str):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]
 
 # -------------------------------
# Parser
# -------------------------------
def extract_label_category(text: str):
    """Extract Label and Category from assistant content."""
    label_match = re.search(
        r"(?:Assigned Label|Label):\s*\[?(Bias|No Bias)\]?",
        text,
        re.IGNORECASE
    )

    category_match = re.search(
        r"Category:\s*\[?([^\]\n\r]+)\]?",
        text,
        re.IGNORECASE
    )

    if not label_match:
        raise ValueError(f"Cannot extract label from: {text}")
    if not category_match:
        raise ValueError(f"Cannot extract category from: {text}")

    label = label_match.group(1).strip().title()   # Normalize "No bias" -> "No Bias"
    category = category_match.group(1).strip()
    return label, category
 
 # -------------------------------
# Load JSONL
# -------------------------------

TEST_JSONL_PATH = os.getenv("TEST_JSONL_PATH")
ID_MAPPING_PATH = os.getenv("ID_MAPPING_PATH")


# Create a new json file with ids (for manual checks on predicted labels later on...)
with open(TEST_JSONL_PATH, "r") as fin, open(ID_MAPPING_PATH, "w") as fout:
    for idx, line in enumerate(fin):
        temp_id = idx + 1
        obj = json.loads(line)
        obj_with_id = {"id": temp_id, **obj}  
        fout.write(json.dumps(obj_with_id) + "\n")
 
print(f"Created {ID_MAPPING_PATH} with IDs for manual checking.")
 
 
test_inputs = []
ground_truth_map = {}
 
with open(TEST_JSONL_PATH, "r") as f:
    for idx, line in enumerate(f):
        temp_id = idx + 1
        obj = json.loads(line)
        test_inputs.append(obj)
 
        # Ground truth from assistant role
        assistant_msgs = [m for m in obj["messages"] if m["role"] == "assistant"]
        label, category = extract_label_category(assistant_msgs[0]["content"])
        ground_truth_map[temp_id] = {"label": label, "category": category}
  
  
  # -------------------------------
# Predict with LLM and print
# -------------------------------

predicted_map = {}
 
for idx, obj in enumerate(test_inputs):
    temp_id = idx + 1
    user_msgs = [m["content"] for m in obj["messages"] if m["role"] == "user"]
    user_text = user_msgs[0]
 
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=wrap_input_for_model(user_text),
        temperature=0,
        max_tokens=50
    )
    output_text = response.choices[0].message.content.strip()
    pred_label, pred_category = extract_label_category(output_text)
    predicted_map[temp_id] = {"label": pred_label, "category": pred_category}
 
    # ---- Clean print of predictions vs ground truth
    true_label = ground_truth_map[temp_id]["label"]
    true_category = ground_truth_map[temp_id]["category"]
    print(f"{temp_id} | {output_text} | Predicted: Label={pred_label}, Category={pred_category} | "
          f"True: Label={true_label}, Category={true_category}")
 
 
# -------------------------------
# Label metrics
# -------------------------------
y_true_label = [ground_truth_map[i]["label"] for i in range(1, len(test_inputs)+1)]
y_pred_label = [predicted_map[i]["label"] for i in range(1, len(test_inputs)+1)]
 
# Normalize capitalization
y_true_label = [l.title() for l in y_true_label]
y_pred_label = [l.title() for l in y_pred_label]
 
accuracy = accuracy_score(y_true_label, y_pred_label)
precision = precision_score(y_true_label, y_pred_label, pos_label="Bias", zero_division=0)
recall = recall_score(y_true_label, y_pred_label, pos_label="Bias", zero_division=0)
f1 = f1_score(y_true_label, y_pred_label, pos_label="Bias", zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true_label, y_pred_label, labels=["No Bias", "Bias"]).ravel()
 
print("\n--- LABEL METRICS ---")
print(f"Accuracy : {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall   : {recall:.5f}")
print(f"F1-score : {f1:.5f}")
print(f"FPR      : {fp / (fp + tn):.5f}")
print(f"FNR      : {fn / (fn + tp):.5f}")
 
 
# -------------------------------
# Category metrics (only correct labels)
# -------------------------------
correct_indices = [i for i in range(len(y_true_label)) if y_true_label[i] == y_pred_label[i]]
y_true_cat = [ground_truth_map[i+1]["category"] for i in correct_indices]
y_pred_cat = [predicted_map[i+1]["category"] for i in correct_indices]
 
if y_true_cat:
    # normalize categories for comparison (optional)
    unique_cats = sorted(list(set(y_true_cat) | set(y_pred_cat)))
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=unique_cats)
    
    cat_accuracy = np.trace(cm) / np.sum(cm)
    precision_cat, recall_cat, f1_cat, _ = precision_recall_fscore_support(
        y_true_cat, y_pred_cat, average="macro", zero_division=0
    )
    
    total = np.sum(cm)
    fnr_list = []
    fpr_list = []
    
    for i in range(len(unique_cats)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = total - TP - FN - FP

        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        fnr_list.append(fnr)
        fpr_list.append(fpr)

    macro_fnr_cat = np.mean(fnr_list)
    macro_fpr_cat = np.mean(fpr_list)
 
    print("\n--- CATEGORY METRICS (on correct labels only) ---")
    print(f"Accuracy : {cat_accuracy:.5f}")
    print(f"Precision: {precision_cat:.5f}")
    print(f"Recall   : {recall_cat:.5f}")
    print(f"F1-score : {f1_cat:.5f}")
    print(f"FPR      : {macro_fpr_cat:.5f}")
    print(f"FNR      : {macro_fnr_cat:.5f}")
