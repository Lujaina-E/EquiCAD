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


load_dotenv("/Users/lujainaeldelebshany/Desktop/EquiCAD/app/backend/.env")

# -------------------------------
# configuration
# -------------------------------
api_key = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")
 
# SYSTEM_MESSAGE = """You are a biomedical assistant that detects sex-based discrimination against women in Coronary Artery Disease (CAD) research.
 
# Analyze the input and respond in EXACTLY this format (no extra text):
 
# Label: [Bias OR No Bias]
# Category: [one category from the list below]
 
# If Label is "Bias", Category must be ONE of:
# - Sampling Bias
# - Diagnostic Uncertainty / Bias
# - Symptom Misinterpretation
 
# If Label is "No Bias", Category must be ONE of:
# - Biological / Physiological Differences
# - Factual / Neutral Observed Outcome
 
# CRITICAL RULES:
# 1. Use EXACT category names as written above
# 2. Choose the MOST applicable category - you MUST select one
# 3. Output ONLY the two lines specified - no explanations, no reasoning, no additional text
# 4. If input is ambiguous, choose the most likely category based on available evidence
# """


SYSTEM_MESSAGE = """
You are a biomedical assistant that detects sex-based discrimination against women in Coronary Artery Disease (CAD) research.

Analyze the input and respond in EXACTLY this format (no extra text):

Label: [Bias OR No Bias]
Category: [one category from the list below]

If Label is "Bias", Category must be ONE of:
- Sampling Bias
- Diagnostic Uncertainty / Bias
- Symptom Misinterpretation

If Label is "No bias", Category must be ONE of:
- Biological / Physiological Differences
- Factual / Neutral Observed Outcome

CRITICAL RULES:
1. Use EXACT category names as written above
2. Choose the MOST applicable category - you MUST select one
3. Output ONLY the two lines specified - no explanations, no reasoning, no additional text
4. If input is ambiguous, choose the most likely category based on available evidence

Few-shot examples:

Example 1:
{
  "Text": "While both genders reported chest pain as the most frequent complaint, women more often added atypical symptoms. These vague complaints may have led clinicians to broaden the differential diagnosis and include non-cardiac diseases. Women reported a history of CAD less frequently than men. 
After accounting for these variables, women were still less likely to undergo diagnostic angiography than men.",
  "Assigned Label": "Bias",
  "Category": "Symptom Misinterpretation"
}

Example 2:
{
"Text": "Obstructive coronary artery disease is more common in men, while nonobstructive coronary artery disease is more common in women.",
"Assigned Label": "No bias",
"Category": "Biological / Physiological Differences"
}

Example 3:
{
"Caption": "Table 3. Clinical Event Rate and Hazard Ratio for Women Versus Men",
"Table": {
"Columns": [
{ "Key": "clinical_event", "Label": "Clinical Event" },

{ "Key": "women_events", "Label": "Women (n = 148)", "SubLabel": "No. of events (%)" },
{ "Key": "women_km", "Label": "Women (n = 148)", "SubLabel": "10-y KM Rate (95% CI), %" },
{ "Key": "women_rate", "Label": "Women (n = 148)", "SubLabel": "Event Rate per Person-Year" },

{ "Key": "men_events", "Label": "Men (n = 1064)", "SubLabel": "No. of events (%)" },
{ "Key": "men_km", "Label": "Men (n = 1064)", "SubLabel": "10-y KM Rate (95% CI), %" },
{ "Key": "men_rate", "Label": "Men (n = 1064)", "SubLabel": "Event Rate per Person-Year" },

{ "Key": "model", "Label": "Model" },
{ "Key": "hazard_ratio", "Label": "Hazard Ratio (95% CI)" },
{ "Key": "p_value", "Label": "P Value" }
],
"RowGroups": [
{ "Key": "mortality", "Label": "Mortality Outcomes" },
{ "Key": "hospitalization", "Label": "Hospitalization Outcomes" }
],
"Rows": [
{
"Group": "mortality",
"Label": "All-cause mortality (Unadjusted)",
"Cells": {
"women_events": "73 (49.3)",
"women_km": "49.0 (40.8–57.3)",
"women_rate": "0.073",
"men_events": "684 (64.3)",
"men_km": "65.8 (62.7–68.8)",
"men_rate": "0.105",
"model": "Unadjusted",
"hazard_ratio": "0.70 (0.55–0.89)",
"p_value": "0.003"
}
},
{
"Group": "mortality",
"Label": "All-cause mortality (Adjusted)",
"Cells": {
"women_events": "73 (49.3)",
"women_km": "49.0 (40.8–57.3)",
"women_rate": "0.073",
"men_events": "684 (64.3)",
"men_km": "65.8 (62.7–68.8)",
"men_rate": "0.105",
"model": "Adjusted",
"hazard_ratio": "0.67 (0.52–0.86)",
"p_value": "0.002"
}
},
{
"Group": "mortality",
"Label": "Cardiovascular mortality (Unadjusted)",
"Cells": {
"women_events": "48 (32.4)",
"women_km": "34.3 (26.3–42.3)",
"women_rate": "0.048",
"men_events": "496 (46.6)",
"men_km": "52.3 (48.9–55.8)",
"men_rate": "0.076",
"model": "Unadjusted",
"hazard_ratio": "0.64 (0.48–0.86)",
"p_value": "0.003"
}
},
{
"Group": "mortality",
"Label": "Cardiovascular mortality (Adjusted)",
"Cells": {
"women_events": "48 (32.4)",
"women_km": "34.3 (26.3–42.3)",
"women_rate": "0.048",
"men_events": "496 (46.6)",
"men_km": "52.3 (48.9–55.8)",
"men_rate": "0.076",
"model": "Adjusted",
"hazard_ratio": "0.65 (0.48–0.89)",
"p_value": "0.006"
}
},
{
"Group": "hospitalization",
"Label": "Mortality or cardiovascular hospitalization (Unadjusted)",
"Cells": {
"women_events": "112 (75.7)",
"women_km": "76.6 (68.5–84.6)",
"women_rate": "0.111",
"men_events": "879 (82.6)",
"men_km": "85.2 (82.2–88.2)",
"men_rate": "0.135",
"model": "Unadjusted",
"hazard_ratio": "0.87 (0.72–1.06)",
"p_value": "0.180"
}
},
{
"Group": "hospitalization",
"Label": "Mortality or cardiovascular hospitalization (Adjusted)",
"Cells": {
"women_events": "112 (75.7)",
"women_km": "76.6 (68.5–84.6)",
"women_rate": "0.111",
"men_events": "879 (82.6)",
"men_km": "85.2 (82.2–88.2)",
"men_rate": "0.135",
"model": "Adjusted",
"hazard_ratio": "0.86 (0.70–1.05)",
"p_value": "0.144"
}
},
{
"Group": "mortality",
"Label": "Sudden cardiac death (Unadjusted)",
"Cells": {
"women_events": "21 (14.2)",
"women_km": "17.6 (10.7–24.5)",
"women_rate": "0.021",
"men_events": "249 (23.4)",
"men_km": "30.2 (26.8–33.7)",
"men_rate": "0.038",
"model": "Unadjusted",
"hazard_ratio": "0.56 (0.36–0.87)",
"p_value": "0.011"
}
},
{
"Group": "mortality",
"Label": "Sudden cardiac death (Adjusted)",
"Cells": {
"women_events": "21 (14.2)",
"women_km": "17.6 (10.7–24.5)",
"women_rate": "0.021",
"men_events": "249 (23.4)",
"men_km": "30.2 (26.8–33.7)",
"men_rate": "0.038",
"model": "Adjusted",
"hazard_ratio": "0.62 (0.39–0.97)",
"p_value": "0.038"
}
},
{
"Group": "mortality",
"Label": "HF death (Unadjusted)",
"Cells": {
"women_events": "10 (6.8)",
"women_km": "7.5 (2.7–12.3)",
"women_rate": "0.010",
"men_events": "148 (13.9)",
"men_km": "21.9 (18.5–25.4)",
"men_rate": "0.023",
"model": "Unadjusted",
"hazard_ratio": "0.44 (0.23–0.83)",
"p_value": "0.012"
}
},
{
"Group": "mortality",
"Label": "HF death (Adjusted)",
"Cells": {
"women_events": "10 (6.8)",
"women_km": "7.5 (2.7–12.3)",
"women_rate": "0.010",
"men_events": "148 (13.9)",
"men_km": "21.9 (18.5–25.4)",
"men_rate": "0.023",
"model": "Adjusted",
"hazard_ratio": "0.40 (0.21–0.77)",
"p_value": "0.007"
}
}
]
},
"Assigned Label": "Bias",
"Category": "Sampling Bias"
}


Example 4:
{
"Caption": "Table 4. Sex-Specific Differences and Recommendations in Guidelines",
"Table": {
"Columns": [
{ "Key": "document", "Label": "Clinical Practice Document" },
{ "Key": "differences", "Label": "Noted Sex-Specific Differences" },
{ "Key": "recommendations", "Label": "Noted Sex-Specific Recommendations" },
{ "Key": "knowledge_gaps", "Label": "Knowledge Gaps" }
],
"RowGroups": [
{ "Key": "guidelines", "Label": "Clinical Guidelines" }
],
"Rows": [
{
"Group": "guidelines",
"Label": "2013 ACC/AHA STEMI guideline",
"Cells": {
"document": "2013 ACC/AHA STEMI guideline",
"differences": "30% are women remain undertreated",
"recommendations": "None",
"knowledge_gaps": "Prehospital delay bleeding risks"
}
},
{
"Group": "guidelines",
"Label": "2014 ACC/AHA NSTEMI-ACS",
"Cells": {
"document": "2014 ACC/AHA NSTEMI-ACS",
"differences": "Pregnancy: revascularization if life-threatening complications",
"recommendations": "Early invasive strategy for high risk features",
"knowledge_gaps": "Antithrombotic dosing; Myocardial infarction with nonobstructive coronary arteries"
}
},
{
"Group": "guidelines",
"Label": "2012, 2014 update ACC/AHA stable ischemic heart disease",
"Cells": {
"document": "2012, 2014 update ACC/AHA stable ischemic heart disease",
"differences": "None for PCI, medications, CABG",
"recommendations": "Avoid estrogen replacement therapy in postmenopausal women",
"knowledge_gaps": "Nonobstructive disease diagnosis, treatment"
}
},
{
"Group": "guidelines",
"Label": "2011 ACC/AHA PCI",
"Cells": {
"document": "2011 ACC/AHA PCI",
"differences": "Higher in-hospital mortality; Higher procedural complications",
"recommendations": "None",
"knowledge_gaps": "Vascular access and bleeding risks"
}
},
{
"Group": "guidelines",
"Label": "2011 ACC/AHA CABG",
"Cells": {
"document": "2011 ACC/AHA CABG",
"differences": "Higher perioperative morbidity/mortality; Similar long-term outcomes",
"recommendations": "None; Most data extrapolated from men",
"knowledge_gaps": "Mitigating bleeding risks; Improving complete revascularization"
}
},
{
"Group": "guidelines",
"Label": "2020 ESC ACS without STEMI",
"Cells": {
"document": "2020 ESC ACS without STEMI",
"differences": "None. Noted to follow same treatment",
"recommendations": "Careful antithrombotic dosing periprocedural",
"knowledge_gaps": "Nonobstructive disease"
}
}
]
},
"Assigned Label": "Bias",
"Category": "Diagnostic Uncertainty / Bias"
}


Example 5:
{
"Caption": "Figure 2. Pooled crude and adjusted odds ratios of symptoms experienced by women relative to men. OR indicates odds ratio.",
"Figure": {
"FigureType": "forest_plot",
"OverallDescription": "This figure presents pooled crude and adjusted odds ratios (ORs) with 95% confidence intervals for a range of symptoms, comparing their occurrence in women relative to men. Odds ratios greater than 1 indicate symptoms more commonly reported by women, whereas odds ratios less than 1 indicate symptoms more commonly reported by men. Results are shown on a logarithmic scale and are based on varying numbers of contributing studies.",
"Axes": {
"x_axis": {
"label": "Odds ratio (log scale)",
"reference_line": "OR = 1 (no difference between women and men)",
"directionality": {
"left": "More common in men",
"right": "More common in women"
}
}
},
"Legend": {
"Symbols": {
"Square": "Point estimate of odds ratio",
"Horizontal line": "95% confidence interval"
}
},
"Panels": [
{
"PanelID": "crude_estimates",
"Title": "Crude odds ratios",
"Symptoms": [
{
"Symptom": "Pain between shoulder blades",
"Number_of_studies": 15,
"OR_95_CI": "2.15 [1.95, 2.37]"
},
{
"Symptom": "Neck pain",
"Number_of_studies": 7,
"OR_95_CI": "1.83 [1.60, 2.10]"
},
{
"Symptom": "Palpitations",
"Number_of_studies": 10,
"OR_95_CI": "1.80 [1.44, 2.26]"
},
{
"Symptom": "Jaw pain",
"Number_of_studies": 11,
"OR_95_CI": "1.75 [1.42, 2.17]"
},
{
"Symptom": "Nausea or vomiting",
"Number_of_studies": 19,
"OR_95_CI": "1.64 [1.48, 1.82]"
},
{
"Symptom": "Fatigue",
"Number_of_studies": 11,
"OR_95_CI": "1.36 [1.22, 1.52]"
},
{
"Symptom": "Shortness of breath",
"Number_of_studies": 22,
"OR_95_CI": "1.34 [1.21, 1.48]"
},
{
"Symptom": "Indigestion",
"Number_of_studies": 5,
"OR_95_CI": "1.31 [0.95, 1.81]"
},
{
"Symptom": "Dizziness or lightheadedness",
"Number_of_studies": 9,
"OR_95_CI": "1.28 [1.15, 1.44]"
},
{
"Symptom": "Syncope",
"Number_of_studies": 11,
"OR_95_CI": "1.24 [1.09, 1.42]"
},
{
"Symptom": "Stomach or epigastric pain",
"Number_of_studies": 11,
"OR_95_CI": "1.20 [0.94, 1.53]"
},
{
"Symptom": "Right arm or shoulder pain",
"Number_of_studies": 8,
"OR_95_CI": "1.09 [0.88, 1.35]"
},
{
"Symptom": "Left arm or shoulder pain",
"Number_of_studies": 12,
"OR_95_CI": "1.06 [0.88, 1.27]"
},
{
"Symptom": "Diaphoresis",
"Number_of_studies": 19,
"OR_95_CI": "0.84 [0.76, 0.94]"
},
{
"Symptom": "Chest pain",
"Number_of_studies": 26,
"OR_95_CI": "0.70 [0.63, 0.78]"
}
]
},
{
"PanelID": "adjusted_estimates",
"Title": "Adjusted odds ratios",
"Symptoms": [
{
"Symptom": "Pain between shoulder blades",
"Number_of_studies": 9,
"OR_95_CI": "1.89 [1.27, 2.82]"
},
{
"Symptom": "Neck pain",
"Number_of_studies": 4,
"OR_95_CI": "1.71 [1.00, 2.93]"
},
{
"Symptom": "Palpitations",
"Number_of_studies": 3,
"OR_95_CI": "1.91 [0.91, 4.00]"
},
{
"Symptom": "Jaw pain",
"Number_of_studies": 4,
"OR_95_CI": "1.67 [1.01, 2.78]"
},
{
"Symptom": "Nausea or vomiting",
"Number_of_studies": 10,
"OR_95_CI": "1.63 [1.21, 2.19]"
},
{
"Symptom": "Fatigue",
"Number_of_studies": 6,
"OR_95_CI": "1.34 [0.94, 1.90]"
},
{
"Symptom": "Shortness of breath",
"Number_of_studies": 11,
"OR_95_CI": "1.22 [1.01, 1.48]"
},
{
"Symptom": "Indigestion",
"Number_of_studies": 2,
"OR_95_CI": "1.55 [0.63, 3.83]"
},
{
"Symptom": "Dizziness or lightheadedness",
"Number_of_studies": 5,
"OR_95_CI": "1.41 [0.96, 2.07]"
},
{
"Symptom": "Syncope",
"Number_of_studies": 5,
"OR_95_CI": "1.08 [0.75, 1.56]"
},
{
"Symptom": "Stomach or epigastric pain",
"Number_of_studies": 6,
"OR_95_CI": "0.96 [0.75, 1.23]"
},
{
"Symptom": "Right arm or shoulder pain",
"Number_of_studies": 6,
"OR_95_CI": "1.03 [0.77, 1.38]"
},
{
"Symptom": "Left arm or shoulder pain",
"Number_of_studies": 8,
"OR_95_CI": "1.13 [0.93, 1.38]"
},
{
"Symptom": "Diaphoresis",
"Number_of_studies": 8,
"OR_95_CI": "0.75 [0.72, 0.78]"
},
{
"Symptom": "Chest pain",
"Number_of_studies": 8,
"OR_95_CI": "0.67 [0.62, 0.73]"
}
]
}
]
}, "Assigned Label": "No bias",
"Category": "Factual / Neutral Observed Outcome"
}
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
