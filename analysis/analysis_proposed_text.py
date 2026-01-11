import os
import torch
import pandas as pd
import shap
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import defaultdict

# Device Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ===================================================================
# 1. Configuration
# ===================================================================
# --- User Defined Paths ---
MODEL_DIR = "/path/to/model_save_auprc/"
TEST_DATA_PATH = "/path/to/test_phish_email_list.csv"
OUTPUT_HTML_DIR = "./shap_html_outputs"

# --- Model & Prompt Settings ---
MODEL_MAX_LENGTH = 512
PROMPT_QUESTION = "Label this as 1(phishing) or 0(safe)."
TOP_K_TOKENS = 5  # Number of top tokens to display

# ===================================================================
# 2. Initialization
# ===================================================================
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load Model and Tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load Data
df_test = pd.read_csv(TEST_DATA_PATH)
print(f"Test data loaded: {len(df_test)} samples.")

# Initialize SHAP Javascript (for notebook environments)
shap.initjs()

# Create Pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=model.device,
    top_k=None,
    truncation=True,
    max_length=MODEL_MAX_LENGTH
)

# Create Explainer
explainer = shap.Explainer(classifier, shap.maskers.Text(tokenizer))

# Create Output Directory
os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)

# ===================================================================
# 3. Helper Functions
# ===================================================================

def display_token_report(shap_values_sample, class_label, top_n=5, title="Token Contribution"):
    """Prints a text-based waterfall report of top tokens."""
    print(f"\n{title}")
    
    # Extract data for the specific class
    shap_explanation = shap_values_sample[:, class_label]
    tokens = shap_explanation.data
    values = shap_explanation.values
    
    # Filter and pair tokens
    token_shap_pairs = [
        (token.replace("Ġ", "").strip(), value) 
        for token, value in zip(tokens, values) 
        if token.replace("Ġ", "").strip()
    ]
    
    # Sort
    positive = sorted([p for p in token_shap_pairs if p[1] > 0], key=lambda x: x[1], reverse=True)
    negative = sorted([p for p in token_shap_pairs if p[1] < 0], key=lambda x: x[1])

    print("-" * 50)
    print(f"  Top {top_n} Tokens (Pushing towards '{class_label}'):")
    print("-" * 50)
    for token, value in positive[:top_n]:
        print(f"  {token:<20} | SHAP Value: {value:+.6f}")

    print("\n" + "-" * 50)
    print(f"  Top {top_n} Tokens (Pulling away from '{class_label}'):")
    print("-" * 50)
    for token, value in negative[:top_n]:
        print(f"  {token:<20} | SHAP Value: {value:+.6f}")
    print("-" * 50)

def save_html_plots(shap_slice, filename_prefix):
    """Saves SHAP text and force plots as HTML files."""
    # Text Plot
    html_content = shap.plots.text(shap_slice, display=False)
    full_html = f"""
    <html><head><meta charset="utf-8"><title>SHAP Text Plot</title>
    <style>body {{ font-family: sans-serif; padding: 20px; }}</style>
    </head><body>{html_content}</body></html>
    """
    text_path = os.path.join(OUTPUT_HTML_DIR, f"{filename_prefix}_text_plot.html")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_html)    
    print(f"Saved plots to: {OUTPUT_HTML_DIR} ({filename_prefix})")

# ===================================================================
# 4. Local Explainability (Single Sample Analysis)
# ===================================================================
print("\n===== Local Explainability Analysis =====")

try:
    phish_sample = df_test[df_test['label'] == 1].iloc[1]
    legit_sample = df_test[df_test['label'] == 0].iloc[0]
except IndexError:
    raise ValueError("Dataset must contain at least one example of label 0 and 1.")

# Prepare inputs
inputs = [
    f"{PROMPT_QUESTION} {phish_sample['data']}",
    f"{PROMPT_QUESTION} {legit_sample['data']}"
]

print("Calculating SHAP values for local samples...")
shap_values_local = explainer(inputs, silent=True)

# --- Analysis: Phishing Sample ---
print(f"\n--- Analyzing Phishing Sample (ID: {phish_sample.name}) ---")
display_token_report(shap_values_local[0], "LABEL_1", top_n=TOP_K_TOKENS, title="Contribution to 'Phishing' (LABEL_1)")
save_html_plots(shap_values_local[0, :, "LABEL_1"], "phishing_sample")

# --- Analysis: Legit Sample ---
print(f"\n--- Analyzing Safe Sample (ID: {legit_sample.name}) ---")
display_token_report(shap_values_local[1], "LABEL_0", top_n=TOP_K_TOKENS, title="Contribution to 'Safe' (LABEL_0)")
save_html_plots(shap_values_local[1, :, "LABEL_0"], "legit_sample")

# ===================================================================
# 5. Global Explainability (Full Dataset)
# ===================================================================
print(f"\n===== Global Explainability Analysis (Total Samples: {len(df_test)}) =====")

# Prepare all texts
all_texts = [f"{PROMPT_QUESTION} {row['data']}" for _, row in df_test.iterrows()]

print("Calculating SHAP values for the entire test set (this may take time)...")
global_shap_values = explainer(all_texts, silent=True)

# Focus on the 'Phishing' class (LABEL_1)
shap_values_phishing = global_shap_values[:, :, "LABEL_1"]

print("Aggregating global token importance...")
token_shap_values = defaultdict(list)

# Aggregate values
for i in range(len(shap_values_phishing.values)):
    for token, val in zip(shap_values_phishing.data[i], shap_values_phishing.values[i]):
        clean_token = token.replace("Ġ", "").strip()
        if clean_token:
            token_shap_values[clean_token].append(val)

# Calculate metrics
feature_importances = []
for token, values in token_shap_values.items():
    feature_importances.append({
        'Token': token,
        'Avg_Abs_SHAP': np.abs(values).mean(),
        'Count': len(values)
    })

# Create DataFrame and Sort
importance_df = pd.DataFrame(feature_importances)
sorted_df = importance_df.sort_values(by='Avg_Abs_SHAP', ascending=False)

print(f"\n--- Top {TOP_K_TOKENS} Global Tokens (Impact on Phishing Prediction) ---")
print(sorted_df.head(TOP_K_TOKENS).reset_index(drop=True).to_string())