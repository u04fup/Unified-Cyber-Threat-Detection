import os
import re
import numpy as np
import pandas as pd
import shap
import torch
from IPython.display import display
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

# Device Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ===================================================================
# 1. Configuration
# ===================================================================
# --- User Defined Paths ---
MODEL_DIR = "/path/to/model_save_auprc/"
TEST_DATA_PATH = "/path/to/UNSW_NB15_testing-set_multi_list.csv"     # Serialized data

PROMPT_QUESTION = "Classify the following network event."
TOP_N_FEATURES = 5  # Limit for reporting top features

# ===================================================================
# 2. Load Model, Tokenizer, and Data
# ===================================================================
print(f"\nLoading model and tokenizer from '{MODEL_DIR}'...")
peft_config = PeftConfig.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=10,
    ignore_mismatched_sizes=True
)
model = PeftModel.from_pretrained(model, MODEL_DIR)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"Loading serialized test data from '{TEST_DATA_PATH}'...")
df_test = pd.read_csv(TEST_DATA_PATH)

# Retrieve class labels from the model config
class_labels = list(model.config.id2label.values())
print(f"Model loaded. Detected {len(class_labels)} classes: {class_labels}")

N_SAMPLES_FOR_GLOBAL = len(df_test)

# ===================================================================
# 3. Initialize SHAP Explainer
# ===================================================================
print("\nCreating Hugging Face Pipeline...")
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=model.device,
    top_k=None  # Return scores for all classes
)

print("Creating SHAP Explainer...")
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(classifier, masker)

# ===================================================================
# 4. Local Interpretability (Single Sample Analysis)
# ===================================================================
print("\n===== Phase 1: Local Interpretability =====")

# --- 4.1 Parse Feature Names ---
# We extract feature names from the first data string to map tokens back to features.
print("Parsing feature names...")
first_data_string = df_test['data'].iloc[0]
original_feature_names = [match.strip() for match in re.findall(r'(\w+):', first_data_string)]
feature_name_set = set(original_feature_names)
print(f"Parsed {len(original_feature_names)} feature names.")

# --- 4.2 Helper Function: Feature-Level Report ---
def display_top_features_report_local(shap_explanation, feature_name_set, top_n=5, title="Top Contributing Features:"):
    """
    Aggregates Token-level SHAP values into Feature-level SHAP values.
    It sums up values for tokens like 'proto', ':', 'udp', ';' into a single 'proto' feature score.
    """
    print(f"\n{title}")
    
    tokens = shap_explanation.data
    shap_vals = shap_explanation.values
    
    feature_shap_pairs = []
    current_feature_name = None
    current_feature_shaps = []

    # Iterate through tokens to aggregate values
    for j, token in enumerate(tokens):
        cleaned_token = token.replace('Ġ', '').strip()

        # Check if the token marks the start of a known feature
        if cleaned_token in feature_name_set:
            # If we were tracking a feature, save it
            if current_feature_name and current_feature_shaps:
                aggregated_shap = sum(current_feature_shaps)
                feature_shap_pairs.append((current_feature_name, aggregated_shap))
            
            # Start tracking new feature
            current_feature_name = cleaned_token
            current_feature_shaps = [shap_vals[j] if shap_vals[j] is not None else 0]
        
        # Add values to the currently tracked feature (including values like ':' or data values)
        elif current_feature_name:
            if shap_vals[j] is not None:
                current_feature_shaps.append(shap_vals[j])

    # Handle the last feature
    if current_feature_name and current_feature_shaps:
        aggregated_shap = sum(current_feature_shaps)
        feature_shap_pairs.append((current_feature_name, aggregated_shap))

    # Sort and Display
    positive_contributors = sorted([p for p in feature_shap_pairs if p[1] > 0], key=lambda x: x[1], reverse=True)
    negative_contributors = sorted([p for p in feature_shap_pairs if p[1] < 0], key=lambda x: x[1])

    print("-" * 50)
    print(f"   Top {top_n} Features (Pushing towards target):")
    print("-" * 50)
    if not positive_contributors:
        print("   (None)")
    else:
        for feature, value in positive_contributors[:top_n]:
            print(f"   {feature:<20} | SHAP Value: {value:+.6f}")

    print("\n" + "-" * 50)
    print(f"   Top {top_n} Features (Pulling away from target):")
    print("-" * 50)
    if not negative_contributors:
        print("   (None)")
    else:
        for feature, value in negative_contributors[:top_n]:
            print(f"   {feature:<20} | SHAP Value: {value:+.6f}")
    print("-" * 50)

# --- 4.3 Execute Local Analysis ---

# Select samples
normal_label_str = "Normal"
try:
    normal_sample = df_test[df_test['label'] == normal_label_str].iloc[0]
    attack_sample = df_test[df_test['label'] != normal_label_str].iloc[0]
except IndexError:
    raise RuntimeError("Error: Could not find both 'Normal' and 'Attack' samples in the test set.")

inputs_to_explain = [
    f"{PROMPT_QUESTION} {normal_sample['data']}",
    f"{PROMPT_QUESTION} {attack_sample['data']}"
]

print(f"Analyzing Normal Sample (Label: {normal_sample['label']})")
print(f"Analyzing Attack Sample (Label: {attack_sample['label']})")

shap_values_single = explainer(inputs_to_explain)

# 1. Analyze Normal Sample (Target: 'Normal' / LABEL_6)
normal_model_label = class_labels[6] 
print(f"\n--- Normal Sample Analysis (Target: {normal_model_label}) ---")
display(shap.plots.text(shap_values_single[0, :, normal_model_label]))

display_top_features_report_local(
    shap_values_single[0, :, normal_model_label],
    feature_name_set,
    top_n=TOP_N_FEATURES,
    title=f"Top Contributing Features for '{normal_model_label}' (Normal Sample):"
)

# 2. Analyze Attack Sample (Target: 'Reconnaissance' / LABEL_7)
attack_model_label = class_labels[7] 
print(f"\n--- Attack Sample Analysis (Target: {attack_model_label}) ---")
display(shap.plots.text(shap_values_single[1, :, attack_model_label]))

display_top_features_report_local(
    shap_values_single[1, :, attack_model_label],
    feature_name_set,
    top_n=TOP_N_FEATURES,
    title=f"Top Contributing Features for '{attack_model_label}' (Attack Sample):"
)

# ===================================================================
# 5. Global Feature Importance (Aggregation)
# ===================================================================
print("\n===== Phase 2: Global Feature Importance =====")
print(f"Calculating SHAP values for {N_SAMPLES_FOR_GLOBAL} samples...")

subset_texts = [f"{PROMPT_QUESTION} {row['data']}" for _, row in df_test.head(N_SAMPLES_FOR_GLOBAL).iterrows()]
global_shap_values = explainer(subset_texts)

# ===================================================================
# 6. Aggregating Token-SHAP to Feature-SHAP
# ===================================================================
print("\n===== Phase 3: Aggregating SHAP Values Across All Classes =====")

feature_map = {name: i for i, name in enumerate(original_feature_names)}
num_samples = N_SAMPLES_FOR_GLOBAL
num_features = len(original_feature_names)
num_classes = len(class_labels)

# Initialize 3D array: (samples, features, classes)
aggregated_shap_values_all_classes = np.zeros((num_samples, num_features, num_classes))

print("Aggregating SHAP values...")

for c, label_name in enumerate(class_labels):
    # Get SHAP values for current class
    shap_exp_for_plot = global_shap_values[:, :, label_name]

    for i in range(num_samples):
        tokens = shap_exp_for_plot.data[i]
        shap_vals = shap_exp_for_plot.values[i]
        
        current_feature_name = None
        current_feature_shaps = []

        for j, token in enumerate(tokens):
            cleaned_token = token.replace('Ġ', '').strip()

            if cleaned_token in feature_name_set:
                # Save previous feature
                if current_feature_name and current_feature_shaps:
                    if current_feature_name in feature_map:
                        feature_idx = feature_map[current_feature_name]
                        aggregated_shap_values_all_classes[i, feature_idx, c] = sum(current_feature_shaps)
                
                # Start new feature
                current_feature_name = cleaned_token
                current_feature_shaps = [shap_vals[j] if shap_vals[j] is not None else 0]
            
            elif current_feature_name:
                if shap_vals[j] is not None:
                    current_feature_shaps.append(shap_vals[j])

        # Save last feature
        if current_feature_name and current_feature_shaps and current_feature_name in feature_map:
            feature_idx = feature_map[current_feature_name]
            aggregated_shap_values_all_classes[i, feature_idx, c] = sum(current_feature_shaps)

print("Aggregation complete.")

# ===================================================================
# 7. Reporting Global Importance
# ===================================================================
print(f"\n--- Top {TOP_N_FEATURES} Overall Global Feature Importance ---")
print("Ranking features by Mean Absolute SHAP value across all classes.")

# Calculate Mean Absolute SHAP across Samples (axis 0) and Classes (axis 2)
mean_abs_shap = np.mean(np.abs(aggregated_shap_values_all_classes), axis=(0, 2))

df_feature_importance = pd.DataFrame({
    'Feature': original_feature_names,
    'Mean Abs SHAP': mean_abs_shap
})

df_feature_importance_sorted = df_feature_importance.sort_values(
    by='Mean Abs SHAP',
    ascending=False
).reset_index(drop=True)

display(df_feature_importance_sorted.head(TOP_N_FEATURES))
print("\nGlobal analysis complete.")