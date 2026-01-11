import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import time
import gc

start_time = time.time()

# Device Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. Load Datasets ---
# PLEASE UPDATE THESE PATHS TO YOUR ACTUAL DATA LOCATIONS
train_path = "/path/to/UNSW_NB15_training-set.parquet"
test_path = "/path/to/UNSW_NB15_testing-set.parquet"

print("Loading data...")
try:
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
except FileNotFoundError:
    print("\nError: Training or testing file not found.")
    print("Please verify the file paths.\n")
    exit()

print(f"Training data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")
print("-" * 80)

# --- 2. Prepare Features and Target ---
print("Preparing features and target...")
target_col = 'attack_cat'
drop_cols = ['label']

X_train = df_train.drop(columns=[target_col] + drop_cols)
y_train = df_train[target_col]
X_test = df_test.drop(columns=[target_col] + drop_cols)
y_test = df_test[target_col]

# --- 3. Encode Target Variable ---
print("Encoding target variable...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

class_names = label_encoder.classes_
num_classes = len(class_names)
print(f"Target classes ({num_classes}): {class_names}")

# --- 4. Preprocess Features ---
print("Preprocessing features...")
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align columns
train_cols = X_train.columns
X_test = X_test.reindex(columns=train_cols, fill_value=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# --- 5. Model Training ---
print("Training XGBoost model...")
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    random_state=0,
    subsample=0.8,
    tree_method='hist',
    device='cuda'
)

xgb_classifier.fit(X_train_scaled, y_train_encoded)
print("Training complete.")
print("-" * 80)

# ===================================================================
# 6. SHAP Explainability Analysis
# ===================================================================

def display_multiclass_report(shap_explanation, class_index, class_names, top_n=5, title="Feature Contributions"):
    """
    Extracts and formats SHAP data for a specific class in a multi-class problem.
    """
    target_class_name = class_names[class_index]
    print(f"\n{title} (Contribution to class '{target_class_name}')")
    
    feature_names = shap_explanation.feature_names
    shap_values_for_class = shap_explanation.values[:, class_index]
    
    feature_shap_pairs = [
        (name, val) 
        for name, val in zip(feature_names, shap_values_for_class) 
        if val != 0
    ]
    
    positive_contributors = sorted([p for p in feature_shap_pairs if p[1] > 0], key=lambda x: x[1], reverse=True)
    negative_contributors = sorted([p for p in feature_shap_pairs if p[1] < 0], key=lambda x: x[1])

    print("-" * 60)
    print(f"  Top {top_n} Features Pushing TOWARDS '{target_class_name}':")
    print("-" * 60)
    if not positive_contributors:
        print("  (None)")
    else:
        for feature, value in positive_contributors[:top_n]:
            print(f"  {feature:<30} | SHAP Value: {value:+.6f}")

    print("\n" + "-" * 60)
    print(f"  Top {top_n} Features Pushing AWAY from '{target_class_name}':")
    print("-" * 60)
    if not negative_contributors:
        print("  (None)")
    else:
        for feature, value in negative_contributors[:top_n]:
            print(f"  {feature:<30} | SHAP Value: {value:+.6f}")
    print("-" * 60)

print("\n===== SHAP Analysis (XGBoost Multi-Class) =====")

# --- 6.1 Create Explainer ---
print("Creating SHAP explainer...")
explainer = shap.Explainer(xgb_classifier, X_train_scaled)

# --- 6.2 Calculate SHAP Values ---
print("Calculating SHAP values for test set...")
shap_values = explainer(X_test_scaled)

# --- 6.3 Local Interpretability ---
print("\n--- Local Interpretability (Single Sample) ---")

# Identify indices for specific classes
try:
    class_names_list = class_names.tolist()
    normal_class_index = class_names_list.index('Normal')
    attack_class_index = class_names_list.index('Generic')
except ValueError:
    print("Warning: Specific classes not found. Defaulting to indices 0 and 1.")
    normal_class_index = 0
    attack_class_index = 1

# Select samples
try:
    attack_idx = np.where(y_test_encoded == attack_class_index)[0][0]
    normal_idx = np.where(y_test_encoded == normal_class_index)[0][0]
except IndexError:
    print("Error: Could not find samples for specified classes.")
    exit()

# Analyze Attack Sample
display_multiclass_report(
    shap_values[attack_idx], 
    attack_class_index, 
    class_names, 
    top_n=5,
    title=f"Sample Analysis (Index: {attack_idx})"
)

# Analyze Normal Sample
display_multiclass_report(
    shap_values[normal_idx], 
    normal_class_index, 
    class_names, 
    top_n=5,
    title=f"Sample Analysis (Index: {normal_idx})"
)

# --- 6.4 Global Interpretability ---
print("\n--- Global Interpretability ---")

TOP_N_GLOBAL = 5 

# Calculate mean absolute SHAP value per feature
mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))

shap_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Mean Abs SHAP Value': mean_abs_shap
})

sorted_df = shap_df.sort_values(by='Mean Abs SHAP Value', ascending=False)
full_top_n_df = sorted_df.head(TOP_N_GLOBAL).reset_index(drop=True)
full_top_n_df.index += 1 

print(f"\nTop {TOP_N_GLOBAL} Features by Global Importance:")
print(full_top_n_df.to_string())
print("\n" + "="*80)

# ===================================================================
# 7. Evaluation
# ===================================================================
def evaluate_xgboost_multiclass(y_true, y_prob, num_classes):
    print("\n" + "="*20 + " Model Evaluation " + "="*20)

    y_pred = np.argmax(y_prob, axis=1)
    y_true_one_hot = label_binarize(y_true, classes=np.arange(num_classes))

    # Metrics (Decimal format)
    accuracy = accuracy_score(y_true, y_pred)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Fix probability shape if necessary
    if y_prob.shape[1] != num_classes:
        if y_prob.shape[1] == 1 and num_classes == 2:
             y_prob_fixed = np.hstack([1 - y_prob, y_prob])
        else:
             y_prob_fixed = y_prob
    else:
        y_prob_fixed = y_prob

    auroc_macro = roc_auc_score(y_true, y_prob_fixed, multi_class='ovr', average='macro')
    auprc_macro = average_precision_score(y_true_one_hot, y_prob_fixed, average='macro')

    # Print Report
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Recall (macro)       : {recall_macro:.4f}")
    print(f"Precision (macro)    : {precision_macro:.4f}")
    print(f"F1-score (macro)     : {f1_macro:.4f}")
    print(f"MCC                  : {mcc:.4f}")
    print(f"AUROC (macro, ovr)   : {auroc_macro:.4f}")
    print(f"AUPRC (macro)        : {auprc_macro:.4f}")
    print("="*58)

# --- 8. Prediction & Evaluation ---
print(f"\nPredicting and Evaluating on test set...")
y_probabilities = xgb_classifier.predict_proba(X_test_scaled)

evaluate_xgboost_multiclass(y_test_encoded, y_probabilities, num_classes)

# --- 9. Final Timing ---
elapsed = time.time() - start_time
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"\nTotal time elapsed: {int(h)}h {int(m)}m {int(s)}s")

gc.collect()