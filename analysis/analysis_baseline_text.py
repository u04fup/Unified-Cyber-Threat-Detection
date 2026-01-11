import time
import pandas as pd
import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, f1_score, confusion_matrix,
    accuracy_score, roc_auc_score, average_precision_score
)
from IPython.display import display

# ===================================================================
# 1. Configuration & Initialization
# ===================================================================
start_time = time.time()

TRAIN_PATH = "/path/to/train_phish_email_list.csv"
TEST_PATH = "/path/to/test_phish_email_list.csv"

TOP_N_TOKENS = 5
RANDOM_SEED = 0

print(f"TF-IDF + Logistic Regression Phishing Email Classification (SHAP Analysis)")
print("="*80)

# ===================================================================
# 2. Load Data
# ===================================================================
print("Step 1: Loading data...")
try:
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
except FileNotFoundError:
    raise FileNotFoundError("Please ensure train.csv and test.csv are in the specified paths.")

df_train['data'].fillna('', inplace=True)
df_test['data'].fillna('', inplace=True)

y_train_full = df_train['label']
y_true = df_test['label'].values

print(f"Training data shape (Full): {df_train.shape}")
print(f"Test data shape: {df_test.shape}")
print("-" * 80)

# ===================================================================
# 3. TF-IDF Vectorization
# ===================================================================
print("Step 2: Performing TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    max_features=15000, 
    stop_words='english', 
    ngram_range=(1, 2), 
    sublinear_tf=True
)

X_train_tfidf_full = vectorizer.fit_transform(df_train['data'])
X_test_tfidf = vectorizer.transform(df_test['data'])

print(f"TF-IDF features shape (Full Train): {X_train_tfidf_full.shape}")
print(f"TF-IDF features shape (Test): {X_test_tfidf.shape}")
print("-" * 80)

# ===================================================================
# 4. Model Training
# ===================================================================
print("Step 3: Subsampling and Training Logistic Regression model...")

# Use 80% of training data for model fitting
X_sub, _, y_sub, _ = train_test_split(
    X_train_tfidf_full, y_train_full, 
    train_size=0.8, 
    random_state=RANDOM_SEED, 
    stratify=y_train_full
)

print(f"Training on subsampled data shape: {X_sub.shape}")

model = LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, max_iter=1000)
model.fit(X_sub, y_sub)

print("Model training complete.")
print("-" * 80)

# ===================================================================
# 5. SHAP Explainability Analysis
# ===================================================================

def display_baseline_report(shap_explanation, top_n=5, title="Top Contributing Tokens:"):
    print(f"\n{title}")
    
    feature_names = shap_explanation.feature_names
    shap_values = shap_explanation.values
    
    token_shap_pairs = [
        (name, val) 
        for name, val in zip(feature_names, shap_values) 
        if val != 0
    ]
    
    positive_contributors = sorted([p for p in token_shap_pairs if p[1] > 0], key=lambda x: x[1], reverse=True)
    negative_contributors = sorted([p for p in token_shap_pairs if p[1] < 0], key=lambda x: x[1])

    print("-" * 50)
    print(f"  Top {top_n} Tokens Pushing towards 'Phishing' (Red):")
    print("-" * 50)
    if not positive_contributors:
        print("  (None)")
    else:
        for token, value in positive_contributors[:top_n]:
            print(f"  {token:<20} | SHAP Value: {value:+.6f}")

    print("\n" + "-" * 50)
    print(f"  Top {top_n} Tokens Pushing towards 'Safe' (Blue):")
    print("-" * 50)
    if not negative_contributors:
        print("  (None)")
    else:
        for token, value in negative_contributors[:top_n]:
            print(f"  {token:<20} | SHAP Value: {value:+.6f}")
    print("-" * 50)

print("\n===== SHAP Explainability Analysis =====")

# --- 5.1 Create SHAP Explainer ---
print("Step 5.1: Creating SHAP explainer...")
explainer = shap.Explainer(model, X_sub, feature_names=vectorizer.get_feature_names_out())

# --- 5.2 Calculate SHAP Values for Test Set ---
print("Step 5.2: Calculating SHAP values...")
shap_values = explainer(X_test_tfidf)

# --- 5.3 Local Analysis: Single Sample ---
print("\n--- Phase 1: Local Explainability ---")
shap.initjs()

try:
    phish_idx = np.where(y_true == 1)[0][1]
    legit_idx = np.where(y_true == 0)[0][0]
except IndexError:
    raise ValueError("Dataset must contain at least one example of label 0 and 1.")

# Analyze Phishing Sample
print(f"\n--- Analyzing Phishing Email (Index: {phish_idx}) ---")
print("Email snippet:", df_test['data'].iloc[phish_idx][:200] + "...")
display_baseline_report(shap_values[phish_idx], top_n=TOP_N_TOKENS)
print("\n[Visual] Interactive Force Plot (Phishing):")
display(shap.plots.force(shap_values[phish_idx]))

# Analyze Legitimate Sample
print(f"\n--- Analyzing Safe Email (Index: {legit_idx}) ---")
print("Email snippet:", df_test['data'].iloc[legit_idx][:200] + "...")
display_baseline_report(shap_values[legit_idx], top_n=TOP_N_TOKENS)
print("\n[Visual] Interactive Force Plot (Safe):")
display(shap.plots.force(shap_values[legit_idx]))

# --- 5.4 Global Analysis: Full Test Set ---
print("\n--- Phase 2: Global Explainability ---")

mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

shap_df = pd.DataFrame({
    'Token': vectorizer.get_feature_names_out(),
    'Mean Abs SHAP Value': mean_abs_shap
})

sorted_df = shap_df.sort_values(by='Mean Abs SHAP Value', ascending=False)
full_top_n_df = sorted_df.head(TOP_N_TOKENS).reset_index(drop=True)
full_top_n_df.index += 1

print(f"\n--- Global Top {TOP_N_TOKENS} Tokens (Highest Average Impact) ---")
print(full_top_n_df.to_string())
print("\n" + "="*80)

# ===================================================================
# 6. Prediction
# ===================================================================
print("\nStep 6: Predicting probabilities...")
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
y_pred_default = (y_prob >= 0.5).astype(np.int32)

# ===================================================================
# 7. Evaluation
# ===================================================================
print("\n=== Evaluating TF-IDF + Logistic Regression ===")

cm = confusion_matrix(y_true, y_pred_default)
tn, fp, fn, tp = cm.ravel()

recall_val    = recall_score(y_true, y_pred_default)
spec_val      = (tn / (tn + fp)) if (tn + fp) else 0.0
precision_val = precision_score(y_true, y_pred_default, zero_division=0)
npv_val       = (tn / (tn + fn)) if (tn + fn) else 0.0
f1_val        = f1_score(y_true, y_pred_default)
acc_val       = accuracy_score(y_true, y_pred_default)
auroc_val     = roc_auc_score(y_true, y_prob)
auprc_val     = average_precision_score(y_true, y_prob)

print(f"Confusion Matrix (threshold=0.5) TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Recall        : {recall_val:.4f}")
print(f"Specificity   : {spec_val:.4f}")
print(f"Precision     : {precision_val:.4f}")
print(f"NPV           : {npv_val:.4f}")
print(f"F1-score      : {f1_val:.4f}")
print(f"Accuracy      : {acc_val:.4f}")
print(f"AUROC         : {auroc_val:.4f}")
print(f"AUPRC         : {auprc_val:.4f}")