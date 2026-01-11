import os
import gc
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    recall_score, precision_score, f1_score, confusion_matrix,
    accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
)

# ==============================================================================
#   USER CONFIGURATION SECTION
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Hardware Settings
# ------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------------------------
# 2. Task Selection
# ------------------------------------------------------------------------------
# Select Task ID: "T1", "T2", "T3", "T4", "T5", "T6"
TASK_ID = "T1"

# ------------------------------------------------------------------------------
# 3. INTERNAL CONFIGURATION (Auto-selected based on TASK_ID)
# ------------------------------------------------------------------------------

if TASK_ID == "T1":
    DATA_PATH = "/path/to/test_phish_email_list.csv"
    PROMPT_TEXT = "Label this as 1(phishing) or 0(safe)."
    IS_MULTICLASS = False
elif TASK_ID == "T2":
    DATA_PATH = "/path/to/test_malicious_phish_list.csv"
    PROMPT_TEXT = "Benign(0) or Cyberthreat(1)?"
    IS_MULTICLASS = False
elif TASK_ID == "T3":
    DATA_PATH = "/path/to/test_malicious_phish_multi_list.csv"
    PROMPT_TEXT = "Classify this URL."
    IS_MULTICLASS = True
elif TASK_ID == "T4":
    DATA_PATH = "/path/to/test_creditcard_timesplit_list.csv"
    PROMPT_TEXT = "fraud(1) or otherwise(0)?"
    IS_MULTICLASS = False
elif TASK_ID == "T5":
    DATA_PATH = "/path/to/UNSW_NB15_testing-set_list.csv"
    PROMPT_TEXT = "Normal(0) or Attack(1)?"
    IS_MULTICLASS = False
elif TASK_ID == "T6":
    DATA_PATH = "/path/to/UNSW_NB15_testing-set_multi_list.csv"
    PROMPT_TEXT = "Classify the following network event."
    IS_MULTICLASS = True
else:
    raise ValueError(f"Invalid TASK_ID: {TASK_ID}")

# Define the specific models to evaluate
MODEL_DIRS = [
    "/path/to/model_save_auprc/",       # Best LoRA model
    # "/path/to/model_save_full_auprc/" # Best Full Fine-Tuning model
]

# ==============================================================================
#   MAIN LOGIC
# ==============================================================================

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")
print(f"Task: {TASK_ID} | Multiclass: {IS_MULTICLASS}")

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

print(f"Loading test data from: {DATA_PATH}")
df_test = pd.read_csv(DATA_PATH)

# Handle Label Encoding
if IS_MULTICLASS:
    df_test['label'] = df_test['label'].astype('category').cat.codes
    NUM_LABELS = df_test['label'].nunique()
    print(f"Detected {NUM_LABELS} unique labels for multi-class task.")
else:
    NUM_LABELS = 2
    print("Binary classification (2 labels).")

# ------------------------------------------------------------------------------
# 2. Encoding Function
# ------------------------------------------------------------------------------
def encode_data(data_texts, data_labels, tokenizer):
    input_ids_list, attention_masks_list = [], []
    
    for text in data_texts:
        full_input = f"{PROMPT_TEXT} {text}"
        encoded = tokenizer.encode_plus(
            full_input,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids_list.append(encoded["input_ids"])
        attention_masks_list.append(encoded["attention_mask"])

    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_masks_list, dim=0)
    labels = torch.tensor(data_labels.values, dtype=torch.long)

    return TensorDataset(input_ids, attention_mask, labels)

# ------------------------------------------------------------------------------
# 3. Evaluation Function
# ------------------------------------------------------------------------------
def evaluate_model(model_dir):
    if not os.path.exists(model_dir):
        print(f"Warning: Model directory {model_dir} does not exist. Skipping.")
        return None

    print(f"\n=== Evaluating {model_dir} ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=NUM_LABELS).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.eval()

    test_ds = encode_data(df_test["data"], df_test["label"], tokenizer)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print("Warming up GPU...")
    if len(test_ds) > 0:
        dummy_input, dummy_mask, _ = test_ds[0]
        dummy_input = dummy_input.unsqueeze(0).to(device)
        dummy_mask = dummy_mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_input, attention_mask=dummy_mask)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()

    print(f"Starting inference on {len(test_ds)} samples...")
    
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    
    y_true = []
    y_prob_all = []
    
    with torch.no_grad():
        for input_ids, masks, labels in test_dl:
            input_ids = input_ids.to(device)
            masks = masks.to(device)
            
            starter.record()
            
            logits = model(input_ids, attention_mask=masks).logits
            probs = F.softmax(logits, dim=1)
            
            ender.record()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

            y_true.extend(labels.cpu().numpy())
            y_prob_all.extend(probs.cpu().numpy())

    timings = np.array(timings)
    avg_latency_ms = np.mean(timings)
    std_latency_ms = np.std(timings)
    
    print("-" * 40)
    print(f"Inference Latency (Batch=1): {avg_latency_ms:.4f} ms ± {std_latency_ms:.4f} ms")
    print("-" * 40)

    results = {}
    results["avg_latency_ms"] = avg_latency_ms
    results["model_dir"] = os.path.basename(model_dir.rstrip('/'))

    # --- Metrics Calculation ---
    y_true = np.array(y_true)
    y_prob_all = np.array(y_prob_all)
    y_pred = np.argmax(y_prob_all, axis=1)

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["mcc"] = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    if IS_MULTICLASS:
        results["recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results["precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results["f1-score"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results["auroc"] = roc_auc_score(y_true, y_prob_all, multi_class='ovr', average='macro')
        results["auprc"] = average_precision_score(y_true, y_prob_all, average='macro')
        results["specificity"] = "N/A"
        results["npv"] = "N/A"
    else:
        y_prob_pos = y_prob_all[:, 1]
        tn, fp, fn, tp = cm.ravel()
        
        results["recall"] = recall_score(y_true, y_pred)
        results["precision"] = precision_score(y_true, y_pred, zero_division=0)
        results["f1-score"] = f1_score(y_true, y_pred)
        results["auroc"] = roc_auc_score(y_true, y_prob_pos)
        results["auprc"] = average_precision_score(y_true, y_prob_pos)
        results["specificity"] = (tn / (tn + fp)) if (tn + fp) else 0.0
        results["npv"] = (tn / (tn + fn)) if (tn + fn) else 0.0
        
        print(f"Detailed Binary Stats: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print(f"AUPRC: {results['auprc']:.4f} | F1-score: {results['f1-score']:.4f} | Recall: {results['recall']:.4f}")

    return results

# ------------------------------------------------------------------------------
# 4. Execution Loop
# ------------------------------------------------------------------------------
all_metrics = []
for mdir in MODEL_DIRS:
    metrics = evaluate_model(mdir)
    if metrics:
        all_metrics.append(metrics)
    
    torch.cuda.empty_cache()
    gc.collect()

# ------------------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------------------
print("\n============================ OVERALL SUMMARY ============================")
if not all_metrics:
    print("No models were successfully evaluated.")
else:
    summary = sorted(all_metrics, key=lambda x: x["auprc"], reverse=True)
    
    # Header: AUPRC | F1-score | Recall | Prec | AUROC | Acc | MCC
    print(f"{'Model Directory':>25} | {'AUPRC':>6} | {'F1-score':>8} | {'Recall':>6} | {'Prec':>6} | {'AUROC':>6} | {'Acc':>6} | {'MCC':>6}")
    print("-" * 115)
    
    for m in summary:
        print(
            f"{m['model_dir']:>25} | "
            f"{m['auprc']:6.4f} | "      # 1. AUPRC
            f"{m['f1-score']:8.4f} | "   # 2. F1-score
            f"{m['recall']:6.4f} | "     # 3. Recall
            f"{m['precision']:6.4f} | "  # 4. Precision
            f"{m['auroc']:6.4f} | "      # 5. AUROC
            f"{m['accuracy']:6.4f} | "   # 6. Accuracy
            f"{m['mcc']:6.4f}"           # 7. MCC
        )

# ------------------------------------------------------------------------------
# 6. Time Tracking
# ------------------------------------------------------------------------------
elapsed = time.time() - start_time
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"\nTotal time elapsed: {int(h)}h {int(m)}m {int(s)}s")