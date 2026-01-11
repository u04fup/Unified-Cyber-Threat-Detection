import os
import time
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

# ==============================================================================
#   USER CONFIGURATION SECTION
# ==============================================================================

# 1. Hardware & Mode Settings
# ------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Select Task ID: "T1", "T2", "T3", "T4", "T5", "T6"
TASK_ID = "T1" 

# Set Training Mode: True = LoRA (Parameter Efficient), False = Full Fine-Tuning
USE_LORA = True 

# 2. File Paths
# ------------------------------------------------------------------------------
if TASK_ID == "T1": # Phishing Email (Binary)
    DATA_PATH = "/path/to/train_phish_email_list.csv"
    PROMPT_TEXT = "Label this as 1(phishing) or 0(safe)."

elif TASK_ID == "T2": # Malicious URL (Binary)
    DATA_PATH = "/path/to/train_malicious_phish_list.csv"
    PROMPT_TEXT = "Benign(0) or Cyberthreat(1)?"

elif TASK_ID == "T3": # Malicious URL (Multi-class)
    DATA_PATH = "/path/to/train_malicious_phish_multi_list.csv"
    PROMPT_TEXT = "Classify this URL."

elif TASK_ID == "T4": # Credit Card Fraud (Binary - Time Split)
    DATA_PATH = "/path/to/train_creditcard_timesplit_list.csv"
    PROMPT_TEXT = "fraud(1) or otherwise(0)?"

elif TASK_ID == "T5": # UNSW-NB15 (Binary)
    DATA_PATH = "/path/to/UNSW_NB15_training-set_list.csv"
    PROMPT_TEXT = "Normal(0) or Attack(1)?"

elif TASK_ID == "T6": # UNSW-NB15 (Multi-class)
    DATA_PATH = "/path/to/UNSW_NB15_training-set_multi_list.csv"
    PROMPT_TEXT = "Classify the following network event."

else:
    raise ValueError("Invalid TASK_ID")

# 3. Hyperparameters
# ------------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 96
NUM_WORKERS = 18
EPOCHS = 35
PATIENCE = 15

# Learning Rate automatically set based on LoRA usage
LEARNING_RATE = 1e-4 if USE_LORA else 3e-5 

# Output Directory Prefix
DIR_PREFIX = "./model_save_" if USE_LORA else "./model_save_full_"

# ==============================================================================
#   MAIN LOGIC
# ==============================================================================

print(f"\n==== Configuration ====")
print(f"Task: {TASK_ID}")
print(f"Mode: {'LoRA' if USE_LORA else 'Full Fine-Tuning'}")
print(f"Device: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Prompt: {PROMPT_TEXT}")
print("=======================\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Load Data & Determine Labels ---
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Auto-detect label type
if df['label'].dtype == 'object' or TASK_ID in ["T3", "T6"]:
    # Multi-class
    df['label'] = df['label'].astype('category').cat.codes
    num_labels = df['label'].nunique()
    is_multiclass = True
    print(f"Multi-class detected: {num_labels} labels.")
else:
    # Binary
    num_labels = 2
    is_multiclass = False
    print("Binary classification detected.")

# --- 2. Data Splitting ---
if TASK_ID == "T4":
    print("Applying Chronological Split (T4)...")
    val_ratio = 0.2
    split_index = int(len(df) * (1 - val_ratio))
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]
else:
    print("Applying Stratified Random Split...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")

# --- 3. Tokenizer & Model ---
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 512
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

# --- 4. Apply LoRA (If Enabled) ---
if USE_LORA:
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA Adapter Attached.")
    model.print_trainable_parameters()
else:
    print("Full Fine-Tuning Mode.")

# --- 5. Encoding Function ---
def encode_data(data_texts, data_labels, prompt):
    input_ids_list = []
    attention_masks_list = []
    truncation_count = 0

    for text in data_texts:
        full_input = f"{prompt} {text}"
        
        # Check for truncation
        tokenized_check = tokenizer(full_input, truncation=False)
        if len(tokenized_check["input_ids"]) > 512:
            truncation_count += 1

        encoded = tokenizer.encode_plus(
            full_input,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids_list.append(encoded['input_ids'])
        attention_masks_list.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    labels = torch.tensor(data_labels.values, dtype=torch.long)

    return TensorDataset(input_ids, attention_masks, labels), truncation_count

# --- 6. Prepare Datasets ---
print("Encoding datasets...")
train_dataset, train_truncs = encode_data(train_df["data"], train_df["label"], PROMPT_TEXT)
val_dataset, val_truncs = encode_data(val_df["data"], val_df["label"], PROMPT_TEXT)
print(f"Total samples truncated: {train_truncs + val_truncs}")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# --- 7. Optimizer & Scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps, num_cycles=0.5
)

# --- 8. Tracking Variables ---
metrics_map = ["loss", "auroc", "auprc", "mcc", "recall", "f1-score"]
best_metrics = {m: (float('inf') if m == "loss" else 0) for m in metrics_map}
wait_counts = {m: 0 for m in metrics_map}
early_stop_flags = {m: False for m in metrics_map}

# --- 9. Training Loop ---
for epoch in range(EPOCHS):
    start_time = time.time()
    
    # --- TRAIN ---
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, labels = batch

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)

    # --- EVAL ---
    model.eval()
    total_eval_loss = 0
    val_preds, val_labels, val_probs = [], [], []

    for batch in val_dataloader:
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_eval_loss += outputs.loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
            if is_multiclass:
                val_probs.extend(probs.cpu().numpy())
            else:
                val_probs.extend(probs[:, 1].cpu().numpy())

    # --- METRICS CALCULATION ---
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    val_probs = np.array(val_probs)
    avg_val_loss = total_eval_loss / len(val_dataloader)

    if is_multiclass:
        recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        auroc = roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro')
        auprc = average_precision_score(val_labels, val_probs, average='macro')
        f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        specificity = "N/A" 
    else:
        recall = recall_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, zero_division=0)
        specificity = sum((val_labels == 0) & (val_preds == 0)) / sum(val_labels == 0)
        auroc = roc_auc_score(val_labels, val_probs)
        auprc = average_precision_score(val_labels, val_probs)
        f1 = f1_score(val_labels, val_preds)
    
    mcc = matthews_corrcoef(val_labels, val_preds)

    # Print Stats
    duration = time.time() - start_time
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"AUROC: {auroc:.4f} | F1-score: {f1:.4f} | MCC: {mcc:.4f} | Time: {int(duration)}s")

    # --- EARLY STOPPING & SAVING ---
    current_values = {
        "loss": avg_val_loss, "auroc": auroc, "auprc": auprc, 
        "mcc": mcc, "recall": recall, "f1-score": f1
    }

    for metric in metrics_map:
        if early_stop_flags[metric]: continue

        # Check improvement
        improved = (current_values[metric] < best_metrics[metric]) if metric == "loss" else (current_values[metric] > best_metrics[metric])
        
        if improved:
            best_metrics[metric] = current_values[metric]
            wait_counts[metric] = 0
            
            # Save Model
            save_path = f"{DIR_PREFIX}{metric}/"
            if not os.path.exists(save_path): os.makedirs(save_path)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            wait_counts[metric] += 1
            if wait_counts[metric] >= PATIENCE:
                print(f"Early stopping triggered for: {metric}")
                early_stop_flags[metric] = True

    if all(early_stop_flags.values()):
        print("All metrics triggered early stopping. Exiting.")
        break

print("\nTraining Completed. Best Metrics Achieved:")
print(f"Best Loss:     {best_metrics['loss']:.4f}")
print(f"Best AUROC:    {best_metrics['auroc']:.4f}")
print(f"Best AUPRC:    {best_metrics['auprc']:.4f}")
print(f"Best MCC:      {best_metrics['mcc']:.4f}")
print(f"Best Recall:   {best_metrics['recall']:.4f}")
print(f"Best F1-score: {best_metrics['f1-score']:.4f}")